// ============================================================================
// audio.rs — PipeWire / PulseAudio Monitor Source Loopback Tap (F32LE PCM)
// ============================================================================

use anyhow::{anyhow, Context, Result};
use libloading::Library;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use crate::dsp_fft::SharedPcmRing;

const PA_SAMPLE_FLOAT32LE: i32 = 5;
const PA_STREAM_RECORD: i32 = 2;
const SAMPLE_RATE: u32 = 48_000;
const CHANNELS: u8 = 2;
const FRAMES_PER_READ: usize = 512;

#[repr(C)]
struct PaSampleSpec {
    format: i32,
    rate: u32,
    channels: u8,
}

#[repr(C)]
struct PaBufferAttr {
    maxlength: u32,
    tlength: u32,
    prebuf: u32,
    minreq: u32,
    fragsize: u32,
}

#[repr(C)]
struct PaSimpleOpaque {
    _private: [u8; 0],
}

type FnPaSimpleNew = unsafe extern "C" fn(
    server: *const c_char,
    name: *const c_char,
    dir: i32,
    dev: *const c_char,
    stream_name: *const c_char,
    ss: *const PaSampleSpec,
    map: *const c_void,
    attr: *const PaBufferAttr,
    error: *mut c_int,
) -> *mut PaSimpleOpaque;

type FnPaSimpleRead = unsafe extern "C" fn(
    s: *mut PaSimpleOpaque,
    data: *mut c_void,
    bytes: usize,
    error: *mut c_int,
) -> c_int;

type FnPaSimpleFree = unsafe extern "C" fn(s: *mut PaSimpleOpaque);
type FnPaStrerror = unsafe extern "C" fn(error: c_int) -> *const c_char;

struct PulseSimpleHandle {
    _lib: Library,
    conn: *mut PaSimpleOpaque,
    pa_simple_read: FnPaSimpleRead,
    pa_simple_free: FnPaSimpleFree,
    pa_strerror: FnPaStrerror,
}

// `pa_simple` connection handle is exclusively owned by a single capture thread once opened.
unsafe impl Send for PulseSimpleHandle {}

impl PulseSimpleHandle {
    fn open(device: Option<&str>) -> Result<Self> {
        let lib = unsafe {
            Library::new("libpulse-simple.so.0")
                .or_else(|_| Library::new("libpulse-simple.so"))
                .context("Failed to load libpulse-simple.so.0 (PulseAudio/PipeWire client library)")?
        };

        unsafe {
            let pa_simple_new: FnPaSimpleNew = *lib
                .get::<FnPaSimpleNew>(b"pa_simple_new\0")
                .context("Missing symbol pa_simple_new")?;
            let pa_simple_read: FnPaSimpleRead = *lib
                .get::<FnPaSimpleRead>(b"pa_simple_read\0")
                .context("Missing symbol pa_simple_read")?;
            let pa_simple_free: FnPaSimpleFree = *lib
                .get::<FnPaSimpleFree>(b"pa_simple_free\0")
                .context("Missing symbol pa_simple_free")?;
            let pa_strerror: FnPaStrerror = *lib
                .get::<FnPaStrerror>(b"pa_strerror\0")
                .context("Missing symbol pa_strerror")?;

            let app_name = CString::new("Vulkan Audio Studio")?;
            let stream_name = CString::new("System Audio Monitor Tap")?;
            let dev_cstr = device.map(CString::new).transpose()?;

            let ss = PaSampleSpec {
                format: PA_SAMPLE_FLOAT32LE,
                rate: SAMPLE_RATE,
                channels: CHANNELS,
            };

            let frag_bytes = (FRAMES_PER_READ * CHANNELS as usize * std::mem::size_of::<f32>()) as u32;
            let attr = PaBufferAttr {
                maxlength: u32::MAX,
                tlength: u32::MAX,
                prebuf: u32::MAX,
                minreq: u32::MAX,
                fragsize: frag_bytes,
            };

            let mut err: c_int = 0;
            let conn = pa_simple_new(
                std::ptr::null(),
                app_name.as_ptr(),
                PA_STREAM_RECORD,
                dev_cstr
                    .as_ref()
                    .map_or(std::ptr::null(), |s| s.as_ptr()),
                stream_name.as_ptr(),
                &ss,
                std::ptr::null(),
                &attr,
                &mut err,
            );

            if conn.is_null() {
                let msg = CStr::from_ptr(pa_strerror(err)).to_string_lossy();
                return Err(anyhow!(
                    "pa_simple_new failed for '{}': {}",
                    device.unwrap_or("default"),
                    msg
                ));
            }

            Ok(Self {
                _lib: lib,
                conn,
                pa_simple_read,
                pa_simple_free,
                pa_strerror,
            })
        }
    }

    fn read_f32(&self, byte_buf: &mut [u8]) -> Result<()> {
        let mut err: c_int = 0;
        let rc = unsafe {
            (self.pa_simple_read)(
                self.conn,
                byte_buf.as_mut_ptr() as *mut c_void,
                byte_buf.len(),
                &mut err,
            )
        };
        if rc < 0 {
            let msg = unsafe { CStr::from_ptr((self.pa_strerror)(err)).to_string_lossy() };
            return Err(anyhow!("pa_simple_read error: {msg}"));
        }
        Ok(())
    }
}

impl Drop for PulseSimpleHandle {
    fn drop(&mut self) {
        if !self.conn.is_null() {
            unsafe { (self.pa_simple_free)(self.conn) };
            self.conn = std::ptr::null_mut();
        }
    }
}

/// Discovers available PipeWire/PulseAudio `.monitor` loopback sources.
pub fn list_monitor_sources() -> Vec<String> {
    let Ok(output) = Command::new("pactl")
        .args(["list", "short", "sources"])
        .output()
    else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    let text = String::from_utf8_lossy(&output.stdout);
    text.lines()
        .filter_map(|line| line.split_whitespace().nth(1))
        .filter(|name| name.ends_with(".monitor"))
        .map(ToString::to_string)
        .collect()
}

/// Resolves the default sink's `.monitor` source name when `@DEFAULT_MONITOR@` is unavailable.
fn query_default_sink_monitor() -> Option<String> {
    let output = Command::new("pactl")
        .args(["get-default-sink"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let sink = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if sink.is_empty() {
        None
    } else {
        Some(format!("{sink}.monitor"))
    }
}

/// Captures system audio continuously from PipeWire/PulseAudio monitor into `SharedPcmRing`.
pub struct SystemAudioTap {
    stop_flag: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
    active_source: String,
    last_error: Option<String>,
}

impl SystemAudioTap {
    pub fn start(pcm_ring: SharedPcmRing, preferred_source: Option<&str>) -> Self {
        let mut tap = Self {
            stop_flag: Arc::new(AtomicBool::new(false)),
            worker: None,
            active_source: String::new(),
            last_error: None,
        };
        tap.restart(pcm_ring, preferred_source);
        tap
    }

    pub fn restart(&mut self, pcm_ring: SharedPcmRing, preferred_source: Option<&str>) {
        self.stop();
        self.stop_flag = Arc::new(AtomicBool::new(false));
        if let Ok(mut ring) = pcm_ring.lock() {
            ring.clear();
        }

        let mut candidates: Vec<String> = Vec::new();
        if let Some(pref) = preferred_source.map(str::trim).filter(|s| !s.is_empty() && *s != "auto") {
            candidates.push(pref.to_string());
        }
        candidates.push("@DEFAULT_MONITOR@".to_string());
        if let Some(sink_mon) = query_default_sink_monitor() {
            if !candidates.contains(&sink_mon) {
                candidates.push(sink_mon);
            }
        }
        for mon in list_monitor_sources() {
            if !candidates.contains(&mon) {
                candidates.push(mon);
            }
        }

        let mut opened: Option<(PulseSimpleHandle, String)> = None;
        let mut last_err = None;
        for cand in &candidates {
            match PulseSimpleHandle::open(Some(cand)) {
                Ok(handle) => {
                    opened = Some((handle, cand.clone()));
                    break;
                }
                Err(e) => {
                    last_err = Some(format!("{e}"));
                }
            }
        }

        let Some((handle, source_name)) = opened else {
            self.active_source = "Disconnected".to_string();
            self.last_error = last_err.or_else(|| Some("No monitor source found".to_string()));
            return;
        };

        self.active_source = source_name;
        self.last_error = None;

        let stop = self.stop_flag.clone();
        let worker = thread::spawn(move || {
            let byte_len = FRAMES_PER_READ * CHANNELS as usize * std::mem::size_of::<f32>();
            let mut raw = vec![0u8; byte_len];
            let mut f32_buf = Vec::with_capacity(FRAMES_PER_READ * CHANNELS as usize);

            while !stop.load(Ordering::Relaxed) {
                if handle.read_f32(&mut raw).is_err() {
                    break;
                }
                f32_buf.clear();
                for chunk in raw.chunks_exact(4) {
                    f32_buf.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
                if let Ok(mut ring) = pcm_ring.lock() {
                    ring.push_interleaved(&f32_buf, CHANNELS as usize, SAMPLE_RATE);
                }
            }
        });

        self.worker = Some(worker);
    }

    pub fn stop(&mut self) {
        self.stop_flag.store(true, Ordering::Relaxed);
        if let Some(handle) = self.worker.take() {
            let _ = handle.join();
        }
    }

    pub fn active_source(&self) -> &str {
        &self.active_source
    }

    pub fn last_error(&self) -> Option<&str> {
        self.last_error.as_deref()
    }
}

impl Drop for SystemAudioTap {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pulse_simple_monitor_opens_when_daemon_running() {
        let ring = crate::dsp_fft::new_shared_pcm_ring();
        let mut tap = SystemAudioTap::start(ring.clone(), None);
        if tap.last_error().is_none() {
            std::thread::sleep(std::time::Duration::from_millis(80));
            let mut out = vec![0.0f32; 256];
            let (_sr, _age, pushed) = ring.lock().unwrap().copy_latest(&mut out);
            assert!(pushed > 0, "expected F32LE frames from PipeWire/PulseAudio monitor");
        }
        tap.stop();
    }
}
