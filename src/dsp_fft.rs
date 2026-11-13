// ============================================================================
// dsp_fft.rs — Real-Time PCM Ring Buffer, Windowed FFT & Beat Detection
// ============================================================================

use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub const FFT_SIZE: usize = 2048;
pub const SPECTRUM_BANDS: usize = 64;
pub const WAVEFORM_POINTS: usize = 64;
const RING_CAPACITY: usize = 16384;
const ENERGY_HISTORY_LEN: usize = 43;

/// Fixed-capacity ring buffer receiving mono-downmixed F32LE PCM frames from GStreamer `appsink`.
pub struct PcmRingBuffer {
    samples: VecDeque<f32>,
    sample_rate: u32,
    last_push: Instant,
    total_samples_pushed: u64,
}

impl PcmRingBuffer {
    pub fn new() -> Self {
        Self {
            samples: VecDeque::with_capacity(RING_CAPACITY),
            sample_rate: 44100,
            last_push: Instant::now(),
            total_samples_pushed: 0,
        }
    }

    /// Downmixes interleaved F32LE frames into mono and appends them into the ring buffer.
    pub fn push_interleaved(&mut self, interleaved: &[f32], channels: usize, sample_rate: u32) {
        if channels == 0 || interleaved.is_empty() {
            return;
        }
        if sample_rate > 0 {
            self.sample_rate = sample_rate;
        }

        let inv_ch = 1.0 / channels as f32;
        for frame in interleaved.chunks_exact(channels) {
            let sum: f32 = frame.iter().copied().sum();
            let mono = (sum * inv_ch).clamp(-2.0, 2.0);
            if self.samples.len() >= RING_CAPACITY {
                self.samples.pop_front();
            }
            self.samples.push_back(mono);
            self.total_samples_pushed = self.total_samples_pushed.wrapping_add(1);
        }
        self.last_push = Instant::now();
    }

    /// Copies the most recent `dst.len()` samples into `dst`, zero-padding on the left if needed.
    pub fn copy_latest(&self, dst: &mut [f32]) -> (u32, f32, u64) {
        let len = self.samples.len();
        let need = dst.len();
        if len >= need {
            let start = len - need;
            let (s1, s2) = self.samples.as_slices();
            if start < s1.len() {
                let first_take = (s1.len() - start).min(need);
                dst[..first_take].copy_from_slice(&s1[start..start + first_take]);
                if first_take < need {
                    let rem = need - first_take;
                    dst[first_take..].copy_from_slice(&s2[..rem]);
                }
            } else {
                let s2_start = start - s1.len();
                dst.copy_from_slice(&s2[s2_start..s2_start + need]);
            }
        } else {
            let pad = need - len;
            dst[..pad].fill(0.0);
            let (s1, s2) = self.samples.as_slices();
            dst[pad..pad + s1.len()].copy_from_slice(s1);
            dst[pad + s1.len()..].copy_from_slice(s2);
        }

        let age_sec = self.last_push.elapsed().as_secs_f32();
        (self.sample_rate, age_sec, self.total_samples_pushed)
    }

    pub fn clear(&mut self) {
        self.samples.clear();
    }
}

pub type SharedPcmRing = Arc<Mutex<PcmRingBuffer>>;

pub fn new_shared_pcm_ring() -> SharedPcmRing {
    Arc::new(Mutex::new(PcmRingBuffer::new()))
}

/// Snapshot of real-time audio analysis consumed by the Vulkan uniform buffer and HUD.
#[derive(Clone, Debug)]
pub struct AudioFeatures {
    pub bass: f32,
    pub mid: f32,
    pub treble: f32,
    pub beat_intensity: f32,
    pub overall: f32,
    pub spectrum: [f32; SPECTRUM_BANDS],
    pub waveform: [f32; WAVEFORM_POINTS],
    pub sample_rate: u32,
}

impl Default for AudioFeatures {
    fn default() -> Self {
        Self {
            bass: 0.0,
            mid: 0.0,
            treble: 0.0,
            beat_intensity: 0.0,
            overall: 0.0,
            spectrum: [0.0; SPECTRUM_BANDS],
            waveform: [0.0; WAVEFORM_POINTS],
            sample_rate: 44100,
        }
    }
}

/// Computes Hann-windowed FFT spectrum, frequency-band energies, and adaptive beat detection.
pub struct FftAnalyzer {
    fft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    pcm_scratch: Vec<f32>,
    fft_buf: Vec<Complex<f32>>,
    fft_scratch: Vec<Complex<f32>>,
    magnitudes: Vec<f32>,
    energy_history: VecDeque<f32>,
    last_beat_time: Instant,
    last_seen_seq: u64,
    features: AudioFeatures,
}

impl FftAnalyzer {
    pub fn new() -> Self {
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FFT_SIZE);
        let scratch_len = fft.get_inplace_scratch_len();

        let window: Vec<f32> = (0..FFT_SIZE)
            .map(|i| {
                let phase = 2.0 * std::f32::consts::PI * i as f32 / (FFT_SIZE - 1) as f32;
                0.5 * (1.0 - phase.cos())
            })
            .collect();

        Self {
            fft,
            window,
            pcm_scratch: vec![0.0; FFT_SIZE],
            fft_buf: vec![Complex::new(0.0, 0.0); FFT_SIZE],
            fft_scratch: vec![Complex::new(0.0, 0.0); scratch_len],
            magnitudes: vec![0.0; FFT_SIZE / 2],
            energy_history: VecDeque::with_capacity(ENERGY_HISTORY_LEN),
            last_beat_time: Instant::now(),
            last_seen_seq: 0,
            features: AudioFeatures::default(),
        }
    }

    pub fn update(&mut self, ring: &SharedPcmRing, dt: f32, is_playing: bool) -> &AudioFeatures {
        let dt = dt.clamp(0.001, 0.1);

        let (sample_rate, age_sec, seq) = {
            if let Ok(guard) = ring.lock() {
                guard.copy_latest(&mut self.pcm_scratch)
            } else {
                (self.features.sample_rate, 0.5, self.last_seen_seq)
            }
        };

        self.features.sample_rate = sample_rate.max(8000);

        // If playback is paused or no new frames arrived recently, decay smoothly to idle.
        if !is_playing || age_sec > 0.25 {
            self.decay(dt);
            return &self.features;
        }

        self.last_seen_seq = seq;

        // Downsample latest 1024 samples into 64 crisp waveform points
        let wave_src_start = FFT_SIZE - 1024;
        let stride = 1024 / WAVEFORM_POINTS;
        for i in 0..WAVEFORM_POINTS {
            let s = self.pcm_scratch[wave_src_start + i * stride].clamp(-1.0, 1.0);
            self.features.waveform[i] = self.features.waveform[i] * 0.35 + s * 0.65;
        }

        // Apply Hann window and compute complex FFT
        let inv_n = 2.0 / FFT_SIZE as f32;
        for i in 0..FFT_SIZE {
            self.fft_buf[i] = Complex::new(self.pcm_scratch[i] * self.window[i], 0.0);
        }
        self.fft
            .process_with_scratch(&mut self.fft_buf, &mut self.fft_scratch);

        let nyquist = self.features.sample_rate as f32 * 0.5;
        let bin_hz = self.features.sample_rate as f32 / FFT_SIZE as f32;
        let half = FFT_SIZE / 2;

        for i in 0..half {
            let mag = self.fft_buf[i].norm() * inv_n;
            // Convert linear amplitude to normalized perceptual dB scale (-65 dB .. 0 dB)
            let db = 20.0 * (mag + 1e-6).log10();
            self.magnitudes[i] = ((db + 65.0) / 65.0).clamp(0.0, 1.0);
        }

        // Map FFT bins into 64 logarithmically spaced spectrum bands (28 Hz .. 16 kHz)
        let f_min = 28.0_f32;
        let f_max = nyquist.min(16500.0).max(f_min + 100.0);
        let ratio = f_max / f_min;

        for band in 0..SPECTRUM_BANDS {
            let t0 = band as f32 / SPECTRUM_BANDS as f32;
            let t1 = (band + 1) as f32 / SPECTRUM_BANDS as f32;
            let freq0 = f_min * ratio.powf(t0);
            let freq1 = f_min * ratio.powf(t1);

            let b0 = ((freq0 / bin_hz).floor() as usize).clamp(1, half - 1);
            let b1 = ((freq1 / bin_hz).ceil() as usize).clamp(b0 + 1, half);

            let mut peak = 0.0_f32;
            let mut sum = 0.0_f32;
            for &m in &self.magnitudes[b0..b1] {
                peak = peak.max(m);
                sum += m;
            }
            let avg = sum / (b1 - b0) as f32;
            let tilt = 1.0 + 0.30 * t0;
            let target = ((peak * 0.65 + avg * 0.35) * tilt).clamp(0.0, 1.0);

            let prev = self.features.spectrum[band];
            if target > prev {
                self.features.spectrum[band] = prev * 0.25 + target * 0.75;
            } else {
                self.features.spectrum[band] = (prev * (1.0 - dt * 10.0).max(0.4)).max(target);
            }
        }

        // Compute Bass (20..250 Hz), Mid (250..2500 Hz), Treble (2500..16000 Hz)
        let bass_raw = self.band_energy_hz(20.0, 250.0, bin_hz);
        let mid_raw = self.band_energy_hz(250.0, 2500.0, bin_hz);
        let treble_raw = self.band_energy_hz(2500.0, 16000.0, bin_hz);

        let smooth_up = 1.0 - (-dt * 24.0).exp();
        let smooth_dn = 1.0 - (-dt * 9.0).exp();

        self.features.bass = ema(self.features.bass, bass_raw, smooth_up, smooth_dn);
        self.features.mid = ema(self.features.mid, mid_raw, smooth_up, smooth_dn);
        self.features.treble = ema(self.features.treble, treble_raw, smooth_up, smooth_dn);
        self.features.overall =
            (self.features.bass * 0.45 + self.features.mid * 0.35 + self.features.treble * 0.20)
                .clamp(0.0, 1.0);

        // Adaptive beat detection over bass + low-mid energy
        let instant_energy = (bass_raw * 0.78 + mid_raw * 0.22).clamp(0.0, 1.5);
        let avg_energy = if self.energy_history.is_empty() {
            instant_energy
        } else {
            self.energy_history.iter().sum::<f32>() / self.energy_history.len() as f32
        };

        let variance = if self.energy_history.is_empty() {
            0.0
        } else {
            self.energy_history
                .iter()
                .map(|&e| (e - avg_energy).powi(2))
                .sum::<f32>()
                / self.energy_history.len() as f32
        };

        if self.energy_history.len() >= ENERGY_HISTORY_LEN {
            self.energy_history.pop_front();
        }
        self.energy_history.push_back(instant_energy);

        // Decay existing beat impulse smoothly
        self.features.beat_intensity *= (-dt * 7.5).exp();

        let sensitivity = (1.32 - variance * 1.1).clamp(1.14, 1.45);
        let cooldown_ok = self.last_beat_time.elapsed().as_secs_f32() >= 0.13;
        if cooldown_ok && instant_energy > 0.14 && instant_energy > avg_energy * sensitivity {
            let strength = ((instant_energy - avg_energy) * 3.2).clamp(0.55, 1.0);
            self.features.beat_intensity = self.features.beat_intensity.max(strength);
            self.last_beat_time = Instant::now();
        }

        &self.features
    }

    fn band_energy_hz(&self, low_hz: f32, high_hz: f32, bin_hz: f32) -> f32 {
        let half = self.magnitudes.len();
        let b0 = ((low_hz / bin_hz).floor() as usize).clamp(1, half - 1);
        let b1 = ((high_hz / bin_hz).ceil() as usize).clamp(b0 + 1, half);
        let mut sum = 0.0_f32;
        let mut peak = 0.0_f32;
        for &m in &self.magnitudes[b0..b1] {
            sum += m;
            peak = peak.max(m);
        }
        let mean = sum / (b1 - b0) as f32;
        (mean * 0.55 + peak * 0.45).clamp(0.0, 1.0)
    }

    fn decay(&mut self, dt: f32) {
        let k = (-dt * 8.0).exp();
        self.features.bass *= k;
        self.features.mid *= k;
        self.features.treble *= k;
        self.features.beat_intensity *= (-dt * 9.0).exp();
        self.features.overall *= k;
        for s in &mut self.features.spectrum {
            *s *= k;
        }
        for w in &mut self.features.waveform {
            *w *= (-dt * 12.0).exp();
        }
    }
}

fn ema(current: f32, target: f32, up_alpha: f32, down_alpha: f32) -> f32 {
    let alpha = if target > current {
        up_alpha
    } else {
        down_alpha
    };
    (current + (target - current) * alpha).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_detects_sine_wave_and_bass_energy() {
        let ring = new_shared_pcm_ring();
        let sample_rate = 44100;
        let mut pcm = Vec::with_capacity(FFT_SIZE * 2);
        // Generate 80 Hz stereo bass tone
        for i in 0..FFT_SIZE {
            let v = (2.0 * std::f32::consts::PI * 80.0 * i as f32 / sample_rate as f32).sin() * 0.8;
            pcm.push(v);
            pcm.push(v);
        }
        ring.lock().unwrap().push_interleaved(&pcm, 2, sample_rate);

        let mut analyzer = FftAnalyzer::new();
        let mut bass = 0.0;
        let mut treble = 0.0;
        for _ in 0..12 {
            let feat = analyzer.update(&ring, 0.016, true);
            bass = feat.bass;
            treble = feat.treble;
        }
        assert!(bass > 0.40, "expected strong bass energy, got {}", bass);
        assert!(bass > treble, "bass {} should exceed treble {}", bass, treble);
    }
}
