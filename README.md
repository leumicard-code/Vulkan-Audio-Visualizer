


<img width="650" height="603" alt="image" src="https://github.com/user-attachments/assets/d64c2afd-5602-43b8-b4ea-bbda91f3328d" />
<img width="1410" height="963" alt="image" src="https://github.com/user-attachments/assets/466d7716-e199-4d4d-a0cf-691d98320878" />



# Vulkan Audio Visualizer

Lightweight, high-FPS Linux desktop system-audio visualizer built in **Rust** with **wgpu (Vulkan)**, **FLTK** (`raw-window-handle` 0.6 across X11 & Wayland), **rustfft**, and **PipeWire / PulseAudio Monitor** capture.

## Architecture

1. **System Audio Tap (`src/audio.rs`)**
   - Dynamically loads `libpulse-simple.so.0` (`pa_simple_new` / `pa_simple_read`) in `PA_SAMPLE_FLOAT32LE` stereo 48 kHz low-latency mode (`fragsize = 4096` bytes).
   - Listens directly to `@DEFAULT_MONITOR@` (or any selectable `.monitor` loopback source), reacting to Spotify, YouTube, browsers, VLC, or any playing desktop audio without needing a built-in player.
2. **Real-Time FFT & Beat Engine (`src/dsp_fft.rs`)**
   - 2048-point Hann-windowed `rustfft` feeding 64 logarithmic spectrum bands, 64-point time-domain waveform, bass/mid/treble energy, and adaptive beat impulse detection.
3. **Vulkan Shader Engine (`src/visualizer.rs`)**
   - Attaches a `wgpu::Surface` (`Backends::VULKAN`) directly onto the embedded native FLTK panel via `HasWindowHandle + HasDisplayHandle`.
   - Hot-reloads `.wgsl` shaders safely on disk edits (`F5` or automatic 250 ms polling).
4. **9784 Built-in MilkDrop / Poweramp-Style Shaders (`shaders/`)**
   - Spanning **Chaos**, **Fractal**, **Tunnel**, **Kaleidoscope**, **Particle**, **Waveform**, **Aurora**, **Neon**, **Plasma**, **Starfield**, and **Mandala**.

## Controls

- **Top Menu Bar:** `File` (Monitor Source selection, Hot-Reload, Quit), `Visualizer` (Fullscreen, Prev/Next/Random, Auto-Cycle, UI Theme), `Presets` (categorized quick jump), `Help`.
- **Fullscreen:** `F11`, `F`, or double-click the visualizer surface (`Esc` exits fullscreen).
- **Preset Navigation:** Instant search box + scrollable bottom preset list, `[` / `]` or `Space` to step, `N` / `R` for random, `F5` to reload modified `.wgsl` files.

## Build & Test

```bash
cargo test -- --nocapture
cargo build --release
./target/release/vulkan_audio_studio
```
