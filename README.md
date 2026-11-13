<img width="650" height="603" alt="image" src="https://github.com/user-attachments/assets/d64c2afd-5602-43b8-b4ea-bbda91f3328d" />
<img width="1410" height="963" alt="image" src="https://github.com/user-attachments/assets/466d7716-e199-4d4d-a0cf-691d98320878" />

# Vulkan Audio Visualizer

Lightweight, GPU-accelerated Linux desktop system-audio visualizer built in **Rust** with **wgpu (Vulkan)**, **FLTK**, **rustfft**, and **PipeWire / PulseAudio Monitor** capture, featuring **52 built-in WGSL shaders** plus a native **MilkDrop (`.milk`)** preset engine (EEL2 VM + HLSL-to-GLSL transpiler).

## Architecture

1. **System Audio Tap (`src/audio.rs`)**
   - Dynamically loads `libpulse-simple.so.0` (`pa_simple_new` / `pa_simple_read`) in `PA_SAMPLE_FLOAT32LE` stereo 48 kHz low-latency mode.
   - Listens directly to `@DEFAULT_MONITOR@` (or any selectable `.monitor` loopback source), reacting to Spotify, browsers, VLC, or any desktop audio without needing a built-in media player.
2. **Real-Time FFT & Beat Engine (`src/dsp_fft.rs`)**
   - 2048-point Hann-windowed `rustfft` feeding 64 logarithmic spectrum bands, 64-point time-domain waveform, bass/mid/treble energy, and adaptive beat impulse detection.
3. **Vulkan Shader & MilkDrop Engine (`src/visualizer.rs`, `src/milkdrop/`)**
   - Attaches a `wgpu::Surface` (`Backends::VULKAN`) directly onto the embedded native FLTK panel via `raw-window-handle`.
   - Runs **52 built-in `.wgsl` shaders** out of the box (hot-reloadable with `F5`) and executes external **MilkDrop (`.milk`)** presets natively via a multi-pass ping-pong warp/composite pipeline, per-frame/per-vertex **EEL2** expression evaluator, and **HLSL-to-GLSL** shader translator.
4. **Clean Full-Canvas Interface & Fade-Out Overlay (`src/overlay.rs`)**
   - Full-height visualization canvas with a clean translucent info overlay & watermark that softly fades out on idle and returns on preset switch or `I` press.

## Controls

- **Top Menu Bar:** `File` (Monitor Source selection, Hot-Reload, Quit), `Visualizer` (Fullscreen, Prev/Next/Random, Random MilkDrop `M`, Toggle Overlay `I`, Auto-Cycle, UI Theme), `Presets` (`Browse All Presets...` `Ctrl+P`, `Import MilkDrop Folder...`, category filters), `Help` (`About Vulkan Audio Visualizer...`).
- **Fullscreen:** `F11`, `F`, or double-click the visualizer surface (`Esc` exits fullscreen).
- **Preset Navigation:** `Ctrl+P` or `/` to open the searchable Preset Browser, `[` / `]` or `Space` to step, `N` / `R` for random, `M` for random MilkDrop preset, `I` to show/hide overlay, `F5` to reload custom `.wgsl` files.

## Import MilkDrop (`.milk`) Preset Packs

Open **Presets → Import MilkDrop Folder...** in the menu bar to load any `.milk` preset collection (and matching `textures` directory) from disk — including inside Flatpak via the native XDG Desktop Portal directory picker.

## Build & Run

```bash
cargo test -- --nocapture
cargo build --release
./target/release/vulkan_audio_studio
```

## Trademark & License Notice

- Source code is licensed under the **MIT License** (see [`LICENSE`](LICENSE)).
- *Vulkan* and the Vulkan logo are registered trademarks of the Khronos Group Inc. (public domain SVG via Wikimedia Commons used strictly as an informational attribution graphic inside the About dialog and fade-out overlay). This project is an independent open-source audio visualizer and is not affiliated with, endorsed by, or sponsored by the Khronos Group.
