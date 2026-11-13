// 04_waveform_oscilloscope.wgsl — Dual-trace phosphor oscilloscope + floor FFT bars

struct Uniforms {
    time: f32,
    dt: f32,
    width: f32,
    height: f32,
    bass: f32,
    mid: f32,
    treble: f32,
    beat: f32,
    overall: f32,
    aspect: f32,
    frame: f32,
    _pad: f32,
    spectrum: array<vec4<f32>, 16>,
    waveform: array<vec4<f32>, 16>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    let x = f32((vid << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(vid & 2u) * 2.0 - 1.0;
    var out: VsOut;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(x * 0.5 + 0.5, 0.5 - y * 0.5);
    return out;
}

fn sample_spec(t: f32) -> f32 {
    let x = clamp(t, 0.0, 0.999) * 63.0;
    let idx = u32(x);
    let nxt = min(idx + 1u, 63u);
    return mix(u.spectrum[idx / 4u][idx & 3u], u.spectrum[nxt / 4u][nxt & 3u], fract(x));
}

fn sample_wave(t: f32) -> f32 {
    let x = clamp(t, 0.0, 0.999) * 63.0;
    let idx = u32(x);
    let nxt = min(idx + 1u, 63u);
    return mix(u.waveform[idx / 4u][idx & 3u], u.waveform[nxt / 4u][nxt & 3u], fract(x));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let p = uv * 2.0 - 1.0;

    // Subtle CRT graticule grid
    let gx = abs(fract(uv.x * 16.0) - 0.5);
    let gy = abs(fract(uv.y * 10.0) - 0.5);
    let grid = smoothstep(0.03, 0.0, min(gx, gy)) * 0.08;
    var col = vec3<f32>(0.015, 0.025, 0.045) + vec3<f32>(0.05, 0.16, 0.25) * grid;

    // Bottom mirrored FFT spectrum reflection
    let bar_x = fract(uv.x * 64.0);
    let spec_h = sample_spec(uv.x) * 0.42;
    let floor_dist = 1.0 - uv.y;
    if floor_dist < spec_h {
        let bar_mask = smoothstep(0.08, 0.25, bar_x) * smoothstep(0.92, 0.75, bar_x);
        let grad = mix(vec3<f32>(0.0, 0.75, 1.0), vec3<f32>(0.95, 0.2, 0.75), floor_dist / max(spec_h, 0.01));
        col += grad * bar_mask * 0.45;
    }

    // Primary real-time PCM oscilloscope trace
    let w0 = sample_wave(uv.x) * (0.48 + 0.25 * u.bass);
    let d0 = abs(p.y - w0);
    let beam0 = exp(-d0 * (48.0 - 16.0 * u.beat)) + 0.5 * exp(-d0 * 12.0);
    let core0 = smoothstep(0.012, 0.002, d0);

    // Secondary phase-shifted harmonic trace
    let w1 = sample_wave(fract(uv.x + 0.18)) * 0.36 + 0.08 * sin(uv.x * 18.0 + u.time * 6.0) * u.treble;
    let d1 = abs(p.y - w1);
    let beam1 = exp(-d1 * 40.0);

    // Circular Lissajous radar overlay in center
    let pc = (uv - 0.5) * vec2<f32>(u.aspect, 1.0) * 2.0;
    let ang = atan2(pc.y, pc.x) / 6.283185 + 0.5;
    let rad_target = 0.38 + sample_wave(ang) * 0.22 + u.beat * 0.08;
    let d_ring = abs(length(pc) - rad_target);
    let ring_glow = exp(-d_ring * 55.0) * (0.4 + 0.9 * u.mid + u.beat);

    col += vec3<f32>(0.12, 0.95, 0.85) * beam0 * 0.9;
    col += vec3<f32>(0.9, 1.0, 1.0) * core0 * 0.95;
    col += vec3<f32>(1.0, 0.25, 0.78) * beam1 * (0.45 + u.treble * 0.8);
    col += vec3<f32>(0.35, 0.65, 1.0) * ring_glow * 0.65;

    // Subtle CRT scanline shimmer
    let scan = 0.92 + 0.08 * sin(uv.y * u.height * 1.5);
    col *= scan;
    col = col / (1.0 + col * 0.3);
    return vec4<f32>(col, 1.0);
}
