// 30_radial_scope_bloom.wgsl — Poweramp-style polar waveform ring + glowing mirror spectrum

struct Uniforms {
    time: f32, dt: f32, width: f32, height: f32,
    bass: f32, mid: f32, treble: f32, beat: f32,
    overall: f32, aspect: f32, frame: f32, _pad: f32,
    spectrum: array<vec4<f32>, 16>,
    waveform: array<vec4<f32>, 16>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;
struct VsOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };

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
    let p = (in.uv - 0.5) * vec2<f32>(u.aspect, 1.0) * 2.0;
    let r = length(p);
    let a = atan2(p.y, p.x);
    let norm_a = fract(a / 6.283185 + 0.5);
    let sym_a = abs(norm_a * 2.0 - 1.0);

    let wave = sample_wave(sym_a);
    let spec = sample_spec(sym_a);

    let r_wave = 0.48 + wave * (0.24 + 0.14 * u.bass) + 0.05 * u.beat;
    let r_spec = 0.56 + spec * 0.34;

    let g_wave = exp(-abs(r - r_wave) * (45.0 - 15.0 * u.beat));
    let g_spec = exp(-abs(r - r_spec) * 32.0) * (0.4 + spec);

    var col = vec3<f32>(0.015, 0.02, 0.05);
    col += vec3<f32>(0.15, 0.92, 1.0) * g_wave * 1.1;
    col += vec3<f32>(1.0, 0.25, 0.72) * g_spec * 0.9;
    col += vec3<f32>(0.3, 0.5, 1.0) * exp(-r * 3.0) * (0.25 + u.bass * 0.8);
    return vec4<f32>(col / (1.0 + col * 0.3), 1.0);
}
