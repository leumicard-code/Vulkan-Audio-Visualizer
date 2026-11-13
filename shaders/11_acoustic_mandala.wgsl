// 11_acoustic_mandala.wgsl — Cymatics sacred-geometry rings driven by 64 FFT bands

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

fn palette(t: f32) -> vec3<f32> {
    return 0.5 + 0.5 * cos(6.28318 * (t + vec3<f32>(0.0, 0.25, 0.55)));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let p = (in.uv - 0.5) * vec2<f32>(u.aspect, 1.0) * 2.0;
    let r = length(p);
    let a = atan2(p.y, p.x);

    var col = vec3<f32>(0.015, 0.015, 0.035);

    for (var i = 1; i <= 7; i = i + 1) {
        let fi = f32(i);
        let sym = 4.0 + fi * 2.0;
        let spec = sample_spec(fi / 8.0);
        let rot = u.time * (0.25 + 0.06 * fi) * select(-1.0, 1.0, (i & 1) == 0);
        let petal = cos((a + rot) * sym) * (0.04 + 0.07 * spec + 0.03 * u.beat);
        let target_r = fi * 0.12 + petal + u.bass * 0.03;
        let d = abs(r - target_r);
        let ring = exp(-d * (65.0 - 20.0 * u.beat)) * (0.4 + 1.3 * spec + 0.6 * u.beat);
        col += palette(fi * 0.14 + u.time * 0.08 + spec * 0.2) * ring;
    }

    // Interference cymatics field
    let wave = sin(r * 24.0 - u.time * 4.0) * cos(a * 12.0 + u.time * 1.5);
    col += vec3<f32>(0.2, 0.6, 1.0) * max(wave, 0.0) * (0.08 + 0.25 * u.treble) * exp(-r * 1.1);

    col = col / (1.0 + col * 0.35);
    return vec4<f32>(col, 1.0);
}
