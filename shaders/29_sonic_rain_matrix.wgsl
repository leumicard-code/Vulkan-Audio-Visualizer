// 29_sonic_rain_matrix.wgsl — Vertical cascading digital streaks synced to 64 FFT bands

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

fn hash11(p: f32) -> f32 {
    return fract(sin(p * 127.1) * 43758.5453);
}

fn sample_spec(t: f32) -> f32 {
    let x = clamp(t, 0.0, 0.999) * 63.0;
    let idx = u32(x);
    let nxt = min(idx + 1u, 63u);
    return mix(u.spectrum[idx / 4u][idx & 3u], u.spectrum[nxt / 4u][nxt & 3u], fract(x));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let cols = 48.0;
    let cx = floor(in.uv.x * cols);
    let fx = fract(in.uv.x * cols);
    let rnd = hash11(cx);
    let spec = sample_spec(cx / cols);

    let speed = (1.2 + rnd * 1.8) * (0.7 + spec * 1.5 + u.beat * 0.6);
    let y_head = fract(in.uv.y + u.time * speed * 0.35 + rnd * 7.0);
    let trail = pow(1.0 - y_head, 4.5);
    let mask = smoothstep(0.15, 0.4, fx) * smoothstep(0.85, 0.6, fx);
    let seg = step(0.18, fract(in.uv.y * 36.0));

    let base_col = mix(vec3<f32>(0.05, 0.95, 0.45), vec3<f32>(0.15, 0.85, 1.0), spec);
    var col = vec3<f32>(0.008, 0.02, 0.015) + base_col * trail * mask * seg * (0.45 + 1.4 * spec);
    col += vec3<f32>(0.85, 1.0, 0.9) * exp(-y_head * 28.0) * mask * (0.5 + u.beat);
    return vec4<f32>(col / (1.0 + col * 0.3), 1.0);
}
