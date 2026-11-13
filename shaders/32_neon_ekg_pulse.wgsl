// 32_neon_ekg_pulse.wgsl — Multi-channel glowing synth monitor with beat sweep

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

fn sample_wave(t: f32) -> f32 {
    let x = clamp(t, 0.0, 0.999) * 63.0;
    let idx = u32(x);
    let nxt = min(idx + 1u, 63u);
    return mix(u.waveform[idx / 4u][idx & 3u], u.waveform[nxt / 4u][nxt & 3u], fract(x));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let sweep = fract(u.time * 0.55);
    let trail = fract(sweep - uv.x);
    let fade = exp(-trail * 3.2);

    var col = vec3<f32>(0.01, 0.025, 0.03);
    for (var ch = 0; ch < 3; ch = ch + 1) {
        let fch = f32(ch);
        let y_off = 0.25 + fch * 0.25;
        let w = sample_wave(fract(uv.x + fch * 0.27)) * (0.14 + 0.08 * u.bass + 0.06 * u.beat);
        let beam = exp(-abs(uv.y - (y_off + w)) * 85.0) * (0.35 + 0.95 * fade);
        let tint = mix(vec3<f32>(0.1, 1.0, 0.6), vec3<f32>(0.2, 0.7, 1.0), fch * 0.5);
        col += tint * beam;
    }

    // Sweeping cursor beam
    col += vec3<f32>(0.7, 1.0, 0.95) * exp(-abs(uv.x - sweep) * 90.0) * (0.25 + 0.5 * u.beat);
    return vec4<f32>(col / (1.0 + col * 0.3), 1.0);
}
