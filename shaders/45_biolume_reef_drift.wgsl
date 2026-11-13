// Biolume Reef Drift - Deep-sea glowing plasma jellyfish currents
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
    var pos = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    var out: VsOut;
    out.pos = vec4<f32>(pos[vid], 0.0, 1.0);
    out.uv = pos[vid] * 0.5 + 0.5;
    return out;
}

fn spec(idx: i32) -> f32 {
    let i = clamp(idx, 0, 63);
    return u.spectrum[i / 4][i % 4];
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let p = (in.uv - 0.5) * vec2<f32>(u.aspect, 1.0);
    let t = u.time * 0.7;

    var col = vec3<f32>(0.01, 0.025, 0.06);
    for (var i = 0; i < 7; i++) {
        let fi = f32(i);
        let center = vec2<f32>(
            sin(t * (0.4 + fi * 0.07) + fi * 1.9) * 0.55,
            cos(t * (0.3 + fi * 0.09) + fi * 2.3) * 0.35
        );
        let d = length(p - center);
        let s = spec((i * 8 + 4) % 64);
        let rad = 0.08 + s * 0.14 + u.bass * 0.05;
        let pulse = sin(d * 26.0 - t * 4.5 + fi);
        let glow = exp(-d * (6.5 - u.mid * 2.0)) * (0.55 + 0.45 * pulse);
        let ring = exp(-abs(d - rad) * 35.0) * (0.5 + u.beat * 0.8);

        let c_teal = vec3<f32>(0.05, 0.95, 0.85);
        let c_pink = vec3<f32>(1.0, 0.25, 0.68);
        let tint = mix(c_teal, c_pink, fract(fi * 0.43 + u.treble * 0.4));
        col += tint * (glow * 0.45 + ring * 0.6);
    }

    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
