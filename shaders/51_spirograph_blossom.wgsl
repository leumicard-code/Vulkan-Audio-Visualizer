// Spirograph Blossom - Parametric rose curves & glowing orbital harmonics
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
    let r = length(p);
    let a = atan2(p.y, p.x);
    let t = u.time * 0.6;

    var col = vec3<f32>(0.015, 0.015, 0.03);
    let s_bass = spec(2);
    let s_mid = spec(20);
    let s_hi = spec(44);

    for (var k = 0; k < 7; k++) {
        let fk = f32(k);
        let n = 3.0 + fk;
        let phase = t * (0.4 + fk * 0.12) * select(-1.0, 1.0, (k % 2) == 0);
        let amp = 0.22 + 0.12 * sin(t + fk) + u.bass * 0.08 + s_mid * 0.08;
        let target_r = amp * abs(sin(n * a * 0.5 + phase)) + 0.08;
        let d = abs(r - target_r);
        let glow = (0.003 + u.beat * 0.004 + s_hi * 0.003) / max(d, 0.0014);

        let pal = 0.5 + 0.5 * cos(vec3<f32>(0.0, 2.09, 4.18) + fk * 0.7 + t * 0.9);
        col += pal * glow * (0.18 + s_bass * 0.12);
    }

    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
