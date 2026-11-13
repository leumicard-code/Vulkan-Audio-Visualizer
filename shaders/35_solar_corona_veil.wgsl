// Solar Corona Veil - Radiant solar aurora flares & magnetic coronal loops
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
    let t = u.time;

    let norm_a = fract(a / 6.2831853 + 0.5);
    let s = spec(i32(norm_a * 63.0));

    var corona = 0.0;
    for (var k = 0; k < 5; k++) {
        let fk = f32(k);
        let freq = 4.0 + fk * 3.0;
        let swirl = a * freq + t * (0.5 - fk * 0.22) + sin(r * 10.0 - t * 2.0) * (0.6 + u.mid);
        let arc = pow(max(0.5 + 0.5 * sin(swirl), 0.0), 3.0);
        let radial = exp(-abs(r - (0.24 + fk * 0.07 + s * 0.14 + u.bass * 0.08)) * 12.0);
        corona += arc * radial * (0.7 + u.treble * 0.6);
    }

    let core = exp(-max(r - 0.14 - u.beat * 0.04, 0.0) * 14.0);
    let dark_disc = smoothstep(0.13, 0.17 + u.bass * 0.02, r);

    let c1 = vec3<f32>(1.0, 0.45, 0.08);
    let c2 = vec3<f32>(1.0, 0.9, 0.35);
    let c3 = vec3<f32>(0.85, 0.15, 0.55);

    var col = vec3<f32>(0.02, 0.01, 0.03);
    col += mix(c1, c3, clamp(r * 1.8, 0.0, 1.0)) * corona * 1.1;
    col += c2 * core * dark_disc * 0.9;
    col += vec3<f32>(1.0, 0.6, 0.2) * smoothstep(0.17, 0.14, r) * (0.2 + u.beat * 0.5);

    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
