// Cymatic Lattice - Chladni acoustic resonance nodal patterns
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
    let p = (in.uv - 0.5) * vec2<f32>(u.aspect, 1.0) * 2.2;
    let t = u.time * 0.35;
    let r = length(p);

    let n = 3.0 + floor(2.0 * (0.5 + 0.5 * sin(t * 0.7)) + u.bass * 3.0);
    let m = 5.0 + floor(3.0 * (0.5 + 0.5 * cos(t * 0.5)) + u.mid * 3.0);
    let pi = 3.14159265;

    // Chladni standing-wave nodal equation: cos(n*pi*x)*cos(m*pi*y) - cos(m*pi*x)*cos(n*pi*y) = 0
    let c1 = cos(n * pi * p.x * 0.5 + t * 0.4) * cos(m * pi * p.y * 0.5 - t * 0.3);
    let c2 = cos(m * pi * p.x * 0.5 - t * 0.3) * cos(n * pi * p.y * 0.5 + t * 0.4);
    let nodal = abs(c1 - c2);

    let s = spec(i32(clamp(r * 25.0, 0.0, 63.0)));
    let line_glow = exp(-nodal * (9.5 - u.beat * 3.5));
    let dust = smoothstep(0.32, 0.0, nodal) * (0.3 + s * 0.9);

    let c_gold = vec3<f32>(1.0, 0.82, 0.35);
    let c_aqua = vec3<f32>(0.12, 0.88, 1.0);
    let c_violet = vec3<f32>(0.55, 0.2, 0.95);

    var col = vec3<f32>(0.015, 0.015, 0.03);
    col += mix(c_aqua, c_gold, clamp(r * 0.7, 0.0, 1.0)) * line_glow * (0.7 + u.treble * 0.8);
    col += c_violet * dust * 0.45 * exp(-r * 0.65);

    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
