// Electric Arc Storm - High-voltage plasma arcs crackling to treble & beats
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

fn hash11(n: f32) -> f32 {
    return fract(sin(n) * 43758.5453123);
}

fn noise1(x: f32) -> f32 {
    let i = floor(x);
    let f = fract(x);
    let u_s = f * f * (3.0 - 2.0 * f);
    return mix(hash11(i), hash11(i + 1.0), u_s) * 2.0 - 1.0;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let p = (in.uv - 0.5) * vec2<f32>(u.aspect, 1.0);
    let t = u.time;
    let r = length(p);

    var col = vec3<f32>(0.015, 0.02, 0.055);
    for (var i = 0; i < 6; i++) {
        let fi = f32(i);
        let ang = fi * 1.0472 + t * 0.35;
        let ca = cos(ang);
        let sa = sin(ang);
        let q = vec2<f32>(p.x * ca - p.y * sa, p.x * sa + p.y * ca);

        let jitter = noise1(q.x * 9.0 + t * 14.0 + fi * 17.0) * (0.06 + u.treble * 0.14)
                   + noise1(q.x * 23.0 - t * 22.0 + fi * 7.0) * (0.025 + u.beat * 0.05);
        let s = spec((i * 10) % 64);
        let d = abs(q.y - jitter * (0.6 + s));
        let arc = (0.004 + u.beat * 0.006 + s * 0.005) / max(d, 0.0012);

        let tint = mix(vec3<f32>(0.25, 0.65, 1.0), vec3<f32>(0.85, 0.45, 1.0), fract(fi * 0.31));
        col += tint * arc * 0.25;
    }

    // Tesla coil center spark
    let core = exp(-r * (10.0 - u.bass * 4.5)) * (0.7 + u.beat * 0.9);
    col += vec3<f32>(0.75, 0.92, 1.0) * core;
    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
