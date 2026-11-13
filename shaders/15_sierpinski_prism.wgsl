// 15_sierpinski_prism.wgsl — Folding triangular IFS fractal lattice

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

fn rot2(p: vec2<f32>, a: f32) -> vec2<f32> {
    let c = cos(a);
    let s = sin(a);
    return vec2<f32>(c * p.x - s * p.y, s * p.x + c * p.y);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    var p = (in.uv - 0.5) * vec2<f32>(u.aspect, 1.0) * (2.6 - 0.4 * u.beat);
    p = rot2(p, u.time * 0.25 + u.mid * 0.35);

    var scale = 1.0;
    var trap = 1e6;
    let n1 = normalize(vec2<f32>(1.0, 1.73205));
    let n2 = normalize(vec2<f32>(-1.0, 1.73205));

    for (var i = 0; i < 8; i = i + 1) {
        p.x = abs(p.x);
        p -= 2.0 * min(0.0, dot(p, n1)) * n1;
        p -= 2.0 * min(0.0, dot(p, n2)) * n2;
        p = rot2(p, 0.08 * sin(u.time + f32(i) * 0.5) + u.treble * 0.15);
        p = p * 2.0 - vec2<f32>(0.0, 1.15 + 0.12 * u.bass);
        scale *= 2.0;
        trap = min(trap, abs(length(p) - 0.45) / scale);
    }

    let d = length(p) / scale;
    let glow = exp(-d * 18.0) + exp(-trap * 28.0);
    let hue = 0.5 + 0.5 * cos(6.28318 * (trap * 6.0 + u.time * 0.2 + vec3<f32>(0.0, 0.3, 0.65)));
    var col = hue * glow * (0.45 + 0.85 * u.overall + 0.8 * u.beat);
    return vec4<f32>(col / (1.0 + col * 0.35), 1.0);
}
