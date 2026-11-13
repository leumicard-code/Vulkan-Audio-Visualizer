// 16_apollonian_foam.wgsl — Circle-inversion Apollonian gasket fractal

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

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    var p = (in.uv - 0.5) * vec2<f32>(u.aspect, 1.0) * (2.2 - 0.35 * u.bass);
    let ang = u.time * 0.18;
    p = vec2<f32>(cos(ang) * p.x - sin(ang) * p.y, sin(ang) * p.x + cos(ang) * p.y);

    var scale = 1.0;
    var acc = 0.0;
    for (var i = 0; i < 8; i = i + 1) {
        p = 2.0 * fract(p * 0.5 + 0.5) - 1.0;
        let k = (1.18 + 0.14 * sin(u.time * 0.7) + 0.18 * u.beat) / max(dot(p, p), 0.08);
        p *= k;
        scale *= k;
        acc += exp(-abs(length(p) - 0.95) * 8.0);
    }

    let dist = abs(p.y) / max(scale, 1.0);
    let edge = exp(-dist * 120.0);
    let pal = 0.5 + 0.5 * cos(6.28318 * (acc * 0.12 + u.time * 0.15 + vec3<f32>(0.1, 0.42, 0.78)));
    let col = pal * (0.25 * acc + 1.1 * edge) * (0.6 + u.mid + u.beat);
    return vec4<f32>(col / (1.0 + col * 0.35), 1.0);
}
