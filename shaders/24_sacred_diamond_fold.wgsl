// 24_sacred_diamond_fold.wgsl — Sharp 6-fold neon diamond lattice

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
    var p = (in.uv - 0.5) * vec2<f32>(u.aspect, 1.0) * 2.0;
    let r = length(p);
    var a = atan2(p.y, p.x) + u.time * 0.25;
    let sec = 6.283185 / 6.0;
    a = abs(fract(a / sec) - 0.5) * sec;
    p = vec2<f32>(cos(a), sin(a)) * r;

    var col = vec3<f32>(0.0);
    for (var i = 0; i < 6; i = i + 1) {
        let fi = f32(i);
        p = abs(p) - vec2<f32>(0.28 + 0.08 * u.bass, 0.16 + 0.05 * u.mid);
        let c = cos(0.55 + u.time * 0.2);
        let s = sin(0.55 + u.time * 0.2);
        p = vec2<f32>(c * p.x - s * p.y, s * p.x + c * p.y);
        let dm = abs(abs(p.x) + abs(p.y) - 0.32);
        let line = exp(-dm * (28.0 - 10.0 * u.beat));
        let hue = 0.5 + 0.5 * cos(6.28318 * (fi * 0.18 + u.time * 0.14 + vec3<f32>(0.0, 0.35, 0.7)));
        col += hue * line * (0.45 + u.treble + u.beat);
    }
    return vec4<f32>(col / (1.0 + col * 0.35), 1.0);
}
