// 23_liquid_mirror_lotus.wgsl — 12-fold blooming liquid lotus kaleidoscope

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
    let p0 = (in.uv - 0.5) * vec2<f32>(u.aspect, 1.0) * 2.0;
    let r = length(p0);
    var a = atan2(p0.y, p0.x) - u.time * 0.18;
    let sec = 6.283185 / 12.0;
    a = abs(fract(a / sec) - 0.5) * sec;

    var p = vec2<f32>(cos(a), sin(a)) * r;
    var col = vec3<f32>(0.0);
    for (var i = 0; i < 5; i = i + 1) {
        let fi = f32(i);
        p = vec2<f32>(
            sin(p.x * 2.2 + u.time + fi) + cos(p.y * 1.9 - u.time * 0.7),
            cos(p.y * 2.3 - u.time * 0.8 + fi) - sin(p.x * 1.7)
        ) * (0.72 + 0.15 * u.bass);
        let d = abs(length(p) - 0.55 - 0.2 * u.beat);
        let glow = exp(-d * 10.0);
        let hue = 0.5 + 0.5 * cos(6.28318 * (fi * 0.22 + r * 0.5 + u.time * 0.1 + vec3<f32>(0.1, 0.45, 0.82)));
        col += hue * glow * (0.35 + 0.5 * u.overall + 0.5 * u.beat);
    }
    return vec4<f32>(col / (1.0 + col * 0.35), 1.0);
}
