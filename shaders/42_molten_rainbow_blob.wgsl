// Molten Rainbow Blob - Domain-warped chromatic plasma with bass ripples
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

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    var p = (in.uv - 0.5) * vec2<f32>(u.aspect, 1.0) * (2.6 - u.bass * 0.5);
    let t = u.time * 0.65;

    for (var i = 1; i < 6; i++) {
        let fi = f32(i);
        p.x += (0.55 / fi) * sin(fi * p.y + t * 1.4 + u.mid * 1.5);
        p.y += (0.55 / fi) * cos(fi * p.x + t * 1.2 + u.treble * 1.2);
    }

    let r = length(p);
    let v = sin(p.x * 2.2 + p.y * 1.8 + t) + cos(r * 3.0 - u.bass * 4.0);
    let pal = 0.5 + 0.5 * cos(vec3<f32>(0.0, 2.1, 4.2) + v * 2.2 + t * 0.8);
    let sheen = pow(max(0.5 + 0.5 * sin(v * 3.0), 0.0), 4.0) * (0.25 + u.beat * 0.65);

    let col = pal * (0.55 + u.overall * 0.65) + vec3<f32>(1.0, 0.92, 0.85) * sheen;
    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
