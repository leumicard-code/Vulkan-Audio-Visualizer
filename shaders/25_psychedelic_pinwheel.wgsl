// 25_psychedelic_pinwheel.wgsl — Rotating spiral pinwheel blades with RGB dispersion

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
    let p = (in.uv - 0.5) * vec2<f32>(u.aspect, 1.0) * 2.0;
    let r = length(p);
    let a = atan2(p.y, p.x);

    let twist = r * (5.0 + 3.0 * u.bass) - u.time * 2.0;
    let f_r = sin(a * 9.0 + twist + 0.25 * u.beat);
    let f_g = sin(a * 9.0 + twist + 0.15);
    let f_b = sin(a * 9.0 + twist + 0.32 - 0.25 * u.treble);

    let ring = sin(r * 16.0 - u.time * 4.0) * 0.5 + 0.5;
    let col = vec3<f32>(abs(f_r), abs(f_g), abs(f_b))
            * (0.45 + 0.65 * ring)
            * (0.55 + 0.75 * u.overall + 0.65 * u.beat);
    return vec4<f32>(col / (1.0 + col * 0.3), 1.0);
}
