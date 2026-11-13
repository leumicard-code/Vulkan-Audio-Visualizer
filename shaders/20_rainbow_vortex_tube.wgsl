// 20_rainbow_vortex_tube.wgsl — MilkDrop psychedelic rainbow spiral tunnel

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
    let center = vec2<f32>(sin(u.time * 0.8), cos(u.time * 0.6)) * 0.15 * (0.5 + u.mid);
    let p = (in.uv - 0.5) * vec2<f32>(u.aspect, 1.0) * 2.0 - center;
    let r = max(length(p), 0.02);
    let a = atan2(p.y, p.x);

    let spiral = a * 4.0 + 3.0 / r - u.time * (3.2 + u.bass * 3.0);
    let bands = 0.5 + 0.5 * sin(spiral + sin(a * 6.0 - u.time * 2.0) * (0.5 + u.treble));
    let rainbow = 0.5 + 0.5 * cos(6.28318 * (spiral * 0.08 + vec3<f32>(0.0, 0.33, 0.67)));

    var col = rainbow * pow(bands, 1.4) * (0.7 + 0.8 * u.overall + 0.8 * u.beat);
    col *= smoothstep(0.02, 0.2, r);
    return vec4<f32>(col / (1.0 + col * 0.3), 1.0);
}
