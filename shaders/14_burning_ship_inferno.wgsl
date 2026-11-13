// 14_burning_ship_inferno.wgsl — Burning Ship fractal with fiery plasma coastline

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
    let scale = 0.065 * (1.0 - 0.25 * u.bass - 0.15 * u.beat);
    let c = vec2<f32>(-1.762, -0.028) + vec2<f32>(p.x, -p.y) * scale
          + vec2<f32>(sin(u.time * 0.3), cos(u.time * 0.25)) * 0.008;

    var z = vec2<f32>(0.0);
    var iter = 0.0;
    for (var i = 0; i < 64; i = i + 1) {
        z = vec2<f32>(abs(z.x), abs(z.y));
        z = vec2<f32>(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
        let m2 = dot(z, z);
        if m2 > 32.0 {
            iter = f32(i) + 1.0 - log2(max(log2(m2), 1e-4));
            break;
        }
        iter = f32(i);
    }

    let t = clamp(iter / 64.0, 0.0, 1.0);
    let fire = vec3<f32>(
        pow(t, 0.55) * 1.4,
        pow(t, 1.35) * (0.95 + 0.5 * u.mid),
        pow(t, 2.6) * (0.5 + 1.2 * u.treble)
    );
    let boost = 0.7 + 0.9 * u.bass + 0.9 * u.beat;
    let col = fire * smoothstep(0.03, 0.9, t) * boost;
    return vec4<f32>(col / (1.0 + col * 0.35), 1.0);
}
