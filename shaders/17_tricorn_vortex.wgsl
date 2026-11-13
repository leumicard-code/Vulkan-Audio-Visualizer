// 17_tricorn_vortex.wgsl — Conjugate complex Tricorn fractal with swirling orbit traps

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
    let p = (in.uv - 0.5) * vec2<f32>(u.aspect, 1.0) * (2.4 - 0.4 * u.bass);
    let ang = u.time * 0.22;
    let c = vec2<f32>(cos(ang) * p.x - sin(ang) * p.y, sin(ang) * p.x + cos(ang) * p.y)
          + vec2<f32>(-0.25, 0.0);

    var z = vec2<f32>(0.15 * sin(u.time * 0.5), 0.15 * cos(u.time * 0.4)) * (0.4 + u.mid);
    var trap = 1e6;
    var iter = 0.0;
    for (var i = 0; i < 48; i = i + 1) {
        // Tricorn uses complex conjugate z_conj = (z.x, -z.y)
        z = vec2<f32>(z.x * z.x - z.y * z.y, -2.0 * z.x * z.y) + c;
        trap = min(trap, length(z - vec2<f32>(0.5 * cos(u.time), 0.5 * sin(u.time))));
        let m2 = dot(z, z);
        if m2 > 36.0 {
            iter = f32(i) + 1.0 - log2(max(log2(m2), 1e-4));
            break;
        }
        iter = f32(i);
    }

    let norm = iter / 48.0;
    let c1 = 0.5 + 0.5 * cos(6.28318 * (norm * 2.0 + u.time * 0.2 + vec3<f32>(0.0, 0.35, 0.72)));
    let c2 = vec3<f32>(1.0, 0.28, 0.75) * exp(-trap * (5.0 - 2.0 * u.beat));
    let col = c1 * smoothstep(0.05, 0.9, norm) * (0.7 + u.bass) + c2 * (0.4 + u.treble + u.beat);
    return vec4<f32>(col / (1.0 + col * 0.3), 1.0);
}
