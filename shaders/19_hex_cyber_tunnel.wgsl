// 19_hex_cyber_tunnel.wgsl — Sci-fi hexagonal corridor flying forward with audio speed

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

fn hex_dist(p_in: vec2<f32>) -> f32 {
    let p = abs(p_in);
    return max(dot(p, normalize(vec2<f32>(1.0, 1.73205))), p.x);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    var p = (in.uv - 0.5) * vec2<f32>(u.aspect, 1.0) * 2.0;
    let rot = u.time * 0.2 + 0.2 * sin(u.time * 0.5);
    p = vec2<f32>(cos(rot) * p.x - sin(rot) * p.y, sin(rot) * p.x + cos(rot) * p.y);

    let hd = max(hex_dist(p), 0.025);
    let depth = 1.0 / hd + u.time * (2.4 + u.bass * 2.5);
    let slice = fract(depth * 0.85);
    let ring_id = floor(depth * 0.85);

    let frame_edge = smoothstep(0.08, 0.01, abs(slice - 0.5));
    let ang = atan2(p.y, p.x);
    let spokes = smoothstep(0.06, 0.01, abs(fract(ang / 1.047197) - 0.5));

    let hue = 0.5 + 0.5 * cos(6.28318 * (ring_id * 0.09 + u.time * 0.1 + vec3<f32>(0.0, 0.33, 0.66)));
    var col = hue * (frame_edge * (0.8 + u.beat * 1.4) + spokes * 0.35 * (0.4 + u.treble));
    col += vec3<f32>(0.05, 0.85, 1.0) * exp(-hd * 4.5) * (0.3 + u.bass);
    col *= smoothstep(0.02, 0.16, hd);
    return vec4<f32>(col / (1.0 + col * 0.35), 1.0);
}
