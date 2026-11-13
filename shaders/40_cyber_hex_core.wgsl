// Cyber Hex Core - Pulsing hexagonal honeycomb matrix with spectrum waves
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

fn spec(idx: i32) -> f32 {
    let i = clamp(idx, 0, 63);
    return u.spectrum[i / 4][i % 4];
}

fn hex_dist(p: vec2<f32>) -> f32 {
    let q = abs(p);
    return max(dot(q, normalize(vec2<f32>(1.0, 1.7320508))), q.x);
}

fn hex_coords(uv: vec2<f32>) -> vec4<f32> {
    let r = vec2<f32>(1.0, 1.7320508);
    let h = r * 0.5;
    let a = uv - r * floor(uv / r + 0.5);
    let b = (uv - h) - r * floor((uv - h) / r + 0.5);
    let gv = select(b, a, dot(a, a) < dot(b, b));
    let id = uv - gv;
    let edge = 0.5 - hex_dist(gv);
    return vec4<f32>(edge, length(id), id.x, id.y);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let p = (in.uv - 0.5) * vec2<f32>(u.aspect, 1.0);
    let zoom = 9.0 - u.bass * 1.6;
    let hc = hex_coords(p * zoom);

    let edge = hc.x;
    let cell_r = hc.y / zoom;
    let b_idx = i32(clamp(cell_r * 85.0, 0.0, 63.0));
    let s = spec(b_idx);

    let ring = 0.5 + 0.5 * sin(cell_r * 16.0 - u.time * 4.0 - u.bass * 3.0);
    let border = smoothstep(0.08 + u.beat * 0.04, 0.01, edge);
    let fill = smoothstep(0.02, 0.25, edge) * (s * 0.9 + ring * 0.25);

    let c_neon = mix(vec3<f32>(0.0, 0.9, 1.0), vec3<f32>(1.0, 0.15, 0.75), clamp(cell_r * 1.6 + u.treble * 0.5, 0.0, 1.0));
    var col = vec3<f32>(0.015, 0.015, 0.035);
    col += c_neon * border * (0.45 + s * 0.8 + u.beat * 0.5);
    col += c_neon * 0.35 * fill;

    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
