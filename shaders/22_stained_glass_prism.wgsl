// 22_stained_glass_prism.wgsl — Mirrored 10-fold Voronoi crystal stained glass

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

fn hash22(p: vec2<f32>) -> vec2<f32> {
    let q = vec2<f32>(dot(p, vec2<f32>(127.1, 311.7)), dot(p, vec2<f32>(269.5, 183.3)));
    return fract(sin(q) * 43758.5453);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    var uv = (in.uv - 0.5) * vec2<f32>(u.aspect, 1.0) * 2.0;
    let r = length(uv);
    var a = atan2(uv.y, uv.x) + u.time * 0.15;
    let sector = 6.283185 / 10.0;
    a = abs(fract(a / sector) - 0.5) * sector;
    uv = vec2<f32>(cos(a), sin(a)) * r * (3.2 - 0.5 * u.bass) + vec2<f32>(u.time * 0.25, 0.0);

    let id = floor(uv);
    let gv = fract(uv);
    var md1 = 8.0;
    var md2 = 8.0;
    var best_cell = vec2<f32>(0.0);

    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            let offs = vec2<f32>(f32(x), f32(y));
            let n = hash22(id + offs);
            let pt = offs + 0.5 + 0.42 * sin(u.time * 1.4 + 6.2831 * n) - gv;
            let d = length(pt);
            if d < md1 {
                md2 = md1;
                md1 = d;
                best_cell = n;
            } else if d < md2 {
                md2 = d;
            }
        }
    }

    let edge = smoothstep(0.06, 0.01, md2 - md1);
    let facet = 0.5 + 0.5 * cos(6.28318 * (best_cell.x + u.time * 0.12 + vec3<f32>(0.0, 0.35, 0.72)));
    var col = facet * (0.45 + 0.7 * (1.0 - md1) + 0.6 * u.mid);
    col += vec3<f32>(0.9, 0.95, 1.0) * edge * (0.45 + u.treble + u.beat);
    return vec4<f32>(col / (1.0 + col * 0.3), 1.0);
}
