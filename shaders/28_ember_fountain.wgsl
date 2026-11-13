// 28_ember_fountain.wgsl — Rising incandescent sparks driven by bass energy

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
    let uv = (in.uv - 0.5) * vec2<f32>(u.aspect, 1.0);
    var col = vec3<f32>(0.03, 0.01, 0.01) + vec3<f32>(0.25, 0.05, 0.01) * (1.0 - in.uv.y) * (0.4 + u.bass);

    let rise = u.time * (1.4 + u.bass * 1.5);
    for (var l = 0; l < 3; l = l + 1) {
        let fl = f32(l);
        let p = uv * (7.0 + fl * 5.0) + vec2<f32>(sin(uv.y * 4.0 + u.time) * 0.35, -rise * (1.0 + 0.3 * fl));
        let id = floor(p);
        let gv = fract(p) - 0.5;
        for (var y = -1; y <= 1; y = y + 1) {
            for (var x = -1; x <= 1; x = x + 1) {
                let offs = vec2<f32>(f32(x), f32(y));
                let n = hash22(id + offs + fl * 41.0);
                let pt = offs + 0.38 * (n - 0.5) - gv;
                let d = length(pt * vec2<f32>(1.3, 0.7));
                let spark = exp(-d / (0.035 + 0.045 * u.beat + 0.03 * u.treble));
                let heat = mix(vec3<f32>(1.0, 0.25, 0.05), vec3<f32>(1.0, 0.9, 0.4), n.x);
                col += heat * spark * (0.55 + u.overall);
            }
        }
    }
    return vec4<f32>(col / (1.0 + col * 0.35), 1.0);
}
