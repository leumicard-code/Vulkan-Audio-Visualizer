// 26_firefly_meadow.wgsl — Warm drifting firefly bokeh particles reacting to treble & beat

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
    var col = mix(vec3<f32>(0.01, 0.03, 0.04), vec3<f32>(0.02, 0.06, 0.05), 1.0 - in.uv.y);

    for (var layer = 0; layer < 3; layer = layer + 1) {
        let fl = f32(layer);
        let p = uv * (5.0 + fl * 3.5) + vec2<f32>(u.time * (0.15 + fl * 0.08), -u.time * 0.1);
        let id = floor(p);
        let gv = fract(p) - 0.5;
        for (var y = -1; y <= 1; y = y + 1) {
            for (var x = -1; x <= 1; x = x + 1) {
                let offs = vec2<f32>(f32(x), f32(y));
                let n = hash22(id + offs + fl * 19.0);
                let pt = offs + 0.35 * vec2<f32>(sin(u.time * 1.3 + n.x * 6.28), cos(u.time * 1.1 + n.y * 6.28)) - gv;
                let d = length(pt);
                let blink = 0.35 + 0.65 * sin(u.time * (3.0 + n.x * 3.0) + n.y * 6.28);
                let sz = 0.04 + 0.07 * u.treble + 0.05 * u.beat;
                let glow = exp(-d / sz) * blink;
                let tint = mix(vec3<f32>(0.95, 0.85, 0.25), vec3<f32>(0.35, 1.0, 0.55), n.y);
                col += tint * glow * (0.5 + u.overall);
            }
        }
    }
    return vec4<f32>(col / (1.0 + col * 0.35), 1.0);
}
