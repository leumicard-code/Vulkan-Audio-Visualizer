// Deep Warp Starfield - Multi-layer hyperspace stars streaking on beats
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

fn hash22(p: vec2<f32>) -> vec2<f32> {
    let q = vec2<f32>(dot(p, vec2<f32>(127.1, 311.7)), dot(p, vec2<f32>(269.5, 183.3)));
    return fract(sin(q) * 43758.5453);
}

fn star_layer(uv: vec2<f32>, seed: f32) -> vec3<f32> {
    var col = vec3<f32>(0.0);
    let gv = fract(uv) - 0.5;
    let id = floor(uv);

    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let offs = vec2<f32>(f32(x), f32(y));
            let cell = id + offs;
            let rnd = hash22(cell + seed);
            let pos = offs + (rnd - 0.5) * 0.7;
            let d = length(gv - pos);
            let size = 0.025 + rnd.x * 0.04 + u.beat * 0.025;
            let star = size / max(d, 0.008);
            let flare = max(0.0, 1.0 - abs((gv.x - pos.x) * (gv.y - pos.y)) * 180.0) * smoothstep(0.45, 0.0, d);
            let hue = 0.55 + 0.45 * sin(vec3<f32>(0.0, 2.1, 4.2) + rnd.y * 6.28 + u.time * 0.5);
            col += hue * (star * 0.12 + flare * (0.25 + u.treble * 0.8)) * smoothstep(0.55, 0.05, d);
        }
    }
    return col;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let p = (in.uv - 0.5) * vec2<f32>(u.aspect, 1.0);
    let t = u.time * (0.35 + u.bass * 0.45);
    let rot_a = u.time * 0.12 + u.mid * 0.25;
    let ca = cos(rot_a);
    let sa = sin(rot_a);
    let rp = vec2<f32>(p.x * ca - p.y * sa, p.x * sa + p.y * ca);

    var col = vec3<f32>(0.01, 0.01, 0.035);
    for (var i = 0; i < 6; i++) {
        let fi = f32(i);
        let depth = fract(fi / 6.0 - t * 0.45);
        let scale = mix(18.0, 1.4, depth);
        let fade = depth * smoothstep(1.0, 0.75, depth);
        col += star_layer(rp * scale + vec2<f32>(fi * 13.7, fi * 7.3), fi * 19.0) * fade;
    }

    // Cosmic dust core
    let nebula = exp(-length(p) * 3.0) * (0.15 + u.overall * 0.35);
    col += vec3<f32>(0.25, 0.12, 0.55) * nebula;
    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
