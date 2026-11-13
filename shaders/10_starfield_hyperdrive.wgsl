// 10_starfield_hyperdrive.wgsl — 3D warp-speed starfield & beat-driven hyperdrive tunnel

struct Uniforms {
    time: f32,
    dt: f32,
    width: f32,
    height: f32,
    bass: f32,
    mid: f32,
    treble: f32,
    beat: f32,
    overall: f32,
    aspect: f32,
    frame: f32,
    _pad: f32,
    spectrum: array<vec4<f32>, 16>,
    waveform: array<vec4<f32>, 16>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    let x = f32((vid << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(vid & 2u) * 2.0 - 1.0;
    var out: VsOut;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(x * 0.5 + 0.5, 0.5 - y * 0.5);
    return out;
}

fn hash21(p: vec2<f32>) -> f32 {
    var q = fract(p * vec2<f32>(123.34, 456.21));
    q += dot(q, q + 45.32);
    return fract(q.x * q.y);
}

fn star_layer(uv: vec2<f32>) -> vec3<f32> {
    var col = vec3<f32>(0.0);
    let gv = fract(uv) - 0.5;
    let id = floor(uv);
    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            let offs = vec2<f32>(f32(x), f32(y));
            let n = hash21(id + offs);
            let size = fract(n * 345.32);
            let p = gv - offs - vec2<f32>(n, fract(n * 34.0)) + 0.5;
            let d = length(p);
            var star = 0.025 / max(d, 0.008);
            star *= smoothstep(0.6, 0.05, d);
            let tint = sin(vec3<f32>(0.2, 0.5, 0.9) * fract(n * 2345.2) * 12.3) * 0.5 + 0.5;
            col += star * size * tint;
        }
    }
    return col;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    var uv = (in.uv - 0.5) * vec2<f32>(u.aspect, 1.0) * 2.0;
    let r = length(uv);
    let t = u.time * 0.25 + u.bass * 0.15;

    // Warp angle rotation
    let rot = u.time * 0.12 + 0.25 * u.mid;
    let c = cos(rot);
    let s = sin(rot);
    uv = vec2<f32>(c * uv.x - s * uv.y, s * uv.x + c * uv.y);

    var col = vec3<f32>(0.01, 0.015, 0.04);
    for (var i = 0.0; i < 1.0; i += 0.2) {
        let depth = fract(i + t);
        let scale = mix(18.0, 0.5, depth);
        let fade = depth * smoothstep(1.0, 0.85, depth);
        let streak = uv * scale + i * 453.2;
        col += star_layer(streak) * fade * (0.8 + u.treble * 0.8 + u.beat * 1.2);
    }

    // Radial hyperdrive blue-shift glow
    let radial_glow = exp(-r * 2.4) * (0.25 + u.bass * 0.8 + u.beat * 0.9);
    col += vec3<f32>(0.2, 0.55, 1.0) * radial_glow;
    col = col / (1.0 + col * 0.35);
    return vec4<f32>(col, 1.0);
}
