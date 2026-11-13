// 13_mandelbrot_reactor.wgsl — Deep Mandelbrot coastline zoom pulsing with bass & beat

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

fn pal(t: f32) -> vec3<f32> {
    return 0.5 + 0.5 * cos(6.28318 * (t + vec3<f32>(0.0, 0.25, 0.58)));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let p = (in.uv - 0.5) * vec2<f32>(u.aspect, 1.0) * 2.0;
    let zoom = exp(-2.2 - 1.6 * sin(u.time * 0.22) - 0.45 * u.bass - 0.25 * u.beat);
    let ang = u.time * 0.14;
    let rp = vec2<f32>(cos(ang) * p.x - sin(ang) * p.y, sin(ang) * p.x + cos(ang) * p.y);
    let c = vec2<f32>(-0.7436438, 0.1318259) + rp * zoom;

    var z = vec2<f32>(0.0);
    var iter = 0.0;
    var min_d = 1e6;
    for (var i = 0; i < 80; i = i + 1) {
        z = vec2<f32>(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
        let m2 = dot(z, z);
        min_d = min(min_d, abs(length(z) - 0.5));
        if m2 > 64.0 {
            iter = f32(i) + 1.0 - log2(max(log2(m2), 1e-4));
            break;
        }
        iter = f32(i);
    }

    let t = iter / 80.0;
    let edge = smoothstep(0.02, 0.85, t);
    var col = pal(t * 3.0 + u.time * 0.12 + u.mid * 0.3) * edge * (0.8 + u.bass + u.beat);
    col += vec3<f32>(0.15, 0.75, 1.0) * exp(-min_d * 10.0) * (0.25 + u.treble + u.beat);
    return vec4<f32>(col / (1.0 + col * 0.3), 1.0);
}
