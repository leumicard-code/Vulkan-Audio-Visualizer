// 08_fractal_julia_pulse.wgsl — Iridescent audio-modulated Julia set fractal

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

fn palette(t: f32) -> vec3<f32> {
    return 0.5 + 0.5 * cos(6.28318 * (t + vec3<f32>(0.15, 0.42, 0.78)));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let zoom = 1.45 - 0.22 * u.bass - 0.16 * u.beat;
    let p = (in.uv - 0.5) * vec2<f32>(u.aspect, 1.0) * 2.0 * zoom;
    let rot = u.time * 0.18;
    let cr = cos(rot);
    let sr = sin(rot);
    var z = vec2<f32>(cr * p.x - sr * p.y, sr * p.x + cr * p.y);

    let c = vec2<f32>(
        0.37 * cos(u.time * 0.42 + u.mid * 0.4) - 0.36,
        0.37 * sin(u.time * 0.39 + u.bass * 0.5) + 0.18
    );

    var iter = 0.0;
    var trap = 1e6;
    for (var i = 0; i < 64; i = i + 1) {
        z = vec2<f32>(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
        let m2 = dot(z, z);
        trap = min(trap, abs(z.x * z.y));
        if m2 > 32.0 {
            iter = f32(i) + 1.0 - log2(max(log2(m2), 1e-4));
            break;
        }
        iter = f32(i);
    }

    let norm = iter / 64.0;
    let trap_glow = exp(-trap * (8.0 - 3.5 * u.treble));
    var col = palette(norm * 2.2 + u.time * 0.15) * smoothstep(0.02, 0.9, norm);
    col += palette(trap * 2.0 + u.time * 0.3 + 0.5) * trap_glow * (0.55 + u.beat * 1.2 + u.treble);

    col = col / (1.0 + col * 0.35);
    return vec4<f32>(col, 1.0);
}
