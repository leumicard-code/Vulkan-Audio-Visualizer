// 03_plasma_tunnel.wgsl — MilkDrop-style polar warp plasma tunnel

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

fn sample_spec(t: f32) -> f32 {
    let x = clamp(t, 0.0, 0.999) * 63.0;
    let idx = u32(x);
    let nxt = min(idx + 1u, 63u);
    return mix(u.spectrum[idx / 4u][idx & 3u], u.spectrum[nxt / 4u][nxt & 3u], fract(x));
}

fn palette(t: f32) -> vec3<f32> {
    return 0.5 + 0.5 * cos(6.28318 * (t + vec3<f32>(0.0, 0.33, 0.67)));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let drift = vec2<f32>(sin(u.time * 0.7), cos(u.time * 0.53)) * (0.14 + 0.18 * u.mid);
    let p = (in.uv - 0.5) * vec2<f32>(u.aspect, 1.0) * 2.0 - drift;

    let r = max(length(p), 0.035);
    let a = atan2(p.y, p.x);

    let speed = u.time * (1.8 + u.bass * 1.5) + u.beat * 0.35;
    let inv_r = 0.65 / r + speed;
    let twist = a + 0.55 * sin(inv_r * 0.65 - u.time) * (0.5 + u.mid);

    let sector = fract(twist / 6.283185 + 0.5);
    let spec = sample_spec(abs(sector * 2.0 - 1.0));

    let p1 = sin(inv_r * 4.5 + sin(twist * 5.0 + u.time));
    let p2 = cos(twist * 7.0 - inv_r * 2.8 + u.treble * 4.0);
    let p3 = sin((p.x * p.x - p.y * p.y) * 3.0 + u.time * 1.7);
    let plasma = (p1 + p2 + p3) * 0.3333;

    let rib = smoothstep(0.88, 0.98, abs(sin(inv_r * 3.0))) * (0.4 + spec * 1.4 + u.beat);
    let hue = plasma * 0.45 + inv_r * 0.08 + u.time * 0.07;

    var col = palette(hue) * (0.45 + 0.65 * abs(plasma) + 0.5 * spec);
    col += palette(hue + 0.35) * rib * 1.3;

    // Center warp core glow on beat
    let core = exp(-r * 3.5) * (0.4 + u.bass * 1.2 + u.beat * 1.5);
    col += vec3<f32>(0.8, 0.9, 1.0) * core;

    // Depth fog near center singularity
    col *= smoothstep(0.02, 0.22, r);
    col = col / (1.0 + col * 0.35);
    return vec4<f32>(col, 1.0);
}
