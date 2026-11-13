// 01_lorenz_attractor.wgsl — Chaotic 3D Lorenz Attractor with audio-reactive neon trails

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

fn rot_y(p: vec3<f32>, a: f32) -> vec3<f32> {
    let c = cos(a);
    let s = sin(a);
    return vec3<f32>(c * p.x + s * p.z, p.y, -s * p.x + c * p.z);
}

fn rot_x(p: vec3<f32>, a: f32) -> vec3<f32> {
    let c = cos(a);
    let s = sin(a);
    return vec3<f32>(p.x, c * p.y - s * p.z, s * p.y + c * p.z);
}

fn sd_segment(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
    return length(pa - ba * h);
}

fn palette(t: f32) -> vec3<f32> {
    let a = vec3<f32>(0.5, 0.5, 0.55);
    let b = vec3<f32>(0.5, 0.45, 0.45);
    let c = vec3<f32>(1.0, 1.0, 1.0);
    let d = vec3<f32>(0.05, 0.35, 0.72);
    return a + b * cos(6.28318 * (c * t + d));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let p = (in.uv - 0.5) * vec2<f32>(u.aspect, 1.0) * 2.0;

    // Subtle reactive radial nebula backdrop
    let r = length(p);
    let bg_wave = 0.5 + 0.5 * sin(r * 6.0 - u.time * 1.4 + u.bass * 3.0);
    var col = vec3<f32>(0.02, 0.03, 0.07) + vec3<f32>(0.03, 0.05, 0.14) * bg_wave * (0.3 + u.bass);

    // Chaotic Lorenz parameters modulated gently by audio
    let sigma = 10.0 + u.mid * 1.5;
    let rho = 28.0 + u.bass * 4.5 + u.beat * 3.0;
    let beta = 8.0 / 3.0;
    let step_dt = 0.011;

    // Warm up trajectory along time-shifted phase
    var l = vec3<f32>(
        0.1 + 0.4 * sin(u.time * 0.37),
        0.2 + 0.4 * cos(u.time * 0.29),
        12.0 + 4.0 * sin(u.time * 0.21)
    );
    for (var w = 0; w < 45; w = w + 1) {
        let dx = sigma * (l.y - l.x);
        let dy = l.x * (rho - l.z) - l.y;
        let dz = l.x * l.y - beta * l.z;
        l += vec3<f32>(dx, dy, dz) * step_dt;
    }

    let yaw = u.time * 0.45 + u.mid * 0.8;
    let pitch = 0.35 * sin(u.time * 0.31) + u.treble * 0.35;
    let scale = 0.036 + 0.008 * u.bass + 0.010 * u.beat;

    var prev_proj = vec2<f32>(0.0);
    var has_prev = false;

    let steps = 110;
    for (var i = 0; i < steps; i = i + 1) {
        let dx = sigma * (l.y - l.x);
        let dy = l.x * (rho - l.z) - l.y;
        let dz = l.x * l.y - beta * l.z;
        l += vec3<f32>(dx, dy, dz) * step_dt;

        var q = (l - vec3<f32>(0.0, 0.0, 25.0)) * scale;
        q = rot_y(q, yaw);
        q = rot_x(q, pitch);

        let persp = 1.8 / max(1.8 + q.z * 0.45, 0.35);
        let proj = q.xy * persp;

        if has_prev {
            let fi = f32(i) / f32(steps);
            let spec = sample_spec(fi);
            let d = sd_segment(p, prev_proj, proj);
            let thickness = 0.004 + 0.006 * u.beat + 0.004 * spec;
            let glow = exp(-d / max(thickness, 0.002)) * (0.35 + 1.15 * spec + 0.7 * u.beat);
            let core = smoothstep(thickness * 1.4, 0.0, d);
            let c = palette(fi * 1.3 + u.time * 0.12 + u.treble * 0.4);
            col += c * glow * 0.11 + vec3<f32>(0.9, 0.96, 1.0) * core * 0.16;
        }
        prev_proj = proj;
        has_prev = true;
    }

    // Vignette and tone mapping
    col *= smoothstep(1.9, 0.25, r);
    col = col / (1.0 + col * 0.45);
    return vec4<f32>(pow(max(col, vec3<f32>(0.0)), vec3<f32>(0.92)), 1.0);
}
