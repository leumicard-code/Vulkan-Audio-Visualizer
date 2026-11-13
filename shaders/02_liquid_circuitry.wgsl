// 02_liquid_circuitry.wgsl — Domain-warped liquid metal & cybernetic PCB signal pulses

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

fn noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let s = f * f * (3.0 - 2.0 * f);
    let a = hash21(i);
    let b = hash21(i + vec2<f32>(1.0, 0.0));
    let c = hash21(i + vec2<f32>(0.0, 1.0));
    let d = hash21(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, s.x), mix(c, d, s.x), s.y);
}

fn fbm(p_in: vec2<f32>) -> f32 {
    var p = p_in;
    var v = 0.0;
    var a = 0.5;
    for (var i = 0; i < 4; i = i + 1) {
        v += a * noise(p);
        p = p * 2.03 + vec2<f32>(17.1, 31.7);
        a *= 0.5;
    }
    return v;
}

fn sample_spec(t: f32) -> f32 {
    let x = clamp(t, 0.0, 0.999) * 63.0;
    let idx = u32(x);
    let nxt = min(idx + 1u, 63u);
    return mix(u.spectrum[idx / 4u][idx & 3u], u.spectrum[nxt / 4u][nxt & 3u], fract(x));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    var p = (in.uv - 0.5) * vec2<f32>(u.aspect, 1.0) * 3.2;
    let t = u.time * 0.45;

    // Domain-warped liquid substrate
    let q = vec2<f32>(
        fbm(p + vec2<f32>(t * 0.7, -t * 0.4)),
        fbm(p + vec2<f32>(-t * 0.5, t * 0.6) + vec2<f32>(5.2, 1.3))
    );
    let warp_amt = 1.1 + u.bass * 1.4 + u.beat * 0.8;
    let r = fbm(p + warp_amt * q + vec2<f32>(t * 0.3, t * 0.2));

    // Procedural PCB trace grid distorted by liquid field
    let gp = (p + q * (0.45 + 0.4 * u.mid)) * 4.5;
    let cell = floor(gp);
    let gv = fract(gp) - 0.5;
    let rnd = hash21(cell);

    var trace_dist = 1.0;
    if rnd < 0.33 {
        trace_dist = min(abs(gv.x), abs(gv.y));
    } else if rnd < 0.66 {
        trace_dist = abs(gv.x + gv.y) * 0.707;
    } else {
        trace_dist = abs(gv.x - gv.y) * 0.707;
    }

    let pad_dist = abs(length(gv) - 0.16);
    let trace_line = smoothstep(0.075, 0.015, trace_dist);
    let pad_ring = smoothstep(0.05, 0.01, pad_dist);

    // Signal pulse traveling along circuit cells
    let spec = sample_spec(fract(rnd * 3.7));
    let pulse = pow(0.5 + 0.5 * sin(u.time * (3.5 + u.treble * 4.0) - (cell.x + cell.y) * 0.9 + rnd * 6.28), 6.0);

    let liquid_col = mix(
        vec3<f32>(0.02, 0.06, 0.12),
        vec3<f32>(0.05, 0.32, 0.46),
        smoothstep(0.25, 0.85, r)
    );

    let neon_cyan = vec3<f32>(0.05, 0.95, 1.0);
    let neon_mag = vec3<f32>(1.0, 0.18, 0.65);
    let trace_col = mix(neon_cyan, neon_mag, u.beat * 0.65 + spec * 0.5);

    var col = liquid_col * (0.6 + 0.8 * u.bass);
    col += trace_col * trace_line * (0.25 + 1.4 * pulse * (0.4 + spec + u.beat));
    col += vec3<f32>(0.6, 0.95, 1.0) * pad_ring * (0.4 + u.treble * 1.2);

    // Subtle specular liquid sheen
    let sheen = smoothstep(0.68, 0.92, r) * (0.2 + 0.5 * u.mid);
    col += vec3<f32>(0.4, 0.85, 1.0) * sheen;

    col = col / (1.0 + col * 0.35);
    return vec4<f32>(col, 1.0);
}
