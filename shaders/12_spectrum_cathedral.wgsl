// 12_spectrum_cathedral.wgsl — Glowing 3D-perspective audio pillars + wet reflection floor

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
    return 0.5 + 0.5 * cos(6.28318 * (t + vec3<f32>(0.0, 0.35, 0.7)));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let p = (uv - 0.5) * vec2<f32>(u.aspect, 1.0);

    // Mirror vertical around floor line y = -0.15
    let floor_y = -0.14;
    let is_floor = p.y < floor_y;
    let y = select(p.y - floor_y, (floor_y - p.y) * 1.15, is_floor);
    let ripple = select(0.0, 0.012 * sin(p.x * 38.0 + u.time * 5.0) * (0.4 + u.bass), is_floor);

    let x_sym = abs(uv.x * 2.0 - 1.0);
    let bands = 32.0;
    let cell_x = fract((uv.x + ripple) * bands);
    let band_t = clamp(x_sym, 0.0, 0.999);
    let energy = sample_spec(band_t);

    let pillar_h = 0.05 + energy * 0.52 + u.beat * 0.06;
    let pillar_w = smoothstep(0.12, 0.28, cell_x) * smoothstep(0.88, 0.72, cell_x);

    var col = vec3<f32>(0.015, 0.02, 0.045);

    // Volumetric light beam behind each pillar
    let beam = exp(-abs(y) * (5.0 - 2.0 * energy)) * pillar_w * energy;
    let bar_col = palette(band_t * 0.8 + u.time * 0.06);
    col += bar_col * beam * 0.55;

    // Solid glowing pillar
    if y >= 0.0 && y < pillar_h {
        let cap = smoothstep(pillar_h - 0.025, pillar_h, y);
        let body = mix(bar_col * 0.7, vec3<f32>(0.95, 0.98, 1.0), cap);
        col = mix(col, body * (0.85 + 0.65 * energy + 0.5 * u.beat), pillar_w);
    }

    // Floor attenuation + horizon glow
    if is_floor {
        col *= 0.42 * exp(-y * 3.5);
    }
    col += vec3<f32>(0.25, 0.65, 1.0) * exp(-abs(p.y - floor_y) * 35.0) * (0.4 + u.beat);

    col = col / (1.0 + col * 0.35);
    return vec4<f32>(col, 1.0);
}
