// 07_neon_cyber_grid.wgsl — Outrun synthwave horizon grid + reactive spectrum peaks

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

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let p = (in.uv - 0.5) * vec2<f32>(u.aspect, 1.0) * 2.0;
    let horizon = -0.05;

    var col = vec3<f32>(0.02, 0.01, 0.06);

    if p.y > horizon {
        // Sky gradient + Retro Sun
        let sky_t = clamp((p.y - horizon) * 0.9, 0.0, 1.0);
        col = mix(vec3<f32>(0.22, 0.04, 0.34), vec3<f32>(0.02, 0.02, 0.09), sky_t);

        let sun_pos = vec2<f32>(0.0, 0.34);
        let sun_r = 0.36 + 0.06 * u.bass + 0.05 * u.beat;
        let sd = length(p - sun_pos);
        if sd < sun_r {
            let stripe = step(0.18, fract((p.y - horizon) * 14.0 - u.time * 0.8));
            let sun_col = mix(vec3<f32>(1.0, 0.15, 0.55), vec3<f32>(1.0, 0.85, 0.2), clamp((p.y - 0.02) * 1.6, 0.0, 1.0));
            col = mix(col, sun_col, stripe);
        } else {
            col += vec3<f32>(1.0, 0.2, 0.6) * exp(-max(sd - sun_r, 0.0) * 6.0) * (0.45 + u.beat);
        }

        // Spectrum mountain silhouette along horizon
        let sym_x = abs(in.uv.x * 2.0 - 1.0);
        let ridge_h = sample_spec(sym_x) * 0.32;
        if p.y - horizon < ridge_h {
            let rim = smoothstep(0.025, 0.0, abs((p.y - horizon) - ridge_h));
            col = mix(vec3<f32>(0.03, 0.01, 0.08), vec3<f32>(0.1, 0.95, 1.0), rim);
        }
    } else {
        // 3D perspective floor grid
        let depth = 0.55 / max(horizon - p.y, 0.015);
        let gx = p.x * depth;
        let gz = depth + u.time * (3.2 + u.bass * 3.5);
        let lx = abs(fract(gx * 0.65) - 0.5);
        let lz = abs(fract(gz * 0.65) - 0.5);
        let line = smoothstep(0.05, 0.0, min(lx, lz));
        let fog = exp(-depth * 0.085);

        let grid_col = mix(vec3<f32>(0.05, 0.9, 1.0), vec3<f32>(1.0, 0.2, 0.8), u.beat * 0.7);
        col = mix(vec3<f32>(0.02, 0.01, 0.06), grid_col, line * fog * (0.85 + u.mid));
        col += vec3<f32>(0.9, 0.2, 0.6) * exp(-(horizon - p.y) * 12.0) * (0.6 + u.beat);
    }

    col = col / (1.0 + col * 0.25);
    return vec4<f32>(col, 1.0);
}
