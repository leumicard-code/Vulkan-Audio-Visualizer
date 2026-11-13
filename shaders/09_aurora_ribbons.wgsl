// 09_aurora_ribbons.wgsl — Flowing northern lights spectrum & waveform ribbons

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

fn sample_wave(t: f32) -> f32 {
    let x = clamp(t, 0.0, 0.999) * 63.0;
    let idx = u32(x);
    let nxt = min(idx + 1u, 63u);
    return mix(u.waveform[idx / 4u][idx & 3u], u.waveform[nxt / 4u][nxt & 3u], fract(x));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let p = uv * 2.0 - 1.0;
    var col = mix(vec3<f32>(0.01, 0.03, 0.07), vec3<f32>(0.02, 0.01, 0.04), uv.y);

    let spec = sample_spec(uv.x);
    let wave = sample_wave(uv.x);

    for (var i = 0; i < 6; i = i + 1) {
        let fi = f32(i);
        let freq = 2.2 + fi * 0.7;
        let phase = u.time * (0.7 + fi * 0.14) + fi * 1.1;
        let curve = 0.25 * sin(p.x * freq + phase)
                  + 0.15 * cos(p.x * (freq * 1.7) - phase * 1.3)
                  + wave * (0.25 + 0.05 * fi)
                  + (spec - 0.25) * 0.32;

        let dy = p.y - (curve + (fi - 2.5) * 0.12);
        let curtain = exp(-max(dy, -0.02) * 5.0) * smoothstep(-0.08, 0.02, dy);
        let core = exp(-abs(dy) * (24.0 - 8.0 * u.beat));
        let rays = 0.65 + 0.35 * sin(uv.x * 55.0 + u.time * 3.0 + fi * 2.0);

        let ribbon_col = mix(
            vec3<f32>(0.05, 0.92, 0.62),
            vec3<f32>(0.55, 0.22, 1.0),
            fi / 5.0
        );
        col += ribbon_col * (core * 0.55 + curtain * 0.22) * rays * (0.5 + spec + u.beat * 0.6);
    }

    col = col / (1.0 + col * 0.35);
    return vec4<f32>(col, 1.0);
}
