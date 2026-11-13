// 31_dna_double_helix.wgsl — Twisted dual waveform helix strands with glowing rungs

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

fn sample_wave(t: f32) -> f32 {
    let x = clamp(t, 0.0, 0.999) * 63.0;
    let idx = u32(x);
    let nxt = min(idx + 1u, 63u);
    return mix(u.waveform[idx / 4u][idx & 3u], u.waveform[nxt / 4u][nxt & 3u], fract(x));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let p = in.uv * 2.0 - 1.0;
    let w = sample_wave(in.uv.x) * (0.28 + 0.18 * u.bass);
    let phase = p.x * 5.0 - u.time * 2.5;
    let amp = 0.28 + 0.12 * u.beat + abs(w) * 0.35;

    let y1 = sin(phase) * amp + w * 0.4;
    let y2 = -sin(phase) * amp + w * 0.4;

    let s1 = exp(-abs(p.y - y1) * 38.0);
    let s2 = exp(-abs(p.y - y2) * 38.0);

    // Base-pair rungs connecting both strands
    let rung_x = smoothstep(0.18, 0.0, abs(fract(phase / 1.5708) - 0.5));
    let between = step(min(y1, y2), p.y) * step(p.y, max(y1, y2));
    let rung = rung_x * between * 0.45 * (0.5 + u.treble);

    var col = vec3<f32>(0.015, 0.018, 0.045);
    col += vec3<f32>(0.1, 0.9, 1.0) * s1 * 1.1;
    col += vec3<f32>(1.0, 0.2, 0.68) * s2 * 1.1;
    col += vec3<f32>(0.65, 0.85, 1.0) * rung;
    return vec4<f32>(col / (1.0 + col * 0.3), 1.0);
}
