// 21_laser_prism_shaft.wgsl — Square perspective prism corridor with spectrum wall bars

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

fn sample_spec(t: f32) -> f32 {
    let x = clamp(t, 0.0, 0.999) * 63.0;
    let idx = u32(x);
    let nxt = min(idx + 1u, 63u);
    return mix(u.spectrum[idx / 4u][idx & 3u], u.spectrum[nxt / 4u][nxt & 3u], fract(x));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    var p = (in.uv - 0.5) * vec2<f32>(u.aspect, 1.0) * 2.0;
    let rot = u.time * 0.16 + 0.15 * u.beat;
    p = vec2<f32>(cos(rot) * p.x - sin(rot) * p.y, sin(rot) * p.x + cos(rot) * p.y);

    let box_d = max(abs(p.x), abs(p.y));
    let z = 0.85 / max(box_d, 0.025) + u.time * (2.6 + u.bass * 2.5);
    let spec = sample_spec(fract(z * 0.12));

    let grid_z = smoothstep(0.07, 0.01, abs(fract(z) - 0.5));
    let diag = smoothstep(0.04, 0.005, abs(abs(p.x) - abs(p.y)));
    let laser_col = mix(vec3<f32>(0.0, 0.95, 1.0), vec3<f32>(1.0, 0.15, 0.72), spec + u.beat * 0.5);

    var col = laser_col * (grid_z * (0.6 + 1.3 * spec) + diag * 0.45 * (0.5 + u.treble));
    col += vec3<f32>(0.5, 0.8, 1.0) * exp(-box_d * 5.0) * (0.3 + u.beat);
    col *= smoothstep(0.02, 0.15, box_d);
    return vec4<f32>(col / (1.0 + col * 0.35), 1.0);
}
