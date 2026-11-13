// 18_wormhole_singularity.wgsl — Relativistic wormhole vortex + glowing accretion ring

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
    let p = (in.uv - 0.5) * vec2<f32>(u.aspect, 1.0) * 2.0;
    let r = max(length(p), 0.01);
    let a = atan2(p.y, p.x);

    let event_horizon = 0.24 + 0.06 * u.bass + 0.05 * u.beat;
    let swirl = a + 1.8 / (r + 0.18) - u.time * (1.2 + u.mid);
    let z = 1.2 / max(r - event_horizon * 0.55, 0.03) + u.time * (2.5 + u.bass * 2.0);
    let spec = sample_spec(abs(fract(swirl / 6.283185) * 2.0 - 1.0));

    let rings = pow(0.5 + 0.5 * sin(z * 4.0 + swirl * 3.0), 3.0);
    let disk = exp(-abs(r - event_horizon - 0.06 * spec) * 26.0);

    let warp_col = mix(vec3<f32>(0.08, 0.45, 1.0), vec3<f32>(1.0, 0.35, 0.12), rings);
    var col = warp_col * rings * (0.4 + spec * 1.4) * smoothstep(event_horizon * 0.75, event_horizon + 0.2, r);
    col += vec3<f32>(1.0, 0.75, 0.35) * disk * (1.1 + u.beat * 1.4);
    col *= smoothstep(event_horizon * 0.58, event_horizon * 0.85, r);
    return vec4<f32>(col / (1.0 + col * 0.35), 1.0);
}
