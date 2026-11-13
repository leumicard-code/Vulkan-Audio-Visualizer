// 27_quantum_orbitals.wgsl — Atomic electron shells & nucleus pulsing on beat

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

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let p = (in.uv - 0.5) * vec2<f32>(u.aspect, 1.0) * 2.0;
    let r = length(p);
    var col = vec3<f32>(0.01, 0.015, 0.04);

    // Pulsing quantum nucleus
    col += vec3<f32>(0.35, 0.75, 1.0) * exp(-r * (8.0 - 3.5 * u.beat)) * (0.8 + u.bass + u.beat);

    for (var i = 0; i < 6; i = i + 1) {
        let fi = f32(i);
        let tilt = fi * 0.5236 + u.time * (0.25 + 0.05 * fi);
        let c = cos(tilt);
        let s = sin(tilt);
        let q = vec2<f32>(c * p.x - s * p.y, s * p.x + c * p.y);
        let ell = vec2<f32>(q.x / (0.68 + 0.1 * u.bass), q.y / 0.24);
        let ring_d = abs(length(ell) - 1.0);
        let ring = exp(-ring_d * 26.0);

        let e_ang = u.time * (2.2 + 0.35 * fi) + fi * 1.4;
        let e_pos = vec2<f32>(cos(e_ang) * (0.68 + 0.1 * u.bass), sin(e_ang) * 0.24);
        let particle = exp(-length(q - e_pos) * (24.0 - 8.0 * u.treble));

        let tint = 0.5 + 0.5 * cos(6.28318 * (fi * 0.16 + vec3<f32>(0.0, 0.35, 0.7)));
        col += tint * (ring * (0.25 + 0.5 * u.mid) + particle * (1.2 + u.beat));
    }
    return vec4<f32>(col / (1.0 + col * 0.35), 1.0);
}
