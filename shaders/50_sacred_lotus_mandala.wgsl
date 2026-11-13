// Sacred Lotus Mandala - Blooming 12-petal sacred geometry driven by FFT
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
    var pos = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    var out: VsOut;
    out.pos = vec4<f32>(pos[vid], 0.0, 1.0);
    out.uv = pos[vid] * 0.5 + 0.5;
    return out;
}

fn spec(idx: i32) -> f32 {
    let i = clamp(idx, 0, 63);
    return u.spectrum[i / 4][i % 4];
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let p = (in.uv - 0.5) * vec2<f32>(u.aspect, 1.0);
    let r = length(p);
    let a = atan2(p.y, p.x);
    let t = u.time;

    var col = vec3<f32>(0.02, 0.01, 0.045);

    for (var ring = 0; ring < 5; ring++) {
        let fr = f32(ring);
        let petals = 6.0 + fr * 4.0;
        let spin = t * (0.25 - fr * 0.08) * select(-1.0, 1.0, (ring % 2) == 0);
        let ang = a + spin;
        let s = spec((ring * 12 + 3) % 64);

        let petal_curve = 0.08 + fr * 0.075 + (0.04 + s * 0.08 + u.bass * 0.03) * abs(cos(ang * petals * 0.5));
        let d = abs(r - petal_curve);
        let line_glow = (0.004 + u.beat * 0.004) / max(d, 0.0016);

        let gold = vec3<f32>(1.0, 0.78, 0.25);
        let lotus = vec3<f32>(0.95, 0.25, 0.65);
        let cyan = vec3<f32>(0.15, 0.85, 1.0);
        let tint = mix(mix(lotus, gold, fr / 4.0), cyan, u.treble * 0.5);
        col += tint * line_glow * 0.22;
    }

    // Center bindu jewel
    let jewel = exp(-r * (14.0 - u.beat * 5.0));
    col += vec3<f32>(1.0, 0.9, 0.6) * jewel;
    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
