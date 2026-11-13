// Magnetic Flux Sky - Polar geomagnetic aurora spirals & particle streaks
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
    let t = u.time * 0.7;

    let s_low = spec(4);
    let s_mid = spec(22);
    let s_hi = spec(48);

    var acc = vec3<f32>(0.01, 0.02, 0.05);
    for (var i = 0; i < 6; i++) {
        let fi = f32(i);
        let twist = a + r * (2.6 + fi * 0.5) - t * (0.4 + fi * 0.1);
        let ribbon = sin(twist * (2.0 + floor(fi * 0.5)) + sin(r * 6.0 - t));
        let band = exp(-abs(ribbon) * (7.0 - u.bass * 3.0)) * exp(-r * 1.4);
        let ray = 0.5 + 0.5 * cos(a * 24.0 + r * 14.0 - t * 3.0);

        let c_emerald = vec3<f32>(0.1, 1.0, 0.65);
        let c_magenta = vec3<f32>(0.85, 0.2, 0.95);
        let c_blue = vec3<f32>(0.2, 0.55, 1.0);
        let col_i = mix(mix(c_emerald, c_blue, fract(fi * 0.37)), c_magenta, s_hi * 0.8 + 0.2 * sin(fi + t));
        acc += col_i * band * (0.4 + 0.6 * ray) * (0.35 + s_low * 0.4 + s_mid * 0.4 + u.beat * 0.3);
    }

    return vec4<f32>(clamp(acc, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
