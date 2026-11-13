// Toxic Acid Swirl - Neon acid green & ultraviolet vortex plasma
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
    let t = u.time * 0.9;

    let twist = a + (2.5 + u.bass * 3.0) * sin(r * 4.5 - t) + r * 3.0;
    let q = vec2<f32>(cos(twist), sin(twist)) * r * 4.0;
    let s = spec(i32(clamp(r * 60.0, 0.0, 63.0)));

    let f1 = sin(q.x + cos(q.y + t)) + cos(q.y * 1.7 - t * 1.3);
    let f2 = sin(length(q) * 2.5 - t * 2.0 + f1 * (1.2 + u.mid));
    let veins = exp(-abs(f2) * (4.5 - u.beat * 1.8));

    let c_acid = vec3<f32>(0.25, 1.0, 0.15);
    let c_uv = vec3<f32>(0.55, 0.08, 0.95);
    let c_cyan = vec3<f32>(0.0, 0.95, 0.75);

    var col = mix(c_uv * 0.35, c_acid, clamp(0.5 + 0.5 * f1, 0.0, 1.0)) * (0.4 + s * 0.8);
    col += c_cyan * veins * (0.65 + u.treble * 0.7);
    col += vec3<f32>(0.9, 1.0, 0.5) * pow(veins, 3.0) * u.beat;

    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
