// Neon Laser Lattice - Crossing volumetric laser beams & prism nodes
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

fn rot2(a: f32) -> mat2x2<f32> {
    let c = cos(a);
    let s = sin(a);
    return mat2x2<f32>(vec2<f32>(c, s), vec2<f32>(-s, c));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let p = (in.uv - 0.5) * vec2<f32>(u.aspect, 1.0);
    let t = u.time;

    var col = vec3<f32>(0.01, 0.01, 0.025);
    for (var i = 0; i < 8; i++) {
        let fi = f32(i);
        let s = spec(i * 7 + 2);
        let ang = fi * 0.3927 + t * (0.2 + (fi % 2.0 - 0.5) * 0.35) + u.mid * 0.4;
        let q = rot2(ang) * p;
        let spacing = 0.24 - u.bass * 0.04;
        let line_d = abs(fract(q.y / spacing + t * 0.25 + fi * 0.13) - 0.5) * spacing;
        let glow = (0.0035 + s * 0.008 + u.beat * 0.004) / max(line_d, 0.0018);

        let hue = sin(vec3<f32>(0.0, 2.09, 4.18) + fi * 0.8 + t * 0.7) * 0.5 + 0.5;
        col += hue * glow * 0.12;
    }

    // Central laser hub pulse
    let core = exp(-length(p) * (6.0 - u.beat * 2.5));
    col += vec3<f32>(0.85, 0.95, 1.0) * core * (0.3 + u.bass * 0.7);
    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
