// Galaxy Spiral Arms - Rotating spiral star clusters with spectrum dust
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

fn hash21(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453123);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let p = (in.uv - 0.5) * vec2<f32>(u.aspect, 1.0);
    let r = length(p);
    let a = atan2(p.y, p.x);
    let t = u.time * 0.5;

    let arms = 3.0;
    let swirl = a * arms - r * (9.5 - u.bass * 2.5) - t * 1.4;
    let arm_wave = pow(max(0.5 + 0.5 * cos(swirl), 0.0), 2.5);
    let s = spec(i32(clamp(r * 75.0, 0.0, 63.0)));

    // Star dust sprinkled along the spiral
    let spiral_uv = vec2<f32>(cos(a + r * 3.0 - t * 0.3), sin(a + r * 3.0 - t * 0.3)) * r * 40.0;
    let dust = step(0.92, hash21(floor(spiral_uv)));

    let bulge = exp(-r * (7.5 - u.beat * 2.2));
    let outer_fade = exp(-r * 2.2);

    let c_core = vec3<f32>(1.0, 0.85, 0.55);
    let c_arm = vec3<f32>(0.25, 0.6, 1.0);
    let c_nebula = vec3<f32>(0.85, 0.2, 0.75);

    var col = vec3<f32>(0.01, 0.01, 0.025);
    col += mix(c_arm, c_nebula, clamp(r * 1.5 + u.mid * 0.4, 0.0, 1.0)) * arm_wave * outer_fade * (0.65 + s * 1.1);
    col += c_core * bulge * (0.9 + u.beat * 0.7);
    col += vec3<f32>(0.9, 0.95, 1.0) * dust * (0.25 + arm_wave * 0.8) * (0.4 + u.treble * 0.9) * outer_fade;

    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
