// Borealis Silk Ribbons - Flowing northern lights curtains driven by spectrum
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
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let p = vec2<f32>((in.uv.x - 0.5) * u.aspect, in.uv.y - 0.5);
    let t = u.time * 0.55;

    // Night sky gradient + stars
    var col = mix(vec3<f32>(0.01, 0.03, 0.07), vec3<f32>(0.02, 0.01, 0.05), in.uv.y);
    let star_grid = floor(p * 75.0);
    let star_rnd = hash21(star_grid);
    if (star_rnd > 0.985) {
        let twinkle = 0.5 + 0.5 * sin(u.time * 3.0 + star_rnd * 40.0 + u.treble * 4.0);
        col += vec3<f32>(0.7, 0.85, 1.0) * twinkle * 0.6;
    }

    // Stacked flowing aurora curtains
    let band_idx = i32(clamp(in.uv.x * 63.0, 0.0, 63.0));
    let s = spec(band_idx);

    for (var i = 0; i < 6; i++) {
        let fi = f32(i);
        let wave_x = p.x * (1.8 + fi * 0.35) + t * (0.4 + fi * 0.12);
        let curtain_y = -0.1 + sin(wave_x) * 0.16 + cos(wave_x * 2.3 - t) * 0.07 + (fi - 2.5) * 0.06;
        let dy = p.y - curtain_y;

        // Vertical shimmer streaking upward
        let shimmer = 0.55 + 0.45 * sin(p.x * 28.0 + t * 2.5 + fi * 2.0 + s * 4.0);
        let height_fade = smoothstep(-0.08, 0.02, dy) * exp(-max(dy, 0.0) * (5.5 - u.bass * 2.2));
        let intensity = height_fade * shimmer * (0.45 + s * 0.9 + u.beat * 0.35);

        let c_green = vec3<f32>(0.1, 0.95, 0.55);
        let c_violet = vec3<f32>(0.65, 0.25, 0.95);
        let c_cyan = vec3<f32>(0.15, 0.75, 1.0);
        let tint = mix(mix(c_green, c_cyan, clamp(dy * 3.0, 0.0, 1.0)), c_violet, smoothstep(0.12, 0.45, dy));
        col += tint * intensity * 0.38;
    }

    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
