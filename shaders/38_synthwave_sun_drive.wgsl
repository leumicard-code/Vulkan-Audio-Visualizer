// Synthwave Sun Drive - Outrun perspective grid, palm horizon & retro sun
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
    let uv = in.uv;
    let p = vec2<f32>((uv.x - 0.5) * u.aspect, uv.y - 0.5);
    let t = u.time;
    let horizon = -0.05;

    var col = vec3<f32>(0.03, 0.01, 0.09);

    if (p.y > horizon) {
        // Sky + reactive sun + skyline spectrum bars
        let sky_t = clamp((p.y - horizon) * 2.0, 0.0, 1.0);
        col = mix(vec3<f32>(0.45, 0.08, 0.38), vec3<f32>(0.02, 0.01, 0.12), sky_t);

        let sun_d = length(p - vec2<f32>(0.0, 0.15));
        let sun_r = 0.21 + u.beat * 0.025;
        if (sun_d < sun_r) {
            let cut = step(0.18 + max(0.0, (0.14 - p.y) * 1.4), fract(p.y * 22.0 - t * 0.3));
            let sun_c = mix(vec3<f32>(1.0, 0.15, 0.55), vec3<f32>(1.0, 0.9, 0.25), clamp((p.y - 0.0) * 3.0, 0.0, 1.0));
            col = mix(col, sun_c, cut);
        } else {
            col += vec3<f32>(1.0, 0.2, 0.6) * exp(-sun_d * 6.0) * (0.4 + u.bass * 0.6);
        }

        // Distant neon equalizer silhouettes along horizon
        let bx = i32(clamp(abs(uv.x - 0.5) * 126.0, 0.0, 63.0));
        let bar_h = spec(bx) * 0.16;
        if (p.y - horizon < bar_h) {
            col = mix(col, vec3<f32>(0.05, 0.02, 0.15), 0.85);
            let edge = smoothstep(0.01, 0.0, abs((p.y - horizon) - bar_h));
            col += vec3<f32>(0.1, 0.95, 1.0) * edge;
        }
    } else {
        // Perspective ground grid scrolling toward camera
        let dy = max(horizon - p.y, 0.008);
        let z = 0.22 / dy;
        let gx = p.x / dy * 0.5;
        let gz = z + t * (2.6 + u.bass * 2.5);

        let fx = abs(fract(gx) - 0.5);
        let fz = abs(fract(gz) - 0.5);
        let lx = exp(-fx * (28.0 - u.treble * 8.0));
        let lz = exp(-fz * (28.0 - u.beat * 8.0));
        let grid = max(lx, lz);
        let fade = smoothstep(0.01, 0.14, dy);

        let road_glow = exp(-abs(p.x) * 4.5) * (0.25 + u.bass * 0.4);
        col = vec3<f32>(0.02, 0.01, 0.06);
        col += vec3<f32>(0.0, 0.9, 1.0) * grid * fade * (0.65 + u.mid * 0.6);
        col += vec3<f32>(1.0, 0.15, 0.7) * road_glow * fade;
    }

    // Horizon laser
    col += vec3<f32>(1.0, 0.25, 0.85) * exp(-abs(p.y - horizon) * 90.0) * (0.7 + u.beat * 0.8);
    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
