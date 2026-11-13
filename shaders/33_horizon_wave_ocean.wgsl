// Horizon Wave Ocean - Layered audio-reactive horizon waves & sun reflection
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

fn wave(idx: i32) -> f32 {
    let i = clamp(idx, 0, 63);
    return u.waveform[i / 4][i % 4];
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let p = vec2<f32>((uv.x - 0.5) * u.aspect, uv.y - 0.5);
    let t = u.time;

    var col = mix(vec3<f32>(0.02, 0.01, 0.08), vec3<f32>(0.08, 0.03, 0.20), clamp(uv.y * 1.4, 0.0, 1.0));

    // Reactive sun behind the waves
    let sun_center = vec2<f32>(0.0, 0.14 + u.bass * 0.04);
    let sun_d = length(p - sun_center);
    let sun_r = 0.22 + u.beat * 0.04;
    let sun_glow = exp(-sun_d * 5.0) * (0.6 + u.mid * 0.8);
    col += vec3<f32>(1.0, 0.35, 0.55) * sun_glow * 0.45;
    if (sun_d < sun_r && p.y > 0.0) {
        let stripe = step(0.25, fract((p.y - t * 0.08) * 18.0));
        let sun_col = mix(vec3<f32>(1.0, 0.25, 0.5), vec3<f32>(1.0, 0.85, 0.2), clamp((p.y + 0.1) * 3.0, 0.0, 1.0));
        col += sun_col * stripe * 0.85;
    }

    // 10 layered ocean waveforms
    let w_idx = i32(clamp(uv.x * 63.0, 0.0, 63.0));
    let w_val = wave(w_idx);
    let s_val = spec(w_idx);

    for (var l = 0; l < 10; l++) {
        let fl = f32(l);
        let depth = fl / 9.0;
        let base_y = 0.48 - depth * 0.42;
        let freq = 4.0 + fl * 1.4;
        let speed = (0.8 + fl * 0.25) * (1.0 + u.mid * 0.5);
        let ripple = sin(uv.x * freq + t * speed + fl * 1.1) * (0.015 + u.bass * 0.03);
        let audio_disp = w_val * (0.04 + (1.0 - depth) * 0.07) + s_val * 0.03 * sin(fl + t);
        let ridge = base_y + ripple + audio_disp;

        if (uv.y < ridge) {
            let water_tint = mix(vec3<f32>(0.02, 0.15, 0.35), vec3<f32>(0.01, 0.04, 0.12), depth);
            col = mix(col, water_tint, 0.88);
        }
        let dist = abs(uv.y - ridge);
        let crest = exp(-dist * 180.0) * (0.55 + u.treble * 0.8 + u.beat * 0.6);
        let crest_col = mix(vec3<f32>(0.1, 0.9, 1.0), vec3<f32>(1.0, 0.3, 0.8), depth);
        col += crest_col * crest;
    }

    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
