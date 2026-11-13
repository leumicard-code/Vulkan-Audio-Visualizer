// Meteor Rain Nova - Diagonal glowing meteor streaks & shock rings
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

fn hash11(n: f32) -> f32 {
    return fract(sin(n) * 43758.5453);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let p = (in.uv - 0.5) * vec2<f32>(u.aspect, 1.0);
    let t = u.time;

    // Slanted coordinate space for diagonal meteors
    let c = 0.866;
    let s = 0.5;
    let rp = vec2<f32>(p.x * c - p.y * s, p.x * s + p.y * c);

    var col = vec3<f32>(0.015, 0.012, 0.045);

    // 16 meteor lanes
    for (var i = 0; i < 16; i++) {
        let fi = f32(i);
        let lane_y = (fi / 15.0 - 0.5) * 1.15;
        let speed = 0.75 + hash11(fi * 3.1) * 1.1 + u.mid * 0.5;
        let head_x = fract(hash11(fi * 9.7) + t * speed) * 2.6 - 1.3;
        let dx = head_x - rp.x;
        let dy = abs(rp.y - lane_y);

        let sp = spec((i * 4) % 64);
        let tail_len = 0.25 + sp * 0.45 + u.bass * 0.2;
        if (dx > 0.0 && dx < tail_len) {
            let fade = pow(max(1.0 - dx / tail_len, 0.0), 2.2);
            let beam = exp(-dy * (160.0 - sp * 60.0)) * fade;
            let hue = 0.5 + 0.5 * sin(vec3<f32>(0.0, 2.0, 4.0) + fi * 0.5 + t);
            col += mix(vec3<f32>(1.0, 0.85, 0.5), hue, 0.65) * beam * (0.9 + u.beat * 0.8);
        }
        let head_glow = exp(-length(vec2<f32>(rp.x - head_x, rp.y - lane_y)) * 55.0);
        col += vec3<f32>(1.0, 0.95, 0.9) * head_glow * (0.6 + u.treble * 0.7);
    }

    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
