// Polar Mist Curtains - Dreamy pastel aurora waves with soft vertical drapes
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
    let t = u.time * 0.45;
    let s = spec(i32(clamp(in.uv.x * 63.0, 0.0, 63.0)));

    var col = vec3<f32>(0.015, 0.025, 0.055);
    for (var i = 0; i < 8; i++) {
        let fi = f32(i);
        let phase = fi * 0.85 + t * (0.7 + fi * 0.08);
        let fold = sin(p.x * (2.2 + fi * 0.3) + phase) * 0.14
                 + cos(p.x * 5.0 - phase * 1.4) * 0.05;
        let y_off = fold + (fi - 3.5) * 0.065;
        let d = abs(p.y - y_off);
        let drape = exp(-d * (14.0 - u.bass * 4.5));
        let rays = 0.6 + 0.4 * sin(p.x * 36.0 + t * 3.0 + fi);
        let hue_shift = sin(vec3<f32>(0.0, 2.1, 4.2) + fi * 0.65 + t * 0.6) * 0.5 + 0.5;
        let aura = mix(vec3<f32>(0.15, 0.95, 0.85), hue_shift, 0.55);
        col += aura * drape * rays * (0.22 + s * 0.45 + u.beat * 0.18);
    }

    // Horizon mist glow
    let mist = exp(-abs(in.uv.y - 0.15) * 6.0) * (0.25 + u.overall * 0.4);
    col += vec3<f32>(0.15, 0.45, 0.7) * mist;
    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
