// Arcade Vector Storm - Spinning neon wireframe polygons & shockwave echoes
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

fn sd_ngon(p: vec2<f32>, r: f32, n: f32) -> f32 {
    let an = 3.14159265 / n;
    let bn = atan2(p.y, p.x);
    let sector = floor(0.5 + bn / (2.0 * an)) * (2.0 * an) - bn;
    let d = length(p) * cos(sector) - r * cos(an);
    return d;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let p = (in.uv - 0.5) * vec2<f32>(u.aspect, 1.0);
    let t = u.time;

    var col = vec3<f32>(0.01, 0.01, 0.03);
    for (var i = 0; i < 12; i++) {
        let fi = f32(i);
        let z = fract(fi / 12.0 - t * (0.28 + u.bass * 0.25));
        let scale = mix(0.05, 0.85, pow(1.0 - z, 1.6));
        let ang = t * (0.4 + fi * 0.05) + z * 2.0;
        let c = cos(ang);
        let s_ang = sin(ang);
        let rp = vec2<f32>(p.x * c - p.y * s_ang, p.x * s_ang + p.y * c);

        let sides = 3.0 + f32((i % 4) + 1);
        let sp = spec((i * 5) % 64);
        let d = abs(sd_ngon(rp, scale * (1.0 + sp * 0.22), sides));
        let glow = (0.004 + u.beat * 0.005) / max(d, 0.0015);

        let neon = sin(vec3<f32>(0.5, 2.5, 4.5) + fi * 0.55 + t) * 0.5 + 0.5;
        col += neon * glow * (1.0 - z) * 0.22;
    }

    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
