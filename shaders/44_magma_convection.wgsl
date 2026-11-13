// Magma Convection - Turbulent volcanic lava cells pulsing with bass heat
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

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    var p = (in.uv - 0.5) * vec2<f32>(u.aspect, 1.0) * 3.2;
    let t = u.time * 0.5;

    var w = vec2<f32>(0.0);
    for (var i = 0; i < 4; i++) {
        let fi = f32(i) + 1.0;
        w.x += sin(p.y * fi * 1.4 + t * fi * 0.6 + w.y) / fi;
        w.y += cos(p.x * fi * 1.4 - t * fi * 0.5 + w.x) / fi;
        p += w * (0.28 + u.mid * 0.14);
    }

    let heat = 0.5 + 0.5 * sin(p.x * 2.5 + cos(p.y * 2.5 - t));
    let cracks = pow(max(1.0 - abs(sin(p.x + p.y)), 0.0), 6.0);
    let intensity = clamp(heat * (0.65 + u.bass * 0.7) + cracks * (0.5 + u.beat * 0.9), 0.0, 1.4);

    let crust = vec3<f32>(0.07, 0.02, 0.02);
    let lava = vec3<f32>(0.95, 0.18, 0.03);
    let core = vec3<f32>(1.0, 0.88, 0.35);

    var col = mix(crust, lava, smoothstep(0.15, 0.65, intensity));
    col = mix(col, core, smoothstep(0.68, 1.15, intensity));
    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
