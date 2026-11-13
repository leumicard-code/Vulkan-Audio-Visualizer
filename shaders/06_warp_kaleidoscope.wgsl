// 06_warp_kaleidoscope.wgsl — MilkDrop symmetric N-fold geometric kaleidoscope

struct Uniforms {
    time: f32,
    dt: f32,
    width: f32,
    height: f32,
    bass: f32,
    mid: f32,
    treble: f32,
    beat: f32,
    overall: f32,
    aspect: f32,
    frame: f32,
    _pad: f32,
    spectrum: array<vec4<f32>, 16>,
    waveform: array<vec4<f32>, 16>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    let x = f32((vid << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(vid & 2u) * 2.0 - 1.0;
    var out: VsOut;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(x * 0.5 + 0.5, 0.5 - y * 0.5);
    return out;
}

fn palette(t: f32) -> vec3<f32> {
    return 0.5 + 0.5 * cos(6.28318 * (t + vec3<f32>(0.02, 0.34, 0.68)));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    var uv = (in.uv - 0.5) * vec2<f32>(u.aspect, 1.0) * 2.0;
    let uv0 = uv;

    // Polar mirror folding
    let folds = 8.0;
    var a = atan2(uv.y, uv.x) + u.time * 0.22;
    let r = length(uv);
    let sector = 6.283185 / folds;
    a = abs(fract(a / sector) - 0.5) * sector;
    uv = vec2<f32>(cos(a), sin(a)) * r;

    var final_col = vec3<f32>(0.0);
    for (var i = 0; i < 4; i = i + 1) {
        let fi = f32(i);
        uv = fract(uv * (1.45 + 0.15 * u.bass)) - 0.5;
        var d = length(uv) * exp(-length(uv0));
        let col = palette(length(uv0) + fi * 0.38 + u.time * 0.35 + u.mid * 0.4);
        d = sin(d * (8.0 + u.treble * 4.0) + u.time * 1.8) / 8.0;
        d = abs(d);
        d = pow(0.011 / max(d, 0.002), 1.25);
        final_col += col * d * (0.35 + 0.65 * u.overall + 0.55 * u.beat);
    }

    final_col = final_col / (1.0 + final_col * 0.35);
    return vec4<f32>(final_col, 1.0);
}
