// 05_particle_field.wgsl — Multi-layer procedural audio-reactive GPU particle bokeh

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

fn hash22(p: vec2<f32>) -> vec2<f32> {
    let q = vec2<f32>(dot(p, vec2<f32>(127.1, 311.7)), dot(p, vec2<f32>(269.5, 183.3)));
    return fract(sin(q) * 43758.5453);
}

fn sample_spec(t: f32) -> f32 {
    let x = clamp(t, 0.0, 0.999) * 63.0;
    let idx = u32(x);
    let nxt = min(idx + 1u, 63u);
    return mix(u.spectrum[idx / 4u][idx & 3u], u.spectrum[nxt / 4u][nxt & 3u], fract(x));
}

fn particle_layer(uv: vec2<f32>, scale: f32, seed: f32, speed: f32) -> vec3<f32> {
    let p = uv * scale + vec2<f32>(u.time * speed * 0.25, -u.time * speed * 0.15);
    let id = floor(p);
    let gv = fract(p) - 0.5;
    var acc = vec3<f32>(0.0);

    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            let offs = vec2<f32>(f32(x), f32(y));
            let cell = id + offs;
            let n = hash22(cell + seed);
            let orbit = vec2<f32>(
                sin(u.time * (1.2 + n.x * 2.0) + n.y * 6.28),
                cos(u.time * (1.1 + n.y * 2.0) + n.x * 6.28)
            ) * (0.28 + 0.18 * u.beat);

            let pos = offs + orbit - gv;
            let d = length(pos);
            let spec = sample_spec(n.x);
            let radius = 0.05 + 0.12 * spec + 0.08 * u.beat;
            let sparkle = exp(-d / max(radius, 0.02));
            let hue = 0.5 + 0.5 * cos(6.28318 * (n.y + u.time * 0.12 + vec3<f32>(0.0, 0.28, 0.62)));
            acc += hue * sparkle * sparkle * (0.35 + 1.4 * spec);
        }
    }
    return acc;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let uv = (in.uv - 0.5) * vec2<f32>(u.aspect, 1.0);
    let r = length(uv);

    // Radial burst distortion on beat
    let zoom = 1.0 - 0.12 * u.beat - 0.06 * u.bass;
    let p = uv * zoom;

    var col = vec3<f32>(0.015, 0.018, 0.045);
    col += particle_layer(p, 4.0, 11.3, 0.8) * 0.55;
    col += particle_layer(p, 7.5, 37.9, 1.25) * 0.45;
    col += particle_layer(p, 12.0, 71.7, 1.8) * 0.35;

    // Beat shockwave ring
    let ring_r = fract(u.time * 0.55) * 1.4;
    let shock = exp(-abs(r - ring_r) * 28.0) * u.beat * 0.8;
    col += vec3<f32>(0.35, 0.85, 1.0) * shock;

    col *= smoothstep(1.5, 0.15, r);
    col = col / (1.0 + col * 0.4);
    return vec4<f32>(col, 1.0);
}
