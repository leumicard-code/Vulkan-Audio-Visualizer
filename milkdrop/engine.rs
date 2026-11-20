// ============================================================================
// milkdrop/engine.rs — Native MilkDrop (.milk) Renderer on Vulkan (wgpu)
// ============================================================================
// Executes MilkDrop 1 & MilkDrop 2 presets:
//   - EEL2 equations (per_frame_init, per_frame, per_pixel, waves, shapes)
//   - Warped ping-pong feedback mesh (48x36) with MilkDrop 2 HLSL->GLSL
//     shaders or MilkDrop 1 default fallback shader
//   - Motion vectors, borders, darken-center, custom shapes, custom waves,
//     and 8-mode main waveform
//   - 6-pass separable blur pyramid (blur1..3)
//   - Composite pass (comp shader or MilkDrop 1 gamma/echo/effects)
// ============================================================================

use anyhow::Result;
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use wgpu::util::DeviceExt;

use crate::dsp_fft::AudioFeatures;
use crate::milkdrop::eel::{self, Globals, Program, SharedGlobals, VarId, Vm};
use crate::milkdrop::hlsl::{self, MdShaderUniforms, ShaderInputs, ShaderKind};
use crate::milkdrop::preset::{CustomShapeDef, CustomWaveDef, MilkPreset};
use crate::milkdrop::textures::{NoiseSet, TextureCatalog};

const MESH_X: usize = 48;
const MESH_Y: usize = 36;

// ============================================================================
// WGSL Shaders (Shared Mesh VS, 2D Batch Primitives, Blur, MilkDrop 1 Fallbacks)
// ============================================================================

const MD_SHARED_VS_WGSL: &str = r#"
struct VsIn {
    @location(0) pos: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) uv_orig: vec2<f32>,
    @location(3) rad_ang: vec2<f32>,
    @location(4) hue_shader: vec4<f32>,
};

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) v_uv: vec2<f32>,
    @location(1) v_uv_orig: vec2<f32>,
    @location(2) v_rad_ang: vec2<f32>,
    @location(3) v_hue_shader: vec3<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.pos = vec4<f32>(in.pos, 0.0, 1.0);
    out.v_uv = in.uv;
    out.v_uv_orig = in.uv_orig;
    out.v_rad_ang = in.rad_ang;
    out.v_hue_shader = in.hue_shader.xyz;
    return out;
}
"#;

const MD_DEFAULT_SHADER_WGSL: &str = r#"
struct DefaultUniforms {
    gamma_adj: f32,
    decay: f32,
    echo_zoom: f32,
    echo_alpha: f32,
    echo_orient: u32,
    shader_mix: f32,
    flags: u32, // bit0: brighten, bit1: darken, bit2: solarize, bit3: invert
    _pad: f32,
};

@group(0) @binding(0) var<uniform> d: DefaultUniforms;
@group(0) @binding(1) var tex_main: texture_2d<f32>;
@group(0) @binding(2) var samp_main: sampler;

struct FsIn {
    @location(0) v_uv: vec2<f32>,
    @location(1) v_uv_orig: vec2<f32>,
    @location(2) v_rad_ang: vec2<f32>,
    @location(3) v_hue_shader: vec3<f32>,
};

@fragment
fn fs_default_warp(in: FsIn) -> @location(0) vec4<f32> {
    var c = textureSample(tex_main, samp_main, in.v_uv).rgb;
    if (d.shader_mix > 0.001) {
        c = mix(c, c * in.v_hue_shader, clamp(d.shader_mix, 0.0, 1.0));
    }
    c = clamp(c * d.decay, vec3<f32>(0.0), vec3<f32>(1.0));
    return vec4<f32>(c, 1.0);
}

@fragment
fn fs_default_comp(in: FsIn) -> @location(0) vec4<f32> {
    var ret = textureSample(tex_main, samp_main, in.v_uv_orig).rgb;
    if (d.echo_alpha > 0.001) {
        var euv = (in.v_uv_orig - vec2<f32>(0.5)) / max(d.echo_zoom, 0.05) + vec2<f32>(0.5);
        if ((d.echo_orient & 1u) != 0u) { euv.x = 1.0 - euv.x; }
        if ((d.echo_orient & 2u) != 0u) { euv.y = 1.0 - euv.y; }
        let echo_col = textureSample(tex_main, samp_main, clamp(euv, vec2<f32>(0.0), vec2<f32>(1.0))).rgb;
        ret = mix(ret, echo_col, clamp(d.echo_alpha, 0.0, 1.0));
    }
    ret = clamp(ret * max(d.gamma_adj, 0.05), vec3<f32>(0.0), vec3<f32>(1.0));
    if ((d.flags & 1u) != 0u) { ret = sqrt(clamp(ret, vec3<f32>(0.0), vec3<f32>(1.0))); }
    if ((d.flags & 2u) != 0u) { ret = ret * ret; }
    if ((d.flags & 4u) != 0u) {
        let lum = dot(ret, vec3<f32>(0.32, 0.49, 0.29));
        let sol = ret * 1.5;
        ret = select(sol, vec3<f32>(1.0) - sol, lum > 0.5);
    }
    if ((d.flags & 8u) != 0u) { ret = vec3<f32>(1.0) - ret; }
    return vec4<f32>(clamp(ret, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
"#;

const MD_DRAW_WGSL: &str = r#"
struct DrawIn {
    @location(0) pos: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
};

struct DrawOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@group(0) @binding(0) var tex_shape: texture_2d<f32>;
@group(0) @binding(1) var samp_shape: sampler;

@vertex
fn vs_draw(in: DrawIn) -> DrawOut {
    var out: DrawOut;
    out.pos = vec4<f32>(in.pos, 0.0, 1.0);
    out.uv = in.uv;
    out.color = in.color;
    return out;
}

@fragment
fn fs_solid(in: DrawOut) -> @location(0) vec4<f32> {
    return in.color;
}

@fragment
fn fs_textured(in: DrawOut) -> @location(0) vec4<f32> {
    let t = textureSample(tex_shape, samp_shape, in.uv);
    return vec4<f32>(t.rgb * in.color.rgb, in.color.a);
}
"#;

const MD_BLUR_WGSL: &str = r#"
struct BlurUniforms {
    dir_step: vec2<f32>,
    min_val: f32,
    max_val: f32,
    is_pack: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> bu: BlurUniforms;
@group(0) @binding(1) var src_tex: texture_2d<f32>;
@group(0) @binding(2) var src_samp: sampler;

struct BlurOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_blur(@builtin(vertex_index) vid: u32) -> BlurOut {
    let pts = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>( 3.0,  1.0),
        vec2<f32>(-1.0,  1.0),
    );
    let p = pts[vid];
    var out: BlurOut;
    out.pos = vec4<f32>(p, 0.0, 1.0);
    out.uv = vec2<f32>(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
    return out;
}

@fragment
fn fs_blur(in: BlurOut) -> @location(0) vec4<f32> {
    let d = bu.dir_step;
    let c =
        textureSample(src_tex, src_samp, in.uv - 2.0 * d).rgb * 0.06136 +
        textureSample(src_tex, src_samp, in.uv - 1.0 * d).rgb * 0.24477 +
        textureSample(src_tex, src_samp, in.uv).rgb           * 0.38774 +
        textureSample(src_tex, src_samp, in.uv + 1.0 * d).rgb * 0.24477 +
        textureSample(src_tex, src_samp, in.uv + 2.0 * d).rgb * 0.06136;
    if (bu.is_pack != 0u) {
        let span = max(bu.max_val - bu.min_val, 0.001);
        let packed = clamp((c - vec3<f32>(bu.min_val)) / span, vec3<f32>(0.0), vec3<f32>(1.0));
        return vec4<f32>(packed, 1.0);
    }
    return vec4<f32>(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
"#;

// ============================================================================
// GPU Structs
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct WarpVertex {
    pos: [f32; 2],
    uv: [f32; 2],
    uv_orig: [f32; 2],
    rad_ang: [f32; 2],
    hue_shader: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DrawVertex {
    pos: [f32; 2],
    uv: [f32; 2],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DefaultUniforms {
    gamma_adj: f32,
    decay: f32,
    echo_zoom: f32,
    echo_alpha: f32,
    echo_orient: u32,
    shader_mix: f32,
    flags: u32,
    _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BlurUniforms {
    dir_step: [f32; 2],
    min_val: f32,
    max_val: f32,
    is_pack: u32,
    _pad: [u32; 3],
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DrawMode {
    SolidAlpha,
    SolidAdditive,
    TexAlpha,
    TexAdditive,
}

struct DrawBatch {
    mode: DrawMode,
    start: u32,
    count: u32,
}

struct BatchCollector {
    verts: Vec<DrawVertex>,
    batches: Vec<DrawBatch>,
}

impl BatchCollector {
    fn new() -> Self {
        Self {
            verts: Vec::with_capacity(16384),
            batches: Vec::with_capacity(64),
        }
    }

    fn clear(&mut self) {
        self.verts.clear();
        self.batches.clear();
    }

    fn push_tri(&mut self, mode: DrawMode, a: DrawVertex, b: DrawVertex, c: DrawVertex) {
        let start = self.verts.len() as u32;
        self.verts.extend_from_slice(&[a, b, c]);
        if let Some(last) = self.batches.last_mut() {
            if last.mode == mode && last.start + last.count == start {
                last.count += 3;
                return;
            }
        }
        self.batches.push(DrawBatch {
            mode,
            start,
            count: 3,
        });
    }

    fn push_quad(&mut self, mode: DrawMode, v0: DrawVertex, v1: DrawVertex, v2: DrawVertex, v3: DrawVertex) {
        self.push_tri(mode, v0, v1, v2);
        self.push_tri(mode, v0, v2, v3);
    }

    fn push_line(&mut self, mode: DrawMode, x0: f32, y0: f32, x1: f32, y1: f32, half_w_ndc_x: f32, half_w_ndc_y: f32, col: [f32; 4]) {
        let dx = x1 - x0;
        let dy = y1 - y0;
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-6 {
            let v0 = DrawVertex { pos: [x0 - half_w_ndc_x, y0 - half_w_ndc_y], uv: [0.0, 0.0], color: col };
            let v1 = DrawVertex { pos: [x0 + half_w_ndc_x, y0 - half_w_ndc_y], uv: [1.0, 0.0], color: col };
            let v2 = DrawVertex { pos: [x0 + half_w_ndc_x, y0 + half_w_ndc_y], uv: [1.0, 1.0], color: col };
            let v3 = DrawVertex { pos: [x0 - half_w_ndc_x, y0 + half_w_ndc_y], uv: [0.0, 1.0], color: col };
            self.push_quad(mode, v0, v1, v2, v3);
            return;
        }
        let nx = (-dy / len) * half_w_ndc_x;
        let ny = (dx / len) * half_w_ndc_y;
        let v0 = DrawVertex { pos: [x0 - nx, y0 - ny], uv: [0.0, 0.0], color: col };
        let v1 = DrawVertex { pos: [x0 + nx, y0 + ny], uv: [1.0, 0.0], color: col };
        let v2 = DrawVertex { pos: [x1 + nx, y1 + ny], uv: [1.0, 1.0], color: col };
        let v3 = DrawVertex { pos: [x1 - nx, y1 - ny], uv: [0.0, 1.0], color: col };
        self.push_quad(mode, v0, v1, v2, v3);
    }
}

// ============================================================================
// EEL Variable Bindings
// ============================================================================

struct MainVarIds {
    time: VarId,
    fps: VarId,
    frame: VarId,
    progress: VarId,
    bass: VarId,
    mid: VarId,
    treb: VarId,
    vol: VarId,
    bass_att: VarId,
    mid_att: VarId,
    treb_att: VarId,
    vol_att: VarId,
    aspectx: VarId,
    aspecty: VarId,
    meshx: VarId,
    meshy: VarId,
    pixelsx: VarId,
    pixelsy: VarId,
    x: VarId,
    y: VarId,
    rad: VarId,
    ang: VarId,
    zoom: VarId,
    rot: VarId,
    cx: VarId,
    cy: VarId,
    dx: VarId,
    dy: VarId,
    warp: VarId,
    sx: VarId,
    sy: VarId,
    zoomexp: VarId,
    warpanimspeed: VarId,
    decay: VarId,
    gamma: VarId,
    echo_zoom: VarId,
    echo_alpha: VarId,
    echo_orient: VarId,
    wave_r: VarId,
    wave_g: VarId,
    wave_b: VarId,
    wave_a: VarId,
    wave_x: VarId,
    wave_y: VarId,
    wave_mystery: VarId,
    wave_mode: VarId,
    wave_usedots: VarId,
    wave_thick: VarId,
    wave_brighten: VarId,
    ob_size: VarId,
    ob_r: VarId,
    ob_g: VarId,
    ob_b: VarId,
    ob_a: VarId,
    ib_size: VarId,
    ib_r: VarId,
    ib_g: VarId,
    ib_b: VarId,
    ib_a: VarId,
    mv_x: VarId,
    mv_y: VarId,
    mv_dx: VarId,
    mv_dy: VarId,
    mv_l: VarId,
    mv_r: VarId,
    mv_g: VarId,
    mv_b: VarId,
    mv_a: VarId,
    b1n: VarId,
    b2n: VarId,
    b3n: VarId,
    b1x: VarId,
    b2x: VarId,
    b3x: VarId,
    b1ed: VarId,
    darken_center: VarId,
    darken: VarId,
    brighten: VarId,
    solarize: VarId,
    invert: VarId,
    q: [VarId; 32],
}

impl MainVarIds {
    fn resolve(vm: &mut Vm) -> Self {
        Self {
            time: vm.var("time"),
            fps: vm.var("fps"),
            frame: vm.var("frame"),
            progress: vm.var("progress"),
            bass: vm.var("bass"),
            mid: vm.var("mid"),
            treb: vm.var("treb"),
            vol: vm.var("vol"),
            bass_att: vm.var("bass_att"),
            mid_att: vm.var("mid_att"),
            treb_att: vm.var("treb_att"),
            vol_att: vm.var("vol_att"),
            aspectx: vm.var("aspectx"),
            aspecty: vm.var("aspecty"),
            meshx: vm.var("meshx"),
            meshy: vm.var("meshy"),
            pixelsx: vm.var("pixelsx"),
            pixelsy: vm.var("pixelsy"),
            x: vm.var("x"),
            y: vm.var("y"),
            rad: vm.var("rad"),
            ang: vm.var("ang"),
            zoom: vm.var("zoom"),
            rot: vm.var("rot"),
            cx: vm.var("cx"),
            cy: vm.var("cy"),
            dx: vm.var("dx"),
            dy: vm.var("dy"),
            warp: vm.var("warp"),
            sx: vm.var("sx"),
            sy: vm.var("sy"),
            zoomexp: vm.var("zoomexp"),
            warpanimspeed: vm.var("warpanimspeed"),
            decay: vm.var("decay"),
            gamma: vm.var("gamma"),
            echo_zoom: vm.var("echo_zoom"),
            echo_alpha: vm.var("echo_alpha"),
            echo_orient: vm.var("echo_orient"),
            wave_r: vm.var("wave_r"),
            wave_g: vm.var("wave_g"),
            wave_b: vm.var("wave_b"),
            wave_a: vm.var("wave_a"),
            wave_x: vm.var("wave_x"),
            wave_y: vm.var("wave_y"),
            wave_mystery: vm.var("wave_mystery"),
            wave_mode: vm.var("wave_mode"),
            wave_usedots: vm.var("wave_usedots"),
            wave_thick: vm.var("wave_thick"),
            wave_brighten: vm.var("wave_brighten"),
            ob_size: vm.var("ob_size"),
            ob_r: vm.var("ob_r"),
            ob_g: vm.var("ob_g"),
            ob_b: vm.var("ob_b"),
            ob_a: vm.var("ob_a"),
            ib_size: vm.var("ib_size"),
            ib_r: vm.var("ib_r"),
            ib_g: vm.var("ib_g"),
            ib_b: vm.var("ib_b"),
            ib_a: vm.var("ib_a"),
            mv_x: vm.var("mv_x"),
            mv_y: vm.var("mv_y"),
            mv_dx: vm.var("mv_dx"),
            mv_dy: vm.var("mv_dy"),
            mv_l: vm.var("mv_l"),
            mv_r: vm.var("mv_r"),
            mv_g: vm.var("mv_g"),
            mv_b: vm.var("mv_b"),
            mv_a: vm.var("mv_a"),
            b1n: vm.var("b1n"),
            b2n: vm.var("b2n"),
            b3n: vm.var("b3n"),
            b1x: vm.var("b1x"),
            b2x: vm.var("b2x"),
            b3x: vm.var("b3x"),
            b1ed: vm.var("b1ed"),
            darken_center: vm.var("darken_center"),
            darken: vm.var("darken"),
            brighten: vm.var("brighten"),
            solarize: vm.var("solarize"),
            invert: vm.var("invert"),
            q: std::array::from_fn(|i| vm.var(&format!("q{}", i + 1))),
        }
    }
}

struct WaveVarIds {
    time: VarId,
    fps: VarId,
    frame: VarId,
    bass: VarId,
    mid: VarId,
    treb: VarId,
    bass_att: VarId,
    mid_att: VarId,
    treb_att: VarId,
    samples: VarId,
    sample: VarId,
    value1: VarId,
    value2: VarId,
    x: VarId,
    y: VarId,
    r: VarId,
    g: VarId,
    b: VarId,
    a: VarId,
    q: [VarId; 32],
    t: [VarId; 8],
}

impl WaveVarIds {
    fn resolve(vm: &mut Vm) -> Self {
        Self {
            time: vm.var("time"),
            fps: vm.var("fps"),
            frame: vm.var("frame"),
            bass: vm.var("bass"),
            mid: vm.var("mid"),
            treb: vm.var("treb"),
            bass_att: vm.var("bass_att"),
            mid_att: vm.var("mid_att"),
            treb_att: vm.var("treb_att"),
            samples: vm.var("samples"),
            sample: vm.var("sample"),
            value1: vm.var("value1"),
            value2: vm.var("value2"),
            x: vm.var("x"),
            y: vm.var("y"),
            r: vm.var("r"),
            g: vm.var("g"),
            b: vm.var("b"),
            a: vm.var("a"),
            q: std::array::from_fn(|i| vm.var(&format!("q{}", i + 1))),
            t: std::array::from_fn(|i| vm.var(&format!("t{}", i + 1))),
        }
    }
}

struct ShapeVarIds {
    time: VarId,
    fps: VarId,
    frame: VarId,
    bass: VarId,
    mid: VarId,
    treb: VarId,
    bass_att: VarId,
    mid_att: VarId,
    treb_att: VarId,
    instance: VarId,
    num_inst: VarId,
    x: VarId,
    y: VarId,
    rad: VarId,
    ang: VarId,
    sides: VarId,
    textured: VarId,
    additive: VarId,
    thickoutline: VarId,
    tex_ang: VarId,
    tex_zoom: VarId,
    r: VarId,
    g: VarId,
    b: VarId,
    a: VarId,
    r2: VarId,
    g2: VarId,
    b2: VarId,
    a2: VarId,
    border_r: VarId,
    border_g: VarId,
    border_b: VarId,
    border_a: VarId,
    q: [VarId; 32],
    t: [VarId; 8],
}

impl ShapeVarIds {
    fn resolve(vm: &mut Vm) -> Self {
        Self {
            time: vm.var("time"),
            fps: vm.var("fps"),
            frame: vm.var("frame"),
            bass: vm.var("bass"),
            mid: vm.var("mid"),
            treb: vm.var("treb"),
            bass_att: vm.var("bass_att"),
            mid_att: vm.var("mid_att"),
            treb_att: vm.var("treb_att"),
            instance: vm.var("instance"),
            num_inst: vm.var("num_inst"),
            x: vm.var("x"),
            y: vm.var("y"),
            rad: vm.var("rad"),
            ang: vm.var("ang"),
            sides: vm.var("sides"),
            textured: vm.var("textured"),
            additive: vm.var("additive"),
            thickoutline: vm.var("thickoutline"),
            tex_ang: vm.var("tex_ang"),
            tex_zoom: vm.var("tex_zoom"),
            r: vm.var("r"),
            g: vm.var("g"),
            b: vm.var("b"),
            a: vm.var("a"),
            r2: vm.var("r2"),
            g2: vm.var("g2"),
            b2: vm.var("b2"),
            a2: vm.var("a2"),
            border_r: vm.var("border_r"),
            border_g: vm.var("border_g"),
            border_b: vm.var("border_b"),
            border_a: vm.var("border_a"),
            q: std::array::from_fn(|i| vm.var(&format!("q{}", i + 1))),
            t: std::array::from_fn(|i| vm.var(&format!("t{}", i + 1))),
        }
    }
}

struct CompiledWave {
    def: CustomWaveDef,
    vm: Vm,
    ids: WaveVarIds,
    per_frame: Program,
    per_point: Program,
    init_t_snap: [f64; 8],
}

struct CompiledShape {
    def: CustomShapeDef,
    vm: Vm,
    ids: ShapeVarIds,
    per_frame: Program,
    init_t_snap: [f64; 8],
}

struct ActivePreset {
    def: MilkPreset,
    _globals: SharedGlobals,
    main_vm: Vm,
    main_ids: MainVarIds,
    per_frame_prog: Program,
    per_pixel_prog: Program,
    init_q: [f64; 32],
    waves: [CompiledWave; 4],
    shapes: [CompiledShape; 4],
    warp_pipeline: Option<wgpu::RenderPipeline>,
    comp_pipeline: Option<wgpu::RenderPipeline>,
    warp_user_texsize: [[f32; 4]; 8],
    comp_user_texsize: [[f32; 4]; 8],
    warp_slot_names: [Option<String>; 8],
    comp_slot_names: [Option<String>; 8],
    warp_bgs: [wgpu::BindGroup; 2],
    comp_bgs: [wgpu::BindGroup; 2],
    rand_preset: [f32; 4],
    frame_idx: u64,
    preset_start_time: f32,
}

// ============================================================================
// MilkDropEngine
// ============================================================================

struct BlurLevel {
    tmp_tex: wgpu::Texture,
    tmp_view: wgpu::TextureView,
    out_tex: wgpu::Texture,
    out_view: wgpu::TextureView,
    bg_h: wgpu::BindGroup,
    bg_v: wgpu::BindGroup,
    u_buf_h: wgpu::Buffer,
    u_buf_v: wgpu::Buffer,
    w: u32,
    h: u32,
}

pub struct MilkDropEngine {
    target_format: wgpu::TextureFormat,
    width: u32,
    height: u32,

    // Shared MilkDrop shader pipeline layout & vertex module
    md_bgl: wgpu::BindGroupLayout,
    md_pipeline_layout: wgpu::PipelineLayout,
    md_vs_module: wgpu::ShaderModule,
    md_warp_uniform_buf: wgpu::Buffer,
    md_comp_uniform_buf: wgpu::Buffer,

    // Default MilkDrop 1 pipelines & uniform buffer
    def_bgl: wgpu::BindGroupLayout,
    def_pl: wgpu::PipelineLayout,
    def_uniform_buf: wgpu::Buffer,
    def_warp_pipeline: wgpu::RenderPipeline,
    def_comp_pipeline: wgpu::RenderPipeline,
    def_bgs_wrap: [wgpu::BindGroup; 2],
    def_bgs_clamp: [wgpu::BindGroup; 2],

    // 2D Overlay batch pipelines
    draw_solid_alpha: wgpu::RenderPipeline,
    draw_solid_add: wgpu::RenderPipeline,
    draw_tex_alpha: wgpu::RenderPipeline,
    draw_tex_add: wgpu::RenderPipeline,
    draw_bgl: wgpu::BindGroupLayout,
    draw_vbuf: wgpu::Buffer,
    draw_vbuf_cap: usize,
    draw_bg_ping: [wgpu::BindGroup; 2],

    // Blur pyramid pipelines
    blur_bgl: wgpu::BindGroupLayout,
    blur_pl: wgpu::PipelineLayout,
    blur_pipeline: wgpu::RenderPipeline,
    blurs: [BlurLevel; 3],
    blur0_bg_h: [wgpu::BindGroup; 2],

    // Warp + comp geometry buffers
    warp_verts: Vec<WarpVertex>,
    warp_vbuf: wgpu::Buffer,
    comp_vbuf: wgpu::Buffer,
    mesh_ibuf: wgpu::Buffer,
    mesh_index_count: u32,

    // Ping-pong canvas textures (0 & 1)
    ping_tex: [wgpu::Texture; 2],
    ping_view: [wgpu::TextureView; 2],
    ping_idx: usize,
    clear_pending: bool,

    // Samplers
    samp_fw: wgpu::Sampler,
    samp_fc: wgpu::Sampler,
    samp_pw: wgpu::Sampler,
    samp_pc: wgpu::Sampler,

    // Fixed noise textures (kept alive by Texture structs)
    _noise_textures: Vec<wgpu::Texture>,
    noise_views: [wgpu::TextureView; 6], // lq, lq_lite, mq, hq, vol_lq, vol_hq
    default_user_tex: wgpu::Texture,
    default_user_view: wgpu::TextureView,

    // Texture pack catalog + GPU cache
    catalog: TextureCatalog,
    gpu_user_cache: HashMap<String, (wgpu::Texture, wgpu::TextureView, [f32; 4])>,

    // Audio envelope state
    bass_att: f32,
    mid_att: f32,
    treb_att: f32,
    vol_att: f32,
    rng_state: u64,
    last_time: f32,

    // Active preset state
    active: Option<ActivePreset>,
    last_error: Option<String>,
    collector: BatchCollector,
}

impl MilkDropEngine {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        tex_dirs: &[PathBuf],
    ) -> Self {
        let w = width.max(64);
        let h = height.max(64);

        let md_bgl = create_md_bgl(device);
        let md_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("MD Shader PL"),
            bind_group_layouts: &[Some(&md_bgl)],
            immediate_size: 0,
        });
        let md_vs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MD Shared VS"),
            source: wgpu::ShaderSource::Wgsl(MD_SHARED_VS_WGSL.into()),
        });
        let md_warp_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MD Warp Shader Uniforms"),
            size: std::mem::size_of::<MdShaderUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let md_comp_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MD Comp Shader Uniforms"),
            size: std::mem::size_of::<MdShaderUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let samp_fw = create_sampler(device, "MD FW", wgpu::FilterMode::Linear, wgpu::AddressMode::Repeat);
        let samp_fc = create_sampler(device, "MD FC", wgpu::FilterMode::Linear, wgpu::AddressMode::ClampToEdge);
        let samp_pw = create_sampler(device, "MD PW", wgpu::FilterMode::Nearest, wgpu::AddressMode::Repeat);
        let samp_pc = create_sampler(device, "MD PC", wgpu::FilterMode::Nearest, wgpu::AddressMode::ClampToEdge);

        let (def_bgl, def_pl, def_warp_pipeline, def_comp_pipeline) =
            create_default_pipelines(device, &md_vs_module, target_format);
        let def_uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MD Default Uniforms"),
            contents: bytemuck::bytes_of(&DefaultUniforms {
                gamma_adj: 2.0,
                decay: 0.98,
                echo_zoom: 2.0,
                echo_alpha: 0.0,
                echo_orient: 0,
                shader_mix: 0.0,
                flags: 0,
                _pad: 0.0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let (draw_bgl, draw_solid_alpha, draw_solid_add, draw_tex_alpha, draw_tex_add) =
            create_draw_pipelines(device);
        let initial_draw_cap = 16384;
        let draw_vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MD Draw VBuf"),
            size: (initial_draw_cap * std::mem::size_of::<DrawVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let (blur_bgl, blur_pl, blur_pipeline) = create_blur_pipeline(device);

        let ping_tex = [
            create_canvas_tex(device, "MD Ping 0", w, h),
            create_canvas_tex(device, "MD Ping 1", w, h),
        ];
        let ping_view = [
            ping_tex[0].create_view(&wgpu::TextureViewDescriptor::default()),
            ping_tex[1].create_view(&wgpu::TextureViewDescriptor::default()),
        ];

        let def_bgs_wrap = [
            make_def_bg(device, &def_bgl, &def_uniform_buf, &ping_view[0], &samp_fw),
            make_def_bg(device, &def_bgl, &def_uniform_buf, &ping_view[1], &samp_fw),
        ];
        let def_bgs_clamp = [
            make_def_bg(device, &def_bgl, &def_uniform_buf, &ping_view[0], &samp_fc),
            make_def_bg(device, &def_bgl, &def_uniform_buf, &ping_view[1], &samp_fc),
        ];

        let draw_bg_ping = [
            make_draw_bg(device, &draw_bgl, &ping_view[0], &samp_fw),
            make_draw_bg(device, &draw_bgl, &ping_view[1], &samp_fw),
        ];

        let blurs = create_blur_levels(device, &blur_bgl, &samp_fc, &ping_view[0], w, h);
        let blur0_bg_h = [
            make_blur_bg(device, &blur_bgl, &blurs[0].u_buf_h, &ping_view[0], &samp_fc),
            make_blur_bg(device, &blur_bgl, &blurs[0].u_buf_h, &ping_view[1], &samp_fc),
        ];

        let grid_verts = (MESH_X + 1) * (MESH_Y + 1);
        let mut init_warp_verts = Vec::with_capacity(grid_verts);
        let mut comp_verts = Vec::with_capacity(grid_verts);
        build_default_grid(&mut comp_verts, w as f32, h as f32);
        init_warp_verts.extend_from_slice(&comp_verts);

        let warp_vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MD Warp Mesh VBuf"),
            contents: bytemuck::cast_slice(&init_warp_verts),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let comp_vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MD Comp Mesh VBuf"),
            contents: bytemuck::cast_slice(&comp_verts),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let indices = build_grid_indices();
        let mesh_index_count = indices.len() as u32;
        let mesh_ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MD Mesh IBuf"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let (_noise_textures, noise_views) = upload_noise_set(device, queue);
        let default_user_tex = upload_2d_rgba8(device, queue, "MD Dummy User", 2, 2, &[0u8; 16]);
        let default_user_view = default_user_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let catalog = TextureCatalog::scan(tex_dirs);

        Self {
            target_format,
            width: w,
            height: h,
            md_bgl,
            md_pipeline_layout,
            md_vs_module,
            md_warp_uniform_buf,
            md_comp_uniform_buf,
            def_bgl,
            def_pl,
            def_uniform_buf,
            def_warp_pipeline,
            def_comp_pipeline,
            def_bgs_wrap,
            def_bgs_clamp,
            draw_solid_alpha,
            draw_solid_add,
            draw_tex_alpha,
            draw_tex_add,
            draw_bgl,
            draw_vbuf,
            draw_vbuf_cap: initial_draw_cap,
            draw_bg_ping,
            blur_bgl,
            blur_pl,
            blur_pipeline,
            blurs,
            blur0_bg_h,
            warp_verts: init_warp_verts,
            warp_vbuf,
            comp_vbuf,
            mesh_ibuf,
            mesh_index_count,
            ping_tex,
            ping_view,
            ping_idx: 0,
            clear_pending: true,
            samp_fw,
            samp_fc,
            samp_pw,
            samp_pc,
            _noise_textures,
            noise_views,
            default_user_tex,
            default_user_view,
            catalog,
            gpu_user_cache: HashMap::new(),
            bass_att: 0.0,
            mid_att: 0.0,
            treb_att: 0.0,
            vol_att: 0.0,
            rng_state: 0x243F_6A88_85A3_08D3,
            last_time: 0.0,
            active: None,
            last_error: None,
            collector: BatchCollector::new(),
        }
    }

    pub fn update_texture_dirs(&mut self, dirs: &[PathBuf]) {
        self.catalog = TextureCatalog::scan(dirs);
        self.gpu_user_cache.clear();
    }

    pub fn last_error(&self) -> Option<&str> {
        self.last_error.as_deref()
    }

    pub fn resize(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, width: u32, height: u32) {
        let w = width.max(64);
        let h = height.max(64);
        if w == self.width && h == self.height {
            return;
        }
        self.width = w;
        self.height = h;
        self.ping_tex = [
            create_canvas_tex(device, "MD Ping 0", w, h),
            create_canvas_tex(device, "MD Ping 1", w, h),
        ];
        self.ping_view = [
            self.ping_tex[0].create_view(&wgpu::TextureViewDescriptor::default()),
            self.ping_tex[1].create_view(&wgpu::TextureViewDescriptor::default()),
        ];
        self.def_bgs_wrap = [
            make_def_bg(device, &self.def_bgl, &self.def_uniform_buf, &self.ping_view[0], &self.samp_fw),
            make_def_bg(device, &self.def_bgl, &self.def_uniform_buf, &self.ping_view[1], &self.samp_fw),
        ];
        self.def_bgs_clamp = [
            make_def_bg(device, &self.def_bgl, &self.def_uniform_buf, &self.ping_view[0], &self.samp_fc),
            make_def_bg(device, &self.def_bgl, &self.def_uniform_buf, &self.ping_view[1], &self.samp_fc),
        ];
        self.draw_bg_ping = [
            make_draw_bg(device, &self.draw_bgl, &self.ping_view[0], &self.samp_fw),
            make_draw_bg(device, &self.draw_bgl, &self.ping_view[1], &self.samp_fw),
        ];
        let mut comp_verts = Vec::with_capacity((MESH_X + 1) * (MESH_Y + 1));
        build_default_grid(&mut comp_verts, w as f32, h as f32);
        queue.write_buffer(&self.comp_vbuf, 0, bytemuck::cast_slice(&comp_verts));

        self.rebuild_active_bind_groups(device);
        self.clear_pending = true;
    }

    fn rebuild_active_bind_groups(&mut self, device: &wgpu::Device) {
        let blur_in_view = &self.ping_view[1 - self.ping_idx];
        self.blurs = create_blur_levels(
            device,
            &self.blur_bgl,
            &self.samp_fc,
            blur_in_view,
            self.width,
            self.height,
        );
        self.blur0_bg_h = [
            make_blur_bg(device, &self.blur_bgl, &self.blurs[0].u_buf_h, &self.ping_view[0], &self.samp_fc),
            make_blur_bg(device, &self.blur_bgl, &self.blurs[0].u_buf_h, &self.ping_view[1], &self.samp_fc),
        ];
        if let Some(ref act) = self.active {
            let main_samp = if act.def.tex_wrap { &self.samp_fw } else { &self.samp_fc };
            let warp_bgs = [
                self.build_md_bg(device, &self.md_warp_uniform_buf, &self.ping_view[0], main_samp, &act.warp_slot_names),
                self.build_md_bg(device, &self.md_warp_uniform_buf, &self.ping_view[1], main_samp, &act.warp_slot_names),
            ];
            let comp_bgs = [
                self.build_md_bg(device, &self.md_comp_uniform_buf, &self.ping_view[0], main_samp, &act.comp_slot_names),
                self.build_md_bg(device, &self.md_comp_uniform_buf, &self.ping_view[1], main_samp, &act.comp_slot_names),
            ];
            let act_mut = self.active.as_mut().unwrap();
            act_mut.warp_bgs = warp_bgs;
            act_mut.comp_bgs = comp_bgs;
        }
    }

    pub fn load_preset_file(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        path: &Path,
        preset_seed: u32,
        current_time: f32,
    ) -> Result<()> {
        let preset = MilkPreset::from_file(path)?;
        self.load_preset(device, queue, preset, preset_seed, current_time);
        Ok(())
    }

    pub fn load_preset(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        def: MilkPreset,
        preset_seed: u32,
        current_time: f32,
    ) {
        self.last_error = None;
        let globals = Rc::new(RefCell::new(Globals::new()));
        let mut main_vm = Vm::new(globals.clone());
        let main_ids = MainVarIds::resolve(&mut main_vm);
        seed_main_vm_defaults(&mut main_vm, &main_ids, &def);

        // Compile per_frame_init and execute once
        if !def.per_frame_init_src.trim().is_empty() {
            match eel::compile(&def.per_frame_init_src, &mut main_vm) {
                Ok(prog) => prog.run(&mut main_vm),
                Err(e) => {
                    self.last_error = Some(format!("EEL init error: {e}"));
                }
            }
        }
        let init_q: [f64; 32] = std::array::from_fn(|i| main_vm.get(main_ids.q[i]));

        let per_frame_prog = compile_eel_or_note(&def.per_frame_src, &mut main_vm, &mut self.last_error);
        let per_pixel_prog = compile_eel_or_note(&def.per_pixel_src, &mut main_vm, &mut self.last_error);

        let waves = std::array::from_fn(|i| {
            let wdef = def.waves[i].clone();
            let mut wvm = Vm::new(globals.clone());
            let wids = WaveVarIds::resolve(&mut wvm);
            for k in 0..32 {
                wvm.set(wids.q[k], init_q[k]);
            }
            let iprog = compile_eel_or_note(&wdef.init_src, &mut wvm, &mut self.last_error);
            iprog.run(&mut wvm);
            let init_t_snap = std::array::from_fn(|k| wvm.get(wids.t[k]));
            let pf = compile_eel_or_note(&wdef.per_frame_src, &mut wvm, &mut self.last_error);
            let pp = compile_eel_or_note(&wdef.per_point_src, &mut wvm, &mut self.last_error);
            CompiledWave {
                def: wdef,
                vm: wvm,
                ids: wids,
                per_frame: pf,
                per_point: pp,
                init_t_snap,
            }
        });

        let shapes = std::array::from_fn(|i| {
            let sdef = def.shapes[i].clone();
            let mut svm = Vm::new(globals.clone());
            let sids = ShapeVarIds::resolve(&mut svm);
            for k in 0..32 {
                svm.set(sids.q[k], init_q[k]);
            }
            let iprog = compile_eel_or_note(&sdef.init_src, &mut svm, &mut self.last_error);
            iprog.run(&mut svm);
            let init_t_snap = std::array::from_fn(|k| svm.get(sids.t[k]));
            let pf = compile_eel_or_note(&sdef.per_frame_src, &mut svm, &mut self.last_error);
            CompiledShape {
                def: sdef,
                vm: svm,
                ids: sids,
                per_frame: pf,
                init_t_snap,
            }
        });

        // Translate + validate + build warp & comp fragment shaders if present
        let use_warp = def.ps_version_warp > 0 || (def.ps_version > 0 && def.warp_shader_src.is_some());
        let (warp_pipeline, warp_names) = if use_warp {
            if let Some(ref wsrc) = def.warp_shader_src {
                self.compile_md_shader(device, queue, wsrc, ShaderKind::Warp, preset_seed)
            } else {
                (None, Vec::new())
            }
        } else {
            (None, Vec::new())
        };

        let use_comp = def.ps_version_comp > 0 || (def.ps_version > 0 && def.comp_shader_src.is_some());
        let (comp_pipeline, comp_names) = if use_comp {
            if let Some(ref csrc) = def.comp_shader_src {
                self.compile_md_shader(device, queue, csrc, ShaderKind::Comp, preset_seed)
            } else {
                (None, Vec::new())
            }
        } else {
            (None, Vec::new())
        };

        let mut warp_slot_names: [Option<String>; 8] = Default::default();
        let mut warp_user_texsize = [[256.0, 256.0, 1.0 / 256.0, 1.0 / 256.0]; 8];
        for ut in warp_names {
            if ut.slot < 8 {
                let sz = self.ensure_user_tex(device, queue, &ut.name, preset_seed);
                warp_user_texsize[ut.slot] = sz;
                warp_slot_names[ut.slot] = Some(ut.name);
            }
        }

        let mut comp_slot_names: [Option<String>; 8] = Default::default();
        let mut comp_user_texsize = [[256.0, 256.0, 1.0 / 256.0, 1.0 / 256.0]; 8];
        for ut in comp_names {
            if ut.slot < 8 {
                let sz = self.ensure_user_tex(device, queue, &ut.name, preset_seed);
                comp_user_texsize[ut.slot] = sz;
                comp_slot_names[ut.slot] = Some(ut.name);
            }
        }

        let main_samp = if def.tex_wrap { &self.samp_fw } else { &self.samp_fc };
        let warp_bgs = [
            self.build_md_bg(device, &self.md_warp_uniform_buf, &self.ping_view[0], main_samp, &warp_slot_names),
            self.build_md_bg(device, &self.md_warp_uniform_buf, &self.ping_view[1], main_samp, &warp_slot_names),
        ];
        let comp_bgs = [
            self.build_md_bg(device, &self.md_comp_uniform_buf, &self.ping_view[0], main_samp, &comp_slot_names),
            self.build_md_bg(device, &self.md_comp_uniform_buf, &self.ping_view[1], main_samp, &comp_slot_names),
        ];

        let rand_preset = [
            rand_f32_seed(preset_seed.wrapping_mul(747796405).wrapping_add(1)),
            rand_f32_seed(preset_seed.wrapping_mul(2891336453).wrapping_add(2)),
            rand_f32_seed(preset_seed.wrapping_mul(1664525).wrapping_add(3)),
            rand_f32_seed(preset_seed.wrapping_mul(1013904223).wrapping_add(4)),
        ];

        self.clear_pending = true;
        self.active = Some(ActivePreset {
            def,
            _globals: globals,
            main_vm,
            main_ids,
            per_frame_prog,
            per_pixel_prog,
            init_q,
            waves,
            shapes,
            warp_pipeline,
            comp_pipeline,
            warp_user_texsize,
            comp_user_texsize,
            warp_slot_names,
            comp_slot_names,
            warp_bgs,
            comp_bgs,
            rand_preset,
            frame_idx: 0,
            preset_start_time: current_time,
        });
    }

    fn compile_md_shader(
        &mut self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        src: &str,
        kind: ShaderKind,
        _preset_seed: u32,
    ) -> (Option<wgpu::RenderPipeline>, Vec<hlsl::UserTexture>) {
        let tr = match hlsl::translate(src, kind) {
            Ok(t) => t,
            Err(e) => {
                self.last_error = Some(format!("{kind:?} translate fallback: {e}"));
                return (None, Vec::new());
            }
        };

        // Pre-validate with naga on CPU so malformed shaders never trip uncaptured wgpu validation errors
        let mut fe = wgpu::naga::front::glsl::Frontend::default();
        let opts = wgpu::naga::front::glsl::Options::from(wgpu::naga::ShaderStage::Fragment);
        let module = match fe.parse(&opts, &tr.glsl) {
            Ok(m) => m,
            Err(e) => {
                self.last_error = Some(format!("{kind:?} GLSL parse fallback: {e:?}"));
                return (None, Vec::new());
            }
        };
        let mut validator = wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        );
        if let Err(e) = validator.validate(&module) {
            self.last_error = Some(format!("{kind:?} naga validate fallback: {}", e.as_inner()));
            return (None, Vec::new());
        }

        let fs_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(match kind {
                ShaderKind::Warp => "MD Warp GLSL",
                ShaderKind::Comp => "MD Comp GLSL",
            }),
            source: wgpu::ShaderSource::Glsl {
                shader: tr.glsl.into(),
                stage: wgpu::naga::ShaderStage::Fragment,
                defines: Default::default(),
            },
        });

        let target_fmt = match kind {
            ShaderKind::Warp => wgpu::TextureFormat::Rgba8Unorm,
            ShaderKind::Comp => self.target_format,
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(match kind {
                ShaderKind::Warp => "MD Warp Pipeline",
                ShaderKind::Comp => "MD Comp Pipeline",
            }),
            layout: Some(&self.md_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &self.md_vs_module,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[Some(warp_vertex_layout())],
            },
            fragment: Some(wgpu::FragmentState {
                module: &fs_mod,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_fmt,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        (Some(pipeline), tr.user_textures)
    }

    fn ensure_user_tex(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        name: &str,
        preset_seed: u32,
    ) -> [f32; 4] {
        let clean = name.to_ascii_lowercase();
        let cache_key = if clean.starts_with("rand") {
            format!("{clean}_{preset_seed}")
        } else {
            clean
        };
        if let Some((_, _, sz)) = self.gpu_user_cache.get(&cache_key) {
            return *sz;
        }
        let img = self.catalog.load_named(name, preset_seed);
        let tex = upload_2d_rgba8(device, queue, name, img.width, img.height, &img.rgba);
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        let fw = img.width.max(1) as f32;
        let fh = img.height.max(1) as f32;
        let sz = [fw, fh, 1.0 / fw, 1.0 / fh];
        self.gpu_user_cache.insert(cache_key, (tex, view, sz));
        sz
    }

    fn build_md_bg(
        &self,
        device: &wgpu::Device,
        uniform_buf: &wgpu::Buffer,
        main_view: &wgpu::TextureView,
        main_samp: &wgpu::Sampler,
        user_names: &[Option<String>; 8],
    ) -> wgpu::BindGroup {
        let user_views: [&wgpu::TextureView; 8] = std::array::from_fn(|s| {
            if let Some(ref name) = user_names[s] {
                let clean = name.to_ascii_lowercase();
                if let Some((_, v, _)) = self.gpu_user_cache.get(&clean) {
                    return v;
                }
                if let Some((_, (_, v, _))) = self
                    .gpu_user_cache
                    .iter()
                    .find(|(k, _)| k.starts_with(&clean))
                {
                    return v;
                }
            }
            &self.default_user_view
        });

        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MD BindGroup"),
            layout: &self.md_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(main_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.blurs[0].out_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.blurs[1].out_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&self.blurs[2].out_view) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&self.noise_views[0]) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&self.noise_views[1]) },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&self.noise_views[2]) },
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&self.noise_views[3]) },
                wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(&self.noise_views[4]) },
                wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(&self.noise_views[5]) },
                wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::TextureView(user_views[0]) },
                wgpu::BindGroupEntry { binding: 12, resource: wgpu::BindingResource::TextureView(user_views[1]) },
                wgpu::BindGroupEntry { binding: 13, resource: wgpu::BindingResource::TextureView(user_views[2]) },
                wgpu::BindGroupEntry { binding: 14, resource: wgpu::BindingResource::TextureView(user_views[3]) },
                wgpu::BindGroupEntry { binding: 15, resource: wgpu::BindingResource::TextureView(user_views[4]) },
                wgpu::BindGroupEntry { binding: 16, resource: wgpu::BindingResource::TextureView(user_views[5]) },
                wgpu::BindGroupEntry { binding: 17, resource: wgpu::BindingResource::TextureView(user_views[6]) },
                wgpu::BindGroupEntry { binding: 18, resource: wgpu::BindingResource::TextureView(user_views[7]) },
                wgpu::BindGroupEntry { binding: 20, resource: wgpu::BindingResource::Sampler(&self.samp_fw) },
                wgpu::BindGroupEntry { binding: 21, resource: wgpu::BindingResource::Sampler(&self.samp_fc) },
                wgpu::BindGroupEntry { binding: 22, resource: wgpu::BindingResource::Sampler(&self.samp_pw) },
                wgpu::BindGroupEntry { binding: 23, resource: wgpu::BindingResource::Sampler(&self.samp_pc) },
                wgpu::BindGroupEntry { binding: 24, resource: wgpu::BindingResource::Sampler(main_samp) },
            ],
        })
    }

    fn next_rand_01(&mut self) -> f32 {
        let mut x = self.rng_state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng_state = x;
        ((x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32) / 16_777_215.0
    }

    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        time: f32,
        dt: f32,
        features: &AudioFeatures,
    ) {
        let _ = dt;
        let dt_safe = (time - self.last_time).clamp(0.001, 0.25);
        self.last_time = time;
        let fps = (1.0 / dt_safe).clamp(1.0, 240.0);

        // Smooth MilkDrop-style *_att envelopes
        let bass = (features.bass * 1.4).max(0.0);
        let mid = (features.mid * 1.4).max(0.0);
        let treb = (features.treble * 1.4).max(0.0);
        let vol = (bass + mid + treb) / 3.0;
        let att_k = (dt_safe * 2.5).clamp(0.01, 0.25);
        self.bass_att += (bass - self.bass_att) * att_k;
        self.mid_att += (mid - self.mid_att) * att_k;
        self.treb_att += (treb - self.treb_att) * att_k;
        self.vol_att += (vol - self.vol_att) * att_k;

        let rand_frame = [
            self.next_rand_01(),
            self.next_rand_01(),
            self.next_rand_01(),
            self.next_rand_01(),
        ];

        // Clear ping-pong canvases on first frame after load/resize
        if self.clear_pending {
            for view in &self.ping_view {
                let _rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("MD Clear Ping"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
            }
            self.clear_pending = false;
        }

        let Some(mut act) = self.active.take() else {
            return;
        };

        act.frame_idx += 1;
        let progress = ((time - act.preset_start_time) / 30.0).clamp(0.0, 1.0);
        let w_f = self.width as f32;
        let h_f = self.height as f32;
        let (ax, ay) = if w_f >= h_f { (1.0, h_f / w_f) } else { (w_f / h_f, 1.0) };
        let aspect = [ax, ay, 1.0 / ax.max(1e-4), 1.0 / ay.max(1e-4)];

        // 1. Run per_frame EEL
        seed_main_vm_defaults(&mut act.main_vm, &act.main_ids, &act.def);
        for k in 0..32 {
            act.main_vm.set(act.main_ids.q[k], act.init_q[k]);
        }
        act.main_vm.set(act.main_ids.time, time as f64);
        act.main_vm.set(act.main_ids.fps, fps as f64);
        act.main_vm.set(act.main_ids.frame, act.frame_idx as f64);
        act.main_vm.set(act.main_ids.progress, progress as f64);
        act.main_vm.set(act.main_ids.bass, bass as f64);
        act.main_vm.set(act.main_ids.mid, mid as f64);
        act.main_vm.set(act.main_ids.treb, treb as f64);
        act.main_vm.set(act.main_ids.vol, vol as f64);
        act.main_vm.set(act.main_ids.bass_att, self.bass_att as f64);
        act.main_vm.set(act.main_ids.mid_att, self.mid_att as f64);
        act.main_vm.set(act.main_ids.treb_att, self.treb_att as f64);
        act.main_vm.set(act.main_ids.vol_att, self.vol_att as f64);
        act.main_vm.set(act.main_ids.aspectx, ax as f64);
        act.main_vm.set(act.main_ids.aspecty, ay as f64);
        act.main_vm.set(act.main_ids.meshx, MESH_X as f64);
        act.main_vm.set(act.main_ids.meshy, MESH_Y as f64);
        act.main_vm.set(act.main_ids.pixelsx, w_f as f64);
        act.main_vm.set(act.main_ids.pixelsy, h_f as f64);

        act.per_frame_prog.run(&mut act.main_vm);

        let q_out: [f32; 32] = std::array::from_fn(|i| act.main_vm.get(act.main_ids.q[i]) as f32);
        let q_out_f64: [f64; 32] = std::array::from_fn(|i| q_out[i] as f64);

        // Read dynamic preset parameters modified by per_frame
        let decay = act.main_vm.get(act.main_ids.decay) as f32;
        let gamma_adj = act.main_vm.get(act.main_ids.gamma) as f32;
        let echo_zoom = act.main_vm.get(act.main_ids.echo_zoom) as f32;
        let echo_alpha = act.main_vm.get(act.main_ids.echo_alpha) as f32;
        let echo_orient = (act.main_vm.get(act.main_ids.echo_orient).max(0.0) as u32) & 3;
        let warpanimspeed = act.main_vm.get(act.main_ids.warpanimspeed) as f32;
        let zoomexp_pf = act.main_vm.get(act.main_ids.zoomexp) as f32;
        let blur_min = [
            act.main_vm.get(act.main_ids.b1n) as f32,
            act.main_vm.get(act.main_ids.b2n) as f32,
            act.main_vm.get(act.main_ids.b3n) as f32,
        ];
        let blur_max = [
            act.main_vm.get(act.main_ids.b1x) as f32,
            act.main_vm.get(act.main_ids.b2x) as f32,
            act.main_vm.get(act.main_ids.b3x) as f32,
        ];

        // 2. Build Warp Mesh via per_pixel EEL
        compute_warp_mesh(
            &mut self.warp_verts,
            &mut act,
            time,
            warpanimspeed,
            zoomexp_pf,
            ax,
            ay,
        );
        queue.write_buffer(&self.warp_vbuf, 0, bytemuck::cast_slice(&self.warp_verts));

        let prev_idx = self.ping_idx;
        let curr_idx = 1 - prev_idx;

        // 3. Warp Pass: prev_tex -> curr_tex
        let warp_inputs = ShaderInputs {
            time,
            fps,
            frame: act.frame_idx as f32,
            progress,
            bass,
            mid,
            treb,
            vol,
            bass_att: self.bass_att,
            mid_att: self.mid_att,
            treb_att: self.treb_att,
            vol_att: self.vol_att,
            q: q_out,
            texsize: [w_f, h_f],
            aspect,
            rand_preset: act.rand_preset,
            rand_frame,
            blur_min,
            blur_max,
            user_texsize: act.warp_user_texsize,
        };
        queue.write_buffer(
            &self.md_warp_uniform_buf,
            0,
            bytemuck::bytes_of(&MdShaderUniforms::from_inputs(&warp_inputs)),
        );

        let mut flags = 0u32;
        if act.main_vm.get(act.main_ids.brighten) >= 0.5 { flags |= 1; }
        if act.main_vm.get(act.main_ids.darken) >= 0.5 { flags |= 2; }
        if act.main_vm.get(act.main_ids.solarize) >= 0.5 { flags |= 4; }
        if act.main_vm.get(act.main_ids.invert) >= 0.5 { flags |= 8; }

        let def_uniforms = DefaultUniforms {
            gamma_adj,
            decay: decay.clamp(0.0, 1.2),
            echo_zoom,
            echo_alpha,
            echo_orient,
            shader_mix: act.def.shader_mix,
            flags,
            _pad: 0.0,
        };
        queue.write_buffer(&self.def_uniform_buf, 0, bytemuck::bytes_of(&def_uniforms));

        let def_bgs = if act.def.tex_wrap { &self.def_bgs_wrap } else { &self.def_bgs_clamp };

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("MD Warp Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.ping_view[curr_idx],
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rpass.set_vertex_buffer(0, self.warp_vbuf.slice(..));
            rpass.set_index_buffer(self.mesh_ibuf.slice(..), wgpu::IndexFormat::Uint32);
            if let Some(ref pipe) = act.warp_pipeline {
                rpass.set_pipeline(pipe);
                rpass.set_bind_group(0, &act.warp_bgs[prev_idx], &[]);
            } else {
                rpass.set_pipeline(&self.def_warp_pipeline);
                rpass.set_bind_group(0, &def_bgs[prev_idx], &[]);
            }
            rpass.draw_indexed(0..self.mesh_index_count, 0, 0..1);
        }

        // 4. Overlay 2D Primitives onto curr_tex (borders, motion vectors, shapes, waves, main wave)
        self.collector.clear();
        emit_borders_and_darken(&mut self.collector, &act, ax, ay);
        emit_motion_vectors(&mut self.collector, &act, &self.warp_verts, w_f, h_f);
        emit_custom_shapes(
            &mut self.collector,
            &mut act,
            time,
            fps,
            bass,
            mid,
            treb,
            self.bass_att,
            self.mid_att,
            self.treb_att,
            &q_out_f64,
            w_f,
            h_f,
            ax,
            ay,
        );
        emit_custom_waves(
            &mut self.collector,
            &mut act,
            features,
            time,
            fps,
            bass,
            mid,
            treb,
            self.bass_att,
            self.mid_att,
            self.treb_att,
            &q_out_f64,
            w_f,
            h_f,
        );
        emit_main_waveform(&mut self.collector, &act, features, time, vol, w_f, h_f);

        if !self.collector.verts.is_empty() {
            if self.collector.verts.len() > self.draw_vbuf_cap {
                self.draw_vbuf_cap = self.collector.verts.len().next_power_of_two();
                self.draw_vbuf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("MD Draw VBuf"),
                    size: (self.draw_vbuf_cap * std::mem::size_of::<DrawVertex>()) as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
            }
            queue.write_buffer(&self.draw_vbuf, 0, bytemuck::cast_slice(&self.collector.verts));

            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("MD Draw Overlay"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.ping_view[curr_idx],
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rpass.set_vertex_buffer(0, self.draw_vbuf.slice(..));
            for batch in &self.collector.batches {
                let pipe = match batch.mode {
                    DrawMode::SolidAlpha => &self.draw_solid_alpha,
                    DrawMode::SolidAdditive => &self.draw_solid_add,
                    DrawMode::TexAlpha => &self.draw_tex_alpha,
                    DrawMode::TexAdditive => &self.draw_tex_add,
                };
                rpass.set_pipeline(pipe);
                if matches!(batch.mode, DrawMode::TexAlpha | DrawMode::TexAdditive) {
                    rpass.set_bind_group(0, &self.draw_bg_ping[prev_idx], &[]);
                }
                rpass.draw(batch.start..batch.start + batch.count, 0..1);
            }
        }

        // 5. Separable 6-pass Blur Pyramid (curr_tex -> blur1 -> blur2 -> blur3)
        for lvl in 0..3 {
            let b = &self.blurs[lvl];
            let bg_h = if lvl == 0 { &self.blur0_bg_h[curr_idx] } else { &b.bg_h };
            let step_x = 1.3 / b.w.max(1) as f32;
            let step_y = 1.3 / b.h.max(1) as f32;
            queue.write_buffer(
                &b.u_buf_h,
                0,
                bytemuck::bytes_of(&BlurUniforms {
                    dir_step: [step_x, 0.0],
                    min_val: blur_min[lvl],
                    max_val: blur_max[lvl],
                    is_pack: 0,
                    _pad: [0; 3],
                }),
            );
            queue.write_buffer(
                &b.u_buf_v,
                0,
                bytemuck::bytes_of(&BlurUniforms {
                    dir_step: [0.0, step_y],
                    min_val: blur_min[lvl],
                    max_val: blur_max[lvl],
                    is_pack: 1,
                    _pad: [0; 3],
                }),
            );
            // H pass -> tmp
            {
                let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("MD Blur H"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &b.tmp_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                rp.set_pipeline(&self.blur_pipeline);
                rp.set_bind_group(0, bg_h, &[]);
                rp.draw(0..3, 0..1);
            }
            // V pass -> out
            {
                let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("MD Blur V"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &b.out_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                rp.set_pipeline(&self.blur_pipeline);
                rp.set_bind_group(0, &b.bg_v, &[]);
                rp.draw(0..3, 0..1);
            }
        }

        // 6. Composite Pass: curr_tex (+ blur1..3) -> final target_view
        if let Some(ref comp_pipe) = act.comp_pipeline {
            let comp_inputs = ShaderInputs {
                user_texsize: act.comp_user_texsize,
                ..warp_inputs
            };
            queue.write_buffer(
                &self.md_comp_uniform_buf,
                0,
                bytemuck::bytes_of(&MdShaderUniforms::from_inputs(&comp_inputs)),
            );
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("MD Comp Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rpass.set_vertex_buffer(0, self.comp_vbuf.slice(..));
            rpass.set_index_buffer(self.mesh_ibuf.slice(..), wgpu::IndexFormat::Uint32);
            rpass.set_pipeline(comp_pipe);
            rpass.set_bind_group(0, &act.comp_bgs[curr_idx], &[]);
            rpass.draw_indexed(0..self.mesh_index_count, 0, 0..1);
        } else {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("MD Default Comp Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rpass.set_vertex_buffer(0, self.comp_vbuf.slice(..));
            rpass.set_index_buffer(self.mesh_ibuf.slice(..), wgpu::IndexFormat::Uint32);
            rpass.set_pipeline(&self.def_comp_pipeline);
            rpass.set_bind_group(0, &def_bgs[curr_idx], &[]);
            rpass.draw_indexed(0..self.mesh_index_count, 0, 0..1);
        }

        self.ping_idx = curr_idx;
        self.active = Some(act);
    }
}

// ============================================================================
// Warp Mesh & EEL Geometry Emitters
// ============================================================================

fn compile_eel_or_note(src: &str, vm: &mut Vm, err_slot: &mut Option<String>) -> Program {
    if src.trim().is_empty() {
        return Program::empty();
    }
    match eel::compile(src, vm) {
        Ok(p) => p,
        Err(e) => {
            if err_slot.is_none() {
                *err_slot = Some(format!("EEL compile warning: {e}"));
            }
            Program::empty()
        }
    }
}

fn seed_main_vm_defaults(vm: &mut Vm, ids: &MainVarIds, def: &MilkPreset) {
    vm.set(ids.zoom, def.zoom as f64);
    vm.set(ids.rot, def.rot as f64);
    vm.set(ids.cx, def.cx as f64);
    vm.set(ids.cy, def.cy as f64);
    vm.set(ids.dx, def.dx as f64);
    vm.set(ids.dy, def.dy as f64);
    vm.set(ids.warp, def.warp as f64);
    vm.set(ids.sx, def.sx as f64);
    vm.set(ids.sy, def.sy as f64);
    vm.set(ids.zoomexp, def.zoom_exponent as f64);
    vm.set(ids.warpanimspeed, def.warp_anim_speed as f64);
    vm.set(ids.decay, def.decay as f64);
    vm.set(ids.gamma, def.gamma_adj as f64);
    vm.set(ids.echo_zoom, def.echo_zoom as f64);
    vm.set(ids.echo_alpha, def.echo_alpha as f64);
    vm.set(ids.echo_orient, def.echo_orient as f64);
    vm.set(ids.wave_r, def.wave_r as f64);
    vm.set(ids.wave_g, def.wave_g as f64);
    vm.set(ids.wave_b, def.wave_b as f64);
    vm.set(ids.wave_a, def.wave_alpha as f64);
    vm.set(ids.wave_x, def.wave_x as f64);
    vm.set(ids.wave_y, def.wave_y as f64);
    vm.set(ids.wave_mystery, def.wave_param as f64);
    vm.set(ids.wave_mode, def.wave_mode as f64);
    vm.set(ids.wave_usedots, if def.wave_dots { 1.0 } else { 0.0 });
    vm.set(ids.wave_thick, if def.wave_thick { 1.0 } else { 0.0 });
    vm.set(ids.wave_brighten, if def.maximize_wave_color { 1.0 } else { 0.0 });
    vm.set(ids.ob_size, def.ob_size as f64);
    vm.set(ids.ob_r, def.ob_r as f64);
    vm.set(ids.ob_g, def.ob_g as f64);
    vm.set(ids.ob_b, def.ob_b as f64);
    vm.set(ids.ob_a, def.ob_a as f64);
    vm.set(ids.ib_size, def.ib_size as f64);
    vm.set(ids.ib_r, def.ib_r as f64);
    vm.set(ids.ib_g, def.ib_g as f64);
    vm.set(ids.ib_b, def.ib_b as f64);
    vm.set(ids.ib_a, def.ib_a as f64);
    vm.set(ids.mv_x, def.mv_x as f64);
    vm.set(ids.mv_y, def.mv_y as f64);
    vm.set(ids.mv_dx, def.mv_dx as f64);
    vm.set(ids.mv_dy, def.mv_dy as f64);
    vm.set(ids.mv_l, def.mv_l as f64);
    vm.set(ids.mv_r, def.mv_r as f64);
    vm.set(ids.mv_g, def.mv_g as f64);
    vm.set(ids.mv_b, def.mv_b as f64);
    vm.set(ids.mv_a, def.mv_a as f64);
    vm.set(ids.b1n, def.b1n as f64);
    vm.set(ids.b2n, def.b2n as f64);
    vm.set(ids.b3n, def.b3n as f64);
    vm.set(ids.b1x, def.b1x as f64);
    vm.set(ids.b2x, def.b2x as f64);
    vm.set(ids.b3x, def.b3x as f64);
    vm.set(ids.b1ed, def.b1ed as f64);
    vm.set(ids.darken_center, if def.darken_center { 1.0 } else { 0.0 });
    vm.set(ids.darken, if def.darken { 1.0 } else { 0.0 });
    vm.set(ids.brighten, if def.brighten { 1.0 } else { 0.0 });
    vm.set(ids.solarize, if def.solarize { 1.0 } else { 0.0 });
    vm.set(ids.invert, if def.invert { 1.0 } else { 0.0 });
}

fn hue_shader_for(t: f32) -> [f32; 4] {
    [
        0.5 + 0.5 * (t * 0.71).sin(),
        0.5 + 0.5 * (t * 0.93 + 2.094).sin(),
        0.5 + 0.5 * (t * 1.17 + 4.188).sin(),
        1.0,
    ]
}

fn compute_warp_mesh(
    out: &mut Vec<WarpVertex>,
    act: &mut ActivePreset,
    time: f32,
    warp_anim_speed: f32,
    zoomexp_pf: f32,
    ax: f32,
    ay: f32,
) {
    out.clear();
    let pf_zoom = act.main_vm.get(act.main_ids.zoom) as f32;
    let pf_rot = act.main_vm.get(act.main_ids.rot) as f32;
    let pf_cx = act.main_vm.get(act.main_ids.cx) as f32;
    let pf_cy = act.main_vm.get(act.main_ids.cy) as f32;
    let pf_dx = act.main_vm.get(act.main_ids.dx) as f32;
    let pf_dy = act.main_vm.get(act.main_ids.dy) as f32;
    let pf_warp = act.main_vm.get(act.main_ids.warp) as f32;
    let pf_sx = act.main_vm.get(act.main_ids.sx) as f32;
    let pf_sy = act.main_vm.get(act.main_ids.sy) as f32;

    let has_pp = !act.def.per_pixel_src.trim().is_empty();
    let snap = if has_pp { Some(act.main_vm.snapshot()) } else { None };

    let warp_time_v = time * warp_anim_speed;
    let f_warp_scale_inv = 1.0 / act.def.warp_scale.abs().max(0.01);

    for iy in 0..=MESH_Y {
        let fy = iy as f32 / MESH_Y as f32;
        for ix in 0..=MESH_X {
            let fx = ix as f32 / MESH_X as f32;
            let dx_c = (fx - 0.5) * 2.0 * ax;
            let dy_c = (fy - 0.5) * 2.0 * ay;
            let rad = (dx_c * dx_c + dy_c * dy_c).sqrt();
            let ang = dy_c.atan2(dx_c);

            let (zoom, rot, cx, cy, dx, dy, warp, sx, sy) = if let Some(ref s) = snap {
                act.main_vm.restore(s);
                act.main_vm.set(act.main_ids.x, fx as f64);
                act.main_vm.set(act.main_ids.y, fy as f64);
                act.main_vm.set(act.main_ids.rad, rad as f64);
                act.main_vm.set(act.main_ids.ang, ang as f64);
                act.per_pixel_prog.run(&mut act.main_vm);
                (
                    act.main_vm.get(act.main_ids.zoom) as f32,
                    act.main_vm.get(act.main_ids.rot) as f32,
                    act.main_vm.get(act.main_ids.cx) as f32,
                    act.main_vm.get(act.main_ids.cy) as f32,
                    act.main_vm.get(act.main_ids.dx) as f32,
                    act.main_vm.get(act.main_ids.dy) as f32,
                    act.main_vm.get(act.main_ids.warp) as f32,
                    act.main_vm.get(act.main_ids.sx) as f32,
                    act.main_vm.get(act.main_ids.sy) as f32,
                )
            } else {
                (pf_zoom, pf_rot, pf_cx, pf_cy, pf_dx, pf_dy, pf_warp, pf_sx, pf_sy)
            };

            let z_safe = zoom.clamp(0.05, 20.0);
            let z_exp = z_safe.powf(zoomexp_pf.clamp(0.05, 8.0).powf(rad * 2.0 - 1.0));
            let inv_zoom = 1.0 / z_exp.clamp(0.02, 50.0);
            let sx_inv = if sx.abs() > 1e-4 { 1.0 / sx } else { 1.0 };
            let sy_inv = if sy.abs() > 1e-4 { 1.0 / sy } else { 1.0 };

            let du = (fx - cx) * inv_zoom * sx_inv;
            let dv = (fy - cy) * inv_zoom * sy_inv;
            let (sr, cr) = rot.sin_cos();

            let mut u = cx + du * cr + dv * sr - dx;
            let mut v = cy - du * sr + dv * cr - dy;

            if warp.abs() > 1e-5 {
                let w_amt = warp * 0.0035;
                let wx = (u - 0.5) * f_warp_scale_inv * ax;
                let wy = (v - 0.5) * f_warp_scale_inv * ay;
                u += w_amt * (warp_time_v * 0.375 + wy * 8.0 + (wx * 5.0 + warp_time_v * 0.25).sin()).sin();
                v += w_amt * (warp_time_v * 0.313 + wx * 8.0 + (wy * 5.0 + warp_time_v * 0.19).cos()).cos();
            }

            let corner_phase = time + fx * 1.9 + fy * 2.3;
            out.push(WarpVertex {
                pos: [fx * 2.0 - 1.0, 1.0 - fy * 2.0],
                uv: [u, v],
                uv_orig: [fx, fy],
                rad_ang: [rad, ang],
                hue_shader: hue_shader_for(corner_phase),
            });
        }
    }
    if let Some(ref s) = snap {
        act.main_vm.restore(s);
    }
}

fn emit_borders_and_darken(col: &mut BatchCollector, act: &ActivePreset, _ax: f32, _ay: f32) {
    let ob_size = (act.main_vm.get(act.main_ids.ob_size) as f32).clamp(0.0, 0.5);
    let ob_a = (act.main_vm.get(act.main_ids.ob_a) as f32).clamp(0.0, 1.0);
    if ob_size > 0.0005 && ob_a > 0.002 {
        let c = [
            act.main_vm.get(act.main_ids.ob_r) as f32,
            act.main_vm.get(act.main_ids.ob_g) as f32,
            act.main_vm.get(act.main_ids.ob_b) as f32,
            ob_a,
        ];
        emit_frame_border(col, 0.0, ob_size, c);
    }

    let ib_size = (act.main_vm.get(act.main_ids.ib_size) as f32).clamp(0.0, 0.5);
    let ib_a = (act.main_vm.get(act.main_ids.ib_a) as f32).clamp(0.0, 1.0);
    if ib_size > 0.0005 && ib_a > 0.002 {
        let c = [
            act.main_vm.get(act.main_ids.ib_r) as f32,
            act.main_vm.get(act.main_ids.ib_g) as f32,
            act.main_vm.get(act.main_ids.ib_b) as f32,
            ib_a,
        ];
        emit_frame_border(col, ob_size, ob_size + ib_size, c);
    }

    if act.main_vm.get(act.main_ids.darken_center) >= 0.5 {
        let center = DrawVertex { pos: [0.0, 0.0], uv: [0.5, 0.5], color: [0.0, 0.0, 0.0, 0.28] };
        let r = 0.09_f32;
        let pts = [(0.0, r), (r, 0.0), (0.0, -r), (-r, 0.0)];
        for i in 0..4 {
            let p0 = pts[i];
            let p1 = pts[(i + 1) % 4];
            col.push_tri(
                DrawMode::SolidAlpha,
                center,
                DrawVertex { pos: [p0.0, p0.1], uv: [0.5, 0.5], color: [0.0, 0.0, 0.0, 0.0] },
                DrawVertex { pos: [p1.0, p1.1], uv: [0.5, 0.5], color: [0.0, 0.0, 0.0, 0.0] },
            );
        }
    }
}

fn emit_frame_border(col: &mut BatchCollector, r_in: f32, r_out: f32, rgba: [f32; 4]) {
    let i0 = -1.0 + r_in * 2.0;
    let i1 = 1.0 - r_in * 2.0;
    let o0 = -1.0 + r_out * 2.0;
    let o1 = 1.0 - r_out * 2.0;
    if o0 >= o1 || i0 >= i1 {
        return;
    }
    let mk = |px: f32, py: f32| DrawVertex { pos: [px, py], uv: [0.0, 0.0], color: rgba };
    // Top, Bottom, Left, Right strips
    col.push_quad(DrawMode::SolidAlpha, mk(i0, i1), mk(i1, i1), mk(o1, o1), mk(o0, o1));
    col.push_quad(DrawMode::SolidAlpha, mk(o0, o0), mk(o1, o0), mk(i1, i0), mk(i0, i0));
    col.push_quad(DrawMode::SolidAlpha, mk(i0, i0), mk(o0, o0), mk(o0, o1), mk(i0, i1));
    col.push_quad(DrawMode::SolidAlpha, mk(o1, o0), mk(i1, i0), mk(i1, i1), mk(o1, o1));
}

fn emit_motion_vectors(
    col: &mut BatchCollector,
    act: &ActivePreset,
    warp_verts: &[WarpVertex],
    w_f: f32,
    h_f: f32,
) {
    let mv_a = (act.main_vm.get(act.main_ids.mv_a) as f32).clamp(0.0, 1.0);
    if mv_a <= 0.002 {
        return;
    }
    let nx = (act.main_vm.get(act.main_ids.mv_x) as usize).clamp(1, 64);
    let ny = (act.main_vm.get(act.main_ids.mv_y) as usize).clamp(1, 48);
    let mv_dx = act.main_vm.get(act.main_ids.mv_dx) as f32;
    let mv_dy = act.main_vm.get(act.main_ids.mv_dy) as f32;
    let mv_l = (act.main_vm.get(act.main_ids.mv_l) as f32).clamp(0.0, 10.0);
    let c = [
        act.main_vm.get(act.main_ids.mv_r) as f32,
        act.main_vm.get(act.main_ids.mv_g) as f32,
        act.main_vm.get(act.main_ids.mv_b) as f32,
        mv_a,
    ];
    let hw_x = 1.6 / w_f;
    let hw_y = 1.6 / h_f;

    for iy in 0..ny {
        let fy = (iy as f32 + 0.5) / ny as f32;
        let my = ((fy * MESH_Y as f32).round() as usize).min(MESH_Y);
        for ix in 0..nx {
            let fx = (ix as f32 + 0.5) / nx as f32;
            let mx = ((fx * MESH_X as f32).round() as usize).min(MESH_X);
            let wv = warp_verts[my * (MESH_X + 1) + mx];

            let x0 = (fx + mv_dx) * 2.0 - 1.0;
            let y0 = 1.0 - (fy + mv_dy) * 2.0;
            let du = (fx - wv.uv[0]) * mv_l;
            let dv = (fy - wv.uv[1]) * mv_l;
            let x1 = x0 + du * 2.0;
            let y1 = y0 - dv * 2.0;
            col.push_line(DrawMode::SolidAlpha, x0, y0, x1, y1, hw_x, hw_y, c);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_custom_shapes(
    col: &mut BatchCollector,
    act: &mut ActivePreset,
    time: f32,
    fps: f32,
    bass: f32,
    mid: f32,
    treb: f32,
    bass_att: f32,
    mid_att: f32,
    treb_att: f32,
    q_vals: &[f64; 32],
    w_f: f32,
    h_f: f32,
    ax: f32,
    ay: f32,
) {
    let hw_x = 1.5 / w_f;
    let hw_y = 1.5 / h_f;

    for shape in &mut act.shapes {
        if !shape.def.enabled {
            continue;
        }
        let n_inst = shape.def.num_inst.clamp(1, 1024);
        for inst in 0..n_inst {
            shape.vm.set(shape.ids.time, time as f64);
            shape.vm.set(shape.ids.fps, fps as f64);
            shape.vm.set(shape.ids.frame, act.frame_idx as f64);
            shape.vm.set(shape.ids.bass, bass as f64);
            shape.vm.set(shape.ids.mid, mid as f64);
            shape.vm.set(shape.ids.treb, treb as f64);
            shape.vm.set(shape.ids.bass_att, bass_att as f64);
            shape.vm.set(shape.ids.mid_att, mid_att as f64);
            shape.vm.set(shape.ids.treb_att, treb_att as f64);
            shape.vm.set(shape.ids.instance, inst as f64);
            shape.vm.set(shape.ids.num_inst, n_inst as f64);
            for k in 0..32 {
                shape.vm.set(shape.ids.q[k], q_vals[k]);
            }
            for k in 0..8 {
                shape.vm.set(shape.ids.t[k], shape.init_t_snap[k]);
            }

            let d = &shape.def;
            shape.vm.set(shape.ids.x, d.x as f64);
            shape.vm.set(shape.ids.y, d.y as f64);
            shape.vm.set(shape.ids.rad, d.rad as f64);
            shape.vm.set(shape.ids.ang, d.ang as f64);
            shape.vm.set(shape.ids.sides, d.sides as f64);
            shape.vm.set(shape.ids.textured, if d.textured { 1.0 } else { 0.0 });
            shape.vm.set(shape.ids.additive, if d.additive { 1.0 } else { 0.0 });
            shape.vm.set(shape.ids.thickoutline, if d.thick_outline { 1.0 } else { 0.0 });
            shape.vm.set(shape.ids.tex_ang, d.tex_ang as f64);
            shape.vm.set(shape.ids.tex_zoom, d.tex_zoom as f64);
            shape.vm.set(shape.ids.r, d.r as f64);
            shape.vm.set(shape.ids.g, d.g as f64);
            shape.vm.set(shape.ids.b, d.b as f64);
            shape.vm.set(shape.ids.a, d.a as f64);
            shape.vm.set(shape.ids.r2, d.r2 as f64);
            shape.vm.set(shape.ids.g2, d.g2 as f64);
            shape.vm.set(shape.ids.b2, d.b2 as f64);
            shape.vm.set(shape.ids.a2, d.a2 as f64);
            shape.vm.set(shape.ids.border_r, d.border_r as f64);
            shape.vm.set(shape.ids.border_g, d.border_g as f64);
            shape.vm.set(shape.ids.border_b, d.border_b as f64);
            shape.vm.set(shape.ids.border_a, d.border_a as f64);

            shape.per_frame.run(&mut shape.vm);

            let sx = shape.vm.get(shape.ids.x) as f32;
            let sy = shape.vm.get(shape.ids.y) as f32;
            let rad = (shape.vm.get(shape.ids.rad) as f32).clamp(0.0, 2.0);
            let ang = shape.vm.get(shape.ids.ang) as f32;
            let sides = (shape.vm.get(shape.ids.sides).round() as usize).clamp(3, 100);
            let textured = shape.vm.get(shape.ids.textured) >= 0.5;
            let additive = shape.vm.get(shape.ids.additive) >= 0.5;
            let thick = shape.vm.get(shape.ids.thickoutline) >= 0.5;
            let tex_ang = shape.vm.get(shape.ids.tex_ang) as f32;
            let tex_zoom = (shape.vm.get(shape.ids.tex_zoom) as f32).abs().max(1e-3);

            let c_in = [
                shape.vm.get(shape.ids.r) as f32,
                shape.vm.get(shape.ids.g) as f32,
                shape.vm.get(shape.ids.b) as f32,
                (shape.vm.get(shape.ids.a) as f32).clamp(0.0, 1.0),
            ];
            let c_out = [
                shape.vm.get(shape.ids.r2) as f32,
                shape.vm.get(shape.ids.g2) as f32,
                shape.vm.get(shape.ids.b2) as f32,
                (shape.vm.get(shape.ids.a2) as f32).clamp(0.0, 1.0),
            ];
            let c_bd = [
                shape.vm.get(shape.ids.border_r) as f32,
                shape.vm.get(shape.ids.border_g) as f32,
                shape.vm.get(shape.ids.border_b) as f32,
                (shape.vm.get(shape.ids.border_a) as f32).clamp(0.0, 1.0),
            ];

            let cx_ndc = sx * 2.0 - 1.0;
            let cy_ndc = sy * 2.0 - 1.0;
            let rx = rad * 2.0 * ay;
            let ry = rad * 2.0 * ax;

            let mode = match (textured, additive) {
                (false, false) => DrawMode::SolidAlpha,
                (false, true) => DrawMode::SolidAdditive,
                (true, false) => DrawMode::TexAlpha,
                (true, true) => DrawMode::TexAdditive,
            };

            let center_v = DrawVertex {
                pos: [cx_ndc, cy_ndc],
                uv: [0.5, 0.5],
                color: c_in,
            };

            let step = std::f32::consts::TAU / sides as f32;
            let (sta, cta) = tex_ang.sin_cos();
            let mut prev_pos = [0.0_f32; 2];
            let mut first_pos = [0.0_f32; 2];

            if c_in[3] > 0.002 || c_out[3] > 0.002 {
                let mut prev_v = center_v;
                let mut first_v = center_v;
                for s in 0..=sides {
                    let th = ang + step * (s as f32);
                    let px = cx_ndc + th.cos() * rx;
                    let py = cy_ndc + th.sin() * ry;
                    let t_th = th + tex_ang - ang;
                    let tu = 0.5 + (t_th.cos() * cta + t_th.sin() * sta) * (0.5 / tex_zoom);
                    let tv = 0.5 - (t_th.sin() * cta - t_th.cos() * sta) * (0.5 / tex_zoom);
                    let v = DrawVertex {
                        pos: [px, py],
                        uv: [tu, tv],
                        color: c_out,
                    };
                    if s == 0 {
                        first_v = v;
                        first_pos = [px, py];
                    } else {
                        col.push_tri(mode, center_v, prev_v, v);
                    }
                    prev_v = v;
                    let _ = first_v;
                    let _ = (prev_pos, first_pos);
                    prev_pos = [px, py];
                }
            }

            if c_bd[3] > 0.002 {
                let bw_x = if thick { hw_x * 2.2 } else { hw_x };
                let bw_y = if thick { hw_y * 2.2 } else { hw_y };
                let bmode = if additive { DrawMode::SolidAdditive } else { DrawMode::SolidAlpha };
                let mut lp = [0.0_f32; 2];
                for s in 0..=sides {
                    let th = ang + step * (s as f32);
                    let px = cx_ndc + th.cos() * rx;
                    let py = cy_ndc + th.sin() * ry;
                    if s > 0 {
                        col.push_line(bmode, lp[0], lp[1], px, py, bw_x, bw_y, c_bd);
                    }
                    lp = [px, py];
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_custom_waves(
    col: &mut BatchCollector,
    act: &mut ActivePreset,
    features: &AudioFeatures,
    time: f32,
    fps: f32,
    bass: f32,
    mid: f32,
    treb: f32,
    bass_att: f32,
    mid_att: f32,
    treb_att: f32,
    q_vals: &[f64; 32],
    w_f: f32,
    h_f: f32,
) {
    let base_hw_x = 1.4 / w_f;
    let base_hw_y = 1.4 / h_f;

    for wave in &mut act.waves {
        if !wave.def.enabled {
            continue;
        }
        wave.vm.set(wave.ids.time, time as f64);
        wave.vm.set(wave.ids.fps, fps as f64);
        wave.vm.set(wave.ids.frame, act.frame_idx as f64);
        wave.vm.set(wave.ids.bass, bass as f64);
        wave.vm.set(wave.ids.mid, mid as f64);
        wave.vm.set(wave.ids.treb, treb as f64);
        wave.vm.set(wave.ids.bass_att, bass_att as f64);
        wave.vm.set(wave.ids.mid_att, mid_att as f64);
        wave.vm.set(wave.ids.treb_att, treb_att as f64);
        for k in 0..32 {
            wave.vm.set(wave.ids.q[k], q_vals[k]);
        }
        for k in 0..8 {
            wave.vm.set(wave.ids.t[k], wave.init_t_snap[k]);
        }

        let d = &wave.def;
        wave.vm.set(wave.ids.samples, d.samples as f64);
        wave.vm.set(wave.ids.r, d.r as f64);
        wave.vm.set(wave.ids.g, d.g as f64);
        wave.vm.set(wave.ids.b, d.b as f64);
        wave.vm.set(wave.ids.a, d.a as f64);

        wave.per_frame.run(&mut wave.vm);

        let samples = (wave.vm.get(wave.ids.samples).round() as usize).clamp(2, 2048);
        let fr = wave.vm.get(wave.ids.r) as f32;
        let fg = wave.vm.get(wave.ids.g) as f32;
        let fb = wave.vm.get(wave.ids.b) as f32;
        let fa = wave.vm.get(wave.ids.a) as f32;

        let mode = if d.additive { DrawMode::SolidAdditive } else { DrawMode::SolidAlpha };
        let hw_x = if d.draw_thick { base_hw_x * 2.2 } else { base_hw_x };
        let hw_y = if d.draw_thick { base_hw_y * 2.2 } else { base_hw_y };

        let mut prev_pt: Option<(f32, f32, [f32; 4])> = None;
        for s in 0..samples {
            let t = s as f32 / (samples - 1) as f32;
            let (v1, v2) = if d.spectrum {
                let sv = sample_slice_catmull(&features.spectrum, t) * d.scaling;
                let sv2 = sample_slice_catmull(&features.spectrum, (t + 0.03).min(1.0)) * d.scaling;
                (sv, sv2)
            } else {
                let wv = sample_slice_catmull(&features.waveform, t) * d.scaling * 0.5;
                let sep_off = (d.sep as f32 * 0.004).clamp(-0.3, 0.3);
                let wv2 = sample_slice_catmull(&features.waveform, (t + sep_off).rem_euclid(1.0)) * d.scaling * 0.5;
                (wv, wv2)
            };

            wave.vm.set(wave.ids.sample, t as f64);
            wave.vm.set(wave.ids.value1, v1 as f64);
            wave.vm.set(wave.ids.value2, v2 as f64);
            wave.vm.set(wave.ids.x, t as f64);
            wave.vm.set(wave.ids.y, 0.5 + v1 as f64 * 0.25);
            wave.vm.set(wave.ids.r, fr as f64);
            wave.vm.set(wave.ids.g, fg as f64);
            wave.vm.set(wave.ids.b, fb as f64);
            wave.vm.set(wave.ids.a, fa as f64);

            wave.per_point.run(&mut wave.vm);

            let px = (wave.vm.get(wave.ids.x) as f32) * 2.0 - 1.0;
            let py = (wave.vm.get(wave.ids.y) as f32) * 2.0 - 1.0;
            let pr = wave.vm.get(wave.ids.r) as f32;
            let pg = wave.vm.get(wave.ids.g) as f32;
            let pb = wave.vm.get(wave.ids.b) as f32;
            let pa = (wave.vm.get(wave.ids.a) as f32).clamp(0.0, 1.0);
            let rgba = [pr, pg, pb, pa];

            if pa > 0.002 {
                if d.use_dots {
                    let dot_x = hw_x * 1.4;
                    let dot_y = hw_y * 1.4;
                    col.push_line(mode, px, py, px, py, dot_x, dot_y, rgba);
                } else if let Some((lx, ly, lc)) = prev_pt {
                    let avg = [
                        (lc[0] + rgba[0]) * 0.5,
                        (lc[1] + rgba[1]) * 0.5,
                        (lc[2] + rgba[2]) * 0.5,
                        (lc[3] + rgba[3]) * 0.5,
                    ];
                    col.push_line(mode, lx, ly, px, py, hw_x, hw_y, avg);
                }
            }
            prev_pt = Some((px, py, rgba));
        }
    }
}

fn emit_main_waveform(
    col: &mut BatchCollector,
    act: &ActivePreset,
    features: &AudioFeatures,
    time: f32,
    vol: f32,
    w_f: f32,
    h_f: f32,
) {
    let mut alpha = (act.main_vm.get(act.main_ids.wave_a) as f32).clamp(0.0, 1.0);
    if act.def.mod_wave_alpha_by_volume {
        let s0 = act.def.mod_wave_alpha_start;
        let s1 = act.def.mod_wave_alpha_end.max(s0 + 1e-4);
        alpha *= ((vol - s0) / (s1 - s0)).clamp(0.0, 1.0);
    }
    if alpha <= 0.002 {
        return;
    }

    let mut wr = act.main_vm.get(act.main_ids.wave_r) as f32;
    let mut wg = act.main_vm.get(act.main_ids.wave_g) as f32;
    let mut wb = act.main_vm.get(act.main_ids.wave_b) as f32;
    if act.main_vm.get(act.main_ids.wave_brighten) >= 0.5 {
        let m = wr.max(wg).max(wb);
        if m > 1e-4 {
            wr /= m;
            wg /= m;
            wb /= m;
        }
    }
    let rgba = [wr.clamp(0.0, 2.0), wg.clamp(0.0, 2.0), wb.clamp(0.0, 2.0), alpha];

    let wx = act.main_vm.get(act.main_ids.wave_x) as f32 * 2.0 - 1.0;
    let wy = act.main_vm.get(act.main_ids.wave_y) as f32 * 2.0 - 1.0;
    let scale = act.def.wave_scale * 0.25;
    let mystery = act.main_vm.get(act.main_ids.wave_mystery) as f32;
    let wmode = (act.main_vm.get(act.main_ids.wave_mode).round() as u32) & 7;
    let thick = act.main_vm.get(act.main_ids.wave_thick) >= 0.5;
    let dots = act.main_vm.get(act.main_ids.wave_usedots) >= 0.5;

    let draw_mode = if act.def.additive_waves {
        DrawMode::SolidAdditive
    } else {
        DrawMode::SolidAlpha
    };
    let hw_x = if thick { 3.0 / w_f } else { 1.4 / w_f };
    let hw_y = if thick { 3.0 / h_f } else { 1.4 / h_f };
    let asp = (h_f / w_f).clamp(0.25, 4.0);

    let pts_count = 160usize;
    let mut prev: Option<(f32, f32)> = None;
    for i in 0..pts_count {
        let f = i as f32 / (pts_count - 1) as f32;
        let s0 = sample_slice_catmull(&features.waveform, f) * scale;
        let s1 = sample_slice_catmull(&features.waveform, (f + 0.15).rem_euclid(1.0)) * scale;

        let (px, py) = match wmode {
            0 => {
                // Polar circle
                let a = f * std::f32::consts::TAU + mystery * 2.0 + time * 0.2;
                let r = (0.35 + s0).max(0.02);
                (wx + a.cos() * r * asp, wy + a.sin() * r)
            }
            1 => {
                // XY Lissajous spiral
                let a = f * std::f32::consts::TAU * 2.0 + time * 0.3;
                let r = 0.28 + s0 * 0.8 + mystery * 0.12 * f;
                (wx + a.cos() * r * asp, wy + a.sin() * r)
            }
            2 => {
                // Centered X-Y scope
                (wx + s0 * 1.4, wy + s1 * 1.4)
            }
            3 => {
                // Dual Lissajous scatter
                (wx + (f - 0.5) * 1.1 + s0 * 0.6, wy + s1 * 1.1)
            }
            4 => {
                // Centered horizontal wave
                (wx + (f * 2.0 - 1.0) * 0.9, wy + s0 * 0.9)
            }
            5 => {
                // Flower / lobe polar curve
                let a = f * std::f32::consts::TAU;
                let lobes = (mystery * 4.0).round() + 3.0;
                let r = (0.3 + 0.08 * (a * lobes + time).cos() + s0).max(0.02);
                (wx + a.cos() * r * asp, wy + a.sin() * r)
            }
            6 => {
                // Vertical / tilted horizon wave
                let (st, ct) = (mystery * 1.57).sin_cos();
                let u = (f - 0.5) * 1.6;
                let v = s0;
                (wx + u * ct - v * st, wy + u * st + v * ct)
            }
            _ => {
                // Double horizontal ribbon
                let dir = if i % 2 == 0 { 1.0 } else { -1.0 };
                (wx + (f - 0.5) * 1.7, wy + dir * (0.04 + s0.abs() * 0.9))
            }
        };

        if dots {
            col.push_line(draw_mode, px, py, px, py, hw_x * 1.5, hw_y * 1.5, rgba);
        } else if let Some((lx, ly)) = prev {
            col.push_line(draw_mode, lx, ly, px, py, hw_x, hw_y, rgba);
        }
        prev = Some((px, py));
    }
}

fn sample_slice_catmull(slice: &[f32], t: f32) -> f32 {
    if slice.is_empty() {
        return 0.0;
    }
    let last = slice.len() - 1;
    let fidx = t.clamp(0.0, 1.0) * last as f32;
    let i1 = (fidx.floor() as usize).min(last);
    let u = fidx - i1 as f32;
    let p0 = slice[i1.saturating_sub(1)];
    let p1 = slice[i1];
    let p2 = slice[(i1 + 1).min(last)];
    let p3 = slice[(i1 + 2).min(last)];
    0.5 * ((2.0 * p1)
        + (-p0 + p2) * u
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u * u
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u * u * u)
}

fn rand_f32_seed(seed: u32) -> f32 {
    let mut x = seed;
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    (x & 0xFFFF) as f32 / 65535.0
}

// ============================================================================
// GPU Resource Constructors
// ============================================================================

fn build_default_grid(out: &mut Vec<WarpVertex>, w: f32, h: f32) {
    out.clear();
    let (ax, ay) = if w >= h { (1.0, h / w) } else { (w / h, 1.0) };
    for iy in 0..=MESH_Y {
        let fy = iy as f32 / MESH_Y as f32;
        for ix in 0..=MESH_X {
            let fx = ix as f32 / MESH_X as f32;
            let dx_c = (fx - 0.5) * 2.0 * ax;
            let dy_c = (fy - 0.5) * 2.0 * ay;
            out.push(WarpVertex {
                pos: [fx * 2.0 - 1.0, 1.0 - fy * 2.0],
                uv: [fx, fy],
                uv_orig: [fx, fy],
                rad_ang: [(dx_c * dx_c + dy_c * dy_c).sqrt(), dy_c.atan2(dx_c)],
                hue_shader: [1.0, 1.0, 1.0, 1.0],
            });
        }
    }
}

fn build_grid_indices() -> Vec<u32> {
    let stride = (MESH_X + 1) as u32;
    let mut idx = Vec::with_capacity(MESH_X * MESH_Y * 6);
    for iy in 0..MESH_Y as u32 {
        for ix in 0..MESH_X as u32 {
            let a = iy * stride + ix;
            let b = a + 1;
            let c = a + stride;
            let d = c + 1;
            idx.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }
    idx
}

fn warp_vertex_layout() -> wgpu::VertexBufferLayout<'static> {
    const ATTRS: [wgpu::VertexAttribute; 5] = [
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 0, shader_location: 0 },
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 8, shader_location: 1 },
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 16, shader_location: 2 },
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 24, shader_location: 3 },
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 32, shader_location: 4 },
    ];
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<WarpVertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &ATTRS,
    }
}

fn draw_vertex_layout() -> wgpu::VertexBufferLayout<'static> {
    const ATTRS: [wgpu::VertexAttribute; 3] = [
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 0, shader_location: 0 },
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 8, shader_location: 1 },
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 16, shader_location: 2 },
    ];
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<DrawVertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &ATTRS,
    }
}

fn create_md_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let mut entries = Vec::with_capacity(24);
    entries.push(wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    });
    for b in 1..=18 {
        let dim = if b == 9 || b == 10 {
            wgpu::TextureViewDimension::D3
        } else {
            wgpu::TextureViewDimension::D2
        };
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: dim,
                multisampled: false,
            },
            count: None,
        });
    }
    for b in 20..=24 {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        });
    }
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("MD Shader BGL"),
        entries: &entries,
    })
}

fn create_sampler(
    device: &wgpu::Device,
    label: &str,
    filter: wgpu::FilterMode,
    addr: wgpu::AddressMode,
) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some(label),
        address_mode_u: addr,
        address_mode_v: addr,
        address_mode_w: addr,
        mag_filter: filter,
        min_filter: filter,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        ..Default::default()
    })
}

fn create_canvas_tex(device: &wgpu::Device, label: &str, width: u32, height: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}

fn upload_2d_rgba8(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    width: u32,
    height: u32,
    rgba: &[u8],
) -> wgpu::Texture {
    let size = wgpu::Extent3d {
        width: width.max(1),
        height: height.max(1),
        depth_or_array_layers: 1,
    };
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * size.width),
            rows_per_image: Some(size.height),
        },
        size,
    );
    tex
}

fn upload_3d_rgba8(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    dim: u32,
    rgba: &[u8],
) -> wgpu::Texture {
    let size = wgpu::Extent3d {
        width: dim,
        height: dim,
        depth_or_array_layers: dim,
    };
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * dim),
            rows_per_image: Some(dim),
        },
        size,
    );
    tex
}

fn upload_noise_set(device: &wgpu::Device, queue: &wgpu::Queue) -> (Vec<wgpu::Texture>, [wgpu::TextureView; 6]) {
    let n = NoiseSet::generate();
    let texs = vec![
        upload_2d_rgba8(device, queue, "noise_lq", 256, 256, &n.noise_lq),
        upload_2d_rgba8(device, queue, "noise_lq_lite", 32, 32, &n.noise_lq_lite),
        upload_2d_rgba8(device, queue, "noise_mq", 256, 256, &n.noise_mq),
        upload_2d_rgba8(device, queue, "noise_hq", 256, 256, &n.noise_hq),
        upload_3d_rgba8(device, queue, "noisevol_lq", 32, &n.noisevol_lq),
        upload_3d_rgba8(device, queue, "noisevol_hq", 32, &n.noisevol_hq),
    ];
    let views = [
        texs[0].create_view(&wgpu::TextureViewDescriptor::default()),
        texs[1].create_view(&wgpu::TextureViewDescriptor::default()),
        texs[2].create_view(&wgpu::TextureViewDescriptor::default()),
        texs[3].create_view(&wgpu::TextureViewDescriptor::default()),
        texs[4].create_view(&wgpu::TextureViewDescriptor::default()),
        texs[5].create_view(&wgpu::TextureViewDescriptor::default()),
    ];
    (texs, views)
}

fn create_default_pipelines(
    device: &wgpu::Device,
    vs_mod: &wgpu::ShaderModule,
    target_format: wgpu::TextureFormat,
) -> (
    wgpu::BindGroupLayout,
    wgpu::PipelineLayout,
    wgpu::RenderPipeline,
    wgpu::RenderPipeline,
) {
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("MD Default BGL"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("MD Default PL"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let fs_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("MD Default FS WGSL"),
        source: wgpu::ShaderSource::Wgsl(MD_DEFAULT_SHADER_WGSL.into()),
    });

    let make = |label: &str, entry: &str, fmt: wgpu::TextureFormat| {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: vs_mod,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[Some(warp_vertex_layout())],
            },
            fragment: Some(wgpu::FragmentState {
                module: &fs_mod,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: fmt,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        })
    };

    let warp = make("MD Default Warp", "fs_default_warp", wgpu::TextureFormat::Rgba8Unorm);
    let comp = make("MD Default Comp", "fs_default_comp", target_format);
    (bgl, pl, warp, comp)
}

fn create_draw_pipelines(
    device: &wgpu::Device,
) -> (
    wgpu::BindGroupLayout,
    wgpu::RenderPipeline,
    wgpu::RenderPipeline,
    wgpu::RenderPipeline,
    wgpu::RenderPipeline,
) {
    let sh = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("MD Draw WGSL"),
        source: wgpu::ShaderSource::Wgsl(MD_DRAW_WGSL.into()),
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("MD Draw BGL"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });
    let solid_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("MD Solid Draw PL"),
        bind_group_layouts: &[],
        immediate_size: 0,
    });
    let tex_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("MD Tex Draw PL"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });

    let additive_blend = wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::SrcAlpha,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        },
        alpha: wgpu::BlendComponent::OVER,
    };

    let make = |label: &str, pl: &wgpu::PipelineLayout, entry: &str, blend: wgpu::BlendState| {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(pl),
            vertex: wgpu::VertexState {
                module: &sh,
                entry_point: Some("vs_draw"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[Some(draw_vertex_layout())],
            },
            fragment: Some(wgpu::FragmentState {
                module: &sh,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: Some(blend),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        })
    };

    (
        bgl,
        make("MD Draw Solid Alpha", &solid_pl, "fs_solid", wgpu::BlendState::ALPHA_BLENDING),
        make("MD Draw Solid Add", &solid_pl, "fs_solid", additive_blend),
        make("MD Draw Tex Alpha", &tex_pl, "fs_textured", wgpu::BlendState::ALPHA_BLENDING),
        make("MD Draw Tex Add", &tex_pl, "fs_textured", additive_blend),
    )
}

fn make_def_bg(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    u_buf: &wgpu::Buffer,
    main_view: &wgpu::TextureView,
    main_samp: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("MD Default BG"),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: u_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(main_view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(main_samp) },
        ],
    })
}

fn make_draw_bg(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    view: &wgpu::TextureView,
    samp: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("MD Draw BG"),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(samp) },
        ],
    })
}

fn create_blur_pipeline(
    device: &wgpu::Device,
) -> (wgpu::BindGroupLayout, wgpu::PipelineLayout, wgpu::RenderPipeline) {
    let sh = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("MD Blur WGSL"),
        source: wgpu::ShaderSource::Wgsl(MD_BLUR_WGSL.into()),
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("MD Blur BGL"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("MD Blur PL"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("MD Blur Pipeline"),
        layout: Some(&pl),
        vertex: wgpu::VertexState {
            module: &sh,
            entry_point: Some("vs_blur"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &sh,
            entry_point: Some("fs_blur"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });
    (bgl, pl, pipeline)
}

fn make_blur_bg(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    u_buf: &wgpu::Buffer,
    src_view: &wgpu::TextureView,
    samp: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("MD Blur BG"),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: u_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(src_view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(samp) },
        ],
    })
}

fn create_blur_levels(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    samp: &wgpu::Sampler,
    init_src_view: &wgpu::TextureView,
    full_w: u32,
    full_h: u32,
) -> [BlurLevel; 3] {
    let dims = [
        ((full_w / 2).max(32), (full_h / 2).max(32)),
        ((full_w / 4).max(16), (full_h / 4).max(16)),
        ((full_w / 8).max(8), (full_h / 8).max(8)),
    ];
    let mut levels: Vec<BlurLevel> = Vec::with_capacity(3);
    for (idx, &(bw, bh)) in dims.iter().enumerate() {
        let tmp_tex = create_canvas_tex(device, &format!("MD BlurTmp {idx}"), bw, bh);
        let tmp_view = tmp_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let out_tex = create_canvas_tex(device, &format!("MD BlurOut {idx}"), bw, bh);
        let out_view = out_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let init_u = BlurUniforms {
            dir_step: [1.0 / bw as f32, 0.0],
            min_val: 0.0,
            max_val: 1.0,
            is_pack: 0,
            _pad: [0; 3],
        };
        let u_buf_h = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MD Blur U H"),
            contents: bytemuck::bytes_of(&init_u),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let u_buf_v = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MD Blur U V"),
            contents: bytemuck::bytes_of(&init_u),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let src_v = if idx == 0 {
            init_src_view
        } else {
            &levels[idx - 1].out_view
        };
        let bg_h = make_blur_bg(device, bgl, &u_buf_h, src_v, samp);
        let bg_v = make_blur_bg(device, bgl, &u_buf_v, &tmp_view, samp);
        levels.push(BlurLevel {
            tmp_tex,
            tmp_view,
            out_tex,
            out_view,
            bg_h,
            bg_v,
            u_buf_h,
            u_buf_v,
            w: bw,
            h: bh,
        });
    }
    levels.try_into().ok().unwrap()
}
