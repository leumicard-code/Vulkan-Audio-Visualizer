// ============================================================================
// milkdrop/preset.rs — .milk Preset Parser (Params, EEL Blocks, Shaders)
// ============================================================================

use anyhow::{Context, Result};
use std::fs;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct CustomWaveDef {
    pub enabled: bool,
    pub samples: usize,
    pub sep: i32,
    pub spectrum: bool,
    pub use_dots: bool,
    pub draw_thick: bool,
    pub additive: bool,
    pub scaling: f32,
    pub smoothing: f32,
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
    pub init_src: String,
    pub per_frame_src: String,
    pub per_point_src: String,
}

impl Default for CustomWaveDef {
    fn default() -> Self {
        Self {
            enabled: false,
            samples: 512,
            sep: 0,
            spectrum: false,
            use_dots: false,
            draw_thick: false,
            additive: false,
            scaling: 1.0,
            smoothing: 0.5,
            r: 1.0,
            g: 1.0,
            b: 1.0,
            a: 1.0,
            init_src: String::new(),
            per_frame_src: String::new(),
            per_point_src: String::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct CustomShapeDef {
    pub enabled: bool,
    pub sides: u32,
    pub additive: bool,
    pub thick_outline: bool,
    pub textured: bool,
    pub num_inst: usize,
    pub x: f32,
    pub y: f32,
    pub rad: f32,
    pub ang: f32,
    pub tex_ang: f32,
    pub tex_zoom: f32,
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
    pub r2: f32,
    pub g2: f32,
    pub b2: f32,
    pub a2: f32,
    pub border_r: f32,
    pub border_g: f32,
    pub border_b: f32,
    pub border_a: f32,
    pub init_src: String,
    pub per_frame_src: String,
}

impl Default for CustomShapeDef {
    fn default() -> Self {
        Self {
            enabled: false,
            sides: 4,
            additive: false,
            thick_outline: false,
            textured: false,
            num_inst: 1,
            x: 0.5,
            y: 0.5,
            rad: 0.1,
            ang: 0.0,
            tex_ang: 0.0,
            tex_zoom: 1.0,
            r: 1.0,
            g: 0.0,
            b: 0.0,
            a: 1.0,
            r2: 0.0,
            g2: 1.0,
            b2: 0.0,
            a2: 0.0,
            border_r: 1.0,
            border_g: 1.0,
            border_b: 1.0,
            border_a: 0.1,
            init_src: String::new(),
            per_frame_src: String::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct MilkPreset {
    pub ps_version: u32,
    pub ps_version_warp: u32,
    pub ps_version_comp: u32,

    pub gamma_adj: f32,
    pub decay: f32,
    pub echo_zoom: f32,
    pub echo_alpha: f32,
    pub echo_orient: u32,
    pub wave_mode: u32,
    pub additive_waves: bool,
    pub wave_dots: bool,
    pub wave_thick: bool,
    pub mod_wave_alpha_by_volume: bool,
    pub maximize_wave_color: bool,
    pub tex_wrap: bool,
    pub darken_center: bool,
    pub red_blue_stereo: bool,
    pub brighten: bool,
    pub darken: bool,
    pub solarize: bool,
    pub invert: bool,

    pub wave_alpha: f32,
    pub wave_scale: f32,
    pub wave_smoothing: f32,
    pub wave_param: f32,
    pub mod_wave_alpha_start: f32,
    pub mod_wave_alpha_end: f32,

    pub warp_anim_speed: f32,
    pub warp_scale: f32,
    pub zoom_exponent: f32,
    pub shader_mix: f32,

    pub zoom: f32,
    pub rot: f32,
    pub cx: f32,
    pub cy: f32,
    pub dx: f32,
    pub dy: f32,
    pub warp: f32,
    pub sx: f32,
    pub sy: f32,

    pub wave_r: f32,
    pub wave_g: f32,
    pub wave_b: f32,
    pub wave_x: f32,
    pub wave_y: f32,

    pub ob_size: f32,
    pub ob_r: f32,
    pub ob_g: f32,
    pub ob_b: f32,
    pub ob_a: f32,

    pub ib_size: f32,
    pub ib_r: f32,
    pub ib_g: f32,
    pub ib_b: f32,
    pub ib_a: f32,

    pub mv_x: f32,
    pub mv_y: f32,
    pub mv_dx: f32,
    pub mv_dy: f32,
    pub mv_l: f32,
    pub mv_r: f32,
    pub mv_g: f32,
    pub mv_b: f32,
    pub mv_a: f32,

    pub b1n: f32,
    pub b2n: f32,
    pub b3n: f32,
    pub b1x: f32,
    pub b2x: f32,
    pub b3x: f32,
    pub b1ed: f32,

    pub per_frame_init_src: String,
    pub per_frame_src: String,
    pub per_pixel_src: String,

    pub waves: [CustomWaveDef; 4],
    pub shapes: [CustomShapeDef; 4],

    pub warp_shader_src: Option<String>,
    pub comp_shader_src: Option<String>,
}

impl Default for MilkPreset {
    fn default() -> Self {
        Self {
            ps_version: 0,
            ps_version_warp: 0,
            ps_version_comp: 0,
            gamma_adj: 2.0,
            decay: 0.98,
            echo_zoom: 2.0,
            echo_alpha: 0.0,
            echo_orient: 0,
            wave_mode: 0,
            additive_waves: false,
            wave_dots: false,
            wave_thick: false,
            mod_wave_alpha_by_volume: false,
            maximize_wave_color: true,
            tex_wrap: true,
            darken_center: false,
            red_blue_stereo: false,
            brighten: false,
            darken: false,
            solarize: false,
            invert: false,
            wave_alpha: 0.8,
            wave_scale: 1.0,
            wave_smoothing: 0.75,
            wave_param: 0.0,
            mod_wave_alpha_start: 0.75,
            mod_wave_alpha_end: 0.95,
            warp_anim_speed: 1.0,
            warp_scale: 1.0,
            zoom_exponent: 1.0,
            shader_mix: 0.0,
            zoom: 1.0,
            rot: 0.0,
            cx: 0.5,
            cy: 0.5,
            dx: 0.0,
            dy: 0.0,
            warp: 1.0,
            sx: 1.0,
            sy: 1.0,
            wave_r: 1.0,
            wave_g: 1.0,
            wave_b: 1.0,
            wave_x: 0.5,
            wave_y: 0.5,
            ob_size: 0.01,
            ob_r: 0.0,
            ob_g: 0.0,
            ob_b: 0.0,
            ob_a: 0.0,
            ib_size: 0.01,
            ib_r: 0.25,
            ib_g: 0.25,
            ib_b: 0.25,
            ib_a: 0.0,
            mv_x: 12.0,
            mv_y: 9.0,
            mv_dx: 0.0,
            mv_dy: 0.0,
            mv_l: 0.9,
            mv_r: 1.0,
            mv_g: 1.0,
            mv_b: 1.0,
            mv_a: 0.0,
            b1n: 0.0,
            b2n: 0.0,
            b3n: 0.0,
            b1x: 1.0,
            b2x: 1.0,
            b3x: 1.0,
            b1ed: 0.25,
            per_frame_init_src: String::new(),
            per_frame_src: String::new(),
            per_pixel_src: String::new(),
            waves: std::array::from_fn(|_| CustomWaveDef::default()),
            shapes: std::array::from_fn(|_| CustomShapeDef::default()),
            warp_shader_src: None,
            comp_shader_src: None,
        }
    }
}

fn parse_f32(v: &str, default: f32) -> f32 {
    v.trim().parse::<f32>().ok().filter(|x| x.is_finite()).unwrap_or(default)
}

fn parse_u32(v: &str, default: u32) -> u32 {
    v.trim()
        .parse::<u32>()
        .ok()
        .or_else(|| v.trim().parse::<f32>().ok().map(|f| f.max(0.0) as u32))
        .unwrap_or(default)
}

fn parse_bool(v: &str) -> bool {
    parse_f32(v, 0.0) >= 0.5
}

fn join_numbered(mut items: Vec<(u32, String)>) -> String {
    items.sort_by_key(|(n, _)| *n);
    let mut out = String::new();
    for (_, line) in items {
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(&line);
    }
    out
}

impl MilkPreset {
    pub fn from_file(path: &Path) -> Result<Self> {
        let bytes = fs::read(path)
            .with_context(|| format!("Failed to read MilkDrop preset '{}'", path.display()))?;
        let text = String::from_utf8_lossy(&bytes);
        Ok(Self::parse(&text))
    }

    pub fn parse(src: &str) -> Self {
        let mut p = MilkPreset::default();

        let mut kv_pairs: Vec<(&str, &str)> = Vec::new();
        let mut pf_init: Vec<(u32, String)> = Vec::new();
        let mut pf_lines: Vec<(u32, String)> = Vec::new();
        let mut pp_lines: Vec<(u32, String)> = Vec::new();

        let mut wave_init: [Vec<(u32, String)>; 4] = Default::default();
        let mut wave_pf: [Vec<(u32, String)>; 4] = Default::default();
        let mut wave_pp: [Vec<(u32, String)>; 4] = Default::default();

        let mut shape_init: [Vec<(u32, String)>; 4] = Default::default();
        let mut shape_pf: [Vec<(u32, String)>; 4] = Default::default();

        for raw_line in src.lines() {
            let line = raw_line.trim_end_matches('\r');
            let trimmed = line.trim_start();
            if trimmed.is_empty() || trimmed.starts_with('[') || trimmed.starts_with("//") {
                continue;
            }
            let Some((raw_k, v)) = line.split_once('=') else {
                continue;
            };
            let k = raw_k.trim();
            kv_pairs.push((k, v));

            // Numbered EEL code blocks
            if let Some(rest) = k.strip_prefix("per_frame_init_") {
                if let Ok(n) = rest.parse::<u32>() {
                    pf_init.push((n, v.to_string()));
                }
                continue;
            }
            if let Some(rest) = k.strip_prefix("per_frame_") {
                if let Ok(n) = rest.parse::<u32>() {
                    pf_lines.push((n, v.to_string()));
                }
                continue;
            }
            if let Some(rest) = k.strip_prefix("per_pixel_") {
                if let Ok(n) = rest.parse::<u32>() {
                    pp_lines.push((n, v.to_string()));
                }
                continue;
            }

            if let Some(rest) = k.strip_prefix("wave_") {
                if let Some((idx_s, sub)) = rest.split_once('_') {
                    if let Ok(idx) = idx_s.parse::<usize>() {
                        if idx < 4 {
                            if let Some(n_s) = sub.strip_prefix("init") {
                                if let Ok(n) = n_s.parse::<u32>() {
                                    wave_init[idx].push((n, v.to_string()));
                                    continue;
                                }
                            } else if let Some(n_s) = sub.strip_prefix("per_frame") {
                                if let Ok(n) = n_s.parse::<u32>() {
                                    wave_pf[idx].push((n, v.to_string()));
                                    continue;
                                }
                            } else if let Some(n_s) = sub.strip_prefix("per_point") {
                                if let Ok(n) = n_s.parse::<u32>() {
                                    wave_pp[idx].push((n, v.to_string()));
                                    continue;
                                }
                            }
                        }
                    }
                }
            }

            if let Some(rest) = k.strip_prefix("shape_") {
                if let Some((idx_s, sub)) = rest.split_once('_') {
                    if let Ok(idx) = idx_s.parse::<usize>() {
                        if idx < 4 {
                            if let Some(n_s) = sub.strip_prefix("init") {
                                if let Ok(n) = n_s.parse::<u32>() {
                                    shape_init[idx].push((n, v.to_string()));
                                    continue;
                                }
                            } else if let Some(n_s) = sub.strip_prefix("per_frame") {
                                if let Ok(n) = n_s.parse::<u32>() {
                                    shape_pf[idx].push((n, v.to_string()));
                                    continue;
                                }
                            }
                        }
                    }
                }
            }

            if let Some(rest) = k.strip_prefix("wavecode_") {
                if let Some((idx_s, prop)) = rest.split_once('_') {
                    if let Ok(idx) = idx_s.parse::<usize>() {
                        if idx < 4 {
                            let w = &mut p.waves[idx];
                            match prop {
                                "enabled" => w.enabled = parse_bool(v),
                                "samples" => w.samples = (parse_u32(v, 512) as usize).clamp(2, 2048),
                                "sep" => w.sep = parse_f32(v, 0.0) as i32,
                                "bSpectrum" => w.spectrum = parse_bool(v),
                                "bUseDots" => w.use_dots = parse_bool(v),
                                "bDrawThick" => w.draw_thick = parse_bool(v),
                                "bAdditive" => w.additive = parse_bool(v),
                                "scaling" => w.scaling = parse_f32(v, 1.0),
                                "smoothing" => w.smoothing = parse_f32(v, 0.5),
                                "r" => w.r = parse_f32(v, 1.0),
                                "g" => w.g = parse_f32(v, 1.0),
                                "b" => w.b = parse_f32(v, 1.0),
                                "a" => w.a = parse_f32(v, 1.0),
                                _ => {}
                            }
                            continue;
                        }
                    }
                }
            }

            if let Some(rest) = k.strip_prefix("shapecode_") {
                if let Some((idx_s, prop)) = rest.split_once('_') {
                    if let Ok(idx) = idx_s.parse::<usize>() {
                        if idx < 4 {
                            let s = &mut p.shapes[idx];
                            match prop {
                                "enabled" => s.enabled = parse_bool(v),
                                "sides" => s.sides = parse_u32(v, 4).clamp(3, 100),
                                "additive" => s.additive = parse_bool(v),
                                "thickOutline" => s.thick_outline = parse_bool(v),
                                "textured" => s.textured = parse_bool(v),
                                "num_inst" => s.num_inst = (parse_u32(v, 1) as usize).clamp(1, 1024),
                                "x" => s.x = parse_f32(v, 0.5),
                                "y" => s.y = parse_f32(v, 0.5),
                                "rad" => s.rad = parse_f32(v, 0.1),
                                "ang" => s.ang = parse_f32(v, 0.0),
                                "tex_ang" => s.tex_ang = parse_f32(v, 0.0),
                                "tex_zoom" => s.tex_zoom = parse_f32(v, 1.0),
                                "r" => s.r = parse_f32(v, 1.0),
                                "g" => s.g = parse_f32(v, 0.0),
                                "b" => s.b = parse_f32(v, 0.0),
                                "a" => s.a = parse_f32(v, 1.0),
                                "r2" => s.r2 = parse_f32(v, 0.0),
                                "g2" => s.g2 = parse_f32(v, 1.0),
                                "b2" => s.b2 = parse_f32(v, 0.0),
                                "a2" => s.a2 = parse_f32(v, 0.0),
                                "border_r" => s.border_r = parse_f32(v, 1.0),
                                "border_g" => s.border_g = parse_f32(v, 1.0),
                                "border_b" => s.border_b = parse_f32(v, 1.0),
                                "border_a" => s.border_a = parse_f32(v, 0.1),
                                _ => {}
                            }
                            continue;
                        }
                    }
                }
            }

            match k {
                "PSVERSION" => p.ps_version = parse_u32(v, 0),
                "PSVERSION_WARP" => p.ps_version_warp = parse_u32(v, 0),
                "PSVERSION_COMP" | "PSVERSION_comp" => p.ps_version_comp = parse_u32(v, 0),
                "fGammaAdj" | "gamma" => p.gamma_adj = parse_f32(v, 2.0),
                "fDecay" => p.decay = parse_f32(v, 0.98),
                "fVideoEchoZoom" => p.echo_zoom = parse_f32(v, 2.0),
                "fVideoEchoAlpha" => p.echo_alpha = parse_f32(v, 0.0),
                "nVideoEchoOrientation" => p.echo_orient = parse_u32(v, 0) & 3,
                "nWaveMode" => p.wave_mode = parse_u32(v, 0) & 7,
                "bAdditiveWaves" => p.additive_waves = parse_bool(v),
                "bWaveDots" => p.wave_dots = parse_bool(v),
                "bWaveThick" | "wave_thick" => p.wave_thick = parse_bool(v),
                "bModWaveAlphaByVolume" => p.mod_wave_alpha_by_volume = parse_bool(v),
                "bMaximizeWaveColor" => p.maximize_wave_color = parse_bool(v),
                "bTexWrap" | "wrap" => p.tex_wrap = parse_bool(v),
                "bDarkenCenter" => p.darken_center = parse_bool(v),
                "bRedBlueStereo" => p.red_blue_stereo = parse_bool(v),
                "bBrighten" => p.brighten = parse_bool(v),
                "bDarken" => p.darken = parse_bool(v),
                "bSolarize" => p.solarize = parse_bool(v),
                "bInvert" => p.invert = parse_bool(v),
                "fWaveAlpha" => p.wave_alpha = parse_f32(v, 0.8),
                "fWaveScale" => p.wave_scale = parse_f32(v, 1.0),
                "fWaveSmoothing" => p.wave_smoothing = parse_f32(v, 0.75),
                "fWaveParam" => p.wave_param = parse_f32(v, 0.0),
                "fModWaveAlphaStart" => p.mod_wave_alpha_start = parse_f32(v, 0.75),
                "fModWaveAlphaEnd" => p.mod_wave_alpha_end = parse_f32(v, 0.95),
                "fWarpAnimSpeed" | "fwarpAnimSpeed" => p.warp_anim_speed = parse_f32(v, 1.0),
                "fWarpScale" | "fwarpScale" => p.warp_scale = parse_f32(v, 1.0),
                "fZoomExponent" => p.zoom_exponent = parse_f32(v, 1.0),
                "fShader" => p.shader_mix = parse_f32(v, 0.0),
                "zoom" => p.zoom = parse_f32(v, 1.0),
                "rot" => p.rot = parse_f32(v, 0.0),
                "cx" => p.cx = parse_f32(v, 0.5),
                "cy" => p.cy = parse_f32(v, 0.5),
                "dx" => p.dx = parse_f32(v, 0.0),
                "dy" => p.dy = parse_f32(v, 0.0),
                "warp" => p.warp = parse_f32(v, 1.0),
                "sx" => p.sx = parse_f32(v, 1.0),
                "sy" => p.sy = parse_f32(v, 1.0),
                "wave_r" => p.wave_r = parse_f32(v, 1.0),
                "wave_g" => p.wave_g = parse_f32(v, 1.0),
                "wave_b" => p.wave_b = parse_f32(v, 1.0),
                "wave_x" => p.wave_x = parse_f32(v, 0.5),
                "wave_y" => p.wave_y = parse_f32(v, 0.5),
                "ob_size" => p.ob_size = parse_f32(v, 0.01),
                "ob_r" => p.ob_r = parse_f32(v, 0.0),
                "ob_g" => p.ob_g = parse_f32(v, 0.0),
                "ob_b" => p.ob_b = parse_f32(v, 0.0),
                "ob_a" => p.ob_a = parse_f32(v, 0.0),
                "ib_size" => p.ib_size = parse_f32(v, 0.01),
                "ib_r" => p.ib_r = parse_f32(v, 0.25),
                "ib_g" => p.ib_g = parse_f32(v, 0.25),
                "ib_b" => p.ib_b = parse_f32(v, 0.25),
                "ib_a" => p.ib_a = parse_f32(v, 0.0),
                "nMotionVectorsX" => p.mv_x = parse_f32(v, 12.0),
                "nMotionVectorsY" => p.mv_y = parse_f32(v, 9.0),
                "mv_dx" => p.mv_dx = parse_f32(v, 0.0),
                "mv_dy" => p.mv_dy = parse_f32(v, 0.0),
                "mv_l" => p.mv_l = parse_f32(v, 0.9),
                "mv_r" => p.mv_r = parse_f32(v, 1.0),
                "mv_g" => p.mv_g = parse_f32(v, 1.0),
                "mv_b" => p.mv_b = parse_f32(v, 1.0),
                "mv_a" => p.mv_a = parse_f32(v, 0.0),
                "b1n" => p.b1n = parse_f32(v, 0.0),
                "b2n" => p.b2n = parse_f32(v, 0.0),
                "b3n" => p.b3n = parse_f32(v, 0.0),
                "b1x" => p.b1x = parse_f32(v, 1.0),
                "b2x" => p.b2x = parse_f32(v, 1.0),
                "b3x" => p.b3x = parse_f32(v, 1.0),
                "b1ed" => p.b1ed = parse_f32(v, 0.25),
                _ => {}
            }
        }

        p.per_frame_init_src = join_numbered(pf_init);
        p.per_frame_src = join_numbered(pf_lines);
        p.per_pixel_src = join_numbered(pp_lines);

        for i in 0..4 {
            p.waves[i].init_src = join_numbered(std::mem::take(&mut wave_init[i]));
            p.waves[i].per_frame_src = join_numbered(std::mem::take(&mut wave_pf[i]));
            p.waves[i].per_point_src = join_numbered(std::mem::take(&mut wave_pp[i]));

            p.shapes[i].init_src = join_numbered(std::mem::take(&mut shape_init[i]));
            p.shapes[i].per_frame_src = join_numbered(std::mem::take(&mut shape_pf[i]));
        }

        p.warp_shader_src = crate::milkdrop::hlsl::extract_shader_source(
            &kv_pairs,
            crate::milkdrop::hlsl::ShaderKind::Warp,
        );
        p.comp_shader_src = crate::milkdrop::hlsl::extract_shader_source(
            &kv_pairs,
            crate::milkdrop::hlsl::ShaderKind::Comp,
        );

        p
    }
}
