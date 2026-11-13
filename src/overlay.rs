// ============================================================================
// overlay.rs — Fade-Out Preset HUD Overlay & Khronos Vulkan Logo Watermark
// ============================================================================

use ab_glyph::{Font, FontArc, PxScale, ScaleFont};
use std::path::{Path, PathBuf};
use wgpu::util::DeviceExt;

const WATERMARK_PNG: &[u8] = include_bytes!("../assets/vulkan_logo_watermark.png");

const HOLD_SECONDS: f32 = 3.5;
const FADE_SECONDS: f32 = 0.9;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct OverlayUniforms {
    rect_ndc: [f32; 4], // x_min, y_min, x_max, y_max in NDC
    alpha: f32,
    _pad: [f32; 3],
}

const OVERLAY_WGSL: &str = r#"
struct Uniforms {
    rect_ndc: vec4<f32>,
    alpha: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    let corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 1.0),
    );
    let c = corners[vid];
    let x = mix(u.rect_ndc.x, u.rect_ndc.z, c.x);
    let y = mix(u.rect_ndc.w, u.rect_ndc.y, c.y); // flip Y so uv.y=0 is top
    var out: VsOut;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = c;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let c = textureSample(tex, samp, in.uv);
    return vec4<f32>(c.rgb, c.a * u.alpha);
}
"#;

pub struct OverlayRenderer {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    font: Option<FontArc>,

    // Preset banner card
    banner_w: u32,
    banner_h: u32,
    banner_uniform_buf: wgpu::Buffer,
    banner_bind_group: wgpu::BindGroup,

    // Vulkan watermark
    wm_w: u32,
    wm_h: u32,
    wm_uniform_buf: wgpu::Buffer,
    wm_bind_group: wgpu::BindGroup,

    elapsed_since_trigger: f32,
}

impl OverlayRenderer {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Overlay WGSL"),
            source: wgpu::ShaderSource::Wgsl(OVERLAY_WGSL.into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Overlay BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
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
            label: Some("Overlay PL"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Overlay Pipeline"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Overlay Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let wm_rgba = build_watermark_card();
        let (wm_w, wm_h) = (wm_rgba.0, wm_rgba.1);
        let wm_tex = create_rgba_texture(device, queue, "Vulkan Watermark", wm_w, wm_h, &wm_rgba.2);
        let wm_uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("WM Uniforms"),
            contents: bytemuck::bytes_of(&OverlayUniforms {
                rect_ndc: [0.0; 4],
                alpha: 1.0,
                _pad: [0.0; 3],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let wm_bind_group = make_bg(device, &bgl, &wm_uniform_buf, &wm_tex, &sampler);

        // Initial 1x1 dummy banner until set_preset_info is called
        let banner_tex = create_rgba_texture(device, queue, "Overlay Banner", 2, 2, &[0; 16]);
        let banner_uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Banner Uniforms"),
            contents: bytemuck::bytes_of(&OverlayUniforms {
                rect_ndc: [0.0; 4],
                alpha: 1.0,
                _pad: [0.0; 3],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let banner_bind_group = make_bg(device, &bgl, &banner_uniform_buf, &banner_tex, &sampler);

        Self {
            pipeline,
            bgl,
            sampler,
            font: load_system_font(),
            banner_w: 2,
            banner_h: 2,
            banner_uniform_buf,
            banner_bind_group,
            wm_w,
            wm_h,
            wm_uniform_buf,
            wm_bind_group,
            elapsed_since_trigger: 0.0,
        }
    }

    pub fn trigger(&mut self) {
        self.elapsed_since_trigger = 0.0;
    }

    pub fn set_preset_info(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        index_1based: usize,
        total: usize,
        name: &str,
        category: &str,
        is_milkdrop: bool,
    ) {
        let (w, h, rgba) = self.rasterize_banner(index_1based, total, name, category, is_milkdrop);
        let tex = create_rgba_texture(device, queue, "Overlay Banner", w, h, &rgba);
        self.banner_w = w;
        self.banner_h = h;
        self.banner_bind_group =
            make_bg(device, &self.bgl, &self.banner_uniform_buf, &tex, &self.sampler);
        self.elapsed_since_trigger = 0.0;
    }

    pub fn current_alpha(&self) -> f32 {
        if self.elapsed_since_trigger <= HOLD_SECONDS {
            1.0
        } else {
            let t = (self.elapsed_since_trigger - HOLD_SECONDS) / FADE_SECONDS;
            (1.0 - t).clamp(0.0, 1.0)
        }
    }

    pub fn render(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        viewport_w: u32,
        viewport_h: u32,
        dt: f32,
    ) {
        self.elapsed_since_trigger += dt;
        let alpha = self.current_alpha();
        if alpha <= 0.002 || viewport_w < 64 || viewport_h < 64 {
            return;
        }
        // Smooth ease-out curve
        let smooth_alpha = alpha * alpha * (3.0 - 2.0 * alpha);

        let vw = viewport_w as f32;
        let vh = viewport_h as f32;
        let margin = 18.0_f32;

        // Banner at bottom-left, scaled down gracefully if window is narrow
        let max_bw = (vw - self.wm_w as f32 - margin * 3.0).max(160.0);
        let b_scale = (max_bw / self.banner_w as f32).min(1.0);
        let bw = self.banner_w as f32 * b_scale;
        let bh = self.banner_h as f32 * b_scale;
        let bx0 = margin;
        let by1 = vh - margin;
        let by0 = by1 - bh;
        let bx1 = bx0 + bw;

        let b_rect = px_to_ndc(bx0, by0, bx1, by1, vw, vh);
        queue.write_buffer(
            &self.banner_uniform_buf,
            0,
            bytemuck::bytes_of(&OverlayUniforms {
                rect_ndc: b_rect,
                alpha: smooth_alpha,
                _pad: [0.0; 3],
            }),
        );

        // Vulkan logo watermark at bottom-right
        let wm_scale = if vw < 720.0 { 0.72 } else { 0.85 };
        let ww = self.wm_w as f32 * wm_scale;
        let wh = self.wm_h as f32 * wm_scale;
        let wx1 = vw - margin;
        let wx0 = wx1 - ww;
        let wy1 = vh - margin;
        let wy0 = wy1 - wh;

        let w_rect = px_to_ndc(wx0, wy0, wx1, wy1, vw, vh);
        queue.write_buffer(
            &self.wm_uniform_buf,
            0,
            bytemuck::bytes_of(&OverlayUniforms {
                rect_ndc: w_rect,
                alpha: smooth_alpha * 0.94,
                _pad: [0.0; 3],
            }),
        );

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Overlay Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
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
        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &self.banner_bind_group, &[]);
        rpass.draw(0..6, 0..1);
        rpass.set_bind_group(0, &self.wm_bind_group, &[]);
        rpass.draw(0..6, 0..1);
    }

    fn rasterize_banner(
        &self,
        index_1based: usize,
        total: usize,
        name: &str,
        category: &str,
        is_milkdrop: bool,
    ) -> (u32, u32, Vec<u8>) {
        let title_str = truncate_chars(name, 56);
        let badge_str = if is_milkdrop { "MILKDROP" } else { "WGSL" };
        let sub_str = format!("{index_1based}/{total}  ·  {category}");

        let title_scale = 21.0_f32;
        let sub_scale = 14.0_f32;
        let badge_scale = 12.0_f32;

        let title_w = self.measure_text(&title_str, title_scale);
        let sub_w = self.measure_text(&sub_str, sub_scale);
        let badge_w = self.measure_text(badge_str, badge_scale) + 18.0;

        let w = ((title_w.max(sub_w + badge_w + 10.0) + 34.0).ceil() as u32).clamp(240, 740);
        let h = 68u32;
        let mut buf = vec![0u8; (w * h * 4) as usize];

        let border = if is_milkdrop {
            [226, 46, 72, 210]
        } else {
            [56, 189, 248, 210]
        };
        draw_rounded_card(&mut buf, w, h, 11.0, [11, 15, 24, 210], border);

        // Title line
        self.draw_text(&mut buf, w, h, 16.0, 14.0, &title_str, title_scale, [244, 247, 252]);

        // Format badge pill on second line
        let badge_bg = if is_milkdrop {
            [172, 22, 44, 235]
        } else {
            [14, 116, 168, 235]
        };
        let bw_u = badge_w.ceil() as u32;
        draw_sub_pill(&mut buf, w, h, 16, 40, bw_u, 19, 5.0, badge_bg);
        self.draw_text(
            &mut buf,
            w,
            h,
            25.0,
            43.0,
            badge_str,
            badge_scale,
            [255, 255, 255],
        );

        // Subtitle next to badge
        self.draw_text(
            &mut buf,
            w,
            h,
            16.0 + badge_w + 10.0,
            42.0,
            &sub_str,
            sub_scale,
            [176, 188, 208],
        );

        (w, h, buf)
    }

    fn measure_text(&self, text: &str, px: f32) -> f32 {
        let Some(ref font) = self.font else {
            return text.chars().count() as f32 * (px * 0.62);
        };
        let scaled = font.as_scaled(PxScale::from(px));
        let mut x = 0.0_f32;
        let mut prev = None;
        for ch in text.chars() {
            let gid = scaled.glyph_id(ch);
            if let Some(p) = prev {
                x += scaled.kern(p, gid);
            }
            x += scaled.h_advance(gid);
            prev = Some(gid);
        }
        x
    }

    fn draw_text(
        &self,
        buf: &mut [u8],
        w: u32,
        h: u32,
        x0: f32,
        y0: f32,
        text: &str,
        px: f32,
        rgb: [u8; 3],
    ) {
        let Some(ref font) = self.font else {
            draw_fallback_text(buf, w, h, x0 as i32, y0 as i32, text, rgb);
            return;
        };
        let scaled = font.as_scaled(PxScale::from(px));
        let baseline = y0 + scaled.ascent();
        let mut x = x0;
        let mut prev = None;
        for ch in text.chars() {
            let gid = scaled.glyph_id(ch);
            if let Some(p) = prev {
                x += scaled.kern(p, gid);
            }
            let glyph = gid.with_scale_and_position(PxScale::from(px), ab_glyph::point(x, baseline));
            if let Some(outlined) = font.outline_glyph(glyph) {
                let bounds = outlined.px_bounds();
                outlined.draw(|gx, gy, cov| {
                    let px_x = bounds.min.x as i32 + gx as i32;
                    let px_y = bounds.min.y as i32 + gy as i32;
                    if px_x >= 0 && px_y >= 0 && (px_x as u32) < w && (px_y as u32) < h {
                        blend_px(buf, (px_y as u32 * w + px_x as u32) as usize, rgb, cov);
                    }
                });
            }
            x += scaled.h_advance(gid);
            prev = Some(gid);
        }
    }
}

fn build_watermark_card() -> (u32, u32, Vec<u8>) {
    let img = image::load_from_memory(WATERMARK_PNG)
        .expect("Embedded vulkan_logo_watermark.png must decode")
        .to_rgba8();
    let (iw, ih) = img.dimensions();
    let pad_x = 12u32;
    let pad_y = 8u32;
    let w = iw + pad_x * 2;
    let h = ih + pad_y * 2;
    let mut buf = vec![0u8; (w * h * 4) as usize];
    draw_rounded_card(&mut buf, w, h, 10.0, [11, 15, 24, 185], [172, 22, 44, 175]);

    for y in 0..ih {
        for x in 0..iw {
            let src = img.get_pixel(x, y).0;
            if src[3] == 0 {
                continue;
            }
            let dst_idx = ((y + pad_y) * w + (x + pad_x)) as usize;
            let a = src[3] as f32 / 255.0;
            blend_px(&mut buf, dst_idx, [src[0], src[1], src[2]], a);
        }
    }
    (w, h, buf)
}

fn px_to_ndc(x0: f32, y0: f32, x1: f32, y1: f32, vw: f32, vh: f32) -> [f32; 4] {
    [
        (x0 / vw) * 2.0 - 1.0,
        1.0 - (y1 / vh) * 2.0,
        (x1 / vw) * 2.0 - 1.0,
        1.0 - (y0 / vh) * 2.0,
    ]
}

fn create_rgba_texture(
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
            bytes_per_row: Some(4 * width),
            rows_per_image: Some(height),
        },
        size,
    );
    tex
}

fn make_bg(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    uniform_buf: &wgpu::Buffer,
    tex: &wgpu::Texture,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Overlay BG"),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

fn blend_px(buf: &mut [u8], px_idx: usize, rgb: [u8; 3], cov: f32) {
    let i = px_idx * 4;
    let a = cov.clamp(0.0, 1.0);
    let dst_a = buf[i + 3] as f32 / 255.0;
    let out_a = a + dst_a * (1.0 - a);
    if out_a > 1e-5 {
        for c in 0..3 {
            let s = rgb[c] as f32 * a;
            let d = buf[i + c] as f32 * dst_a * (1.0 - a);
            buf[i + c] = ((s + d) / out_a).clamp(0.0, 255.0) as u8;
        }
        buf[i + 3] = (out_a * 255.0).clamp(0.0, 255.0) as u8;
    }
}

fn draw_rounded_card(buf: &mut [u8], w: u32, h: u32, radius: f32, bg: [u8; 4], border: [u8; 4]) {
    let hw = w as f32 * 0.5;
    let hh = h as f32 * 0.5;
    for y in 0..h {
        for x in 0..w {
            let px = (x as f32 + 0.5 - hw).abs() - (hw - radius - 1.0);
            let py = (y as f32 + 0.5 - hh).abs() - (hh - radius - 1.0);
            let dist = (px.max(0.0).powi(2) + py.max(0.0).powi(2)).sqrt() + px.max(py).min(0.0) - radius;
            if dist <= 0.5 {
                let aa = (0.5 - dist).clamp(0.0, 1.0);
                let is_border = dist > -1.3;
                let col = if is_border { border } else { bg };
                let idx = (y * w + x) as usize * 4;
                buf[idx] = col[0];
                buf[idx + 1] = col[1];
                buf[idx + 2] = col[2];
                buf[idx + 3] = (col[3] as f32 * aa) as u8;
            }
        }
    }
}

fn draw_sub_pill(
    buf: &mut [u8],
    buf_w: u32,
    buf_h: u32,
    x0: u32,
    y0: u32,
    pw: u32,
    ph: u32,
    radius: f32,
    bg: [u8; 4],
) {
    let hw = pw as f32 * 0.5;
    let hh = ph as f32 * 0.5;
    for y in 0..ph {
        if y0 + y >= buf_h {
            continue;
        }
        for x in 0..pw {
            if x0 + x >= buf_w {
                continue;
            }
            let px = (x as f32 + 0.5 - hw).abs() - (hw - radius);
            let py = (y as f32 + 0.5 - hh).abs() - (hh - radius);
            let dist = (px.max(0.0).powi(2) + py.max(0.0).powi(2)).sqrt() + px.max(py).min(0.0) - radius;
            if dist <= 0.5 {
                let aa = (0.5 - dist).clamp(0.0, 1.0) * (bg[3] as f32 / 255.0);
                let idx = ((y0 + y) * buf_w + (x0 + x)) as usize;
                blend_px(buf, idx, [bg[0], bg[1], bg[2]], aa);
            }
        }
    }
}

fn truncate_chars(s: &str, max_chars: usize) -> String {
    let mut chars = s.chars();
    let taken: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{taken}...")
    } else {
        taken
    }
}

fn load_system_font() -> Option<FontArc> {
    let preferred = [
        "/usr/share/fonts/TTF/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/cantarell/Cantarell-Regular.otf",
        "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/run/host/fonts/TTF/NotoSans-Regular.ttf",
        "/run/host/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/run/host/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/run/host/fonts/TTF/dejavu/DejaVuSans.ttf",
        "/run/host/fonts/TTF/DejaVuSans.ttf",
        "/run/host/fonts/cantarell/Cantarell-Regular.otf",
        "/usr/share/fonts/tex-gyre/texgyreheros-regular.otf",
    ];
    for p in preferred {
        if let Ok(bytes) = std::fs::read(p) {
            if let Ok(f) = FontArc::try_from_vec(bytes) {
                return Some(f);
            }
        }
    }
    let found = find_first_ttf(Path::new("/usr/share/fonts"));
    if let Some(p) = found {
        if let Ok(bytes) = std::fs::read(p) {
            if let Ok(f) = FontArc::try_from_vec(bytes) {
                return Some(f);
            }
        }
    }
    None
}

fn find_first_ttf(dir: &Path) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            if let Some(f) = find_first_ttf(&p) {
                return Some(f);
            }
        } else if let Some(name) = p.file_name().and_then(|s| s.to_str()) {
            let lower = name.to_ascii_lowercase();
            if (lower.ends_with("regular.ttf") || lower.ends_with("regular.otf"))
                && !lower.contains("italic")
                && !lower.contains("condensed")
            {
                return Some(p);
            }
        }
    }
    None
}

fn draw_fallback_text(buf: &mut [u8], w: u32, h: u32, x0: i32, y0: i32, text: &str, rgb: [u8; 3]) {
    // Minimal crisp fallback bitmap rectangles if no font file exists on disk
    let mut cx = x0;
    for ch in text.chars() {
        if ch != ' ' {
            for dy in 2..12 {
                for dx in 1..6 {
                    let px = cx + dx;
                    let py = y0 + dy;
                    if px >= 0 && py >= 0 && (px as u32) < w && (py as u32) < h {
                        blend_px(buf, (py as u32 * w + px as u32) as usize, rgb, 0.75);
                    }
                }
            }
        }
        cx += 8;
    }
}
