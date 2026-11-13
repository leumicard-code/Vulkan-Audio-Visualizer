// ============================================================================
// visualizer.rs — wgpu Vulkan Engine with Hot-Reloadable Shader Pipelines
// ============================================================================

use anyhow::{anyhow, Context, Result};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::borrow::Cow;
use std::path::{Path, PathBuf};

use crate::dsp_fft::{AudioFeatures, SPECTRUM_BANDS, WAVEFORM_POINTS};
use crate::milkdrop::engine::MilkDropEngine;
use crate::overlay::OverlayRenderer;

/// Uniform buffer payload matching every MilkDrop WGSL preset's `Uniforms` struct.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VisualizerUniforms {
    pub time: f32,
    pub dt: f32,
    pub width: f32,
    pub height: f32,
    pub bass: f32,
    pub mid: f32,
    pub treble: f32,
    pub beat_intensity: f32,
    pub overall: f32,
    pub aspect: f32,
    pub frame_count: f32,
    pub _pad: f32,
    pub spectrum: [[f32; 4]; 16], // 64 FFT bands packed as 16 vec4<f32>
    pub waveform: [[f32; 4]; 16], // 64 PCM samples packed as 16 vec4<f32>
}

impl VisualizerUniforms {
    pub fn from_features(
        time: f32,
        dt: f32,
        width: u32,
        height: u32,
        frame_count: u64,
        feat: &AudioFeatures,
    ) -> Self {
        let w = width.max(1) as f32;
        let h = height.max(1) as f32;
        let mut spectrum = [[0.0_f32; 4]; 16];
        for i in 0..SPECTRUM_BANDS {
            spectrum[i / 4][i & 3] = feat.spectrum[i];
        }
        let mut waveform = [[0.0_f32; 4]; 16];
        for i in 0..WAVEFORM_POINTS {
            waveform[i / 4][i & 3] = feat.waveform[i];
        }
        Self {
            time,
            dt,
            width: w,
            height: h,
            bass: feat.bass,
            mid: feat.mid,
            treble: feat.treble,
            beat_intensity: feat.beat_intensity,
            overall: feat.overall,
            aspect: w / h,
            frame_count: frame_count as f32,
            _pad: 0.0,
            spectrum,
            waveform,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ActiveMode {
    Wgsl,
    MilkDrop,
}

pub struct VisualizerEngine {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pipeline_layout: wgpu::PipelineLayout,
    active_pipeline: wgpu::RenderPipeline,
    mode: ActiveMode,
    milkdrop: MilkDropEngine,
    overlay: OverlayRenderer,
    overlay_enabled: bool,
    adapter_name: String,
    backend_name: String,
    last_shader_error: Option<String>,
    frame_count: u64,
}

impl VisualizerEngine {
    /// Attaches a Vulkan `wgpu::Surface` to a native GUI window via `raw-window-handle`.
    ///
    /// # Safety
    /// The caller must ensure `window` outlives the returned `VisualizerEngine`.
    pub unsafe fn new<W>(
        window: &W,
        width: u32,
        height: u32,
        initial_wgsl: &str,
        initial_name: &str,
        tex_dirs: &[PathBuf],
        show_overlay: bool,
    ) -> Result<Self>
    where
        W: HasWindowHandle + HasDisplayHandle,
    {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });

        let raw_win = window
            .window_handle()
            .map_err(|e| anyhow!("Failed to query native window handle: {e}"))?
            .as_raw();
        let raw_disp = window
            .display_handle()
            .map_err(|e| anyhow!("Failed to query native display handle: {e}"))?
            .as_raw();

        let surface = instance
            .create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
                raw_display_handle: Some(raw_disp),
                raw_window_handle: raw_win,
            })
            .context("Failed to create Vulkan surface for native GUI panel")?;

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .context("No compatible Vulkan GPU adapter found for this surface")?;

        let info = adapter.get_info();
        let adapter_name = info.name.clone();
        let backend_name = format!("{:?}", info.backend);

        let adapter_limits = adapter.limits();
        let required_limits = wgpu::Limits {
            max_texture_dimension_2d: adapter_limits.max_texture_dimension_2d.min(8192),
            max_sampled_textures_per_shader_stage: adapter_limits
                .max_sampled_textures_per_shader_stage
                .min(32),
            ..wgpu::Limits::default()
        };

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Vulkan Audio Visualizer Device"),
            required_features: wgpu::Features::empty(),
            required_limits,
            ..wgpu::DeviceDescriptor::default()
        }))
        .context("Failed to create logical Vulkan GPU device")?;

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .or_else(|| caps.formats.first().copied())
            .ok_or_else(|| anyhow!("Surface reported zero supported swapchain formats"))?;

        let alpha_mode = caps
            .alpha_modes
            .iter()
            .copied()
            .find(|m| *m == wgpu::CompositeAlphaMode::Opaque)
            .unwrap_or_else(|| caps.alpha_modes[0]);

        let present_mode = if caps.present_modes.contains(&wgpu::PresentMode::Mailbox) {
            wgpu::PresentMode::Mailbox
        } else {
            wgpu::PresentMode::Fifo
        };

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width: width.clamp(1, 8192),
            height: height.clamp(1, 8192),
            present_mode,
            alpha_mode,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let initial_uniforms = VisualizerUniforms::from_features(
            0.0,
            0.016,
            config.width,
            config.height,
            0,
            &AudioFeatures::default(),
        );
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Visualizer Uniforms"),
            size: std::mem::size_of::<VisualizerUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buffer, 0, bytemuck::bytes_of(&initial_uniforms));

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Visualizer BindGroupLayout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Visualizer BindGroup"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Visualizer PipelineLayout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let active_pipeline = Self::compile_pipeline(
            &device,
            &pipeline_layout,
            config.format,
            initial_wgsl,
            initial_name,
        )?;

        let milkdrop = MilkDropEngine::new(
            &device,
            &queue,
            config.format,
            config.width,
            config.height,
            tex_dirs,
        );
        let overlay = OverlayRenderer::new(&device, &queue, config.format);

        Ok(Self {
            surface,
            device,
            queue,
            config,
            uniform_buffer,
            bind_group,
            pipeline_layout,
            active_pipeline,
            mode: ActiveMode::Wgsl,
            milkdrop,
            overlay,
            overlay_enabled: show_overlay,
            adapter_name,
            backend_name,
            last_shader_error: None,
            frame_count: 0,
        })
    }

    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    pub fn backend_name(&self) -> &str {
        &self.backend_name
    }

    pub fn last_error(&self) -> Option<&str> {
        self.last_shader_error
            .as_deref()
            .or_else(|| self.milkdrop.last_error())
    }

    pub fn update_texture_dirs(&mut self, dirs: &[PathBuf]) {
        self.milkdrop.update_texture_dirs(dirs);
    }

    pub fn set_overlay_info(
        &mut self,
        index_1based: usize,
        total: usize,
        name: &str,
        category: &str,
        is_milkdrop: bool,
    ) {
        self.overlay.set_preset_info(
            &self.device,
            &self.queue,
            index_1based,
            total,
            name,
            category,
            is_milkdrop,
        );
    }

    pub fn trigger_overlay(&mut self) {
        self.overlay.trigger();
    }

    pub fn set_overlay_enabled(&mut self, enabled: bool) {
        self.overlay_enabled = enabled;
        if enabled {
            self.overlay.trigger();
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        let w = width.clamp(1, 8192);
        let h = height.clamp(1, 8192);
        if self.config.width != w || self.config.height != h {
            self.config.width = w;
            self.config.height = h;
            self.surface.configure(&self.device, &self.config);
            self.milkdrop.resize(&self.device, &self.queue, w, h);
        }
    }

    /// Hot-reloads a WGSL shader preset safely; keeps the active pipeline on error.
    pub fn load_shader(&mut self, wgsl_source: &str, label: &str) -> Result<()> {
        match Self::compile_pipeline(
            &self.device,
            &self.pipeline_layout,
            self.config.format,
            wgsl_source,
            label,
        ) {
            Ok(pipeline) => {
                self.active_pipeline = pipeline;
                self.mode = ActiveMode::Wgsl;
                self.last_shader_error = None;
                Ok(())
            }
            Err(err) => {
                let msg = format!("{err}");
                self.last_shader_error = Some(msg.clone());
                Err(anyhow!(msg))
            }
        }
    }

    /// Loads a classic MilkDrop (.milk) preset onto the native Rust/Vulkan MilkDrop engine.
    pub fn load_milkdrop(&mut self, path: &Path, preset_seed: u32, current_time: f32) -> Result<()> {
        self.milkdrop
            .load_preset_file(&self.device, &self.queue, path, preset_seed, current_time)?;
        self.mode = ActiveMode::MilkDrop;
        self.last_shader_error = self.milkdrop.last_error().map(str::to_string);
        Ok(())
    }

    fn compile_pipeline(
        device: &wgpu::Device,
        layout: &wgpu::PipelineLayout,
        format: wgpu::TextureFormat,
        wgsl_source: &str,
        label: &str,
    ) -> Result<wgpu::RenderPipeline> {
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(wgsl_source)),
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(layout),
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
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        if let Some(err) = pollster::block_on(scope.pop()) {
            return Err(anyhow!("WGSL error in '{label}': {err}"));
        }

        Ok(pipeline)
    }

    /// Updates uniforms and renders one Vulkan frame onto the native surface.
    pub fn render(&mut self, time: f32, dt: f32, features: &AudioFeatures) {
        self.frame_count = self.frame_count.wrapping_add(1);

        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
            _ => return,
        };

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Visualizer Render Encoder"),
            });

        match self.mode {
            ActiveMode::Wgsl => {
                let uniforms = VisualizerUniforms::from_features(
                    time,
                    dt,
                    self.config.width,
                    self.config.height,
                    self.frame_count,
                    features,
                );
                self.queue
                    .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

                let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Visualizer Render Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
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
                rpass.set_pipeline(&self.active_pipeline);
                rpass.set_bind_group(0, &self.bind_group, &[]);
                rpass.draw(0..3, 0..1);
            }
            ActiveMode::MilkDrop => {
                self.milkdrop.render(
                    &self.device,
                    &self.queue,
                    &mut encoder,
                    &view,
                    time,
                    dt,
                    features,
                );
            }
        }

        if self.overlay_enabled {
            self.overlay.render(
                &self.queue,
                &mut encoder,
                &view,
                self.config.width,
                self.config.height,
                dt,
            );
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        self.queue.present(frame);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::presets::{PresetKind, PresetManager};

    #[test]
    fn all_52_wgsl_shaders_compile_on_vulkan_and_invalid_shader_recovers() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let Ok(adapter) = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        })) else {
            return;
        };
        let Ok((device, _queue)) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
        else {
            return;
        };

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Test BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Test PL"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let mgr = PresetManager::new(None);
        let wgsl_presets: Vec<_> = mgr
            .presets()
            .iter()
            .filter(|p| p.kind == PresetKind::Wgsl)
            .collect();
        assert!(wgsl_presets.len() >= 52);
        for preset in wgsl_presets {
            let res = VisualizerEngine::compile_pipeline(
                &device,
                &layout,
                wgpu::TextureFormat::Bgra8Unorm,
                &preset.source,
                &preset.name,
            );
            assert!(res.is_ok(), "Shader '{}' failed to compile: {:?}", preset.name, res.err());
        }

        // Verify broken WGSL returns Err safely instead of crashing
        let bad = VisualizerEngine::compile_pipeline(
            &device,
            &layout,
            wgpu::TextureFormat::Bgra8Unorm,
            "broken wgsl syntax @@@",
            "bad_shader",
        );
        assert!(bad.is_err());
    }

    #[test]
    fn milkdrop_engine_renders_offscreen_on_vulkan() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let Ok(adapter) = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        })) else {
            return;
        };
        let lim = adapter.limits();
        let req = wgpu::Limits {
            max_texture_dimension_2d: lim.max_texture_dimension_2d.min(8192),
            max_sampled_textures_per_shader_stage: lim.max_sampled_textures_per_shader_stage.min(32),
            ..wgpu::Limits::default()
        };
        let Ok((device, queue)) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_limits: req,
            ..wgpu::DeviceDescriptor::default()
        })) else {
            return;
        };

        let mgr = PresetManager::new(None);
        let mut engine = MilkDropEngine::new(
            &device,
            &queue,
            wgpu::TextureFormat::Bgra8Unorm,
            320,
            240,
            mgr.texture_dirs(),
        );

        let target_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("MD Offscreen Test"),
            size: wgpu::Extent3d { width: 320, height: 240, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Bgra8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let target_view = target_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let feat = AudioFeatures {
            bass: 0.7,
            mid: 0.5,
            treble: 0.4,
            beat_intensity: 0.8,
            overall: 0.6,
            ..AudioFeatures::default()
        };

        // Render real MilkDrop presets (including HLSL warp+comp presets) + overlay under Vulkan validation
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut samples = Vec::new();
        for kw in ["liquid circuitry", "lorenz chaser", "log smell", "liquidfire"] {
            if let Some(p) = mgr
                .presets()
                .iter()
                .find(|p| p.kind == PresetKind::MilkDrop && p.search_key.contains(kw))
            {
                samples.push(p);
            }
        }
        if samples.is_empty() {
            samples.extend(
                mgr.presets()
                    .iter()
                    .filter(|p| p.kind == PresetKind::MilkDrop)
                    .take(3),
            );
        }
        for (i, p) in samples.iter().enumerate() {
            engine
                .load_preset_file(&device, &queue, &p.file_path, i as u32 + 1, i as f32)
                .unwrap_or_else(|e| panic!("Preset '{}' load failed: {e}", p.name));
            assert!(
                engine.last_error().is_none(),
                "Preset '{}' triggered shader fallback: {:?}",
                p.name,
                engine.last_error()
            );
            for f in 0..3 {
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("MD Test Enc"),
                });
                engine.render(
                    &device,
                    &queue,
                    &mut encoder,
                    &target_view,
                    i as f32 + f as f32 * 0.016,
                    0.016,
                    &feat,
                );
                queue.submit(std::iter::once(encoder.finish()));
            }
        }
        let mut overlay = OverlayRenderer::new(&device, &queue, wgpu::TextureFormat::Bgra8Unorm);
        overlay.set_preset_info(&device, &queue, 1, mgr.presets().len(), "Test Preset", "Cat", true);
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Overlay Test Enc"),
        });
        overlay.render(&queue, &mut enc, &target_view, 320, 240, 0.016);
        queue.submit(std::iter::once(enc.finish()));
        if let Some(err) = pollster::block_on(scope.pop()) {
            panic!("Vulkan validation error during MilkDrop/Overlay render: {err}");
        }
    }
}
