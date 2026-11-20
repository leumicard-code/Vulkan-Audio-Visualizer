// ============================================================================
// milkdrop/mod.rs — Native MilkDrop (.milk) Preset Engine on Vulkan (wgpu)
// ============================================================================

pub mod eel;
pub mod engine;
pub mod hlsl;
pub mod preset;
pub mod textures;

#[cfg(test)]
mod corpus_tests {
    use super::*;
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::rc::Rc;
    use std::time::Instant;

    fn collect_milk_files(dir: &std::path::Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else { return };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                collect_milk_files(&p, out);
            } else if p.extension().and_then(|s| s.to_str()).map(|s| s.eq_ignore_ascii_case("milk")).unwrap_or(false) {
                out.push(p);
            }
        }
    }

    #[test]
    #[ignore]
    fn corpus_verify_all_9795_presets() {
        let dir = PathBuf::from("/workspace/milkdrop_presets/cream-of-the-crop");
        let dir = if dir.is_dir() {
            dir
        } else {
            PathBuf::from("/home/shay/code1/milkdrop_presets/cream-of-the-crop")
        };
        let mut files = Vec::new();
        collect_milk_files(&dir, &mut files);
        files.sort();
        assert!(!files.is_empty(), "No .milk files found in {}", dir.display());

        let mut eel_total = 0usize;
        let mut eel_ok = 0usize;
        let mut eel_errs: HashMap<String, usize> = HashMap::new();
        let mut eel_err_samples: Vec<(String, String)> = Vec::new();

        let mut warp_total = 0usize;
        let mut warp_ok = 0usize;
        let mut warp_errs: HashMap<String, usize> = HashMap::new();
        let mut warp_err_samples: Vec<(String, String)> = Vec::new();

        let mut comp_total = 0usize;
        let mut comp_ok = 0usize;
        let mut comp_errs: HashMap<String, usize> = HashMap::new();
        let mut comp_err_samples: Vec<(String, String)> = Vec::new();

        let mut fe = wgpu::naga::front::glsl::Frontend::default();
        let opts = wgpu::naga::front::glsl::Options::from(wgpu::naga::ShaderStage::Fragment);
        let mut validator = wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        );

        let mut per_pixel_ns_total = 0u128;
        let mut per_pixel_runs = 0usize;

        for (fidx, path) in files.iter().enumerate() {
            let p = preset::MilkPreset::from_file(path).unwrap();
            let g = Rc::new(RefCell::new(eel::Globals::new()));
            let mut vm = eel::Vm::new(g.clone());

            let blocks: Vec<&str> = [
                p.per_frame_init_src.as_str(),
                p.per_frame_src.as_str(),
                p.per_pixel_src.as_str(),
            ]
            .into_iter()
            .chain(p.waves.iter().flat_map(|w| [w.init_src.as_str(), w.per_frame_src.as_str(), w.per_point_src.as_str()]))
            .chain(p.shapes.iter().flat_map(|s| [s.init_src.as_str(), s.per_frame_src.as_str()]))
            .collect();

            for (bi, src) in blocks.iter().enumerate() {
                if src.trim().is_empty() {
                    continue;
                }
                eel_total += 1;
                let mut local_vm = if bi >= 3 { eel::Vm::new(g.clone()) } else { eel::Vm::new(g.clone()) };
                let target_vm = if bi < 3 { &mut vm } else { &mut local_vm };
                match eel::compile(src, target_vm) {
                    Ok(prog) => {
                        eel_ok += 1;
                        if bi == 1 {
                            for _ in 0..10 {
                                prog.run(target_vm);
                            }
                        } else if bi == 2 && per_pixel_runs < 200 {
                            let t0 = Instant::now();
                            for _ in 0..100 {
                                prog.run(target_vm);
                            }
                            per_pixel_ns_total += t0.elapsed().as_nanos();
                            per_pixel_runs += 100;
                        }
                    }
                    Err(e) => {
                        let key = e.chars().take(60).collect::<String>();
                        *eel_errs.entry(key).or_insert(0) += 1;
                        if eel_err_samples.len() < 12 {
                            eel_err_samples.push((path.display().to_string(), e));
                        }
                    }
                }
            }

            if let Some(ref wsrc) = p.warp_shader_src {
                warp_total += 1;
                match hlsl::translate(wsrc, hlsl::ShaderKind::Warp) {
                    Ok(tr) => match fe.parse(&opts, &tr.glsl) {
                        Ok(m) => match validator.validate(&m) {
                            Ok(_) => warp_ok += 1,
                            Err(e) => {
                                let emitted = e.emit_to_string(&tr.glsl);
                                let cause = emitted
                                    .lines()
                                    .find(|l| l.trim_start().starts_with("= "))
                                    .unwrap_or("unknown validation error")
                                    .trim();
                                let key = format!("validate: {cause}\n{emitted}");
                                let short: String = format!("validate: {cause}").chars().take(100).collect();
                                *warp_errs.entry(short).or_insert(0) += 1;
                                if warp_err_samples.iter().filter(|(_, m)| m.starts_with("validate:")).count() < 8 {
                                    warp_err_samples.push((path.display().to_string(), key));
                                }
                            }
                        },
                        Err(e) => {
                            let msg = format!("{e:?}");
                            let short = msg.chars().take(90).collect();
                            *warp_errs.entry(short).or_insert(0) += 1;
                            if warp_err_samples.len() < 10 {
                                warp_err_samples.push((path.display().to_string(), msg));
                            }
                        }
                    },
                    Err(e) => {
                        let short = format!("translate: {}", e.chars().take(70).collect::<String>());
                        *warp_errs.entry(short).or_insert(0) += 1;
                        if warp_err_samples.len() < 10 {
                            warp_err_samples.push((path.display().to_string(), e));
                        }
                    }
                }
            }

            if let Some(ref csrc) = p.comp_shader_src {
                comp_total += 1;
                match hlsl::translate(csrc, hlsl::ShaderKind::Comp) {
                    Ok(tr) => match fe.parse(&opts, &tr.glsl) {
                        Ok(m) => match validator.validate(&m) {
                            Ok(_) => comp_ok += 1,
                            Err(e) => {
                                let emitted = e.emit_to_string(&tr.glsl);
                                let cause = emitted
                                    .lines()
                                    .find(|l| l.trim_start().starts_with("= "))
                                    .unwrap_or("unknown validation error")
                                    .trim();
                                let key = format!("validate: {cause}\n{emitted}");
                                let short: String = format!("validate: {cause}").chars().take(100).collect();
                                *comp_errs.entry(short).or_insert(0) += 1;
                                if comp_err_samples.iter().filter(|(_, m)| m.starts_with("validate:")).count() < 8 {
                                    comp_err_samples.push((path.display().to_string(), key));
                                }
                            }
                        },
                        Err(e) => {
                            let msg = format!("{e:?}");
                            let short = msg.chars().take(90).collect();
                            *comp_errs.entry(short).or_insert(0) += 1;
                            if comp_err_samples.len() < 10 {
                                comp_err_samples.push((path.display().to_string(), msg));
                            }
                        }
                    },
                    Err(e) => {
                        let short = format!("translate: {}", e.chars().take(70).collect::<String>());
                        *comp_errs.entry(short).or_insert(0) += 1;
                        if comp_err_samples.len() < 10 {
                            comp_err_samples.push((path.display().to_string(), e));
                        }
                    }
                }
            }

            if fidx == 0 {
                // Verify std140 size of MdUniforms block against Rust struct!
                let sample_glsl = hlsl::translate("shader_body { ret = uv.xyy; }", hlsl::ShaderKind::Warp).unwrap().glsl;
                let m = fe.parse(&opts, &sample_glsl).unwrap();
                for (_h, gv) in m.global_variables.iter() {
                    if gv.name.as_deref() == Some("u_") {
                        let ty = &m.types[gv.ty];
                        let naga_size = ty.inner.size(m.to_ctx()) as usize;
                        assert_eq!(
                            std::mem::size_of::<hlsl::MdShaderUniforms>(),
                            naga_size,
                            "MdShaderUniforms size mismatch with std140 layout!"
                        );
                        println!("Verified MdShaderUniforms size = {naga_size} bytes");
                    }
                }
            }
        }

        println!("=== CORPUS RESULTS ({} presets) ===", files.len());
        println!("EEL : {eel_ok}/{eel_total} ({:.2}%)", 100.0 * eel_ok as f64 / eel_total.max(1) as f64);
        if per_pixel_runs > 0 {
            println!("per_pixel avg: {:.3} us/run", per_pixel_ns_total as f64 / per_pixel_runs as f64 / 1000.0);
        }
        println!("WARP: {warp_ok}/{warp_total} ({:.2}%)", 100.0 * warp_ok as f64 / warp_total.max(1) as f64);
        println!("COMP: {comp_ok}/{comp_total} ({:.2}%)", 100.0 * comp_ok as f64 / comp_total.max(1) as f64);

        let print_top = |label: &str, map: &HashMap<String, usize>, samples: &[(String, String)]| {
            let mut v: Vec<_> = map.iter().collect();
            v.sort_by(|a, b| b.1.cmp(a.1));
            println!("Top {label} errors:");
            for (k, c) in v.iter().take(15) {
                println!("  {c:>5}x  {k}");
            }
            for (f, e) in samples.iter().take(12) {
                println!("  SAMPLE {f}: {}", e.chars().take(360).collect::<String>());
            }
        };
        print_top("EEL", &eel_errs, &eel_err_samples);
        print_top("WARP", &warp_errs, &warp_err_samples);
        print_top("COMP", &comp_errs, &comp_err_samples);
    }
}
