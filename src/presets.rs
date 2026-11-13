// ============================================================================
// presets.rs — Built-in MilkDrop Shader Registry, Search & Hot-Reload Watcher
// ============================================================================

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime};

use crate::config::app_config_dir;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PresetKind {
    Wgsl,
    MilkDrop,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PresetFilter {
    All,
    WgslOnly,
    MilkDropOnly,
}

#[derive(Clone, Debug)]
pub struct ShaderPreset {
    pub id: String,
    pub name: String,
    pub category: String,
    pub kind: PresetKind,
    pub file_path: PathBuf,
    pub source: String,
    pub last_modified: Option<SystemTime>,
    pub search_key: String,
}

struct BuiltinDef {
    id: &'static str,
    name: &'static str,
    category: &'static str,
    file_name: &'static str,
    source: &'static str,
}

const BUILTIN_SHADERS: &[BuiltinDef] = &[
    BuiltinDef {
        id: "01_lorenz_attractor",
        name: "Lorenz Attractor",
        category: "Chaos / 3D",
        file_name: "01_lorenz_attractor.wgsl",
        source: include_str!("../shaders/01_lorenz_attractor.wgsl"),
    },
    BuiltinDef {
        id: "02_liquid_circuitry",
        name: "Liquid Circuitry",
        category: "Cyber / Warp",
        file_name: "02_liquid_circuitry.wgsl",
        source: include_str!("../shaders/02_liquid_circuitry.wgsl"),
    },
    BuiltinDef {
        id: "03_plasma_tunnel",
        name: "Plasma Tunnel",
        category: "MilkDrop / Warp",
        file_name: "03_plasma_tunnel.wgsl",
        source: include_str!("../shaders/03_plasma_tunnel.wgsl"),
    },
    BuiltinDef {
        id: "04_waveform_oscilloscope",
        name: "Waveform Oscilloscope",
        category: "Scope / CRT",
        file_name: "04_waveform_oscilloscope.wgsl",
        source: include_str!("../shaders/04_waveform_oscilloscope.wgsl"),
    },
    BuiltinDef {
        id: "05_particle_field",
        name: "Particle Field",
        category: "Bokeh / Swarm",
        file_name: "05_particle_field.wgsl",
        source: include_str!("../shaders/05_particle_field.wgsl"),
    },
    BuiltinDef {
        id: "06_warp_kaleidoscope",
        name: "Warp Kaleidoscope",
        category: "MilkDrop / Mirror",
        file_name: "06_warp_kaleidoscope.wgsl",
        source: include_str!("../shaders/06_warp_kaleidoscope.wgsl"),
    },
    BuiltinDef {
        id: "07_neon_cyber_grid",
        name: "Neon Cyber Grid",
        category: "Synthwave / 3D",
        file_name: "07_neon_cyber_grid.wgsl",
        source: include_str!("../shaders/07_neon_cyber_grid.wgsl"),
    },
    BuiltinDef {
        id: "08_fractal_julia_pulse",
        name: "Fractal Julia Pulse",
        category: "Fractal / Complex",
        file_name: "08_fractal_julia_pulse.wgsl",
        source: include_str!("../shaders/08_fractal_julia_pulse.wgsl"),
    },
    BuiltinDef {
        id: "09_aurora_ribbons",
        name: "Aurora Ribbons",
        category: "Spectrum / Wave",
        file_name: "09_aurora_ribbons.wgsl",
        source: include_str!("../shaders/09_aurora_ribbons.wgsl"),
    },
    BuiltinDef {
        id: "10_starfield_hyperdrive",
        name: "Starfield Hyperdrive",
        category: "Space / Warp",
        file_name: "10_starfield_hyperdrive.wgsl",
        source: include_str!("../shaders/10_starfield_hyperdrive.wgsl"),
    },
    BuiltinDef {
        id: "11_acoustic_mandala",
        name: "Acoustic Mandala",
        category: "Cymatics / Radial",
        file_name: "11_acoustic_mandala.wgsl",
        source: include_str!("../shaders/11_acoustic_mandala.wgsl"),
    },
    BuiltinDef {
        id: "12_spectrum_cathedral",
        name: "Spectrum Cathedral",
        category: "Spectrum / 3D",
        file_name: "12_spectrum_cathedral.wgsl",
        source: include_str!("../shaders/12_spectrum_cathedral.wgsl"),
    },
    BuiltinDef {
        id: "13_mandelbrot_reactor",
        name: "Mandelbrot Reactor",
        category: "Fractal",
        file_name: "13_mandelbrot_reactor.wgsl",
        source: include_str!("../shaders/13_mandelbrot_reactor.wgsl"),
    },
    BuiltinDef {
        id: "14_burning_ship_inferno",
        name: "Burning Ship Inferno",
        category: "Fractal",
        file_name: "14_burning_ship_inferno.wgsl",
        source: include_str!("../shaders/14_burning_ship_inferno.wgsl"),
    },
    BuiltinDef {
        id: "15_sierpinski_prism",
        name: "Sierpinski Prism",
        category: "Fractal",
        file_name: "15_sierpinski_prism.wgsl",
        source: include_str!("../shaders/15_sierpinski_prism.wgsl"),
    },
    BuiltinDef {
        id: "16_apollonian_foam",
        name: "Apollonian Foam",
        category: "Fractal",
        file_name: "16_apollonian_foam.wgsl",
        source: include_str!("../shaders/16_apollonian_foam.wgsl"),
    },
    BuiltinDef {
        id: "17_tricorn_vortex",
        name: "Tricorn Vortex",
        category: "Fractal",
        file_name: "17_tricorn_vortex.wgsl",
        source: include_str!("../shaders/17_tricorn_vortex.wgsl"),
    },
    BuiltinDef {
        id: "18_wormhole_singularity",
        name: "Wormhole Singularity",
        category: "Tunnel",
        file_name: "18_wormhole_singularity.wgsl",
        source: include_str!("../shaders/18_wormhole_singularity.wgsl"),
    },
    BuiltinDef {
        id: "19_hex_cyber_tunnel",
        name: "Hex Cyber Tunnel",
        category: "Tunnel",
        file_name: "19_hex_cyber_tunnel.wgsl",
        source: include_str!("../shaders/19_hex_cyber_tunnel.wgsl"),
    },
    BuiltinDef {
        id: "20_rainbow_vortex_tube",
        name: "Rainbow Vortex Tube",
        category: "Tunnel",
        file_name: "20_rainbow_vortex_tube.wgsl",
        source: include_str!("../shaders/20_rainbow_vortex_tube.wgsl"),
    },
    BuiltinDef {
        id: "21_laser_prism_shaft",
        name: "Laser Prism Shaft",
        category: "Tunnel",
        file_name: "21_laser_prism_shaft.wgsl",
        source: include_str!("../shaders/21_laser_prism_shaft.wgsl"),
    },
    BuiltinDef {
        id: "22_stained_glass_prism",
        name: "Stained Glass Prism",
        category: "Kaleidoscope",
        file_name: "22_stained_glass_prism.wgsl",
        source: include_str!("../shaders/22_stained_glass_prism.wgsl"),
    },
    BuiltinDef {
        id: "23_liquid_mirror_lotus",
        name: "Liquid Mirror Lotus",
        category: "Kaleidoscope",
        file_name: "23_liquid_mirror_lotus.wgsl",
        source: include_str!("../shaders/23_liquid_mirror_lotus.wgsl"),
    },
    BuiltinDef {
        id: "24_sacred_diamond_fold",
        name: "Sacred Diamond Fold",
        category: "Kaleidoscope",
        file_name: "24_sacred_diamond_fold.wgsl",
        source: include_str!("../shaders/24_sacred_diamond_fold.wgsl"),
    },
    BuiltinDef {
        id: "25_psychedelic_pinwheel",
        name: "Psychedelic Pinwheel",
        category: "Kaleidoscope",
        file_name: "25_psychedelic_pinwheel.wgsl",
        source: include_str!("../shaders/25_psychedelic_pinwheel.wgsl"),
    },
    BuiltinDef {
        id: "26_firefly_meadow",
        name: "Firefly Meadow",
        category: "Particle",
        file_name: "26_firefly_meadow.wgsl",
        source: include_str!("../shaders/26_firefly_meadow.wgsl"),
    },
    BuiltinDef {
        id: "27_quantum_orbitals",
        name: "Quantum Orbitals",
        category: "Particle",
        file_name: "27_quantum_orbitals.wgsl",
        source: include_str!("../shaders/27_quantum_orbitals.wgsl"),
    },
    BuiltinDef {
        id: "28_ember_fountain",
        name: "Ember Fountain",
        category: "Particle",
        file_name: "28_ember_fountain.wgsl",
        source: include_str!("../shaders/28_ember_fountain.wgsl"),
    },
    BuiltinDef {
        id: "29_sonic_rain_matrix",
        name: "Sonic Rain Matrix",
        category: "Particle",
        file_name: "29_sonic_rain_matrix.wgsl",
        source: include_str!("../shaders/29_sonic_rain_matrix.wgsl"),
    },
    BuiltinDef {
        id: "30_radial_scope_bloom",
        name: "Radial Scope Bloom",
        category: "Waveform",
        file_name: "30_radial_scope_bloom.wgsl",
        source: include_str!("../shaders/30_radial_scope_bloom.wgsl"),
    },
    BuiltinDef {
        id: "31_dna_double_helix",
        name: "DNA Double Helix",
        category: "Waveform",
        file_name: "31_dna_double_helix.wgsl",
        source: include_str!("../shaders/31_dna_double_helix.wgsl"),
    },
    BuiltinDef {
        id: "32_neon_ekg_pulse",
        name: "Neon EKG Pulse",
        category: "Waveform",
        file_name: "32_neon_ekg_pulse.wgsl",
        source: include_str!("../shaders/32_neon_ekg_pulse.wgsl"),
    },
    BuiltinDef {
        id: "33_horizon_wave_ocean",
        name: "Horizon Wave Ocean",
        category: "Waveform",
        file_name: "33_horizon_wave_ocean.wgsl",
        source: include_str!("../shaders/33_horizon_wave_ocean.wgsl"),
    },
    BuiltinDef {
        id: "34_borealis_silk_ribbons",
        name: "Borealis Silk Ribbons",
        category: "Aurora",
        file_name: "34_borealis_silk_ribbons.wgsl",
        source: include_str!("../shaders/34_borealis_silk_ribbons.wgsl"),
    },
    BuiltinDef {
        id: "35_solar_corona_veil",
        name: "Solar Corona Veil",
        category: "Aurora",
        file_name: "35_solar_corona_veil.wgsl",
        source: include_str!("../shaders/35_solar_corona_veil.wgsl"),
    },
    BuiltinDef {
        id: "36_polar_mist_curtains",
        name: "Polar Mist Curtains",
        category: "Aurora",
        file_name: "36_polar_mist_curtains.wgsl",
        source: include_str!("../shaders/36_polar_mist_curtains.wgsl"),
    },
    BuiltinDef {
        id: "37_magnetic_flux_sky",
        name: "Magnetic Flux Sky",
        category: "Aurora",
        file_name: "37_magnetic_flux_sky.wgsl",
        source: include_str!("../shaders/37_magnetic_flux_sky.wgsl"),
    },
    BuiltinDef {
        id: "38_synthwave_sun_drive",
        name: "Synthwave Sun Drive",
        category: "Neon",
        file_name: "38_synthwave_sun_drive.wgsl",
        source: include_str!("../shaders/38_synthwave_sun_drive.wgsl"),
    },
    BuiltinDef {
        id: "39_neon_laser_lattice",
        name: "Neon Laser Lattice",
        category: "Neon",
        file_name: "39_neon_laser_lattice.wgsl",
        source: include_str!("../shaders/39_neon_laser_lattice.wgsl"),
    },
    BuiltinDef {
        id: "40_cyber_hex_core",
        name: "Cyber Hex Core",
        category: "Neon",
        file_name: "40_cyber_hex_core.wgsl",
        source: include_str!("../shaders/40_cyber_hex_core.wgsl"),
    },
    BuiltinDef {
        id: "41_arcade_vector_storm",
        name: "Arcade Vector Storm",
        category: "Neon",
        file_name: "41_arcade_vector_storm.wgsl",
        source: include_str!("../shaders/41_arcade_vector_storm.wgsl"),
    },
    BuiltinDef {
        id: "42_molten_rainbow_blob",
        name: "Molten Rainbow Blob",
        category: "Plasma",
        file_name: "42_molten_rainbow_blob.wgsl",
        source: include_str!("../shaders/42_molten_rainbow_blob.wgsl"),
    },
    BuiltinDef {
        id: "43_toxic_acid_swirl",
        name: "Toxic Acid Swirl",
        category: "Plasma",
        file_name: "43_toxic_acid_swirl.wgsl",
        source: include_str!("../shaders/43_toxic_acid_swirl.wgsl"),
    },
    BuiltinDef {
        id: "44_magma_convection",
        name: "Magma Convection",
        category: "Plasma",
        file_name: "44_magma_convection.wgsl",
        source: include_str!("../shaders/44_magma_convection.wgsl"),
    },
    BuiltinDef {
        id: "45_biolume_reef_drift",
        name: "Biolume Reef Drift",
        category: "Plasma",
        file_name: "45_biolume_reef_drift.wgsl",
        source: include_str!("../shaders/45_biolume_reef_drift.wgsl"),
    },
    BuiltinDef {
        id: "46_electric_arc_storm",
        name: "Electric Arc Storm",
        category: "Plasma",
        file_name: "46_electric_arc_storm.wgsl",
        source: include_str!("../shaders/46_electric_arc_storm.wgsl"),
    },
    BuiltinDef {
        id: "47_deep_warp_starfield",
        name: "Deep Warp Starfield",
        category: "Starfield",
        file_name: "47_deep_warp_starfield.wgsl",
        source: include_str!("../shaders/47_deep_warp_starfield.wgsl"),
    },
    BuiltinDef {
        id: "48_galaxy_spiral_arms",
        name: "Galaxy Spiral Arms",
        category: "Starfield",
        file_name: "48_galaxy_spiral_arms.wgsl",
        source: include_str!("../shaders/48_galaxy_spiral_arms.wgsl"),
    },
    BuiltinDef {
        id: "49_meteor_rain_nova",
        name: "Meteor Rain Nova",
        category: "Starfield",
        file_name: "49_meteor_rain_nova.wgsl",
        source: include_str!("../shaders/49_meteor_rain_nova.wgsl"),
    },
    BuiltinDef {
        id: "50_sacred_lotus_mandala",
        name: "Sacred Lotus Mandala",
        category: "Mandala",
        file_name: "50_sacred_lotus_mandala.wgsl",
        source: include_str!("../shaders/50_sacred_lotus_mandala.wgsl"),
    },
    BuiltinDef {
        id: "51_spirograph_blossom",
        name: "Spirograph Blossom",
        category: "Mandala",
        file_name: "51_spirograph_blossom.wgsl",
        source: include_str!("../shaders/51_spirograph_blossom.wgsl"),
    },
    BuiltinDef {
        id: "52_cymatic_lattice",
        name: "Cymatic Lattice",
        category: "Mandala",
        file_name: "52_cymatic_lattice.wgsl",
        source: include_str!("../shaders/52_cymatic_lattice.wgsl"),
    },
];

pub struct PresetManager {
    shader_dir: PathBuf,
    milkdrop_dirs: Vec<PathBuf>,
    texture_dirs: Vec<PathBuf>,
    presets: Vec<ShaderPreset>,
    active_idx: usize,
    last_poll: Instant,
}

impl PresetManager {
    pub fn new(preferred_id: Option<&str>) -> Self {
        Self::with_milkdrop_dirs(preferred_id, &[])
    }

    pub fn with_milkdrop_dirs(preferred_id: Option<&str>, extra_dirs: &[String]) -> Self {
        let shader_dir = resolve_shader_dir();
        let _ = fs::create_dir_all(&shader_dir);

        let user_milk_dir = app_config_dir().join("milkdrop");
        let _ = fs::create_dir_all(&user_milk_dir);

        let mut presets = Vec::new();
        let mut seen_ids = HashMap::new();

        // Seed built-in WGSL shaders onto disk if missing, then load disk version for hot-reloadability
        for def in BUILTIN_SHADERS {
            let path = shader_dir.join(def.file_name);
            if !path.exists() {
                let _ = fs::write(&path, def.source);
            }
            let disk_source = fs::read_to_string(&path).unwrap_or_else(|_| def.source.to_string());
            let mtime = file_mtime(&path);

            seen_ids.insert(def.id.to_string(), presets.len());
            let search_key = format!("{} {} {} wgsl", def.name, def.category, def.id).to_lowercase();
            presets.push(ShaderPreset {
                id: def.id.to_string(),
                name: def.name.to_string(),
                category: def.category.to_string(),
                kind: PresetKind::Wgsl,
                file_path: path,
                source: disk_source,
                last_modified: mtime,
                search_key,
            });
        }

        // Discover any additional custom `.wgsl` files placed in the shader directory
        Self::scan_custom_shaders(&shader_dir, &mut presets, &mut seen_ids);

        // Discover MilkDrop `.milk` preset packs + texture directories
        let (milkdrop_dirs, texture_dirs) = discover_milkdrop_and_texture_dirs(extra_dirs);
        Self::scan_milkdrop_dirs(&milkdrop_dirs, &mut presets, &mut seen_ids);

        let active_idx = preferred_id
            .and_then(|id| presets.iter().position(|p| p.id == id))
            .unwrap_or(0);

        Self {
            shader_dir,
            milkdrop_dirs,
            texture_dirs,
            presets,
            active_idx,
            last_poll: Instant::now(),
        }
    }

    pub fn rescan(&mut self, extra_dirs: &[String]) {
        let active_id = self.presets.get(self.active_idx).map(|p| p.id.clone());
        *self = Self::with_milkdrop_dirs(active_id.as_deref(), extra_dirs);
    }

    pub fn texture_dirs(&self) -> &[PathBuf] {
        &self.texture_dirs
    }

    pub fn milkdrop_dirs(&self) -> &[PathBuf] {
        &self.milkdrop_dirs
    }

    pub fn counts(&self) -> (usize, usize) {
        let milk = self
            .presets
            .iter()
            .filter(|p| p.kind == PresetKind::MilkDrop)
            .count();
        (self.presets.len() - milk, milk)
    }

    fn scan_custom_shaders(
        dir: &Path,
        presets: &mut Vec<ShaderPreset>,
        seen_ids: &mut HashMap<String, usize>,
    ) {
        let Ok(entries) = fs::read_dir(dir) else {
            return;
        };
        let mut custom_paths: Vec<PathBuf> = entries
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| {
                p.is_file()
                    && p.extension()
                        .and_then(|e| e.to_str())
                        .map(|e| e.eq_ignore_ascii_case("wgsl"))
                        .unwrap_or(false)
            })
            .collect();
        custom_paths.sort();

        for path in custom_paths {
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            if seen_ids.contains_key(stem) {
                continue;
            }
            if let Ok(source) = fs::read_to_string(&path) {
                let display_name = prettify_shader_name(stem);
                seen_ids.insert(stem.to_string(), presets.len());
                let search_key = format!("{display_name} custom wgsl {stem}").to_lowercase();
                presets.push(ShaderPreset {
                    id: stem.to_string(),
                    name: display_name,
                    category: "Custom WGSL".to_string(),
                    kind: PresetKind::Wgsl,
                    last_modified: file_mtime(&path),
                    file_path: path,
                    source,
                    search_key,
                });
            }
        }
    }

    fn scan_milkdrop_dirs(
        dirs: &[PathBuf],
        presets: &mut Vec<ShaderPreset>,
        seen_ids: &mut HashMap<String, usize>,
    ) {
        let mut collected: Vec<(String, String, String, PathBuf)> = Vec::new();
        for root in dirs {
            collect_milk_recursive(root, root, &mut collected);
        }
        collected.sort_by(|a, b| a.2.cmp(&b.2).then_with(|| a.1.cmp(&b.1)));

        for (id, name, category, file_path) in collected {
            if seen_ids.contains_key(&id) {
                continue;
            }
            seen_ids.insert(id.clone(), presets.len());
            let search_key = format!("{name} {category} milkdrop").to_lowercase();
            presets.push(ShaderPreset {
                id,
                name,
                category,
                kind: PresetKind::MilkDrop,
                file_path,
                source: String::new(),
                last_modified: None,
                search_key,
            });
        }
    }

    pub fn shader_dir(&self) -> &Path {
        &self.shader_dir
    }

    pub fn presets(&self) -> &[ShaderPreset] {
        &self.presets
    }

    pub fn active_idx(&self) -> usize {
        self.active_idx
    }

    pub fn active_preset(&self) -> &ShaderPreset {
        &self.presets[self.active_idx]
    }

    pub fn set_active_idx(&mut self, idx: usize) -> Option<&ShaderPreset> {
        if idx < self.presets.len() {
            self.active_idx = idx;
            Some(&self.presets[self.active_idx])
        } else {
            None
        }
    }

    pub fn next_preset(&mut self) -> &ShaderPreset {
        self.active_idx = (self.active_idx + 1) % self.presets.len();
        &self.presets[self.active_idx]
    }

    pub fn prev_preset(&mut self) -> &ShaderPreset {
        self.active_idx = if self.active_idx == 0 {
            self.presets.len() - 1
        } else {
            self.active_idx - 1
        };
        &self.presets[self.active_idx]
    }

    /// Instant case-insensitive search filter matching preset name, category, or id.
    pub fn filter_indices(&self, query: &str) -> Vec<usize> {
        self.filter_indices_with(query, PresetFilter::All, None)
    }

    pub fn filter_indices_with(
        &self,
        query: &str,
        kind_filter: PresetFilter,
        top_category: Option<&str>,
    ) -> Vec<usize> {
        let q = query.trim().to_lowercase();
        let words: Vec<&str> = q.split_whitespace().collect();
        self.presets
            .iter()
            .enumerate()
            .filter(|(_, p)| {
                match kind_filter {
                    PresetFilter::All => {}
                    PresetFilter::WgslOnly if p.kind != PresetKind::Wgsl => return false,
                    PresetFilter::MilkDropOnly if p.kind != PresetKind::MilkDrop => return false,
                    _ => {}
                }
                if let Some(cat) = top_category {
                    if !p.category.starts_with(cat) {
                        return false;
                    }
                }
                words.iter().all(|w| p.search_key.contains(w))
            })
            .map(|(i, _)| i)
            .collect()
    }

    pub fn top_categories(&self) -> Vec<(String, usize)> {
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        for p in &self.presets {
            let key = if p.kind == PresetKind::Wgsl {
                "WGSL Shaders".to_string()
            } else {
                p.category
                    .split('/')
                    .take(2)
                    .map(str::trim)
                    .collect::<Vec<_>>()
                    .join(" / ")
            };
            *counts.entry(key).or_insert(0) += 1;
        }
        counts.into_iter().collect()
    }

    /// Checks disk modification timestamps every ~250ms and reloads changed shaders.
    /// Returns `true` if the currently active shader changed on disk.
    pub fn poll_hot_reload(&mut self, force: bool) -> bool {
        if !force && self.last_poll.elapsed().as_millis() < 250 {
            return false;
        }
        self.last_poll = Instant::now();

        let mut active_changed = false;
        for (i, preset) in self.presets.iter_mut().enumerate() {
            if preset.kind != PresetKind::Wgsl && i != self.active_idx {
                continue;
            }
            let mtime = file_mtime(&preset.file_path);
            if force || (mtime.is_some() && mtime != preset.last_modified) {
                if preset.kind == PresetKind::MilkDrop {
                    if preset.last_modified.is_some() || force {
                        preset.last_modified = mtime;
                        if i == self.active_idx {
                            active_changed = true;
                        }
                    } else {
                        preset.last_modified = mtime;
                    }
                } else if let Ok(new_src) = fs::read_to_string(&preset.file_path) {
                    if force || new_src != preset.source {
                        preset.source = new_src;
                        preset.last_modified = mtime;
                        if i == self.active_idx {
                            active_changed = true;
                        }
                    } else {
                        preset.last_modified = mtime;
                    }
                }
            }
        }
        active_changed
    }
}

fn discover_milkdrop_and_texture_dirs(extra_dirs: &[String]) -> (Vec<PathBuf>, Vec<PathBuf>) {
    let mut candidates: Vec<PathBuf> = Vec::new();
    candidates.push(app_config_dir().join("milkdrop"));
    candidates.push(PathBuf::from("/home/shay/code1/milkdrop_presets/cream-of-the-crop"));
    candidates.push(PathBuf::from("/workspace/milkdrop_presets/cream-of-the-crop"));
    candidates.push(PathBuf::from("../milkdrop_presets/cream-of-the-crop"));
    candidates.push(PathBuf::from("milkdrop_presets"));
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(dir.join("milkdrop_presets"));
        }
    }
    for d in extra_dirs {
        if !d.trim().is_empty() {
            candidates.push(PathBuf::from(d));
        }
    }

    let mut seen_canon = HashSet::new();
    let mut milk_dirs = Vec::new();
    let mut tex_dirs = Vec::new();

    for c in candidates {
        if !c.is_dir() {
            continue;
        }
        let key = fs::canonicalize(&c).unwrap_or_else(|_| c.clone());
        if seen_canon.insert(key) {
            milk_dirs.push(c.clone());
            // Check sibling / nested texture directories
            if let Some(parent) = c.parent() {
                for rel in ["textures/textures", "textures"] {
                    let td = parent.join(rel);
                    if td.is_dir() {
                        tex_dirs.push(td);
                    }
                }
            }
            for rel in ["textures", "textures/textures"] {
                let td = c.join(rel);
                if td.is_dir() {
                    tex_dirs.push(td);
                }
            }
        }
    }

    // Standard standalone texture paths
    for td in [
        app_config_dir().join("textures"),
        PathBuf::from("/home/shay/code1/milkdrop_presets/textures/textures"),
        PathBuf::from("/workspace/milkdrop_presets/textures/textures"),
        PathBuf::from("../milkdrop_presets/textures/textures"),
    ] {
        if td.is_dir() && !tex_dirs.contains(&td) {
            tex_dirs.push(td);
        }
    }

    (milk_dirs, tex_dirs)
}

fn collect_milk_recursive(
    root: &Path,
    cur: &Path,
    out: &mut Vec<(String, String, String, PathBuf)>,
) {
    let Ok(entries) = fs::read_dir(cur) else {
        return;
    };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() {
            collect_milk_recursive(root, &p, out);
        } else if p.is_file()
            && p.extension()
                .and_then(|e| e.to_str())
                .map(|e| e.eq_ignore_ascii_case("milk"))
                .unwrap_or(false)
        {
            let Some(stem) = p.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            let rel = p.strip_prefix(root).unwrap_or(&p);
            let id = format!("milk:{}", rel.to_string_lossy());
            let parent_rel = rel
                .parent()
                .map(|pp| pp.to_string_lossy().replace('\\', "/"))
                .unwrap_or_default();
            let category = if parent_rel.is_empty() {
                "MilkDrop".to_string()
            } else {
                format!("MilkDrop / {}", parent_rel.replace('/', " / "))
            };
            out.push((id, stem.to_string(), category, p));
        }
    }
}

fn resolve_shader_dir() -> PathBuf {
    let local_shaders = PathBuf::from("shaders");
    if local_shaders.is_dir() {
        return local_shaders;
    }
    app_config_dir().join("shaders")
}

fn file_mtime(path: &Path) -> Option<SystemTime> {
    fs::metadata(path).ok().and_then(|m| m.modified().ok())
}

fn prettify_shader_name(stem: &str) -> String {
    stem.split('_')
        .filter(|part| !part.is_empty() && !part.chars().all(|c| c.is_ascii_digit()))
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn has_at_least_52_builtin_shaders_and_search_works() {
        let mgr = PresetManager::new(None);
        let (wgsl_count, _milk_count) = mgr.counts();
        assert!(wgsl_count >= 52);
        let lorenz_wgsl = mgr.filter_indices_with("lorenz", PresetFilter::WgslOnly, None);
        assert_eq!(lorenz_wgsl.len(), 1);
        assert_eq!(mgr.presets()[lorenz_wgsl[0]].name, "Lorenz Attractor");
        assert!(!mgr.filter_indices("lorenz").is_empty());
    }
}
