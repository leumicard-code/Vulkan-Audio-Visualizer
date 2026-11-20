// ============================================================================
// milkdrop/textures.rs — Procedural Noise Textures & Texture Pack Loader
// ============================================================================

use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub struct LoadedImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// Generates MilkDrop's standard 2D and 3D RGBA8 noise textures deterministically.
pub struct NoiseSet {
    pub noise_lq: Vec<u8>,      // 256x256
    pub noise_lq_lite: Vec<u8>, // 32x32
    pub noise_mq: Vec<u8>,      // 256x256
    pub noise_hq: Vec<u8>,      // 256x256
    pub noisevol_lq: Vec<u8>,   // 32x32x32
    pub noisevol_hq: Vec<u8>,   // 32x32x32
}

#[inline]
fn hash_u32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

fn gen_2d_lattice_noise(size: usize, cell: usize, seed: u32) -> Vec<u8> {
    let grid = (size / cell).max(1);
    let mut lat = vec![0.0_f32; grid * grid * 4];
    for i in 0..(grid * grid * 4) {
        let h = hash_u32(seed.wrapping_add(i as u32));
        lat[i] = (h & 0xFFFF) as f32 / 65535.0;
    }

    let mut out = vec![0u8; size * size * 4];
    for y in 0..size {
        let fy = (y as f32) / cell as f32;
        let y0 = (fy.floor() as usize) % grid;
        let y1 = (y0 + 1) % grid;
        let ty = fy.fract();
        let sy = ty * ty * (3.0 - 2.0 * ty);

        for x in 0..size {
            let fx = (x as f32) / cell as f32;
            let x0 = (fx.floor() as usize) % grid;
            let x1 = (x0 + 1) % grid;
            let tx = fx.fract();
            let sx = tx * tx * (3.0 - 2.0 * tx);

            let dst = (y * size + x) * 4;
            for c in 0..4 {
                let v00 = lat[(y0 * grid + x0) * 4 + c];
                let v10 = lat[(y0 * grid + x1) * 4 + c];
                let v01 = lat[(y1 * grid + x0) * 4 + c];
                let v11 = lat[(y1 * grid + x1) * 4 + c];
                let a = v00 + (v10 - v00) * sx;
                let b = v01 + (v11 - v01) * sx;
                let v = a + (b - a) * sy;
                out[dst + c] = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }
    }
    out
}

fn gen_3d_lattice_noise(size: usize, cell: usize, seed: u32) -> Vec<u8> {
    let grid = (size / cell).max(1);
    let mut lat = vec![0.0_f32; grid * grid * grid * 4];
    for i in 0..lat.len() {
        let h = hash_u32(seed.wrapping_add(i as u32));
        lat[i] = (h & 0xFFFF) as f32 / 65535.0;
    }

    let mut out = vec![0u8; size * size * size * 4];
    for z in 0..size {
        let fz = z as f32 / cell as f32;
        let z0 = (fz.floor() as usize) % grid;
        let z1 = (z0 + 1) % grid;
        let tz = fz.fract();
        let sz = tz * tz * (3.0 - 2.0 * tz);

        for y in 0..size {
            let fy = y as f32 / cell as f32;
            let y0 = (fy.floor() as usize) % grid;
            let y1 = (y0 + 1) % grid;
            let ty = fy.fract();
            let sy = ty * ty * (3.0 - 2.0 * ty);

            for x in 0..size {
                let fx = x as f32 / cell as f32;
                let x0 = (fx.floor() as usize) % grid;
                let x1 = (x0 + 1) % grid;
                let tx = fx.fract();
                let sx = tx * tx * (3.0 - 2.0 * tx);

                let dst = ((z * size + y) * size + x) * 4;
                for c in 0..4 {
                    let idx = |ix, iy, iz| ((iz * grid + iy) * grid + ix) * 4 + c;
                    let c000 = lat[idx(x0, y0, z0)];
                    let c100 = lat[idx(x1, y0, z0)];
                    let c010 = lat[idx(x0, y1, z0)];
                    let c110 = lat[idx(x1, y1, z0)];
                    let c001 = lat[idx(x0, y0, z1)];
                    let c101 = lat[idx(x1, y0, z1)];
                    let c011 = lat[idx(x0, y1, z1)];
                    let c111 = lat[idx(x1, y1, z1)];

                    let x00 = c000 + (c100 - c000) * sx;
                    let x10 = c010 + (c110 - c010) * sx;
                    let x01 = c001 + (c101 - c001) * sx;
                    let x11 = c011 + (c111 - c011) * sx;

                    let y0v = x00 + (x10 - x00) * sy;
                    let y1v = x01 + (x11 - x01) * sy;
                    let v = y0v + (y1v - y0v) * sz;
                    out[dst + c] = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
                }
            }
        }
    }
    out
}

impl NoiseSet {
    pub fn generate() -> Self {
        Self {
            noise_lq: gen_2d_lattice_noise(256, 1, 0x1337_0001),
            noise_lq_lite: gen_2d_lattice_noise(32, 1, 0x1337_0002),
            noise_mq: gen_2d_lattice_noise(256, 4, 0x1337_0003),
            noise_hq: gen_2d_lattice_noise(256, 8, 0x1337_0004),
            noisevol_lq: gen_3d_lattice_noise(32, 1, 0x1337_0005),
            noisevol_hq: gen_3d_lattice_noise(32, 4, 0x1337_0006),
        }
    }
}

pub struct TextureCatalog {
    by_stem: HashMap<String, PathBuf>,
    ordered_paths: Vec<PathBuf>,
    cache: HashMap<String, LoadedImage>,
}

impl TextureCatalog {
    pub fn scan(dirs: &[PathBuf]) -> Self {
        let mut by_stem = HashMap::new();
        let mut ordered_paths = Vec::new();
        for dir in dirs {
            collect_dir(dir, &mut by_stem, &mut ordered_paths);
        }
        ordered_paths.sort();
        Self {
            by_stem,
            ordered_paths,
            cache: HashMap::new(),
        }
    }

    pub fn load_named(&mut self, name: &str, preset_seed: u32) -> &LoadedImage {
        let clean = normalize_tex_name(name);
        let cache_key = if clean.starts_with("rand") {
            format!("{clean}_{preset_seed}")
        } else {
            clean.clone()
        };

        if !self.cache.contains_key(&cache_key) {
            let img = self.resolve_and_decode(&clean, preset_seed);
            self.cache.insert(cache_key.clone(), img);
        }
        self.cache.get(&cache_key).unwrap()
    }

    fn resolve_and_decode(&self, clean: &str, preset_seed: u32) -> LoadedImage {
        // 1. Direct name match
        if let Some(path) = self.by_stem.get(clean) {
            if let Some(img) = decode_file(path) {
                return img;
            }
        }
        // 2. Partial stem match
        if let Some((_, path)) = self
            .by_stem
            .iter()
            .find(|(k, _)| k.contains(clean) || clean.contains(k.as_str()))
        {
            if let Some(img) = decode_file(path) {
                return img;
            }
        }
        // 3. `rand00`..`rand03` or fallback to an existing pack texture deterministically
        if !self.ordered_paths.is_empty() {
            let mut h = preset_seed;
            for b in clean.bytes() {
                h = hash_u32(h ^ b as u32);
            }
            let path = &self.ordered_paths[(h as usize) % self.ordered_paths.len()];
            if let Some(img) = decode_file(path) {
                return img;
            }
        }
        // 4. Procedural fallback tileable RGBA8 (256x256)
        let mut h = 0x517c_c1b7_u32;
        for b in clean.bytes() {
            h = hash_u32(h ^ b as u32);
        }
        LoadedImage {
            width: 256,
            height: 256,
            rgba: gen_2d_lattice_noise(256, 16, h),
        }
    }
}

fn normalize_tex_name(name: &str) -> String {
    let mut s = name.to_ascii_lowercase();
    for prefix in ["smalltiled_", "tiled_"] {
        if let Some(r) = s.strip_prefix(prefix) {
            s = r.to_string();
        }
    }
    for suffix in ["_smalltiled", "_tiled"] {
        if let Some(r) = s.strip_suffix(suffix) {
            s = r.to_string();
        }
    }
    s
}

fn collect_dir(dir: &Path, by_stem: &mut HashMap<String, PathBuf>, ordered: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() {
            collect_dir(&p, by_stem, ordered);
        } else if p.is_file() {
            let ok_ext = p
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| {
                    matches!(
                        e.to_ascii_lowercase().as_str(),
                        "jpg" | "jpeg" | "png" | "bmp" | "tga"
                    )
                })
                .unwrap_or(false);
            if ok_ext {
                if let Some(stem) = p.file_stem().and_then(|s| s.to_str()) {
                    let key = stem.to_ascii_lowercase();
                    by_stem.entry(key).or_insert_with(|| p.clone());
                    ordered.push(p);
                }
            }
        }
    }
}

fn decode_file(path: &Path) -> Option<LoadedImage> {
    let img = image::open(path).ok()?.to_rgba8();
    let (width, height) = img.dimensions();
    Some(LoadedImage {
        width: width.max(1),
        height: height.max(1),
        rgba: img.into_raw(),
    })
}
