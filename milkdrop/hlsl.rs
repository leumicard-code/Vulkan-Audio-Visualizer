// ============================================================================
// milkdrop/hlsl.rs — MilkDrop 2 HLSL (ps_2_0/ps_3_0) -> GLSL 450 Translator
// ============================================================================

use std::collections::{BTreeSet, HashMap};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShaderKind {
    Warp,
    Comp,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamplerKind {
    Fw,
    Fc,
    Pw,
    Pc,
}

#[derive(Clone, Debug)]
pub struct UserTexture {
    pub slot: usize,
    pub name: String,
    pub sampler: SamplerKind,
}

#[derive(Clone, Debug)]
pub struct TranslatedShader {
    pub glsl: String,
    pub user_textures: Vec<UserTexture>,
}

/// Extracts and joins `warp_N=` or `comp_N=` shader lines from MilkDrop key/value pairs.
/// Strips the leading backtick `` ` `` used by MilkDrop 2.
pub fn extract_shader_source<'a>(milk_kv: &[(&'a str, &'a str)], kind: ShaderKind) -> Option<String> {
    let prefix = match kind {
        ShaderKind::Warp => "warp_",
        ShaderKind::Comp => "comp_",
    };
    let mut lines: Vec<(u32, &'a str)> = Vec::new();
    for &(k, v) in milk_kv {
        if let Some(rest) = k.strip_prefix(prefix) {
            if let Ok(n) = rest.parse::<u32>() {
                let code = v.strip_prefix('`').unwrap_or(v);
                lines.push((n, code));
            }
        }
    }
    if lines.is_empty() {
        return None;
    }
    lines.sort_by_key(|(n, _)| *n);
    let mut out = String::new();
    for (idx, (_, line)) in lines.into_iter().enumerate() {
        if idx > 0 {
            out.push('\n');
        }
        out.push_str(line);
    }
    if out.trim().is_empty() {
        None
    } else {
        Some(out)
    }
}

#[derive(Clone, Debug, Default)]
pub struct ShaderInputs {
    pub time: f32,
    pub fps: f32,
    pub frame: f32,
    pub progress: f32,
    pub bass: f32,
    pub mid: f32,
    pub treb: f32,
    pub vol: f32,
    pub bass_att: f32,
    pub mid_att: f32,
    pub treb_att: f32,
    pub vol_att: f32,
    pub q: [f32; 32],
    pub texsize: [f32; 2],
    pub aspect: [f32; 4],
    pub rand_preset: [f32; 4],
    pub rand_frame: [f32; 4],
    pub blur_min: [f32; 3],
    pub blur_max: [f32; 3],
    pub user_texsize: [[f32; 4]; 8],
}

/// std140-aligned uniform payload matching the generated GLSL `MdUniforms` block.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MdShaderUniforms {
    // 0: (time, fps, frame, progress)
    pub timing: [f32; 4],
    // 1: (bass, mid, treb, vol)
    pub audio: [f32; 4],
    // 2: (bass_att, mid_att, treb_att, vol_att)
    pub audio_att: [f32; 4],
    // 3..10: _qa.._qh (q1..q32)
    pub q_vecs: [[f32; 4]; 8],
    // 11: aspect
    pub aspect: [f32; 4],
    // 12: texsize (w, h, 1/w, 1/h)
    pub texsize: [f32; 4],
    // 13: rand_preset
    pub rand_preset: [f32; 4],
    // 14: rand_frame
    pub rand_frame: [f32; 4],
    // 15..18: roam_cos, roam_sin, slow_roam_cos, slow_roam_sin
    pub roam_cos: [f32; 4],
    pub roam_sin: [f32; 4],
    pub slow_roam_cos: [f32; 4],
    pub slow_roam_sin: [f32; 4],
    // 19: (blur1_min, blur2_min, blur3_min, mip_avg)
    pub blur_min_mip: [f32; 4],
    // 20: (blur1_max, blur2_max, blur3_max, mip_x)
    pub blur_max_mip: [f32; 4],
    // 21: (mip_y, mip_xy, 0, 0)
    pub mip_extra: [f32; 4],
    // 22..29: user texsizes (w, h, 1/w, 1/h)
    pub user_texsize: [[f32; 4]; 8],
    // 30..35: standard texture sizes (noise_lq, noise_lq_lite, noise_mq, noise_hq, noisevol_lq, noisevol_hq)
    pub noise_lq_size: [f32; 4],
    pub noise_lq_lite_size: [f32; 4],
    pub noise_mq_size: [f32; 4],
    pub noise_hq_size: [f32; 4],
    pub noisevol_lq_size: [f32; 4],
    pub noisevol_hq_size: [f32; 4],
    // 36..59: 24 x mat4x3 (each mat4x3 is 4 vec4 columns in std140 = [[f32; 4]; 4])
    pub rot_s: [[[f32; 4]; 4]; 4],
    pub rot_d: [[[f32; 4]; 4]; 4],
    pub rot_f: [[[f32; 4]; 4]; 4],
    pub rot_vf: [[[f32; 4]; 4]; 4],
    pub rot_uf: [[[f32; 4]; 4]; 4],
    pub rot_rand: [[[f32; 4]; 4]; 4],
}

fn build_rot_mat4x3(seed: [f32; 4], phase: f32, freq: f32, translate_scale: f32) -> [[f32; 4]; 4] {
    let a = seed[0] * 6.283185 + phase * freq;
    let b = seed[1] * 6.283185 + phase * (freq * 0.73 + 0.19);
    let c = seed[2] * 6.283185 + phase * (freq * 0.51 - 0.13);

    let (sa, ca) = a.sin_cos();
    let (sb, cb) = b.sin_cos();
    let (sc, cc) = c.sin_cos();

    // Standard 3D Euler rotation matrix (rows 0..2) + translation (row 3)
    let r0 = [cb * cc, cb * sc, -sb, 0.0];
    let r1 = [sa * sb * cc - ca * sc, sa * sb * sc + ca * cc, sa * cb, 0.0];
    let r2 = [ca * sb * cc + sa * sc, ca * sb * sc - sa * cc, ca * cb, 0.0];
    let r3 = [
        (seed[0] + phase * 0.13 * freq).sin() * translate_scale,
        (seed[1] + phase * 0.17 * freq).cos() * translate_scale,
        (seed[2] + phase * 0.11 * freq).sin() * translate_scale,
        0.0,
    ];
    [r0, r1, r2, r3]
}

impl MdShaderUniforms {
    pub fn from_inputs(i: &ShaderInputs) -> Self {
        let tw = i.texsize[0].max(1.0);
        let th = i.texsize[1].max(1.0);

        let mut q_vecs = [[0.0_f32; 4]; 8];
        for g in 0..8 {
            q_vecs[g] = [i.q[g * 4], i.q[g * 4 + 1], i.q[g * 4 + 2], i.q[g * 4 + 3]];
        }

        let t = i.time;
        let roam_cos = [
            0.5 + 0.5 * (t * 0.31 + i.rand_preset[0] * 6.28).cos(),
            0.5 + 0.5 * (t * 0.43 + i.rand_preset[1] * 6.28).cos(),
            0.5 + 0.5 * (t * 0.59 + i.rand_preset[2] * 6.28).cos(),
            0.5 + 0.5 * (t * 0.71 + i.rand_preset[3] * 6.28).cos(),
        ];
        let roam_sin = [
            0.5 + 0.5 * (t * 0.31 + i.rand_preset[0] * 6.28).sin(),
            0.5 + 0.5 * (t * 0.43 + i.rand_preset[1] * 6.28).sin(),
            0.5 + 0.5 * (t * 0.59 + i.rand_preset[2] * 6.28).sin(),
            0.5 + 0.5 * (t * 0.71 + i.rand_preset[3] * 6.28).sin(),
        ];
        let slow_roam_cos = [
            0.5 + 0.5 * (t * 0.07 + i.rand_preset[0] * 6.28).cos(),
            0.5 + 0.5 * (t * 0.11 + i.rand_preset[1] * 6.28).cos(),
            0.5 + 0.5 * (t * 0.13 + i.rand_preset[2] * 6.28).cos(),
            0.5 + 0.5 * (t * 0.17 + i.rand_preset[3] * 6.28).cos(),
        ];
        let slow_roam_sin = [
            0.5 + 0.5 * (t * 0.07 + i.rand_preset[0] * 6.28).sin(),
            0.5 + 0.5 * (t * 0.11 + i.rand_preset[1] * 6.28).sin(),
            0.5 + 0.5 * (t * 0.13 + i.rand_preset[2] * 6.28).sin(),
            0.5 + 0.5 * (t * 0.17 + i.rand_preset[3] * 6.28).sin(),
        ];

        let mut rot_s = [[[0.0; 4]; 4]; 4];
        let mut rot_d = [[[0.0; 4]; 4]; 4];
        let mut rot_f = [[[0.0; 4]; 4]; 4];
        let mut rot_vf = [[[0.0; 4]; 4]; 4];
        let mut rot_uf = [[[0.0; 4]; 4]; 4];
        let mut rot_rand = [[[0.0; 4]; 4]; 4];
        for k in 0..4 {
            let s = [
                i.rand_preset[k],
                i.rand_preset[(k + 1) & 3],
                i.rand_preset[(k + 2) & 3],
                i.rand_preset[(k + 3) & 3],
            ];
            let f = 0.25 + k as f32 * 0.19;
            rot_s[k] = build_rot_mat4x3(s, t * 0.15, f, 0.25);
            rot_d[k] = build_rot_mat4x3(s, t * (0.6 + i.bass_att * 0.25), f * 1.3, 0.4);
            rot_f[k] = build_rot_mat4x3(s, t * 1.1, f * 1.7, 0.5);
            rot_vf[k] = build_rot_mat4x3(s, t * 2.1, f * 2.2, 0.7);
            rot_uf[k] = build_rot_mat4x3(s, t * 3.4, f * 2.9, 1.0);
            rot_rand[k] = build_rot_mat4x3(s, 0.0, f, 1.0);
        }

        let mip_avg = (i.vol * 0.5 + 0.15).clamp(0.0, 1.0);
        Self {
            timing: [i.time, i.fps.max(1.0), i.frame, i.progress],
            audio: [i.bass, i.mid, i.treb, i.vol],
            audio_att: [i.bass_att, i.mid_att, i.treb_att, i.vol_att],
            q_vecs,
            aspect: i.aspect,
            texsize: [tw, th, 1.0 / tw, 1.0 / th],
            rand_preset: i.rand_preset,
            rand_frame: i.rand_frame,
            roam_cos,
            roam_sin,
            slow_roam_cos,
            slow_roam_sin,
            blur_min_mip: [i.blur_min[0], i.blur_min[1], i.blur_min[2], mip_avg],
            blur_max_mip: [i.blur_max[0], i.blur_max[1], i.blur_max[2], 0.5],
            mip_extra: [0.5, 0.25, 0.0, 0.0],
            user_texsize: i.user_texsize,
            noise_lq_size: [256.0, 256.0, 1.0 / 256.0, 1.0 / 256.0],
            noise_lq_lite_size: [32.0, 32.0, 1.0 / 32.0, 1.0 / 32.0],
            noise_mq_size: [256.0, 256.0, 1.0 / 256.0, 1.0 / 256.0],
            noise_hq_size: [256.0, 256.0, 1.0 / 256.0, 1.0 / 256.0],
            noisevol_lq_size: [32.0, 32.0, 1.0 / 32.0, 1.0 / 32.0],
            noisevol_hq_size: [32.0, 32.0, 1.0 / 32.0, 1.0 / 32.0],
            rot_s,
            rot_d,
            rot_f,
            rot_vf,
            rot_uf,
            rot_rand,
        }
    }
}

// ============================================================================
// Preprocessor (#if 0 / #if 1 / #else / #endif / #define)
// ============================================================================

fn strip_comments(src: &str) -> String {
    let b = src.as_bytes();
    let n = b.len();
    let mut out = String::with_capacity(n);
    let mut i = 0;
    while i < n {
        if b[i] == b'/' && i + 1 < n && b[i + 1] == b'/' {
            i += 2;
            while i < n && b[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        if b[i] == b'/' && i + 1 < n && b[i + 1] == b'*' {
            i += 2;
            while i + 1 < n && !(b[i] == b'*' && b[i + 1] == b'/') {
                if b[i] == b'\n' {
                    out.push('\n');
                }
                i += 1;
            }
            i = (i + 2).min(n);
            continue;
        }
        out.push(b[i] as char);
        i += 1;
    }
    out
}

fn preprocess(src: &str) -> String {
    let clean = strip_comments(src);
    let mut defines: HashMap<String, String> = HashMap::new();
    // Keep track of nested #if state: (parent_active, branch_active, branch_taken)
    let mut if_stack: Vec<(bool, bool, bool)> = Vec::new();
    let mut lines_out: Vec<String> = Vec::new();

    for raw_line in clean.lines() {
        let trimmed = raw_line.trim();
        let active = if_stack.last().map_or(true, |x| x.1);

        if let Some(rest) = trimmed.strip_prefix('#') {
            let rest = rest.trim();
            if let Some(cond) = rest.strip_prefix("if").filter(|r| r.is_empty() || r.starts_with(char::is_whitespace) || r.starts_with('(')) {
                let cond_trim = cond.trim();
                let truth = if cond_trim == "0" || cond_trim.eq_ignore_ascii_case("false") {
                    false
                } else {
                    true
                };
                if_stack.push((active, active && truth, truth));
                continue;
            }
            if let Some(name) = rest.strip_prefix("ifdef") {
                let name = name.trim();
                let truth = defines.contains_key(name);
                if_stack.push((active, active && truth, truth));
                continue;
            }
            if let Some(name) = rest.strip_prefix("ifndef") {
                let name = name.trim();
                let truth = !defines.contains_key(name);
                if_stack.push((active, active && truth, truth));
                continue;
            }
            if rest.starts_with("elif") {
                if let Some(top) = if_stack.last_mut() {
                    if top.0 && !top.2 {
                        let cond_trim = rest[4..].trim();
                        let truth = cond_trim != "0";
                        top.1 = truth;
                        top.2 = truth;
                    } else {
                        top.1 = false;
                    }
                }
                continue;
            }
            if rest.starts_with("else") {
                if let Some(top) = if_stack.last_mut() {
                    top.1 = top.0 && !top.2;
                    top.2 = true;
                }
                continue;
            }
            if rest.starts_with("endif") {
                if_stack.pop();
                continue;
            }
            if let Some(def) = rest.strip_prefix("define") {
                if active {
                    let def = def.trim();
                    if !def.is_empty() {
                        if let Some((name, val)) = split_define(def) {
                            // Ignore function-like macros for now unless simple
                            if !name.contains('(') {
                                defines.insert(name.to_string(), val.to_string());
                            }
                        } else if !def.contains('(') {
                            defines.insert(def.to_string(), "1".to_string());
                        }
                    }
                }
                continue;
            }
            if let Some(undef) = rest.strip_prefix("undef") {
                if active {
                    defines.remove(undef.trim());
                }
                continue;
            }
            continue;
        }

        if active {
            lines_out.push(raw_line.to_string());
        }
    }

    let joined = lines_out.join("\n");
    if defines.is_empty() {
        return joined;
    }
    expand_defines(&joined, &defines)
}

fn split_define(s: &str) -> Option<(&str, &str)> {
    let idx = s.find(|c: char| c.is_whitespace())?;
    let name = &s[..idx];
    let val = s[idx..].trim();
    Some((name, val))
}

fn expand_defines(src: &str, defines: &HashMap<String, String>) -> String {
    let mut current = src.to_string();
    for _pass in 0..4 {
        let mut out = String::with_capacity(current.len());
        let b = current.as_bytes();
        let n = b.len();
        let mut i = 0;
        let mut changed = false;
        while i < n {
            if b[i].is_ascii_alphabetic() || b[i] == b'_' {
                let start = i;
                i += 1;
                while i < n && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                    i += 1;
                }
                let word = &current[start..i];
                // Don't expand field names right after `.`
                let after_dot = start > 0 && b[start - 1] == b'.';
                if !after_dot {
                    if let Some(repl) = defines.get(word) {
                        out.push_str(repl);
                        changed = true;
                        continue;
                    }
                }
                out.push_str(word);
            } else {
                out.push(b[i] as char);
                i += 1;
            }
        }
        current = out;
        if !changed {
            break;
        }
    }
    current
}

// ============================================================================
// Lexer
// ============================================================================

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Ident(String),
    FloatLit(String),
    IntLit(i64),
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Semi,
    Comma,
    Dot,
    Question,
    Colon,
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Bang,
    Tilde,
    BitAnd,
    BitOr,
    BitXor,
    LogAnd,
    LogOr,
    Shl,
    Shr,
    Lt,
    Gt,
    Le,
    Ge,
    Eq,
    Ne,
    PlusPlus,
    MinusMinus,
    Assign,
    PlusAssign,
    MinusAssign,
    MulAssign,
    DivAssign,
    ModAssign,
    Eof,
}

fn tokenize(src: &str) -> Result<Vec<Tok>, String> {
    let b = src.as_bytes();
    let n = b.len();
    let mut i = 0;
    let mut toks = Vec::with_capacity(n / 4 + 16);

    while i < n {
        let c = b[i];
        if c.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        // Numbers
        if c.is_ascii_digit() || (c == b'.' && i + 1 < n && b[i + 1].is_ascii_digit()) {
            if c == b'0' && i + 1 < n && (b[i + 1] == b'x' || b[i + 1] == b'X') {
                i += 2;
                let s = i;
                while i < n && b[i].is_ascii_hexdigit() {
                    i += 1;
                }
                let val = i64::from_str_radix(&src[s..i], 16).unwrap_or(0);
                toks.push(Tok::IntLit(val));
                continue;
            }
            let start = i;
            let mut is_float = false;
            while i < n && b[i].is_ascii_digit() {
                i += 1;
            }
            if i < n && b[i] == b'.' {
                is_float = true;
                i += 1;
                while i < n && b[i].is_ascii_digit() {
                    i += 1;
                }
            }
            if i < n && (b[i] == b'e' || b[i] == b'E') {
                is_float = true;
                i += 1;
                if i < n && (b[i] == b'+' || b[i] == b'-') {
                    i += 1;
                }
                while i < n && b[i].is_ascii_digit() {
                    i += 1;
                }
            }
            let num_end = i;
            // Suffixes f, F, h, H, u, U, l, L
            let mut has_float_suffix = false;
            while i < n && matches!(b[i], b'f' | b'F' | b'h' | b'H' | b'u' | b'U' | b'l' | b'L') {
                if matches!(b[i], b'f' | b'F' | b'h' | b'H') {
                    has_float_suffix = true;
                }
                i += 1;
            }
            let raw = &src[start..num_end];
            if is_float || has_float_suffix {
                let mut f = raw.to_string();
                if f.starts_with('.') {
                    f.insert(0, '0');
                }
                if f.ends_with('.') {
                    f.push('0');
                }
                toks.push(Tok::FloatLit(f));
            } else {
                let val = raw.parse::<i64>().unwrap_or(0);
                toks.push(Tok::IntLit(val));
            }
            continue;
        }
        // Identifiers
        if c.is_ascii_alphabetic() || c == b'_' {
            let start = i;
            i += 1;
            while i < n && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
            toks.push(Tok::Ident(src[start..i].to_string()));
            continue;
        }

        let next = if i + 1 < n { b[i + 1] } else { 0 };
        match c {
            b'(' => { toks.push(Tok::LParen); i += 1; }
            b')' => { toks.push(Tok::RParen); i += 1; }
            b'{' => { toks.push(Tok::LBrace); i += 1; }
            b'}' => { toks.push(Tok::RBrace); i += 1; }
            b'[' => { toks.push(Tok::LBracket); i += 1; }
            b']' => { toks.push(Tok::RBracket); i += 1; }
            b';' => { toks.push(Tok::Semi); i += 1; }
            b',' => { toks.push(Tok::Comma); i += 1; }
            b'.' => { toks.push(Tok::Dot); i += 1; }
            b'?' => { toks.push(Tok::Question); i += 1; }
            b':' => { toks.push(Tok::Colon); i += 1; }
            b'~' => { toks.push(Tok::Tilde); i += 1; }
            b'^' => { toks.push(Tok::BitXor); i += 1; }
            b'+' => {
                if next == b'+' { toks.push(Tok::PlusPlus); i += 2; }
                else if next == b'=' { toks.push(Tok::PlusAssign); i += 2; }
                else { toks.push(Tok::Plus); i += 1; }
            }
            b'-' => {
                if next == b'-' { toks.push(Tok::MinusMinus); i += 2; }
                else if next == b'=' { toks.push(Tok::MinusAssign); i += 2; }
                else { toks.push(Tok::Minus); i += 1; }
            }
            b'*' => {
                if next == b'=' { toks.push(Tok::MulAssign); i += 2; }
                else { toks.push(Tok::Star); i += 1; }
            }
            b'/' => {
                if next == b'=' { toks.push(Tok::DivAssign); i += 2; }
                else { toks.push(Tok::Slash); i += 1; }
            }
            b'%' => {
                if next == b'=' { toks.push(Tok::ModAssign); i += 2; }
                else { toks.push(Tok::Percent); i += 1; }
            }
            b'!' => {
                if next == b'=' { toks.push(Tok::Ne); i += 2; }
                else { toks.push(Tok::Bang); i += 1; }
            }
            b'=' => {
                if next == b'=' { toks.push(Tok::Eq); i += 2; }
                else { toks.push(Tok::Assign); i += 1; }
            }
            b'<' => {
                if next == b'<' { toks.push(Tok::Shl); i += 2; }
                else if next == b'=' { toks.push(Tok::Le); i += 2; }
                else { toks.push(Tok::Lt); i += 1; }
            }
            b'>' => {
                if next == b'>' { toks.push(Tok::Shr); i += 2; }
                else if next == b'=' { toks.push(Tok::Ge); i += 2; }
                else { toks.push(Tok::Gt); i += 1; }
            }
            b'&' => {
                if next == b'&' { toks.push(Tok::LogAnd); i += 2; }
                else { toks.push(Tok::BitAnd); i += 1; }
            }
            b'|' => {
                if next == b'|' { toks.push(Tok::LogOr); i += 2; }
                else { toks.push(Tok::BitOr); i += 1; }
            }
            _ => return Err(format!("Unexpected character '{}' in HLSL", c as char)),
        }
    }
    toks.push(Tok::Eof);
    Ok(toks)
}

// ============================================================================
// Type System & Sampler Registry
// ============================================================================

#[derive(Clone, Debug, PartialEq, Eq)]
enum Ty {
    Void,
    Float(u8),   // 1..=4
    Int(u8),     // 1..=4
    Bool(u8),    // 1..=4
    Mat(u8, u8), // rows 2..=4, cols 2..=4 (HLSL dimensions)
    Sampler(String),
    Array(Box<Ty>, usize),
    Unknown,
}

impl Ty {
    fn is_scalar(&self) -> bool {
        matches!(self, Ty::Float(1) | Ty::Int(1) | Ty::Bool(1))
    }

    fn width(&self) -> Option<u8> {
        match *self {
            Ty::Float(w) | Ty::Int(w) | Ty::Bool(w) => Some(w),
            _ => None,
        }
    }

    fn with_width(&self, w: u8) -> Ty {
        match self {
            Ty::Float(_) => Ty::Float(w),
            Ty::Int(_) => Ty::Int(w),
            Ty::Bool(_) => Ty::Bool(w),
            other => other.clone(),
        }
    }

    fn to_glsl(&self) -> String {
        match *self {
            Ty::Void => "void".to_string(),
            Ty::Float(1) => "float".to_string(),
            Ty::Float(w) => format!("vec{w}"),
            Ty::Int(1) => "int".to_string(),
            Ty::Int(w) => format!("ivec{w}"),
            Ty::Bool(1) => "bool".to_string(),
            Ty::Bool(w) => format!("bvec{w}"),
            Ty::Mat(r, c) => {
                if r == c {
                    format!("mat{r}")
                } else {
                    // HLSL floatRxC has R rows, C cols; we store R columns of size C in GLSL matRxC
                    format!("mat{r}x{c}")
                }
            }
            Ty::Array(ref elem, len) => format!("{}[{len}]", elem.to_glsl()),
            Ty::Sampler(_) | Ty::Unknown => "float".to_string(),
        }
    }
}

fn parse_hlsl_type(name: &str) -> Option<Ty> {
    let base = name
        .strip_prefix("float")
        .map(|s| ('f', s))
        .or_else(|| name.strip_prefix("half").map(|s| ('f', s)))
        .or_else(|| name.strip_prefix("fixed").map(|s| ('f', s)))
        .or_else(|| name.strip_prefix("min16float").map(|s| ('f', s)))
        .or_else(|| name.strip_prefix("double").map(|s| ('f', s)))
        .or_else(|| name.strip_prefix("int").map(|s| ('i', s)))
        .or_else(|| name.strip_prefix("uint").map(|s| ('i', s)))
        .or_else(|| name.strip_prefix("bool").map(|s| ('b', s)));

    if let Some((kind, rest)) = base {
        if rest.is_empty() {
            return Some(match kind {
                'f' => Ty::Float(1),
                'i' => Ty::Int(1),
                'b' => Ty::Bool(1),
                _ => unreachable!(),
            });
        }
        let b = rest.as_bytes();
        if b.len() == 1 && (b'1'..=b'4').contains(&b[0]) {
            let w = b[0] - b'0';
            return Some(match kind {
                'f' => Ty::Float(w),
                'i' => Ty::Int(w),
                'b' => Ty::Bool(w),
                _ => unreachable!(),
            });
        }
        if kind == 'f' && b.len() == 3 && b[1] == b'x' && (b'2'..=b'4').contains(&b[0]) && (b'2'..=b'4').contains(&b[2]) {
            return Some(Ty::Mat(b[0] - b'0', b[2] - b'0'));
        }
    }
    match name {
        "void" => Some(Ty::Void),
        "sampler" | "sampler2D" | "sampler3D" | "samplerCUBE" | "SamplerState" | "Texture2D" => {
            Some(Ty::Sampler(String::new()))
        }
        _ => None,
    }
}

#[derive(Clone, Debug)]
struct SamplerBinding {
    tex_expr: String,
    samp_expr: String,
    is_3d: bool,
}

struct SamplerRegistry {
    user_textures: Vec<UserTexture>,
    user_by_key: HashMap<String, usize>,
}

impl SamplerRegistry {
    fn new() -> Self {
        Self {
            user_textures: Vec::new(),
            user_by_key: HashMap::new(),
        }
    }

    fn resolve_sampler(&mut self, ident: &str) -> Result<SamplerBinding, String> {
        let bare = ident.strip_prefix("sampler_").unwrap_or(ident);
        let (samp_kind_opt, name) = if let Some(rest) = bare.strip_prefix("fw_") {
            (Some(SamplerKind::Fw), rest)
        } else if let Some(rest) = bare.strip_prefix("fc_") {
            (Some(SamplerKind::Fc), rest)
        } else if let Some(rest) = bare.strip_prefix("pw_") {
            (Some(SamplerKind::Pw), rest)
        } else if let Some(rest) = bare.strip_prefix("pc_") {
            (Some(SamplerKind::Pc), rest)
        } else {
            (None, bare)
        };

        let samp_expr = match samp_kind_opt {
            Some(SamplerKind::Fw) => "samp_fw",
            Some(SamplerKind::Fc) => "samp_fc",
            Some(SamplerKind::Pw) => "samp_pw",
            Some(SamplerKind::Pc) => "samp_pc",
            None => "",
        };

        match name {
            "main" => Ok(SamplerBinding {
                tex_expr: "tex_main".to_string(),
                samp_expr: if samp_expr.is_empty() { "samp_main" } else { samp_expr }.to_string(),
                is_3d: false,
            }),
            "blur1" => Ok(SamplerBinding {
                tex_expr: "tex_blur1".to_string(),
                samp_expr: "samp_fc".to_string(),
                is_3d: false,
            }),
            "blur2" => Ok(SamplerBinding {
                tex_expr: "tex_blur2".to_string(),
                samp_expr: "samp_fc".to_string(),
                is_3d: false,
            }),
            "blur3" => Ok(SamplerBinding {
                tex_expr: "tex_blur3".to_string(),
                samp_expr: "samp_fc".to_string(),
                is_3d: false,
            }),
            "noise_lq" => Ok(SamplerBinding {
                tex_expr: "tex_noise_lq".to_string(),
                samp_expr: if samp_expr.is_empty() { "samp_fw" } else { samp_expr }.to_string(),
                is_3d: false,
            }),
            "noise_lq_lite" => Ok(SamplerBinding {
                tex_expr: "tex_noise_lq_lite".to_string(),
                samp_expr: if samp_expr.is_empty() { "samp_fw" } else { samp_expr }.to_string(),
                is_3d: false,
            }),
            "noise_mq" => Ok(SamplerBinding {
                tex_expr: "tex_noise_mq".to_string(),
                samp_expr: if samp_expr.is_empty() { "samp_fw" } else { samp_expr }.to_string(),
                is_3d: false,
            }),
            "noise_hq" => Ok(SamplerBinding {
                tex_expr: "tex_noise_hq".to_string(),
                samp_expr: if samp_expr.is_empty() { "samp_fw" } else { samp_expr }.to_string(),
                is_3d: false,
            }),
            "noisevol_lq" => Ok(SamplerBinding {
                tex_expr: "tex_noisevol_lq".to_string(),
                samp_expr: if samp_expr.is_empty() { "samp_fw" } else { samp_expr }.to_string(),
                is_3d: true,
            }),
            "noisevol_hq" => Ok(SamplerBinding {
                tex_expr: "tex_noisevol_hq".to_string(),
                samp_expr: if samp_expr.is_empty() { "samp_fw" } else { samp_expr }.to_string(),
                is_3d: true,
            }),
            other => {
                let kind = samp_kind_opt.unwrap_or(SamplerKind::Fw);
                let slot = self.register_user(other, kind)?;
                let final_samp = match kind {
                    SamplerKind::Fw => "samp_fw",
                    SamplerKind::Fc => "samp_fc",
                    SamplerKind::Pw => "samp_pw",
                    SamplerKind::Pc => "samp_pc",
                };
                Ok(SamplerBinding {
                    tex_expr: format!("tex_user{slot}"),
                    samp_expr: final_samp.to_string(),
                    is_3d: false,
                })
            }
        }
    }

    fn register_user(&mut self, name: &str, sampler: SamplerKind) -> Result<usize, String> {
        let key = format!("{name}:{sampler:?}");
        if let Some(&slot) = self.user_by_key.get(&key) {
            return Ok(slot);
        }
        // Also reuse same slot if name matches without explicit difference
        if let Some(u) = self.user_textures.iter().find(|u| u.name == name) {
            return Ok(u.slot);
        }
        if self.user_textures.len() >= 8 {
            return Err(format!("Preset uses > 8 custom textures (at '{name}')"));
        }
        let slot = self.user_textures.len();
        self.user_textures.push(UserTexture {
            slot,
            name: name.to_string(),
            sampler,
        });
        self.user_by_key.insert(key, slot);
        Ok(slot)
    }

    fn resolve_texsize(&mut self, ident: &str) -> Result<String, String> {
        let name = ident.strip_prefix("texsize_").unwrap_or(ident);
        match name {
            "main" => Ok("u_.texsize".to_string()),
            "noise_lq" => Ok("u_.noise_lq_size".to_string()),
            "noise_lq_lite" => Ok("u_.noise_lq_lite_size".to_string()),
            "noise_mq" => Ok("u_.noise_mq_size".to_string()),
            "noise_hq" => Ok("u_.noise_hq_size".to_string()),
            "noisevol_lq" => Ok("u_.noisevol_lq_size".to_string()),
            "noisevol_hq" => Ok("u_.noisevol_hq_size".to_string()),
            "blur1" | "blur2" | "blur3" => Ok("u_.texsize".to_string()),
            other => {
                // Strip optional fw_/fc_/pw_/pc_ if someone wrote texsize_fw_foo
                let stripped = other
                    .strip_prefix("fw_")
                    .or_else(|| other.strip_prefix("fc_"))
                    .or_else(|| other.strip_prefix("pw_"))
                    .or_else(|| other.strip_prefix("pc_"))
                    .unwrap_or(other);
                let slot = self.register_user(stripped, SamplerKind::Fw)?;
                Ok(format!("u_.user_texsize[{slot}]"))
            }
        }
    }
}

// ============================================================================
// GLSL Identifier Sanitization & Scoped Symbol Table
// ============================================================================

fn safe_glsl_ident(name: &str) -> String {
    match name {
        // GLSL reserved words that MilkDrop shaders occasionally use as local var names
        "sample" | "input" | "output" | "filter" | "active" | "common" | "partition"
        | "patch" | "resource" | "superp" | "subroutine" | "flat" | "smooth"
        | "noperspective" | "centroid" | "layout" | "precision" | "lowp" | "mediump"
        | "highp" | "invariant" | "coherent" | "volatile" | "restrict" | "readonly"
        | "writeonly" | "shared" | "buffer" | "image2D" | "atomic_uint" | "cast"
        | "namespace" | "using" | "sizeof" | "typedef" | "enum" | "class" | "union"
        | "public" | "private" | "protected" | "virtual" | "inline" | "extern"
        | "template" | "this" | "packed" | "noinline" | "hvec2" | "hvec3" | "hvec4"
        | "dvec2" | "dvec3" | "dvec4" | "fvec2" | "fvec3" | "fvec4" | "fixed"
        | "half" | "unsigned" | "long" | "short" | "goto" | "default" | "switch"
        | "case" | "main" => format!("_mdv_{name}"),
        _ => name.to_string(),
    }
}

#[derive(Clone, Debug)]
struct VarInfo {
    glsl_name: String,
    ty: Ty,
}

#[derive(Clone, Debug)]
struct FuncSig {
    ret_ty: Ty,
    params: Vec<Ty>,
}

struct TypedExpr {
    code: String,
    ty: Ty,
}

struct Translator<'a> {
    toks: &'a [Tok],
    pos: usize,
    samplers: SamplerRegistry,
    scopes: Vec<HashMap<String, VarInfo>>,
    funcs: HashMap<String, FuncSig>,
    builtins: HashMap<String, (String, Ty)>,
    mutated_groups: BTreeSet<&'static str>,
    used_q: bool,
    cur_ret_ty: Ty,
}

fn uniform_group_for_ident(name: &str) -> Option<&'static str> {
    match name {
        "time" | "fps" | "frame" | "progress" => Some("timing"),
        "bass" | "mid" | "treb" | "vol" => Some("audio"),
        "bass_att" | "mid_att" | "treb_att" | "vol_att" => Some("audio_att"),
        "aspect" => Some("aspect"),
        "texsize" => Some("texsize"),
        "rand_preset" => Some("rand_preset"),
        "rand_frame" => Some("rand_frame"),
        "roam_cos" => Some("roam_cos"),
        "roam_sin" => Some("roam_sin"),
        "slow_roam_cos" => Some("slow_roam_cos"),
        "slow_roam_sin" => Some("slow_roam_sin"),
        "blur1_min" | "blur2_min" | "blur3_min" | "mip_avg" => Some("blur_min_mip"),
        "blur1_max" | "blur2_max" | "blur3_max" | "mip_x" => Some("blur_max_mip"),
        "mip_y" | "mip_xy" => Some("mip_extra"),
        _ => {
            if let Some(rest) = name.strip_prefix("_q") {
                if rest.len() == 1 && (b'a'..=b'h').contains(&rest.as_bytes()[0]) {
                    return Some("q_vecs");
                }
            }
            if let Some(rest) = name.strip_prefix('q') {
                if let Ok(n) = rest.parse::<u32>() {
                    if (1..=32).contains(&n) {
                        return Some("q_vecs");
                    }
                }
            }
            None
        }
    }
}

fn detect_mutated_uniform_groups(toks: &[Tok]) -> BTreeSet<&'static str> {
    let mut out = BTreeSet::new();
    let is_assign_op = |t: &Tok| {
        matches!(
            t,
            Tok::Assign
                | Tok::PlusAssign
                | Tok::MinusAssign
                | Tok::MulAssign
                | Tok::DivAssign
                | Tok::ModAssign
                | Tok::PlusPlus
                | Tok::MinusMinus
        )
    };
    for i in 0..toks.len() {
        let Tok::Ident(ref name) = toks[i] else {
            continue;
        };
        let Some(grp) = uniform_group_for_ident(name) else {
            continue;
        };
        if i > 0 && matches!(toks[i - 1], Tok::PlusPlus | Tok::MinusMinus) {
            out.insert(grp);
            continue;
        }
        let mut j = i + 1;
        while j < toks.len() {
            match &toks[j] {
                Tok::Dot => {
                    j += 2;
                }
                Tok::LBracket => {
                    let mut depth = 1usize;
                    j += 1;
                    while j < toks.len() && depth > 0 {
                        match toks[j] {
                            Tok::LBracket => depth += 1,
                            Tok::RBracket => depth -= 1,
                            _ => {}
                        }
                        j += 1;
                    }
                }
                t if is_assign_op(t) => {
                    out.insert(grp);
                    break;
                }
                _ => break,
            }
        }
    }
    out
}

impl<'a> Translator<'a> {
    fn new(toks: &'a [Tok]) -> Self {
        let mutated_groups = detect_mutated_uniform_groups(toks);
        let root_for = |grp: &'static str| -> String {
            if mutated_groups.contains(grp) {
                format!("_mw_{grp}")
            } else {
                format!("u_.{grp}")
            }
        };
        let mut builtins = HashMap::new();
        let mut add_b = |hlsl: &str, glsl: &str, ty: Ty| {
            builtins.insert(hlsl.to_string(), (glsl.to_string(), ty));
        };
        let r_timing = root_for("timing");
        add_b("time", &format!("{r_timing}.x"), Ty::Float(1));
        add_b("fps", &format!("{r_timing}.y"), Ty::Float(1));
        add_b("frame", &format!("{r_timing}.z"), Ty::Float(1));
        add_b("progress", &format!("{r_timing}.w"), Ty::Float(1));

        let r_audio = root_for("audio");
        add_b("bass", &format!("{r_audio}.x"), Ty::Float(1));
        add_b("mid", &format!("{r_audio}.y"), Ty::Float(1));
        add_b("treb", &format!("{r_audio}.z"), Ty::Float(1));
        add_b("vol", &format!("{r_audio}.w"), Ty::Float(1));

        let r_audio_att = root_for("audio_att");
        add_b("bass_att", &format!("{r_audio_att}.x"), Ty::Float(1));
        add_b("mid_att", &format!("{r_audio_att}.y"), Ty::Float(1));
        add_b("treb_att", &format!("{r_audio_att}.z"), Ty::Float(1));
        add_b("vol_att", &format!("{r_audio_att}.w"), Ty::Float(1));

        let r_q = root_for("q_vecs");
        for g in 0..8 {
            let qa_name = format!("_q{}", (b'a' + g as u8) as char);
            add_b(&qa_name, &format!("{r_q}[{g}]"), Ty::Float(4));
            for c in 0..4 {
                let qidx = g * 4 + c + 1;
                let comp = ["x", "y", "z", "w"][c];
                add_b(&format!("q{qidx}"), &format!("{r_q}[{g}].{comp}"), Ty::Float(1));
            }
        }

        add_b("aspect", &root_for("aspect"), Ty::Float(4));
        add_b("texsize", &root_for("texsize"), Ty::Float(4));
        add_b("rand_preset", &root_for("rand_preset"), Ty::Float(4));
        add_b("rand_frame", &root_for("rand_frame"), Ty::Float(4));
        add_b("roam_cos", &root_for("roam_cos"), Ty::Float(4));
        add_b("roam_sin", &root_for("roam_sin"), Ty::Float(4));
        add_b("slow_roam_cos", &root_for("slow_roam_cos"), Ty::Float(4));
        add_b("slow_roam_sin", &root_for("slow_roam_sin"), Ty::Float(4));

        let r_bmin = root_for("blur_min_mip");
        add_b("blur1_min", &format!("{r_bmin}.x"), Ty::Float(1));
        add_b("blur2_min", &format!("{r_bmin}.y"), Ty::Float(1));
        add_b("blur3_min", &format!("{r_bmin}.z"), Ty::Float(1));
        add_b("mip_avg", &format!("{r_bmin}.w"), Ty::Float(1));

        let r_bmax = root_for("blur_max_mip");
        add_b("blur1_max", &format!("{r_bmax}.x"), Ty::Float(1));
        add_b("blur2_max", &format!("{r_bmax}.y"), Ty::Float(1));
        add_b("blur3_max", &format!("{r_bmax}.z"), Ty::Float(1));
        add_b("mip_x", &format!("{r_bmax}.w"), Ty::Float(1));

        let r_mipx = root_for("mip_extra");
        add_b("mip_y", &format!("{r_mipx}.x"), Ty::Float(1));
        add_b("mip_xy", &format!("{r_mipx}.y"), Ty::Float(1));

        for (prefix, field) in [
            ("rot_s", "rot_s"),
            ("rot_d", "rot_d"),
            ("rot_f", "rot_f"),
            ("rot_vf", "rot_vf"),
            ("rot_uf", "rot_uf"),
            ("rot_rand", "rot_rand"),
        ] {
            for k in 1..=4 {
                add_b(
                    &format!("{prefix}{k}"),
                    &format!("u_.{field}[{}]", k - 1),
                    Ty::Mat(4, 3),
                );
            }
        }

        add_b("M_PI", "3.14159265359", Ty::Float(1));
        add_b("M_PI_2", "1.57079632679", Ty::Float(1));
        add_b("M_INV_PI_2", "0.63661977237", Ty::Float(1));
        add_b("pi", "3.14159265359", Ty::Float(1));

        // Fragment shader variables
        add_b("uv", "uv", Ty::Float(2));
        add_b("uv_orig", "uv_orig", Ty::Float(2));
        add_b("rad", "rad", Ty::Float(1));
        add_b("ang", "ang", Ty::Float(1));
        add_b("hue_shader", "hue_shader", Ty::Float(3));
        add_b("ret", "ret", Ty::Float(3));

        Self {
            toks,
            pos: 0,
            samplers: SamplerRegistry::new(),
            scopes: vec![HashMap::new()],
            funcs: HashMap::new(),
            builtins,
            mutated_groups,
            used_q: false,
            cur_ret_ty: Ty::Void,
        }
    }

    fn peek(&self) -> &Tok {
        self.toks.get(self.pos).unwrap_or(&Tok::Eof)
    }

    fn peek_at(&self, offset: usize) -> &Tok {
        self.toks.get(self.pos + offset).unwrap_or(&Tok::Eof)
    }

    fn bump(&mut self) -> Tok {
        let t = self.peek().clone();
        if self.pos < self.toks.len() {
            self.pos += 1;
        }
        t
    }

    fn expect(&mut self, expected: &Tok) -> Result<(), String> {
        let got = self.bump();
        if &got == expected {
            Ok(())
        } else {
            Err(format!("Expected {expected:?}, got {got:?}"))
        }
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn declare_var(&mut self, hlsl_name: &str, ty: Ty) -> String {
        let glsl_name = safe_glsl_ident(hlsl_name);
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(
                hlsl_name.to_string(),
                VarInfo {
                    glsl_name: glsl_name.clone(),
                    ty,
                },
            );
        }
        glsl_name
    }

    fn lookup_var(&mut self, name: &str) -> Result<(String, Ty), String> {
        for scope in self.scopes.iter().rev() {
            if let Some(info) = scope.get(name) {
                return Ok((info.glsl_name.clone(), info.ty.clone()));
            }
        }
        if name.starts_with("sampler_") {
            let _ = self.samplers.resolve_sampler(name)?;
            return Ok((name.to_string(), Ty::Sampler(name.to_string())));
        }
        if name.starts_with("texsize_") {
            let expr = self.samplers.resolve_texsize(name)?;
            return Ok((expr, Ty::Float(4)));
        }
        if let Some((glsl, ty)) = self.builtins.get(name) {
            return Ok((glsl.clone(), ty.clone()));
        }
        // Fallback: undeclared MilkDrop variable (some shaders forget a global or assume float)
        Ok((safe_glsl_ident(name), Ty::Float(1)))
    }

    fn skip_qualifiers(&mut self) {
        while let Tok::Ident(ref id) = self.peek().clone() {
            if matches!(
                id.as_str(),
                "static" | "const" | "uniform" | "in" | "out" | "inout" | "precise" | "volatile"
            ) {
                self.bump();
            } else {
                break;
            }
        }
    }

    fn is_type_ahead(&self) -> bool {
        let mut idx = 0;
        while let Tok::Ident(id) = self.peek_at(idx) {
            if matches!(
                id.as_str(),
                "static" | "const" | "uniform" | "in" | "out" | "inout" | "precise" | "volatile"
            ) {
                idx += 1;
                continue;
            }
            return parse_hlsl_type(id).is_some() || id.starts_with("sampler_");
        }
        false
    }

    fn translate_unit(&mut self) -> Result<(String, String), String> {
        let mut globals_glsl = String::new();
        let mut body_glsl = String::new();
        let mut found_body = false;

        while !matches!(self.peek(), Tok::Eof) {
            if matches!(self.peek(), Tok::Semi) {
                self.bump();
                continue;
            }
            if let Tok::Ident(ref id) = self.peek().clone() {
                if id == "shader_body" {
                    self.bump();
                    self.push_scope();
                    // Add MilkDrop body variables to scope so declarations shadowing them work cleanly
                    self.declare_var("uv", Ty::Float(2));
                    self.declare_var("uv_orig", Ty::Float(2));
                    self.declare_var("rad", Ty::Float(1));
                    self.declare_var("ang", Ty::Float(1));
                    self.declare_var("hue_shader", Ty::Float(3));
                    self.declare_var("ret", Ty::Float(3));
                    body_glsl = self.parse_block(false)?;
                    self.pop_scope();
                    found_body = true;
                    continue;
                }
            }
            if found_body && !self.is_type_ahead() {
                break;
            }

            // Skip optional qualifiers
            self.skip_qualifiers();
            if matches!(self.peek(), Tok::Eof) {
                break;
            }

            let ty = match self.peek().clone() {
                Tok::Ident(ref id) => {
                    if id.starts_with("sampler_") && matches!(self.peek_at(1), Tok::Ident(_)) {
                        self.bump();
                        Ty::Sampler(String::new())
                    } else if let Some(t) = parse_hlsl_type(id) {
                        self.bump();
                        t
                    } else {
                        return Err(format!("Expected top-level declaration or shader_body, got '{id}'"));
                    }
                }
                other => return Err(format!("Unexpected token at top level: {other:?}")),
            };

            // Sampler declaration (`sampler sampler_foo [= sampler_state { ... }];`)
            if matches!(ty, Ty::Sampler(_)) {
                let Tok::Ident(name) = self.bump() else {
                    return Err("Expected sampler identifier".to_string());
                };
                let _ = self.samplers.resolve_sampler(&name)?;
                self.declare_var(&name, Ty::Sampler(name.clone()));
                if matches!(self.peek(), Tok::Assign) {
                    self.bump();
                    // Skip `sampler_state { ... }` or RHS identifier
                    if let Tok::Ident(ref s) = self.peek().clone() {
                        if s == "sampler_state" {
                            self.bump();
                        }
                    }
                    if matches!(self.peek(), Tok::LBrace) {
                        self.skip_balanced_braces()?;
                    } else {
                        while !matches!(self.peek(), Tok::Semi | Tok::Eof) {
                            self.bump();
                        }
                    }
                }
                if matches!(self.peek(), Tok::Semi) {
                    self.bump();
                }
                continue;
            }

            // Either function definition or global variable declaration(s)
            let Tok::Ident(name) = self.bump() else {
                return Err("Expected identifier after type".to_string());
            };

            if matches!(self.peek(), Tok::LParen) {
                // Function definition
                self.bump();
                self.push_scope();
                let mut param_tys = Vec::new();
                let mut params_glsl = Vec::new();
                if !matches!(self.peek(), Tok::RParen) {
                    loop {
                        let mut qual = String::new();
                        while let Tok::Ident(ref q) = self.peek().clone() {
                            if matches!(q.as_str(), "in" | "out" | "inout" | "const") {
                                if matches!(q.as_str(), "out" | "inout") {
                                    qual = format!("{q} ");
                                }
                                self.bump();
                            } else {
                                break;
                            }
                        }
                        let pty = match self.bump() {
                            Tok::Ident(ref tid) => parse_hlsl_type(tid)
                                .ok_or_else(|| format!("Unknown param type '{tid}'"))?,
                            other => return Err(format!("Expected param type, got {other:?}")),
                        };
                        let pname = match self.bump() {
                            Tok::Ident(p) => p,
                            other => return Err(format!("Expected param name, got {other:?}")),
                        };
                        // Optional semantic `: TEXCOORD0`
                        if matches!(self.peek(), Tok::Colon) {
                            self.bump();
                            self.bump();
                        }
                        let gname = self.declare_var(&pname, pty.clone());
                        param_tys.push(pty.clone());
                        params_glsl.push(format!("{qual}{} {gname}", pty.to_glsl()));
                        if matches!(self.peek(), Tok::Comma) {
                            self.bump();
                        } else {
                            break;
                        }
                    }
                }
                self.expect(&Tok::RParen)?;
                if matches!(self.peek(), Tok::Colon) {
                    self.bump();
                    self.bump();
                }
                self.funcs.insert(
                    name.clone(),
                    FuncSig {
                        ret_ty: ty.clone(),
                        params: param_tys,
                    },
                );
                let prev_ret = std::mem::replace(&mut self.cur_ret_ty, ty.clone());
                let mut fbody = self.parse_block(false)?;
                self.cur_ret_ty = prev_ret;
                self.pop_scope();

                if ty != Ty::Void {
                    let last_line = fbody.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("");
                    if !last_line.trim_start().starts_with("return") {
                        let fallback = match &ty {
                            Ty::Float(_) | Ty::Int(_) | Ty::Bool(_) | Ty::Mat(_, _) => {
                                format!("{}({})", ty.to_glsl(), match ty {
                                    Ty::Int(_) => "0",
                                    Ty::Bool(_) => "false",
                                    _ => "0.0",
                                })
                            }
                            _ => "0.0".to_string(),
                        };
                        fbody.push_str(&format!("  return {fallback};\n"));
                    }
                }

                let gfn = safe_glsl_ident(&name);
                globals_glsl.push_str(&format!(
                    "{} {gfn}({}) {{\n{fbody}}}\n\n",
                    ty.to_glsl(),
                    params_glsl.join(", ")
                ));
            } else {
                // Global variable(s)
                let decl = self.parse_var_decl_tail(ty, name)?;
                globals_glsl.push_str(&decl);
                globals_glsl.push('\n');
            }
        }

        if !found_body {
            // Check if a `PS(...)` function exists instead
            if let Some(sig) = self.funcs.get("PS").cloned() {
                let _ = sig;
                body_glsl = "  ret = PS(uv).rgb;\n".to_string();
            } else {
                return Err("Missing 'shader_body' block".to_string());
            }
        }
        Ok((globals_glsl, body_glsl))
    }

    fn skip_balanced_braces(&mut self) -> Result<(), String> {
        self.expect(&Tok::LBrace)?;
        let mut depth = 1usize;
        while depth > 0 {
            match self.bump() {
                Tok::LBrace => depth += 1,
                Tok::RBrace => depth -= 1,
                Tok::Eof => return Err("Unclosed '{'".to_string()),
                _ => {}
            }
        }
        Ok(())
    }

    fn parse_var_decl_tail(&mut self, base_ty: Ty, first_name: String) -> Result<String, String> {
        let mut out = String::new();
        let mut name = first_name;
        loop {
            let mut arr_len: Option<usize> = None;
            if matches!(self.peek(), Tok::LBracket) {
                self.bump();
                let n = if let Tok::IntLit(v) = self.peek().clone() {
                    self.bump();
                    v.max(1) as usize
                } else {
                    0
                };
                self.expect(&Tok::RBracket)?;
                arr_len = Some(n);
            }

            // Parse optional initializer
            let init_expr = if matches!(self.peek(), Tok::Assign) {
                self.bump();
                if matches!(self.peek(), Tok::LBrace) {
                    let items = self.parse_brace_list()?;
                    if let Some(n) = arr_len {
                        let actual_len = if n == 0 {
                            let elem_w = base_ty.width().unwrap_or(1) as usize;
                            items.len().div_ceil(elem_w).max(1)
                        } else {
                            n
                        };
                        let ty = Ty::Array(Box::new(base_ty.clone()), actual_len);
                        let gname = self.declare_var(&name, ty.clone());
                        let init_str = self.format_array_init(&base_ty, actual_len, &items)?;
                        out.push_str(&format!("{} {gname} = {init_str}; ", ty.to_glsl()));
                    } else {
                        let gname = self.declare_var(&name, base_ty.clone());
                        let init_str = self.format_brace_init(&base_ty, &items)?;
                        out.push_str(&format!("{} {gname} = {init_str}; ", base_ty.to_glsl()));
                    }
                    None
                } else {
                    let e = self.parse_assign()?;
                    let ce = self.coerce(e, &base_ty)?;
                    let gname = self.declare_var(&name, base_ty.clone());
                    out.push_str(&format!("{} {gname} = {}; ", base_ty.to_glsl(), ce.code));
                    Some(())
                }
            } else {
                let ty = if let Some(n) = arr_len {
                    Ty::Array(Box::new(base_ty.clone()), n.max(1))
                } else {
                    base_ty.clone()
                };
                let gname = self.declare_var(&name, ty.clone());
                out.push_str(&format!("{} {gname}; ", ty.to_glsl()));
                Some(())
            };
            let _ = init_expr;

            if matches!(self.peek(), Tok::Comma) {
                self.bump();
                name = match self.bump() {
                    Tok::Ident(n) => n,
                    other => return Err(format!("Expected variable name after ',', got {other:?}")),
                };
            } else {
                break;
            }
        }
        self.expect(&Tok::Semi)?;
        Ok(out)
    }

    fn parse_brace_list(&mut self) -> Result<Vec<TypedExpr>, String> {
        self.expect(&Tok::LBrace)?;
        let mut items = Vec::new();
        while !matches!(self.peek(), Tok::RBrace | Tok::Eof) {
            if matches!(self.peek(), Tok::LBrace) {
                let sub = self.parse_brace_list()?;
                items.extend(sub);
            } else {
                items.push(self.parse_assign()?);
            }
            if matches!(self.peek(), Tok::Comma) {
                self.bump();
            } else {
                break;
            }
        }
        self.expect(&Tok::RBrace)?;
        Ok(items)
    }

    fn format_brace_init(&mut self, ty: &Ty, items: &[TypedExpr]) -> Result<String, String> {
        match ty {
            Ty::Mat(r, c) => {
                let need = (*r as usize) * (*c as usize);
                let mut args = Vec::with_capacity(need);
                for it in items.iter().take(need) {
                    args.push(self.coerce_clone(it, &Ty::Float(1))?.code);
                }
                while args.len() < need {
                    args.push("0.0".to_string());
                }
                Ok(format!("{}({})", ty.to_glsl(), args.join(", ")))
            }
            Ty::Float(w) | Ty::Int(w) | Ty::Bool(w) => {
                let scalar_ty = ty.with_width(1);
                let args: Vec<String> = items
                    .iter()
                    .take(*w as usize)
                    .map(|it| self.coerce_clone(it, &scalar_ty).map(|c| c.code))
                    .collect::<Result<_, _>>()?;
                Ok(format!("{}({})", ty.to_glsl(), args.join(", ")))
            }
            _ => Err(format!("Unsupported brace init for type {ty:?}")),
        }
    }

    fn format_array_init(&mut self, elem_ty: &Ty, len: usize, items: &[TypedExpr]) -> Result<String, String> {
        let elem_w = elem_ty.width().unwrap_or(1) as usize;
        let scalar_ty = elem_ty.with_width(1);
        let mut elems = Vec::with_capacity(len);
        if items.len() == len && items.first().map_or(false, |it| it.ty == *elem_ty) {
            for it in items {
                elems.push(it.code.clone());
            }
        } else {
            let mut idx = 0;
            for _ in 0..len {
                if elem_w == 1 {
                    let s = if idx < items.len() {
                        self.coerce_clone(&items[idx], &scalar_ty)?.code
                    } else {
                        "0.0".to_string()
                    };
                    idx += 1;
                    elems.push(s);
                } else {
                    let mut comps = Vec::with_capacity(elem_w);
                    for _ in 0..elem_w {
                        let s = if idx < items.len() {
                            self.coerce_clone(&items[idx], &scalar_ty)?.code
                        } else {
                            "0.0".to_string()
                        };
                        idx += 1;
                        comps.push(s);
                    }
                    elems.push(format!("{}({})", elem_ty.to_glsl(), comps.join(", ")));
                }
            }
        }
        Ok(format!("{}[{}]({})", elem_ty.to_glsl(), len, elems.join(", ")))
    }

    fn parse_block(&mut self, new_scope: bool) -> Result<String, String> {
        self.expect(&Tok::LBrace)?;
        if new_scope {
            self.push_scope();
        }
        let mut out = String::new();
        while !matches!(self.peek(), Tok::RBrace | Tok::Eof) {
            let st = self.parse_stmt()?;
            if !st.is_empty() {
                out.push_str("  ");
                out.push_str(&st);
                out.push('\n');
            }
        }
        if new_scope {
            self.pop_scope();
        }
        self.expect(&Tok::RBrace)?;
        Ok(out)
    }

    fn parse_stmt(&mut self) -> Result<String, String> {
        // Empty statement
        if matches!(self.peek(), Tok::Semi) {
            self.bump();
            return Ok(String::new());
        }
        // Nested block
        if matches!(self.peek(), Tok::LBrace) {
            let body = self.parse_block(true)?;
            return Ok(format!("{{\n{body}  }}"));
        }
        // Control flow
        if let Tok::Ident(ref kw) = self.peek().clone() {
            match kw.as_str() {
                "if" => {
                    self.bump();
                    self.expect(&Tok::LParen)?;
                    let cond = self.parse_expr()?;
                    let cond_b = self.to_bool_scalar(cond)?;
                    self.expect(&Tok::RParen)?;
                    let t_stmt = self.parse_stmt_or_block()?;
                    if matches!(self.peek(), Tok::Ident(ref s) if s == "else") {
                        self.bump();
                        let e_stmt = self.parse_stmt_or_block()?;
                        return Ok(format!("if ({}) {t_stmt} else {e_stmt}", cond_b.code));
                    }
                    return Ok(format!("if ({}) {t_stmt}", cond_b.code));
                }
                "while" => {
                    self.bump();
                    self.expect(&Tok::LParen)?;
                    let cond = self.parse_expr()?;
                    let cond_b = self.to_bool_scalar(cond)?;
                    self.expect(&Tok::RParen)?;
                    let body = self.parse_stmt_or_block()?;
                    return Ok(format!("while ({}) {body}", cond_b.code));
                }
                "do" => {
                    self.bump();
                    let body = self.parse_stmt_or_block()?;
                    if matches!(self.peek(), Tok::Ident(ref s) if s == "while") {
                        self.bump();
                    }
                    self.expect(&Tok::LParen)?;
                    let cond = self.parse_expr()?;
                    let cond_b = self.to_bool_scalar(cond)?;
                    self.expect(&Tok::RParen)?;
                    self.expect(&Tok::Semi)?;
                    return Ok(format!("do {body} while ({});", cond_b.code));
                }
                "for" => {
                    self.bump();
                    self.push_scope();
                    self.expect(&Tok::LParen)?;
                    let init = if matches!(self.peek(), Tok::Semi) {
                        self.bump();
                        String::new()
                    } else if self.is_type_ahead() {
                        self.skip_qualifiers();
                        let Tok::Ident(tid) = self.bump() else {
                            return Err("Expected type in for-init".to_string());
                        };
                        let ty = parse_hlsl_type(&tid).ok_or_else(|| format!("Bad type '{tid}'"))?;
                        let Tok::Ident(vname) = self.bump() else {
                            return Err("Expected var name in for-init".to_string());
                        };
                        self.parse_var_decl_tail(ty, vname)?
                    } else {
                        let e = self.parse_expr()?;
                        self.expect(&Tok::Semi)?;
                        format!("{};", e.code)
                    };
                    let init_clean = init.trim_end().trim_end_matches(';');
                    let cond = if matches!(self.peek(), Tok::Semi) {
                        self.bump();
                        "true".to_string()
                    } else {
                        let e = self.parse_expr()?;
                        self.expect(&Tok::Semi)?;
                        self.to_bool_scalar(e)?.code
                    };
                    let step = if matches!(self.peek(), Tok::RParen) {
                        String::new()
                    } else {
                        self.parse_expr()?.code
                    };
                    self.expect(&Tok::RParen)?;
                    let body = self.parse_stmt_or_block()?;
                    self.pop_scope();
                    return Ok(format!("for ({init_clean}; {cond}; {step}) {body}"));
                }
                "return" => {
                    self.bump();
                    if matches!(self.peek(), Tok::Semi) {
                        self.bump();
                        return Ok("return;".to_string());
                    }
                    let e = self.parse_expr()?;
                    self.expect(&Tok::Semi)?;
                    let ce = if self.cur_ret_ty != Ty::Void {
                        self.coerce(e, &self.cur_ret_ty.clone())?
                    } else {
                        e
                    };
                    return Ok(format!("return {};", ce.code));
                }
                "break" => {
                    self.bump();
                    self.expect(&Tok::Semi)?;
                    return Ok("break;".to_string());
                }
                "continue" => {
                    self.bump();
                    self.expect(&Tok::Semi)?;
                    return Ok("continue;".to_string());
                }
                "discard" => {
                    self.bump();
                    self.expect(&Tok::Semi)?;
                    return Ok("discard;".to_string());
                }
                _ => {}
            }
        }

        // Local declaration
        if self.is_type_ahead() {
            self.skip_qualifiers();
            let Tok::Ident(tid) = self.bump() else {
                return Err("Expected type".to_string());
            };
            let ty = parse_hlsl_type(&tid).ok_or_else(|| format!("Unknown type '{tid}'"))?;
            let Tok::Ident(vname) = self.bump() else {
                return Err("Expected variable name".to_string());
            };
            return self.parse_var_decl_tail(ty, vname);
        }

        // Expression statement
        let e = self.parse_expr()?;
        self.expect(&Tok::Semi)?;
        Ok(format!("{};", e.code))
    }

    fn parse_stmt_or_block(&mut self) -> Result<String, String> {
        if matches!(self.peek(), Tok::LBrace) {
            let b = self.parse_block(true)?;
            Ok(format!("{{\n{b}  }}"))
        } else {
            self.parse_stmt()
        }
    }

    fn parse_expr(&mut self) -> Result<TypedExpr, String> {
        let mut lhs = self.parse_assign()?;
        while matches!(self.peek(), Tok::Comma) {
            self.bump();
            let rhs = self.parse_assign()?;
            lhs = TypedExpr {
                code: format!("({}, {})", lhs.code, rhs.code),
                ty: rhs.ty,
            };
        }
        Ok(lhs)
    }

    fn parse_assign(&mut self) -> Result<TypedExpr, String> {
        let lhs = self.parse_ternary()?;
        let op = match self.peek() {
            Tok::Assign => Some("="),
            Tok::PlusAssign => Some("+="),
            Tok::MinusAssign => Some("-="),
            Tok::MulAssign => Some("*="),
            Tok::DivAssign => Some("/="),
            Tok::ModAssign => Some("%="),
            _ => None,
        };
        if let Some(op_str) = op {
            self.bump();
            let rhs = self.parse_assign()?;
            if op_str == "%=" {
                let cr = self.coerce(rhs, &lhs.ty)?;
                return Ok(TypedExpr {
                    code: format!("({} = mod({}, {}))", lhs.code, lhs.code, cr.code),
                    ty: lhs.ty,
                });
            }
            let cr = if matches!(op_str, "*=" | "/=") && lhs.ty.width().unwrap_or(1) > 1 && rhs.ty.is_scalar() {
                self.coerce(rhs, &lhs.ty.with_width(1))?
            } else {
                self.coerce(rhs, &lhs.ty)?
            };
            return Ok(TypedExpr {
                code: format!("({} {op_str} {})", lhs.code, cr.code),
                ty: lhs.ty,
            });
        }
        Ok(lhs)
    }

    fn parse_ternary(&mut self) -> Result<TypedExpr, String> {
        let cond = self.parse_log_or()?;
        if matches!(self.peek(), Tok::Question) {
            self.bump();
            let t_e = self.parse_assign()?;
            self.expect(&Tok::Colon)?;
            let f_e = self.parse_assign()?;
            let out_ty = unify_types(&t_e.ty, &f_e.ty);
            let ct = self.coerce(t_e, &out_ty)?;
            let cf = self.coerce(f_e, &out_ty)?;

            let cw = cond.ty.width().unwrap_or(1);
            if cw > 1 {
                // Component-wise ternary in HLSL -> GLSL mix(f, t, bvecN(cond))
                let out_w = out_ty.width().unwrap_or(cw).min(cw);
                let final_ty = out_ty.with_width(out_w);
                let cb = self.coerce(cond, &Ty::Bool(out_w))?;
                let ct = self.coerce(ct, &final_ty)?;
                let cf = self.coerce(cf, &final_ty)?;
                return Ok(TypedExpr {
                    code: format!("mix({}, {}, {})", cf.code, ct.code, cb.code),
                    ty: final_ty,
                });
            }
            let cond_b = self.to_bool_scalar(cond)?;
            return Ok(TypedExpr {
                code: format!("({} ? {} : {})", cond_b.code, ct.code, cf.code),
                ty: out_ty,
            });
        }
        Ok(cond)
    }

    fn parse_log_or(&mut self) -> Result<TypedExpr, String> {
        let mut lhs = self.parse_log_and()?;
        while matches!(self.peek(), Tok::LogOr) {
            self.bump();
            let rhs = self.parse_log_and()?;
            let a = self.to_bool_scalar(lhs)?;
            let b = self.to_bool_scalar(rhs)?;
            lhs = TypedExpr {
                code: format!("({} || {})", a.code, b.code),
                ty: Ty::Bool(1),
            };
        }
        Ok(lhs)
    }

    fn parse_log_and(&mut self) -> Result<TypedExpr, String> {
        let mut lhs = self.parse_bit_or()?;
        while matches!(self.peek(), Tok::LogAnd) {
            self.bump();
            let rhs = self.parse_bit_or()?;
            let a = self.to_bool_scalar(lhs)?;
            let b = self.to_bool_scalar(rhs)?;
            lhs = TypedExpr {
                code: format!("({} && {})", a.code, b.code),
                ty: Ty::Bool(1),
            };
        }
        Ok(lhs)
    }

    fn parse_bit_or(&mut self) -> Result<TypedExpr, String> {
        let mut lhs = self.parse_bit_xor()?;
        while matches!(self.peek(), Tok::BitOr) {
            self.bump();
            let rhs = self.parse_bit_xor()?;
            lhs = self.bin_int_op("|", lhs, rhs)?;
        }
        Ok(lhs)
    }

    fn parse_bit_xor(&mut self) -> Result<TypedExpr, String> {
        let mut lhs = self.parse_bit_and()?;
        while matches!(self.peek(), Tok::BitXor) {
            self.bump();
            let rhs = self.parse_bit_and()?;
            lhs = self.bin_int_op("^", lhs, rhs)?;
        }
        Ok(lhs)
    }

    fn parse_bit_and(&mut self) -> Result<TypedExpr, String> {
        let mut lhs = self.parse_eq()?;
        while matches!(self.peek(), Tok::BitAnd) {
            self.bump();
            let rhs = self.parse_eq()?;
            lhs = self.bin_int_op("&", lhs, rhs)?;
        }
        Ok(lhs)
    }

    fn parse_eq(&mut self) -> Result<TypedExpr, String> {
        let mut lhs = self.parse_rel()?;
        loop {
            let op = match self.peek() {
                Tok::Eq => ("==", "equal"),
                Tok::Ne => ("!=", "notEqual"),
                _ => break,
            };
            self.bump();
            let rhs = self.parse_rel()?;
            lhs = self.cmp_op(op.0, op.1, lhs, rhs)?;
        }
        Ok(lhs)
    }

    fn parse_rel(&mut self) -> Result<TypedExpr, String> {
        let mut lhs = self.parse_shift()?;
        loop {
            let op = match self.peek() {
                Tok::Lt => ("<", "lessThan"),
                Tok::Gt => (">", "greaterThan"),
                Tok::Le => ("<=", "lessThanEqual"),
                Tok::Ge => (">=", "greaterThanEqual"),
                _ => break,
            };
            self.bump();
            let rhs = self.parse_shift()?;
            lhs = self.cmp_op(op.0, op.1, lhs, rhs)?;
        }
        Ok(lhs)
    }

    fn parse_shift(&mut self) -> Result<TypedExpr, String> {
        let mut lhs = self.parse_add()?;
        loop {
            let op = match self.peek() {
                Tok::Shl => "<<",
                Tok::Shr => ">>",
                _ => break,
            };
            self.bump();
            let rhs = self.parse_add()?;
            lhs = self.bin_int_op(op, lhs, rhs)?;
        }
        Ok(lhs)
    }

    fn parse_add(&mut self) -> Result<TypedExpr, String> {
        let mut lhs = self.parse_mul()?;
        loop {
            let op = match self.peek() {
                Tok::Plus => "+",
                Tok::Minus => "-",
                _ => break,
            };
            self.bump();
            let rhs = self.parse_mul()?;
            lhs = self.bin_arith_op(op, lhs, rhs)?;
        }
        Ok(lhs)
    }

    fn parse_mul(&mut self) -> Result<TypedExpr, String> {
        let mut lhs = self.parse_unary()?;
        loop {
            let op = match self.peek() {
                Tok::Star => "*",
                Tok::Slash => "/",
                Tok::Percent => "%",
                _ => break,
            };
            self.bump();
            let rhs = self.parse_unary()?;
            lhs = self.bin_arith_op(op, lhs, rhs)?;
        }
        Ok(lhs)
    }

    fn parse_unary(&mut self) -> Result<TypedExpr, String> {
        match self.peek().clone() {
            Tok::Minus => {
                self.bump();
                let e = self.parse_unary()?;
                let e = if let Ty::Bool(w) = e.ty {
                    self.coerce(e, &Ty::Float(w))?
                } else {
                    e
                };
                let ty = e.ty.clone();
                Ok(TypedExpr {
                    code: format!("(-{})", e.code),
                    ty,
                })
            }
            Tok::Plus => {
                self.bump();
                let e = self.parse_unary()?;
                if let Ty::Bool(w) = e.ty {
                    self.coerce(e, &Ty::Float(w))
                } else {
                    Ok(e)
                }
            }
            Tok::Bang => {
                self.bump();
                let e = self.parse_unary()?;
                let w = e.ty.width().unwrap_or(1);
                if w == 1 {
                    let b = self.to_bool_scalar(e)?;
                    Ok(TypedExpr {
                        code: format!("(!{})", b.code),
                        ty: Ty::Bool(1),
                    })
                } else {
                    let b = self.coerce(e, &Ty::Bool(w))?;
                    Ok(TypedExpr {
                        code: format!("not({})", b.code),
                        ty: Ty::Bool(w),
                    })
                }
            }
            Tok::Tilde => {
                self.bump();
                let e = self.parse_unary()?;
                let w = e.ty.width().unwrap_or(1);
                let ie = self.coerce(e, &Ty::Int(w))?;
                Ok(TypedExpr {
                    code: format!("(~{})", ie.code),
                    ty: Ty::Int(w),
                })
            }
            Tok::PlusPlus => {
                self.bump();
                let e = self.parse_unary()?;
                let ty = e.ty.clone();
                Ok(TypedExpr {
                    code: format!("(++{})", e.code),
                    ty,
                })
            }
            Tok::MinusMinus => {
                self.bump();
                let e = self.parse_unary()?;
                let ty = e.ty.clone();
                Ok(TypedExpr {
                    code: format!("(--{})", e.code),
                    ty,
                })
            }
            Tok::LParen => {
                // Check for C-style cast `(type)expr`
                if let Some((cast_ty, skip)) = self.check_cast_ahead() {
                    self.pos += skip;
                    let inner = self.parse_unary()?;
                    return self.coerce(inner, &cast_ty);
                }
                self.parse_postfix()
            }
            _ => self.parse_postfix(),
        }
    }

    fn check_cast_ahead(&self) -> Option<(Ty, usize)> {
        if !matches!(self.peek(), Tok::LParen) {
            return None;
        }
        if let Tok::Ident(id) = self.peek_at(1) {
            if matches!(self.peek_at(2), Tok::RParen) {
                if let Some(ty) = parse_hlsl_type(id) {
                    return Some((ty, 3));
                }
            }
        }
        None
    }

    fn parse_postfix(&mut self) -> Result<TypedExpr, String> {
        let mut expr = self.parse_primary()?;
        loop {
            match self.peek().clone() {
                Tok::Dot => {
                    self.bump();
                    let Tok::Ident(field) = self.bump() else {
                        return Err("Expected swizzle/field after '.'".to_string());
                    };
                    if let Some(sw_len) = swizzle_len(&field) {
                        let glsl_sw = normalize_swizzle(&field);
                        // Allow scalar.xxx in HLSL -> vecN(scalar) in GLSL!
                        if expr.ty.is_scalar() {
                            let target = expr.ty.with_width(sw_len);
                            if sw_len == 1 {
                                expr = TypedExpr {
                                    code: expr.code,
                                    ty: target,
                                };
                            } else {
                                expr = TypedExpr {
                                    code: format!("{}({})", target.to_glsl(), expr.code),
                                    ty: target,
                                };
                            }
                        } else {
                            let out_ty = match expr.ty {
                                Ty::Float(_) => Ty::Float(sw_len),
                                Ty::Int(_) => Ty::Int(sw_len),
                                Ty::Bool(_) => Ty::Bool(sw_len),
                                _ => Ty::Float(sw_len),
                            };
                            expr = TypedExpr {
                                code: format!("{}.{glsl_sw}", expr.code),
                                ty: out_ty,
                            };
                        }
                    } else {
                        expr = TypedExpr {
                            code: format!("{}.{field}", expr.code),
                            ty: Ty::Float(1),
                        };
                    }
                }
                Tok::LBracket => {
                    self.bump();
                    let idx = self.parse_expr()?;
                    let idx_i = self.coerce(idx, &Ty::Int(1))?;
                    self.expect(&Tok::RBracket)?;
                    let elem_ty = match expr.ty {
                        Ty::Array(ref el, _) => (**el).clone(),
                        Ty::Mat(_, c) => Ty::Float(c),
                        Ty::Float(_) => Ty::Float(1),
                        Ty::Int(_) => Ty::Int(1),
                        Ty::Bool(_) => Ty::Bool(1),
                        _ => Ty::Float(1),
                    };
                    expr = TypedExpr {
                        code: format!("{}[{}]", expr.code, idx_i.code),
                        ty: elem_ty,
                    };
                }
                Tok::PlusPlus => {
                    self.bump();
                    let ty = expr.ty.clone();
                    expr = TypedExpr {
                        code: format!("({}++)", expr.code),
                        ty,
                    };
                }
                Tok::MinusMinus => {
                    self.bump();
                    let ty = expr.ty.clone();
                    expr = TypedExpr {
                        code: format!("({}--)", expr.code),
                        ty,
                    };
                }
                _ => break,
            }
        }
        Ok(expr)
    }

    fn parse_primary(&mut self) -> Result<TypedExpr, String> {
        match self.bump() {
            Tok::FloatLit(s) => Ok(TypedExpr {
                code: s,
                ty: Ty::Float(1),
            }),
            Tok::IntLit(v) => Ok(TypedExpr {
                code: format!("{v}"),
                ty: Ty::Int(1),
            }),
            Tok::LParen => {
                let e = self.parse_expr()?;
                self.expect(&Tok::RParen)?;
                Ok(TypedExpr {
                    code: format!("({})", e.code),
                    ty: e.ty,
                })
            }
            Tok::Ident(name) => {
                if name == "true" || name == "false" {
                    return Ok(TypedExpr {
                        code: name,
                        ty: Ty::Bool(1),
                    });
                }
                if matches!(self.peek(), Tok::LParen) {
                    self.bump();
                    let args = self.parse_arg_list()?;
                    return self.build_call(&name, args);
                }
                let (glsl, ty) = self.lookup_var(&name)?;
                Ok(TypedExpr { code: glsl, ty })
            }
            other => Err(format!("Unexpected token {other:?} in expression")),
        }
    }

    fn parse_arg_list(&mut self) -> Result<Vec<TypedExpr>, String> {
        let mut args = Vec::new();
        if matches!(self.peek(), Tok::RParen) {
            self.bump();
            return Ok(args);
        }
        loop {
            args.push(self.parse_assign()?);
            if matches!(self.peek(), Tok::Comma) {
                self.bump();
            } else {
                break;
            }
        }
        self.expect(&Tok::RParen)?;
        Ok(args)
    }

    fn build_call(&mut self, name: &str, args: Vec<TypedExpr>) -> Result<TypedExpr, String> {
        // Type constructor e.g. float3(...), float2x2(...)
        if let Some(target_ty) = parse_hlsl_type(name) {
            return self.build_type_ctor(&target_ty, args);
        }

        // User-defined function
        if let Some(sig) = self.funcs.get(name).cloned() {
            let mut cargs = Vec::with_capacity(args.len());
            for (idx, a) in args.into_iter().enumerate() {
                if let Some(pty) = sig.params.get(idx) {
                    cargs.push(self.coerce(a, pty)?.code);
                } else {
                    cargs.push(a.code);
                }
            }
            let gfn = safe_glsl_ident(name);
            return Ok(TypedExpr {
                code: format!("{gfn}({})", cargs.join(", ")),
                ty: sig.ret_ty,
            });
        }

        let lower_name = name.to_ascii_lowercase();
        if let Some(target_ty) = parse_hlsl_type(&lower_name) {
            return self.build_type_ctor(&target_ty, args);
        }

        match lower_name.as_str() {
            "getmain" | "getpixel" => {
                let uv = self.coerce_arg(&args, 0, &Ty::Float(2))?;
                Ok(TypedExpr {
                    code: format!("_md_get_pixel({})", uv.code),
                    ty: Ty::Float(3),
                })
            }
            "getblur1" | "getblur2" | "getblur3" => {
                let uv = self.coerce_arg(&args, 0, &Ty::Float(2))?;
                let idx = &lower_name[7..];
                Ok(TypedExpr {
                    code: format!("_md_get_blur{idx}({})", uv.code),
                    ty: Ty::Float(3),
                })
            }
            "lum" => {
                let a = args.into_iter().next().ok_or("lum requires 1 arg")?;
                let w = a.ty.width().unwrap_or(3);
                if w >= 3 {
                    let v3 = self.coerce(a, &Ty::Float(3))?;
                    Ok(TypedExpr {
                        code: format!("dot({}, vec3(0.32, 0.49, 0.29))", v3.code),
                        ty: Ty::Float(1),
                    })
                } else if w == 2 {
                    let v2 = self.coerce(a, &Ty::Float(2))?;
                    Ok(TypedExpr {
                        code: format!("dot({}, vec2(0.32, 0.49))", v2.code),
                        ty: Ty::Float(1),
                    })
                } else {
                    let f = self.coerce(a, &Ty::Float(1))?;
                    Ok(TypedExpr {
                        code: format!("({} * 0.32)", f.code),
                        ty: Ty::Float(1),
                    })
                }
            }
            "tex2d" | "tex2dproj" | "tex2dgrad" => {
                if args.len() < 2 {
                    return Err(format!("{name} requires at least 2 args"));
                }
                let samp_name = match &args[0].ty {
                    Ty::Sampler(s) => s.clone(),
                    _ => args[0].code.clone(),
                };
                let b = self.samplers.resolve_sampler(&samp_name)?;
                if b.is_3d {
                    let uvw = self.coerce_clone(&args[1], &Ty::Float(3))?;
                    return Ok(TypedExpr {
                        code: format!("texture(sampler3D({}, {}), {})", b.tex_expr, b.samp_expr, uvw.code),
                        ty: Ty::Float(4),
                    });
                }
                let uv = self.coerce_clone(&args[1], &Ty::Float(2))?;
                if args.len() == 4 {
                    let dx = self.coerce_clone(&args[2], &Ty::Float(2))?;
                    let dy = self.coerce_clone(&args[3], &Ty::Float(2))?;
                    return Ok(TypedExpr {
                        code: format!(
                            "textureGrad(sampler2D({}, {}), {}, {}, {})",
                            b.tex_expr, b.samp_expr, uv.code, dx.code, dy.code
                        ),
                        ty: Ty::Float(4),
                    });
                }
                Ok(TypedExpr {
                    code: format!("texture(sampler2D({}, {}), {})", b.tex_expr, b.samp_expr, uv.code),
                    ty: Ty::Float(4),
                })
            }
            "tex2dlod" => {
                if args.len() < 2 {
                    return Err("tex2Dlod requires 2 args".to_string());
                }
                let samp_name = match &args[0].ty {
                    Ty::Sampler(s) => s.clone(),
                    _ => args[0].code.clone(),
                };
                let b = self.samplers.resolve_sampler(&samp_name)?;
                let uv4 = self.coerce_clone(&args[1], &Ty::Float(4))?;
                Ok(TypedExpr {
                    code: format!(
                        "textureLod(sampler2D({}, {}), ({}).xy, ({}).w)",
                        b.tex_expr, b.samp_expr, uv4.code, uv4.code
                    ),
                    ty: Ty::Float(4),
                })
            }
            "tex2dbias" => {
                if args.len() < 2 {
                    return Err("tex2Dbias requires 2 args".to_string());
                }
                let samp_name = match &args[0].ty {
                    Ty::Sampler(s) => s.clone(),
                    _ => args[0].code.clone(),
                };
                let b = self.samplers.resolve_sampler(&samp_name)?;
                let uv4 = self.coerce_clone(&args[1], &Ty::Float(4))?;
                Ok(TypedExpr {
                    code: format!(
                        "texture(sampler2D({}, {}), ({}).xy, ({}).w)",
                        b.tex_expr, b.samp_expr, uv4.code, uv4.code
                    ),
                    ty: Ty::Float(4),
                })
            }
            "tex3d" => {
                if args.len() < 2 {
                    return Err("tex3D requires 2 args".to_string());
                }
                let samp_name = match &args[0].ty {
                    Ty::Sampler(s) => s.clone(),
                    _ => args[0].code.clone(),
                };
                let b = self.samplers.resolve_sampler(&samp_name)?;
                let uvw = self.coerce_clone(&args[1], &Ty::Float(3))?;
                if b.is_3d {
                    Ok(TypedExpr {
                        code: format!("texture(sampler3D({}, {}), {})", b.tex_expr, b.samp_expr, uvw.code),
                        ty: Ty::Float(4),
                    })
                } else {
                    Ok(TypedExpr {
                        code: format!("texture(sampler2D({}, {}), ({}).xy)", b.tex_expr, b.samp_expr, uvw.code),
                        ty: Ty::Float(4),
                    })
                }
            }
            "mul" => self.build_mul(args),
            "lerp" => self.build_nary_float("mix", args, 3),
            "frac" => self.build_nary_float("fract", args, 1),
            "saturate" => {
                let a = args.into_iter().next().ok_or("saturate requires 1 arg")?;
                let ty = self.ensure_float_ty(&a.ty);
                let ca = self.coerce(a, &ty)?;
                Ok(TypedExpr {
                    code: format!("clamp({}, 0.0, 1.0)", ca.code),
                    ty,
                })
            }
            "rsqrt" => {
                let a = args.into_iter().next().ok_or("rsqrt requires 1 arg")?;
                let ty = self.ensure_float_ty(&a.ty);
                let ca = self.coerce(a, &ty)?;
                Ok(TypedExpr {
                    code: format!("inversesqrt(max({}, 1e-6))", ca.code),
                    ty,
                })
            }
            "atan2" => self.build_nary_float("atan", args, 2),
            "fmod" => {
                if args.len() != 2 {
                    return Err("fmod requires 2 args".to_string());
                }
                let ty = self.unify_float(&args[0].ty, &args[1].ty);
                let a = self.coerce_clone(&args[0], &ty)?;
                let b = self.coerce_clone(&args[1], &ty)?;
                let g = ty.to_glsl();
                Ok(TypedExpr {
                    code: format!("(({0}) - ({1}) * trunc(({0}) / (abs({1}) > {g}(1e-6) ? ({1}) : {g}(1e-6))))", a.code, b.code),
                    ty,
                })
            }
            "ddx" => self.build_nary_float("dFdx", args, 1),
            "ddy" => self.build_nary_float("dFdy", args, 1),
            "log10" => {
                let a = args.into_iter().next().ok_or("log10 requires 1 arg")?;
                let ty = self.ensure_float_ty(&a.ty);
                let ca = self.coerce(a, &ty)?;
                Ok(TypedExpr {
                    code: format!("(log(max({}, 1e-6)) * 0.4342944819)", ca.code),
                    ty,
                })
            }
            "pow" => {
                if args.len() != 2 {
                    return Err("pow requires 2 args".to_string());
                }
                let ty = self.unify_float(&args[0].ty, &args[1].ty);
                let a = self.coerce_clone(&args[0], &ty)?;
                let b = self.coerce_clone(&args[1], &ty)?;
                Ok(TypedExpr {
                    code: format!("pow(max({}, 0.0), {})", a.code, b.code),
                    ty,
                })
            }
            "clip" => {
                let a = args.into_iter().next().ok_or("clip requires 1 arg")?;
                let w = a.ty.width().unwrap_or(1);
                let f = self.coerce(a, &Ty::Float(w))?;
                if w == 1 {
                    Ok(TypedExpr {
                        code: format!("if ({} < 0.0) discard", f.code),
                        ty: Ty::Void,
                    })
                } else {
                    Ok(TypedExpr {
                        code: format!("if (any(lessThan({}, vec{w}(0.0)))) discard", f.code),
                        ty: Ty::Void,
                    })
                }
            }
            "dot" => {
                if args.len() != 2 {
                    return Err("dot requires 2 args".to_string());
                }
                let w = args[0].ty.width().unwrap_or(1).max(args[1].ty.width().unwrap_or(1));
                let target = Ty::Float(w);
                let a = self.coerce_clone(&args[0], &target)?;
                let b = self.coerce_clone(&args[1], &target)?;
                if w == 1 {
                    Ok(TypedExpr {
                        code: format!("({} * {})", a.code, b.code),
                        ty: Ty::Float(1),
                    })
                } else {
                    Ok(TypedExpr {
                        code: format!("dot({}, {})", a.code, b.code),
                        ty: Ty::Float(1),
                    })
                }
            }
            "cross" => self.build_fixed_vec("cross", args, 3),
            "length" | "normalize" => {
                let a = args.into_iter().next().ok_or_else(|| format!("{name} requires 1 arg"))?;
                let w = a.ty.width().unwrap_or(1);
                let target = Ty::Float(w);
                let ca = self.coerce(a, &target)?;
                let out_ty = if lower_name == "length" { Ty::Float(1) } else { target };
                if w == 1 {
                    Ok(TypedExpr {
                        code: if lower_name == "length" {
                            format!("abs({})", ca.code)
                        } else {
                            format!("sign({})", ca.code)
                        },
                        ty: Ty::Float(1),
                    })
                } else {
                    Ok(TypedExpr {
                        code: format!("{lower_name}({})", ca.code),
                        ty: out_ty,
                    })
                }
            }
            "distance" => {
                if args.len() != 2 {
                    return Err("distance requires 2 args".to_string());
                }
                let ty = self.unify_float(&args[0].ty, &args[1].ty);
                let a = self.coerce_clone(&args[0], &ty)?;
                let b = self.coerce_clone(&args[1], &ty)?;
                Ok(TypedExpr {
                    code: format!("distance({}, {})", a.code, b.code),
                    ty: Ty::Float(1),
                })
            }
            "any" | "all" => {
                let a = args.into_iter().next().ok_or_else(|| format!("{name} requires 1 arg"))?;
                let w = a.ty.width().unwrap_or(1);
                let cb = self.coerce(a, &Ty::Bool(w))?;
                if w == 1 {
                    Ok(cb)
                } else {
                    Ok(TypedExpr {
                        code: format!("{lower_name}({})", cb.code),
                        ty: Ty::Bool(1),
                    })
                }
            }
            "isfinite" => {
                let a = args.into_iter().next().ok_or("isfinite requires 1 arg")?;
                let ty = self.ensure_float_ty(&a.ty);
                let w = ty.width().unwrap_or(1);
                let ca = self.coerce(a, &ty)?;
                if w == 1 {
                    Ok(TypedExpr {
                        code: format!("(!isinf({0}) && !isnan({0}))", ca.code),
                        ty: Ty::Bool(1),
                    })
                } else {
                    Ok(TypedExpr {
                        code: format!("not(bvec{w}(isinf({0})) )", ca.code),
                        ty: Ty::Bool(w),
                    })
                }
            }
            "sincos" => {
                if args.len() != 3 {
                    return Err("sincos requires 3 args".to_string());
                }
                let ty = self.ensure_float_ty(&args[0].ty);
                let x = self.coerce_clone(&args[0], &ty)?;
                Ok(TypedExpr {
                    code: format!("({1} = sin({0}), {2} = cos({0}))", x.code, args[1].code, args[2].code),
                    ty: Ty::Void,
                })
            }
            "modf" => {
                if args.len() != 2 {
                    return Err("modf requires 2 args".to_string());
                }
                let ty = self.ensure_float_ty(&args[0].ty);
                let x = self.coerce_clone(&args[0], &ty)?;
                let ip = self.coerce_clone(&TypedExpr { code: format!("trunc({})", x.code), ty: ty.clone() }, &args[1].ty)?;
                Ok(TypedExpr {
                    code: format!("({1} = {2}, ({0} - trunc({0})))", x.code, args[1].code, ip.code),
                    ty,
                })
            }
            "atan" => {
                if args.len() == 2 {
                    self.build_nary_float("atan", args, 2)
                } else {
                    self.build_nary_float("atan", args, 1)
                }
            }
            "sin" | "cos" | "tan" | "asin" | "acos" | "sinh" | "cosh" | "tanh"
            | "exp" | "exp2" | "log" | "log2" | "sqrt" | "abs" | "sign"
            | "floor" | "ceil" | "round" | "trunc" | "radians" | "degrees" | "isnan" | "isinf" => {
                let n = args.len();
                let mut res = self.build_nary_float(&lower_name, args, n.max(1))?;
                if matches!(lower_name.as_str(), "isnan" | "isinf") {
                    let w = res.ty.width().unwrap_or(1);
                    res.ty = Ty::Bool(w);
                }
                Ok(res)
            }
            "min" | "max" | "step" | "reflect" | "mod" => self.build_nary_float(&lower_name, args, 2),
            "clamp" | "smoothstep" | "refract" | "mix" => self.build_nary_float(&lower_name, args, 3),
            "transpose" => {
                let a = args.into_iter().next().ok_or("transpose requires 1 arg")?;
                let out_ty = match a.ty {
                    Ty::Mat(r, c) => Ty::Mat(c, r),
                    other => other,
                };
                Ok(TypedExpr {
                    code: format!("transpose({})", a.code),
                    ty: out_ty,
                })
            }
            "determinant" => {
                let a = args.into_iter().next().ok_or("determinant requires 1 arg")?;
                Ok(TypedExpr {
                    code: format!("determinant({})", a.code),
                    ty: Ty::Float(1),
                })
            }
            _ => {
                // Unknown helper function - pass through conservatively
                let cargs: Vec<String> = args.into_iter().map(|a| a.code).collect();
                Ok(TypedExpr {
                    code: format!("{name}({})", cargs.join(", ")),
                    ty: Ty::Float(1),
                })
            }
        }
    }

    fn build_type_ctor(&mut self, target: &Ty, args: Vec<TypedExpr>) -> Result<TypedExpr, String> {
        if let Ty::Mat(r, c) = *target {
            if args.len() == 1 {
                if let Ty::Float(4) = args[0].ty {
                    if r == 2 && c == 2 {
                        return Ok(TypedExpr {
                            code: format!("mat2(({0}).xy, ({0}).zw)", args[0].code),
                            ty: Ty::Mat(2, 2),
                        });
                    }
                }
                if args[0].ty == *target {
                    return Ok(args.into_iter().next().unwrap());
                }
            }
            let cargs: Vec<String> = args.into_iter().map(|a| a.code).collect();
            return Ok(TypedExpr {
                code: format!("{}({})", target.to_glsl(), cargs.join(", ")),
                ty: target.clone(),
            });
        }

        let target_w = target.width().unwrap_or(1);
        let elem_ty = target.with_width(1);
        if args.len() == 1 && args[0].ty.width() == Some(target_w) {
            return self.coerce(args.into_iter().next().unwrap(), target);
        }
        if args.len() == 1 && args[0].ty.is_scalar() {
            let ca = self.coerce(args.into_iter().next().unwrap(), &elem_ty)?;
            return Ok(TypedExpr {
                code: if target_w == 1 {
                    ca.code
                } else {
                    format!("{}({})", target.to_glsl(), ca.code)
                },
                ty: target.clone(),
            });
        }

        // Total component count check: if single wider vector, truncate
        if args.len() == 1 && args[0].ty.width().unwrap_or(1) > target_w {
            return self.coerce(args.into_iter().next().unwrap(), target);
        }

        let mut cargs = Vec::with_capacity(args.len());
        for a in args {
            let aw = a.ty.width().unwrap_or(1);
            let dst = target.with_width(aw);
            cargs.push(self.coerce(a, &dst)?.code);
        }
        Ok(TypedExpr {
            code: format!("{}({})", target.to_glsl(), cargs.join(", ")),
            ty: target.clone(),
        })
    }

    fn build_mul(&mut self, args: Vec<TypedExpr>) -> Result<TypedExpr, String> {
        if args.len() != 2 {
            return Err("mul requires 2 args".to_string());
        }
        let a = &args[0];
        let b = &args[1];
        // HLSL mul(row_vec, M) = GLSL (M * vec) under our row-packed matrix convention
        match (&a.ty, &b.ty) {
            (Ty::Float(v), Ty::Mat(r, c)) | (Ty::Int(v), Ty::Mat(r, c)) => {
                if *r == 4 && *c == 3 && *v == 3 {
                    // Common MilkDrop idiom: mul(float3, float4x3) -> append 1.0 or 3x3
                    let v3 = self.coerce_clone(a, &Ty::Float(3))?;
                    Ok(TypedExpr {
                        code: format!("({} * vec4({}, 1.0))", b.code, v3.code),
                        ty: Ty::Float(3),
                    })
                } else {
                    let va = self.coerce_clone(a, &Ty::Float(*r))?;
                    Ok(TypedExpr {
                        code: format!("({} * {})", b.code, va.code),
                        ty: Ty::Float(*c),
                    })
                }
            }
            (Ty::Mat(r, c), Ty::Float(_)) | (Ty::Mat(r, c), Ty::Int(_)) => {
                let vb = self.coerce_clone(b, &Ty::Float(*c))?;
                Ok(TypedExpr {
                    code: format!("({} * {})", vb.code, a.code),
                    ty: Ty::Float(*r),
                })
            }
            (Ty::Mat(r1, _c1), Ty::Mat(_, c2)) => Ok(TypedExpr {
                code: format!("({} * {})", b.code, a.code),
                ty: Ty::Mat(*r1, *c2),
            }),
            _ => self.bin_arith_op("*", args[0].clone_expr(), args[1].clone_expr()),
        }
    }

    fn build_fixed_vec(&mut self, fn_name: &str, args: Vec<TypedExpr>, w: u8) -> Result<TypedExpr, String> {
        let target = Ty::Float(w);
        let mut cargs = Vec::with_capacity(args.len());
        for a in args {
            cargs.push(self.coerce(a, &target)?.code);
        }
        Ok(TypedExpr {
            code: format!("{fn_name}({})", cargs.join(", ")),
            ty: target,
        })
    }

    fn build_nary_float(&mut self, fn_name: &str, args: Vec<TypedExpr>, expected: usize) -> Result<TypedExpr, String> {
        if args.len() != expected {
            return Err(format!("{fn_name} expects {expected} args, got {}", args.len()));
        }
        let mut ty = Ty::Float(1);
        for a in &args {
            ty = self.unify_float(&ty, &a.ty);
        }
        let mut cargs = Vec::with_capacity(args.len());
        for a in args {
            cargs.push(self.coerce(a, &ty)?.code);
        }
        Ok(TypedExpr {
            code: format!("{fn_name}({})", cargs.join(", ")),
            ty,
        })
    }

    fn coerce_arg(&mut self, args: &[TypedExpr], idx: usize, target: &Ty) -> Result<TypedExpr, String> {
        let a = args.get(idx).ok_or_else(|| format!("Missing arg {idx}"))?;
        self.coerce_clone(a, target)
    }

    fn coerce_clone(&mut self, expr: &TypedExpr, target: &Ty) -> Result<TypedExpr, String> {
        self.coerce(expr.clone_expr(), target)
    }

    fn ensure_float_ty(&self, ty: &Ty) -> Ty {
        match *ty {
            Ty::Float(w) => Ty::Float(w),
            Ty::Int(w) | Ty::Bool(w) => Ty::Float(w),
            Ty::Mat(r, c) => Ty::Mat(r, c),
            _ => Ty::Float(1),
        }
    }

    fn unify_float(&self, a: &Ty, b: &Ty) -> Ty {
        let aw = a.width().unwrap_or(1);
        let bw = b.width().unwrap_or(1);
        let w = if aw == 1 {
            bw
        } else if bw == 1 {
            aw
        } else {
            aw.min(bw)
        };
        Ty::Float(w)
    }

    fn to_bool_scalar(&mut self, e: TypedExpr) -> Result<TypedExpr, String> {
        match e.ty {
            Ty::Bool(1) => Ok(e),
            Ty::Float(1) => Ok(TypedExpr {
                code: format!("({} != 0.0)", e.code),
                ty: Ty::Bool(1),
            }),
            Ty::Int(1) => Ok(TypedExpr {
                code: format!("({} != 0)", e.code),
                ty: Ty::Bool(1),
            }),
            Ty::Bool(_w) => Ok(TypedExpr {
                code: format!("any({})", e.code),
                ty: Ty::Bool(1),
            }),
            Ty::Float(w) => Ok(TypedExpr {
                code: format!("any(notEqual({}, vec{w}(0.0)))", e.code),
                ty: Ty::Bool(1),
            }),
            Ty::Int(w) => Ok(TypedExpr {
                code: format!("any(notEqual({}, ivec{w}(0)))", e.code),
                ty: Ty::Bool(1),
            }),
            _ => Ok(TypedExpr {
                code: format!("(({}) != 0.0)", e.code),
                ty: Ty::Bool(1),
            }),
        }
    }

    fn bin_arith_op(&mut self, op: &str, lhs: TypedExpr, rhs: TypedExpr) -> Result<TypedExpr, String> {
        // Matrix * scalar or scalar * Matrix
        if matches!(lhs.ty, Ty::Mat(_, _)) || matches!(rhs.ty, Ty::Mat(_, _)) {
            if matches!(lhs.ty, Ty::Mat(_, _)) && rhs.ty.is_scalar() {
                let s = self.coerce(rhs, &Ty::Float(1))?;
                return Ok(TypedExpr {
                    code: format!("({} {op} {})", lhs.code, s.code),
                    ty: lhs.ty,
                });
            }
            if lhs.ty.is_scalar() && matches!(rhs.ty, Ty::Mat(_, _)) {
                let s = self.coerce(lhs, &Ty::Float(1))?;
                return Ok(TypedExpr {
                    code: format!("({} {op} {})", s.code, rhs.code),
                    ty: rhs.ty,
                });
            }
            if lhs.ty == rhs.ty {
                return Ok(TypedExpr {
                    code: format!("({} {op} {})", lhs.code, rhs.code),
                    ty: lhs.ty,
                });
            }
        }

        let lw = lhs.ty.width().unwrap_or(1);
        let rw = rhs.ty.width().unwrap_or(1);
        let both_int = matches!(lhs.ty, Ty::Int(_)) && matches!(rhs.ty, Ty::Int(_));
        let out_w = if lw == 1 {
            rw
        } else if rw == 1 {
            lw
        } else {
            lw.min(rw)
        };

        // In HLSL, float2 + float3 truncates; in GLSL scalar+vec works natively if types match base
        let base_is_int = both_int && op != "/";
        let target_l = if base_is_int {
            Ty::Int(if lw == 1 { 1 } else { out_w })
        } else {
            Ty::Float(if lw == 1 { 1 } else { out_w })
        };
        let target_r = if base_is_int {
            Ty::Int(if rw == 1 { 1 } else { out_w })
        } else {
            Ty::Float(if rw == 1 { 1 } else { out_w })
        };
        let out_ty = if base_is_int { Ty::Int(out_w) } else { Ty::Float(out_w) };

        let cl = self.coerce(lhs, &target_l)?;
        let cr = self.coerce(rhs, &target_r)?;

        if op == "%" {
            if base_is_int {
                return Ok(TypedExpr {
                    code: format!("({} % {})", cl.code, cr.code),
                    ty: Ty::Int(out_w),
                });
            }
            return Ok(TypedExpr {
                code: format!("mod({}, {})", cl.code, cr.code),
                ty: Ty::Float(out_w),
            });
        }
        Ok(TypedExpr {
            code: format!("({} {op} {})", cl.code, cr.code),
            ty: out_ty,
        })
    }

    fn bin_int_op(&mut self, op: &str, lhs: TypedExpr, rhs: TypedExpr) -> Result<TypedExpr, String> {
        let lw = lhs.ty.width().unwrap_or(1);
        let rw = rhs.ty.width().unwrap_or(1);
        let w = if lw == 1 { rw } else if rw == 1 { lw } else { lw.min(rw) };
        let cl = self.coerce(lhs, &Ty::Int(if lw == 1 { 1 } else { w }))?;
        let cr = self.coerce(rhs, &Ty::Int(if rw == 1 { 1 } else { w }))?;
        Ok(TypedExpr {
            code: format!("({} {op} {})", cl.code, cr.code),
            ty: Ty::Int(w),
        })
    }

    fn cmp_op(&mut self, scalar_op: &str, vec_fn: &str, lhs: TypedExpr, rhs: TypedExpr) -> Result<TypedExpr, String> {
        let lw = lhs.ty.width().unwrap_or(1);
        let rw = rhs.ty.width().unwrap_or(1);
        let w = if lw == 1 { rw } else if rw == 1 { lw } else { lw.min(rw) };
        let use_int = matches!(lhs.ty, Ty::Int(_)) && matches!(rhs.ty, Ty::Int(_));
        let elem = if use_int { Ty::Int(w) } else { Ty::Float(w) };
        let cl = self.coerce(lhs, &elem)?;
        let cr = self.coerce(rhs, &elem)?;
        if w == 1 {
            Ok(TypedExpr {
                code: format!("({} {scalar_op} {})", cl.code, cr.code),
                ty: Ty::Bool(1),
            })
        } else {
            Ok(TypedExpr {
                code: format!("{vec_fn}({}, {})", cl.code, cr.code),
                ty: Ty::Bool(w),
            })
        }
    }

    fn coerce(&mut self, expr: TypedExpr, target: &Ty) -> Result<TypedExpr, String> {
        if &expr.ty == target || matches!(target, Ty::Unknown) {
            return Ok(expr);
        }
        // Matrix to scalar/vector truncation
        if let Ty::Mat(_, _) = expr.ty {
            match target {
                Ty::Float(1) => {
                    return Ok(TypedExpr {
                        code: format!("({})[0].x", expr.code),
                        ty: Ty::Float(1),
                    })
                }
                Ty::Float(2) => {
                    return Ok(TypedExpr {
                        code: format!("({})[0].xy", expr.code),
                        ty: Ty::Float(2),
                    })
                }
                Ty::Float(3) => {
                    return Ok(TypedExpr {
                        code: format!("({})[0].xyz", expr.code),
                        ty: Ty::Float(3),
                    })
                }
                _ => return Ok(expr),
            }
        }
        let Some(sw) = expr.ty.width() else {
            return Ok(expr);
        };
        let Some(tw) = target.width() else {
            return Ok(expr);
        };

        let same_base = matches!(
            (&expr.ty, target),
            (Ty::Float(_), Ty::Float(_)) | (Ty::Int(_), Ty::Int(_)) | (Ty::Bool(_), Ty::Bool(_))
        );

        if sw == tw && same_base {
            return Ok(expr);
        }

        // Step 1: adjust width
        let width_adjusted = if sw == tw {
            expr.code
        } else if sw == 1 {
            // Scalar -> VecN: cast scalar first to target base then splat
            let scalar_target = target.with_width(1);
            let scalar_cast = if same_base {
                expr.code
            } else {
                format!("{}({})", scalar_target.to_glsl(), expr.code)
            };
            return Ok(TypedExpr {
                code: format!("{}({scalar_cast})", target.to_glsl()),
                ty: target.clone(),
            });
        } else if sw > tw {
            let swiz = match tw {
                1 => "x",
                2 => "xy",
                3 => "xyz",
                _ => "xyzw",
            };
            format!("({}).{swiz}", expr.code)
        } else {
            // sw < tw: pad with 0s
            let mut pad = vec![expr.code];
            for _ in sw..tw {
                pad.push(match target {
                    Ty::Float(_) => "0.0".to_string(),
                    Ty::Int(_) => "0".to_string(),
                    Ty::Bool(_) => "false".to_string(),
                    _ => "0.0".to_string(),
                });
            }
            let same_base_src = expr.ty.with_width(tw);
            let vec_str = format!("{}({})", same_base_src.to_glsl(), pad.join(", "));
            if same_base {
                return Ok(TypedExpr {
                    code: vec_str,
                    ty: target.clone(),
                });
            }
            vec_str
        };

        if same_base {
            Ok(TypedExpr {
                code: width_adjusted,
                ty: target.clone(),
            })
        } else {
            Ok(TypedExpr {
                code: format!("{}({width_adjusted})", target.to_glsl()),
                ty: target.clone(),
            })
        }
    }
}

impl TypedExpr {
    fn clone_expr(&self) -> Self {
        Self {
            code: self.code.clone(),
            ty: self.ty.clone(),
        }
    }
}

fn unify_types(a: &Ty, b: &Ty) -> Ty {
    if a == b {
        return a.clone();
    }
    let aw = a.width().unwrap_or(1);
    let bw = b.width().unwrap_or(1);
    let w = if aw == 1 {
        bw
    } else if bw == 1 {
        aw
    } else {
        aw.min(bw)
    };
    if matches!(a, Ty::Float(_)) || matches!(b, Ty::Float(_)) {
        Ty::Float(w)
    } else if matches!(a, Ty::Int(_)) || matches!(b, Ty::Int(_)) {
        Ty::Int(w)
    } else {
        Ty::Bool(w)
    }
}

fn swizzle_len(s: &str) -> Option<u8> {
    if s.is_empty() || s.len() > 4 {
        return None;
    }
    let xyzw = s.bytes().all(|b| matches!(b, b'x' | b'y' | b'z' | b'w'));
    let rgba = s.bytes().all(|b| matches!(b, b'r' | b'g' | b'b' | b'a'));
    if xyzw || rgba {
        Some(s.len() as u8)
    } else {
        None
    }
}

fn normalize_swizzle(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'r' => 'x',
            b'g' => 'y',
            b'b' => 'z',
            b'a' => 'w',
            other => other as char,
        })
        .collect()
}

pub fn translate(src: &str, kind: ShaderKind) -> Result<TranslatedShader, String> {
    let pre = preprocess(src);
    let toks = tokenize(&pre)?;
    let mut tr = Translator::new(&toks);
    let (globals_glsl, body_glsl) = tr.translate_unit()?;
    let _ = tr.used_q;

    let mut glsl = String::with_capacity(4096 + body_glsl.len());
    glsl.push_str(
        "#version 450

layout(set = 0, binding = 0, std140) uniform MdUniforms {
    vec4 timing;
    vec4 audio;
    vec4 audio_att;
    vec4 q_vecs[8];
    vec4 aspect;
    vec4 texsize;
    vec4 rand_preset;
    vec4 rand_frame;
    vec4 roam_cos;
    vec4 roam_sin;
    vec4 slow_roam_cos;
    vec4 slow_roam_sin;
    vec4 blur_min_mip;
    vec4 blur_max_mip;
    vec4 mip_extra;
    vec4 user_texsize[8];
    vec4 noise_lq_size;
    vec4 noise_lq_lite_size;
    vec4 noise_mq_size;
    vec4 noise_hq_size;
    vec4 noisevol_lq_size;
    vec4 noisevol_hq_size;
    mat4x3 rot_s[4];
    mat4x3 rot_d[4];
    mat4x3 rot_f[4];
    mat4x3 rot_vf[4];
    mat4x3 rot_uf[4];
    mat4x3 rot_rand[4];
} u_;

layout(set = 0, binding = 1) uniform texture2D tex_main;
layout(set = 0, binding = 2) uniform texture2D tex_blur1;
layout(set = 0, binding = 3) uniform texture2D tex_blur2;
layout(set = 0, binding = 4) uniform texture2D tex_blur3;
layout(set = 0, binding = 5) uniform texture2D tex_noise_lq;
layout(set = 0, binding = 6) uniform texture2D tex_noise_lq_lite;
layout(set = 0, binding = 7) uniform texture2D tex_noise_mq;
layout(set = 0, binding = 8) uniform texture2D tex_noise_hq;
layout(set = 0, binding = 9) uniform texture3D tex_noisevol_lq;
layout(set = 0, binding = 10) uniform texture3D tex_noisevol_hq;
layout(set = 0, binding = 11) uniform texture2D tex_user0;
layout(set = 0, binding = 12) uniform texture2D tex_user1;
layout(set = 0, binding = 13) uniform texture2D tex_user2;
layout(set = 0, binding = 14) uniform texture2D tex_user3;
layout(set = 0, binding = 15) uniform texture2D tex_user4;
layout(set = 0, binding = 16) uniform texture2D tex_user5;
layout(set = 0, binding = 17) uniform texture2D tex_user6;
layout(set = 0, binding = 18) uniform texture2D tex_user7;

layout(set = 0, binding = 20) uniform sampler samp_fw;
layout(set = 0, binding = 21) uniform sampler samp_fc;
layout(set = 0, binding = 22) uniform sampler samp_pw;
layout(set = 0, binding = 23) uniform sampler samp_pc;
layout(set = 0, binding = 24) uniform sampler samp_main;

layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec2 v_uv_orig;
layout(location = 2) in vec2 v_rad_ang;
layout(location = 3) in vec3 v_hue_shader;

layout(location = 0) out vec4 o_color;

vec3 _md_get_pixel(vec2 uv_p) {
    return texture(sampler2D(tex_main, samp_main), uv_p).rgb;
}
vec3 _md_get_blur1(vec2 uv_p) {
    return u_.blur_min_mip.x + texture(sampler2D(tex_blur1, samp_fc), uv_p).rgb * (u_.blur_max_mip.x - u_.blur_min_mip.x);
}
vec3 _md_get_blur2(vec2 uv_p) {
    return u_.blur_min_mip.y + texture(sampler2D(tex_blur2, samp_fc), uv_p).rgb * (u_.blur_max_mip.y - u_.blur_min_mip.y);
}
vec3 _md_get_blur3(vec2 uv_p) {
    return u_.blur_min_mip.z + texture(sampler2D(tex_blur3, samp_fc), uv_p).rgb * (u_.blur_max_mip.z - u_.blur_min_mip.z);
}

",
    );

    for grp in &tr.mutated_groups {
        if *grp == "q_vecs" {
            glsl.push_str("vec4 _mw_q_vecs[8];\n");
        } else {
            glsl.push_str(&format!("vec4 _mw_{grp};\n"));
        }
    }
    glsl.push_str(&globals_glsl);

    let _ = kind;
    glsl.push_str(
        "void main() {
  vec2 uv = v_uv;
  vec2 uv_orig = v_uv_orig;
  float rad = v_rad_ang.x;
  float ang = v_rad_ang.y;
  vec3 hue_shader = v_hue_shader;
  vec3 ret = vec3(0.0);
",
    );
    for grp in &tr.mutated_groups {
        if *grp == "q_vecs" {
            glsl.push_str("  for (int _qi = 0; _qi < 8; _qi++) { _mw_q_vecs[_qi] = u_.q_vecs[_qi]; }\n");
        } else {
            glsl.push_str(&format!("  _mw_{grp} = u_.{grp};\n"));
        }
    }
    glsl.push_str(&body_glsl);
    glsl.push_str("  o_color = vec4(ret, 1.0);\n}\n");

    Ok(TranslatedShader {
        glsl,
        user_textures: tr.samplers.user_textures,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_and_translates_typical_warp_and_comp() {
        let kv = vec![
            ("warp_1", "`shader_body {"),
            ("warp_2", "`float corr = texsize.xy * texsize_noise_lq.zw;"),
            ("warp_3", "`float3 noise = tex2D(sampler_noise_lq, corr*uv*0.5 + q1) + 1;"),
            ("warp_4", "`float3 crisp = tex2D(sampler_main, uv + noise.xy * 0.01);"),
            ("warp_5", "`ret = lerp(GetBlur1(uv), crisp, 0.5) * 0.98;"),
            ("warp_6", "`}"),
        ];
        let src = extract_shader_source(&kv, ShaderKind::Warp).unwrap();
        let tr = translate(&src, ShaderKind::Warp).unwrap();
        assert!(tr.glsl.contains("o_color = vec4(ret, 1.0);"));
        assert!(tr.glsl.contains("_md_get_blur1"));

        // Verify with wgpu::naga GLSL parser!
        let mut fe = wgpu::naga::front::glsl::Frontend::default();
        let opts = wgpu::naga::front::glsl::Options::from(wgpu::naga::ShaderStage::Fragment);
        let module = fe.parse(&opts, &tr.glsl).expect("naga glsl parse");
        let mut val = wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        );
        val.validate(&module).expect("naga validate");
    }

    #[test]
    fn translates_user_texture_and_matrix_mul() {
        let src = r#"
sampler sampler_rose;
shader_body {
    float2x2 rot = { cos(q1), -sin(q1), sin(q1), cos(q1) };
    float2 uvm = mul(uv - 0.5, rot) + 0.5;
    ret = tex2D(sampler_rose, uvm).rgb;
    float3 b = GetBlur2(uv_orig);
    ret = saturate(ret + b * 0.3);
}
"#;
        let tr = translate(src, ShaderKind::Comp).unwrap();
        assert_eq!(tr.user_textures.len(), 1);
        assert_eq!(tr.user_textures[0].name, "rose");
        let mut fe = wgpu::naga::front::glsl::Frontend::default();
        let opts = wgpu::naga::front::glsl::Options::from(wgpu::naga::ShaderStage::Fragment);
        let module = fe.parse(&opts, &tr.glsl).expect("naga glsl parse");
        let mut val = wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        );
        val.validate(&module).expect("naga validate");
    }
}
