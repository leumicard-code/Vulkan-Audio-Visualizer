// ============================================================================
// theme.rs — Studio Color Palette & Skin System for FLTK GUI
// ============================================================================

use fltk::enums::Color;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rgb(pub u8, pub u8, pub u8);

impl Rgb {
    pub fn to_fltk(self) -> Color {
        Color::from_rgb(self.0, self.1, self.2)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThemeMode {
    #[default]
    StudioObsidian,
    CyberpunkNeon,
    WarmAnalog,
    MatrixEmerald,
}

pub const ALL_THEMES: [ThemeMode; 4] = [
    ThemeMode::StudioObsidian,
    ThemeMode::CyberpunkNeon,
    ThemeMode::WarmAnalog,
    ThemeMode::MatrixEmerald,
];

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct Theme {
    pub mode: ThemeMode,
    pub name: &'static str,
    pub panel_bg: Rgb,
    pub card_bg: Rgb,
    pub toolbar_bg: Rgb,
    pub btn_bg: Rgb,
    pub border: Rgb,
    pub accent: Rgb,
    pub accent_alt: Rgb,
    pub accent_fg: Rgb,
    pub text_main: Rgb,
    pub text_muted: Rgb,
    pub danger: Rgb,
    pub success: Rgb,
}

impl Theme {
    pub fn get(mode: ThemeMode) -> Self {
        match mode {
            ThemeMode::StudioObsidian => Self {
                mode,
                name: "Studio Obsidian",
                panel_bg: Rgb(14, 18, 28),
                card_bg: Rgb(22, 28, 42),
                toolbar_bg: Rgb(11, 14, 22),
                btn_bg: Rgb(34, 44, 66),
                border: Rgb(48, 62, 92),
                accent: Rgb(56, 189, 248),
                accent_alt: Rgb(168, 85, 247),
                accent_fg: Rgb(10, 14, 22),
                text_main: Rgb(241, 245, 249),
                text_muted: Rgb(148, 163, 184),
                danger: Rgb(244, 63, 94),
                success: Rgb(34, 197, 94),
            },
            ThemeMode::CyberpunkNeon => Self {
                mode,
                name: "Cyberpunk Neon",
                panel_bg: Rgb(10, 10, 18),
                card_bg: Rgb(18, 16, 34),
                toolbar_bg: Rgb(7, 7, 14),
                btn_bg: Rgb(32, 24, 56),
                border: Rgb(68, 44, 112),
                accent: Rgb(0, 242, 255),
                accent_alt: Rgb(255, 42, 165),
                accent_fg: Rgb(8, 8, 14),
                text_main: Rgb(248, 250, 255),
                text_muted: Rgb(192, 132, 252),
                danger: Rgb(255, 56, 100),
                success: Rgb(0, 255, 170),
            },
            ThemeMode::WarmAnalog => Self {
                mode,
                name: "Warm Analog",
                panel_bg: Rgb(24, 19, 16),
                card_bg: Rgb(35, 28, 23),
                toolbar_bg: Rgb(18, 14, 12),
                btn_bg: Rgb(54, 42, 34),
                border: Rgb(82, 64, 50),
                accent: Rgb(245, 158, 11),
                accent_alt: Rgb(239, 68, 68),
                accent_fg: Rgb(22, 16, 10),
                text_main: Rgb(254, 243, 199),
                text_muted: Rgb(196, 160, 126),
                danger: Rgb(239, 68, 68),
                success: Rgb(132, 204, 22),
            },
            ThemeMode::MatrixEmerald => Self {
                mode,
                name: "Matrix Emerald",
                panel_bg: Rgb(9, 16, 12),
                card_bg: Rgb(15, 26, 20),
                toolbar_bg: Rgb(7, 12, 9),
                btn_bg: Rgb(24, 42, 32),
                border: Rgb(38, 72, 54),
                accent: Rgb(16, 185, 129),
                accent_alt: Rgb(52, 211, 153),
                accent_fg: Rgb(7, 14, 10),
                text_main: Rgb(220, 252, 231),
                text_muted: Rgb(110, 231, 183),
                danger: Rgb(248, 113, 113),
                success: Rgb(74, 222, 128),
            },
        }
    }
}
