// ============================================================================
// main.rs — Vulkan System-Audio Visualizer (PipeWire/PulseAudio + wgpu + FLTK)
// ============================================================================

mod audio;
mod config;
mod dsp_fft;
mod milkdrop;
mod overlay;
mod presets;
mod theme;
mod visualizer;

use std::cell::RefCell;
use std::rc::Rc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use fltk::{
    app,
    browser::HoldBrowser,
    button::Button,
    dialog,
    enums::{Align, CallbackTrigger, Color, Event, EventState, FrameType, Key, Shortcut},
    frame::Frame,
    image::PngImage,
    input::Input,
    menu::{Choice, MenuFlag, SysMenuBar},
    prelude::*,
    window::Window,
};
use fltk_theme::{ThemeType, WidgetTheme};

use crate::audio::{list_monitor_sources, SystemAudioTap};
use crate::config::UserConfig;
use crate::dsp_fft::{new_shared_pcm_ring, FftAnalyzer, SharedPcmRing};
use crate::presets::{PresetFilter, PresetKind, PresetManager, ShaderPreset};
use crate::theme::{Theme, ALL_THEMES};
use crate::visualizer::VisualizerEngine;

const MENU_BAR_H: i32 = 28;
const WIN_MIN_W: i32 = 640;
const WIN_MIN_H: i32 = 420;

const VULKAN_ICON_PNG: &[u8] = include_bytes!("../assets/vulkan_icon_128.png");
const VULKAN_ABOUT_PNG: &[u8] = include_bytes!("../assets/vulkan_logo_about.png");

fn sanitize_menu_segment(text: &str) -> String {
    text.replace(['/', '\\', '&', '_'], " ")
}

fn sanitize_choice_label(text: &str) -> String {
    text.replace('/', " > ").replace(['\\', '&', '|'], " ")
}

fn load_vulkan_icon() -> Option<PngImage> {
    PngImage::from_data(VULKAN_ICON_PNG).ok()
}

struct AppState {
    config: UserConfig,
    theme: Theme,
    pcm_ring: SharedPcmRing,
    audio_tap: SystemAudioTap,
    audio_paused: bool,
    fft: FftAnalyzer,
    presets: PresetManager,
    viz: Option<VisualizerEngine>,

    is_fullscreen: bool,
    start_time: Instant,
    last_frame: Instant,
    last_auto_cycle: Instant,
    fps_smooth: f32,
    rng_seq: u64,

    // Core FLTK widgets (100% visualization canvas below top menu bar)
    main_win: Window,
    menu_bar: SysMenuBar,
    viz_win: Window,

    // Optional singleton floating windows
    browser_win: Option<Window>,
    browser_refresh: Option<Rc<dyn Fn()>>,
    about_win: Option<Window>,
}

type SharedState = Rc<RefCell<AppState>>;

fn main() {
    // Wayland crash fix: FLTK commits its own buffers on the child wl_surface that
    // Mesa's Vulkan WSI drives with explicit sync (wp_linux_drm_syncobj). KWin then
    // kills the client with "explicit sync is used, but no acquire point is set".
    // Running FLTK on XWayland gives Vulkan its own X11 surface and avoids the conflict.
    // Override with FLTK_BACKEND=wayland if you want to test native Wayland.
    if std::env::var_os("FLTK_BACKEND").is_none() {
        std::env::set_var("FLTK_BACKEND", "x11");
    }

    let app = app::App::default().with_scheme(app::Scheme::Gtk);
    WidgetTheme::new(ThemeType::Dark).apply();
    app::set_font_size(13);

    let config = UserConfig::load();
    let theme = Theme::get(config.theme);
    let pcm_ring = new_shared_pcm_ring();
    let audio_tap = SystemAudioTap::start(pcm_ring.clone(), config.monitor_source.as_deref());
    let presets = PresetManager::with_milkdrop_dirs(
        Some(&config.active_shader_id),
        &config.milkdrop_dirs,
    );

    let init_w = 1280;
    let init_h = 800;
    let viz_h = init_h - MENU_BAR_H;

    let mut main_win = Window::default()
        .with_size(init_w, init_h)
        .with_label("Vulkan Audio Visualizer");
    main_win.make_resizable(true);
    main_win.size_range(WIN_MIN_W, WIN_MIN_H, 0, 0);
    if let Some(icon) = load_vulkan_icon() {
        main_win.set_icon(Some(icon));
    }

    let mut menu_bar = SysMenuBar::new(0, 0, init_w, MENU_BAR_H, None);
    menu_bar.set_frame(FrameType::FlatBox);
    menu_bar.set_text_size(13);

    // Embedded native child window holding the wgpu Vulkan surface (fills all space below menu bar)
    let mut viz_win = Window::new(0, MENU_BAR_H, init_w, viz_h, None);
    viz_win.set_frame(FrameType::NoBox);
    viz_win.draw(|_| {});
    viz_win.end();

    main_win.end();

    main_win.show();
    viz_win.show();
    app::flush();

    // Always bootstrap the WGSL pipeline with a valid built-in shader, then activate saved preset.
    let init_wgsl = presets
        .presets()
        .iter()
        .find(|p| p.kind == PresetKind::Wgsl)
        .map(|p| (p.source.clone(), p.name.clone()))
        .unwrap_or_else(|| (String::new(), "Default".to_string()));

    let viz = match unsafe {
        VisualizerEngine::new(
            &viz_win,
            init_w as u32,
            viz_h as u32,
            &init_wgsl.0,
            &init_wgsl.1,
            presets.texture_dirs(),
            config.show_overlay,
        )
    } {
        Ok(engine) => Some(engine),
        Err(err) => {
            eprintln!("Vulkan initialization error: {err}");
            None
        }
    };

    let now = Instant::now();
    let state = Rc::new(RefCell::new(AppState {
        config,
        theme,
        pcm_ring,
        audio_tap,
        audio_paused: false,
        fft: FftAnalyzer::new(),
        presets,
        viz,
        is_fullscreen: false,
        start_time: now,
        last_frame: now,
        last_auto_cycle: now,
        fps_smooth: 60.0,
        rng_seq: 1,
        main_win,
        menu_bar,
        viz_win,
        browser_win: None,
        browser_refresh: None,
        about_win: None,
    }));

    {
        let mut st = state.borrow_mut();
        let w = st.main_win.w();
        let h = st.main_win.h();
        apply_layout(&mut st, w, h);
        apply_theme(&mut st);
        let initial_idx = st.presets.active_idx();
        activate_preset(&mut st, initial_idx);
    }
    rebuild_menu_bar(&state);

    wire_events(&state);

    // 60 FPS animation + FFT + hot-reload timer loop
    let state_tick = state.clone();
    app::add_timeout3(1.0 / 60.0, move |handle| {
        if let Ok(mut st) = state_tick.try_borrow_mut() {
            let now = Instant::now();
            let dt = (now - st.last_frame).as_secs_f32().clamp(0.001, 0.1);
            st.last_frame = now;
            let elapsed = (now - st.start_time).as_secs_f32();
            st.fps_smooth = st.fps_smooth * 0.9 + (1.0 / dt) * 0.1;

            if st.presets.poll_hot_reload(false) {
                let idx = st.presets.active_idx();
                let total = st.presets.presets().len();
                let p = st.presets.active_preset().clone();
                if let Some(viz) = st.viz.as_mut() {
                    load_preset_into_viz(viz, &p, idx, total, elapsed);
                }
            }

            let auto_sec = st.config.auto_cycle_sec;
            if auto_sec > 0
                && now.duration_since(st.last_auto_cycle) >= Duration::from_secs(auto_sec as u64)
            {
                st.last_auto_cycle = now;
                select_next_preset(&mut st);
            }

            let ring = st.pcm_ring.clone();
            let is_capturing = !st.audio_paused && st.audio_tap.last_error().is_none();
            let features = st.fft.update(&ring, dt, is_capturing).clone();

            let vw = st.viz_win.w().max(1) as u32;
            let vh = st.viz_win.h().max(1) as u32;
            if let Some(viz) = st.viz.as_mut() {
                viz.resize(vw, vh);
                viz.render(elapsed, dt, &features);
            }
        }
        app::repeat_timeout3(1.0 / 60.0, handle);
    });

    app.run().unwrap();
}

enum DeferredUiAction {
    OpenBrowser,
    OpenAbout,
    ImportFolder,
    RescanAll,
    QuitApp,
}

fn close_all_and_quit(state: &SharedState) {
    if let Ok(mut st) = state.try_borrow_mut() {
        st.audio_tap.stop();
        st.config.save();
        st.viz = None;
        if let Some(mut bw) = st.browser_win.take() {
            bw.hide();
        }
        if let Some(mut aw) = st.about_win.take() {
            aw.hide();
        }
    }
    app::quit();
}

fn wire_events(state: &SharedState) {
    let st_close = state.clone();
    state.borrow_mut().main_win.set_callback(move |_| {
        close_all_and_quit(&st_close);
    });

    let st_resize = state.clone();
    state.borrow_mut().main_win.resize_callback(move |_, _, _, w, h| {
        if let Ok(mut st) = st_resize.try_borrow_mut() {
            apply_layout(&mut st, w, h);
        }
    });

    // Global hotkeys (Ctrl+P, /, Arrows, Space, R, M, G, I, [, ], N, P, F11, F, F5, Esc)
    let st_keys = state.clone();
    state.borrow_mut().main_win.handle(move |_, ev| {
        if ev != Event::KeyDown {
            return false;
        }
        let key = app::event_key();
        let mods = app::event_state();
        let ctrl = mods.contains(EventState::Ctrl);
        let ch = key.to_char().map(|c| c.to_ascii_lowercase());

        if ctrl && ch == Some('q') {
            close_all_and_quit(&st_keys);
            return true;
        }
        if (ctrl && ch == Some('p')) || (!ctrl && ch == Some('/')) {
            open_preset_browser(&st_keys);
            return true;
        }

        let mut action_done = false;
        if let Ok(mut st) = st_keys.try_borrow_mut() {
            match key {
                k if k == Key::F11 => {
                    toggle_fullscreen(&mut st);
                    action_done = true;
                }
                k if k == Key::Escape && st.is_fullscreen => {
                    toggle_fullscreen(&mut st);
                    action_done = true;
                }
                k if k == Key::F5 => {
                    reload_active_preset(&mut st);
                    action_done = true;
                }
                k if k == Key::Left => {
                    select_prev_preset(&mut st);
                    action_done = true;
                }
                k if k == Key::Right => {
                    select_next_preset(&mut st);
                    action_done = true;
                }
                _ if !ctrl => {
                    if let Some(c) = ch {
                        match c {
                            'f' => {
                                toggle_fullscreen(&mut st);
                                action_done = true;
                            }
                            '[' | 'p' => {
                                select_prev_preset(&mut st);
                                action_done = true;
                            }
                            ']' | 'n' => {
                                select_next_preset(&mut st);
                                action_done = true;
                            }
                            ' ' | 'r' => {
                                select_random_preset(&mut st, None);
                                action_done = true;
                            }
                            'm' => {
                                select_random_preset(&mut st, Some(PresetKind::MilkDrop));
                                action_done = true;
                            }
                            'g' => {
                                select_random_preset(&mut st, Some(PresetKind::Wgsl));
                                action_done = true;
                            }
                            'i' => {
                                if let Some(viz) = st.viz.as_mut() {
                                    viz.trigger_overlay();
                                }
                                action_done = true;
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }
        action_done
    });

    // Double-click inside visualization panel toggles fullscreen
    let st_dbl = state.clone();
    state.borrow_mut().viz_win.handle(move |_, ev| {
        if ev == Event::Push && app::event_clicks() {
            if let Ok(mut st) = st_dbl.try_borrow_mut() {
                toggle_fullscreen(&mut st);
            }
            return true;
        }
        false
    });
}

fn apply_layout(st: &mut AppState, w: i32, h: i32) {
    let w = w.max(320);
    let h = h.max(240);

    if st.is_fullscreen {
        st.viz_win.resize(0, 0, w, h);
    } else {
        st.menu_bar.resize(0, 0, w, MENU_BAR_H);
        let viz_h = (h - MENU_BAR_H).max(160);
        st.viz_win.resize(0, MENU_BAR_H, w, viz_h);
    }
}

fn apply_fullscreen_visibility(st: &mut AppState) {
    if st.is_fullscreen {
        st.menu_bar.hide();
    } else {
        st.menu_bar.show();
    }
    let w = st.main_win.w();
    let h = st.main_win.h();
    apply_layout(st, w, h);
    st.main_win.redraw();
}

fn toggle_fullscreen(st: &mut AppState) {
    st.is_fullscreen = !st.is_fullscreen;
    st.main_win.fullscreen(st.is_fullscreen);
    apply_fullscreen_visibility(st);
    if let Some(viz) = st.viz.as_mut() {
        viz.trigger_overlay();
    }
}

fn apply_theme(st: &mut AppState) {
    let t = st.theme.clone();
    st.main_win.set_color(t.panel_bg.to_fltk());
    st.menu_bar.set_color(t.toolbar_bg.to_fltk());
    st.menu_bar.set_text_color(t.text_main.to_fltk());
    st.menu_bar.set_selection_color(t.accent.to_fltk());
    st.main_win.redraw();
}

fn rebuild_menu_bar(state: &SharedState) {
    let (mut menu, overlay_on, audio_paused, auto_sec, active_theme, wgsl_items) = {
        let st = state.borrow();
        let wgsl: Vec<(usize, String, String)> = st
            .presets
            .presets()
            .iter()
            .enumerate()
            .filter(|(_, p)| p.kind == PresetKind::Wgsl)
            .map(|(i, p)| {
                let top_cat = p
                    .category
                    .split('/')
                    .next()
                    .unwrap_or("Shaders")
                    .trim()
                    .to_string();
                (i, sanitize_menu_segment(&top_cat), sanitize_menu_segment(&p.name))
            })
            .collect();
        (
            st.menu_bar.clone(),
            st.config.show_overlay,
            st.audio_paused,
            st.config.auto_cycle_sec,
            st.config.theme,
            wgsl,
        )
    };
    menu.clear();

    // ---- File Menu ----
    let st_auto_src = state.clone();
    menu.add(
        "&File/Monitor Source/Auto-Detect (@DEFAULT_MONITOR@)",
        Shortcut::None,
        MenuFlag::Normal,
        move |_| {
            if let Ok(mut st) = st_auto_src.try_borrow_mut() {
                switch_audio_source(&mut st, None);
            }
        },
    );

    for src in list_monitor_sources() {
        let label = format!("&File/Monitor Source/{}", sanitize_menu_segment(&src));
        let st_src = state.clone();
        menu.add(&label, Shortcut::None, MenuFlag::Normal, move |_| {
            if let Ok(mut st) = st_src.try_borrow_mut() {
                switch_audio_source(&mut st, Some(src.clone()));
            }
        });
    }

    let st_rescan_src = state.clone();
    menu.add(
        "&File/Rescan Monitor Sources",
        Shortcut::None,
        MenuFlag::MenuDivider,
        move |_| rebuild_menu_bar(&st_rescan_src),
    );

    let st_imp = state.clone();
    menu.add(
        "&File/Import MilkDrop Folder...",
        Shortcut::None,
        MenuFlag::Normal,
        move |_| handle_deferred_action(&st_imp, DeferredUiAction::ImportFolder),
    );

    let st_rescan = state.clone();
    menu.add(
        "&File/Rescan All Presets && Textures",
        Shortcut::None,
        MenuFlag::Normal,
        move |_| handle_deferred_action(&st_rescan, DeferredUiAction::RescanAll),
    );

    let st_reload = state.clone();
    menu.add(
        "&File/Reload Active Preset",
        Shortcut::from_key(Key::F5),
        MenuFlag::MenuDivider,
        move |_| {
            if let Ok(mut st) = st_reload.try_borrow_mut() {
                reload_active_preset(&mut st);
            }
        },
    );

    let st_quit = state.clone();
    menu.add(
        "&File/Quit",
        Shortcut::Ctrl | 'q',
        MenuFlag::Normal,
        move |_| handle_deferred_action(&st_quit, DeferredUiAction::QuitApp),
    );

    // ---- Visualizer Menu ----
    let st_fs = state.clone();
    menu.add(
        "&Visualizer/Toggle Fullscreen",
        Shortcut::from_key(Key::F11),
        MenuFlag::MenuDivider,
        move |_| {
            if let Ok(mut st) = st_fs.try_borrow_mut() {
                toggle_fullscreen(&mut st);
            }
        },
    );

    let st_mon = state.clone();
    menu.add(
        "&Visualizer/Audio Monitor Enabled",
        Shortcut::None,
        MenuFlag::Toggle,
        move |m| {
            if let Ok(mut st) = st_mon.try_borrow_mut() {
                let checked = m.mvalue().map(|it| it.value()).unwrap_or(true);
                set_audio_monitor_enabled(&mut st, checked);
            }
        },
    );
    if let Some(mut item) = menu.find_item("&Visualizer/Audio Monitor Enabled") {
        if !audio_paused {
            item.set();
        } else {
            item.clear();
        }
    }

    let st_ov_show = state.clone();
    menu.add(
        "&Visualizer/Show Preset & Vulkan Overlay (I)",
        Shortcut::None,
        MenuFlag::Normal,
        move |_| {
            if let Ok(mut st) = st_ov_show.try_borrow_mut() {
                if let Some(viz) = st.viz.as_mut() {
                    viz.trigger_overlay();
                }
            }
        },
    );

    let st_ov_auto = state.clone();
    menu.add(
        "&Visualizer/Auto-Fade Overlay && Vulkan Watermark",
        Shortcut::None,
        MenuFlag::Toggle | MenuFlag::MenuDivider,
        move |m| {
            if let Ok(mut st) = st_ov_auto.try_borrow_mut() {
                let checked = m.mvalue().map(|it| it.value()).unwrap_or(true);
                st.config.show_overlay = checked;
                st.config.save();
                if let Some(viz) = st.viz.as_mut() {
                    viz.set_overlay_enabled(checked);
                }
            }
        },
    );
    if let Some(mut item) = menu.find_item("&Visualizer/Auto-Fade Overlay && Vulkan Watermark") {
        if overlay_on {
            item.set();
        } else {
            item.clear();
        }
    }

    for (sec, label) in [
        (0u32, "&Visualizer/Auto-Cycle Presets/Off"),
        (15, "&Visualizer/Auto-Cycle Presets/Every 15 seconds"),
        (30, "&Visualizer/Auto-Cycle Presets/Every 30 seconds"),
        (60, "&Visualizer/Auto-Cycle Presets/Every 60 seconds"),
    ] {
        let st_ac = state.clone();
        menu.add(label, Shortcut::None, MenuFlag::Radio, move |_| {
            if let Ok(mut st) = st_ac.try_borrow_mut() {
                st.config.auto_cycle_sec = sec;
                st.last_auto_cycle = Instant::now();
                st.config.save();
            }
        });
        if auto_sec == sec {
            if let Some(mut it) = menu.find_item(label) {
                it.set();
            }
        }
    }

    for mode in ALL_THEMES {
        let t = Theme::get(mode);
        let label = format!("&Visualizer/UI Theme/{}", t.name);
        let st_t = state.clone();
        menu.add(&label, Shortcut::None, MenuFlag::Radio, move |_| {
            if let Ok(mut st) = st_t.try_borrow_mut() {
                st.theme = Theme::get(mode);
                st.config.theme = mode;
                st.config.save();
                apply_theme(&mut st);
            }
        });
        if active_theme == mode {
            if let Some(mut it) = menu.find_item(&label) {
                it.set();
            }
        }
    }

    // ---- Presets Menu ----
    let st_browse = state.clone();
    menu.add(
        "&Presets/Browse && Search Presets... (Ctrl+P or /)",
        Shortcut::Ctrl | 'p',
        MenuFlag::MenuDivider,
        move |_| handle_deferred_action(&st_browse, DeferredUiAction::OpenBrowser),
    );

    let st_p = state.clone();
    menu.add(
        "&Presets/Previous Preset (Left or P)",
        Shortcut::None,
        MenuFlag::Normal,
        move |_| {
            if let Ok(mut st) = st_p.try_borrow_mut() {
                select_prev_preset(&mut st);
            }
        },
    );

    let st_n = state.clone();
    menu.add(
        "&Presets/Next Preset (Right or N)",
        Shortcut::None,
        MenuFlag::Normal,
        move |_| {
            if let Ok(mut st) = st_n.try_borrow_mut() {
                select_next_preset(&mut st);
            }
        },
    );

    let st_ra = state.clone();
    menu.add(
        "&Presets/Random Preset — Any (Space or R)",
        Shortcut::None,
        MenuFlag::Normal,
        move |_| {
            if let Ok(mut st) = st_ra.try_borrow_mut() {
                select_random_preset(&mut st, None);
            }
        },
    );

    let st_rm = state.clone();
    menu.add(
        "&Presets/Random MilkDrop (.milk) Preset (M)",
        Shortcut::None,
        MenuFlag::Normal,
        move |_| {
            if let Ok(mut st) = st_rm.try_borrow_mut() {
                select_random_preset(&mut st, Some(PresetKind::MilkDrop));
            }
        },
    );

    let st_rw = state.clone();
    menu.add(
        "&Presets/Random WGSL Shader (G)",
        Shortcut::None,
        MenuFlag::MenuDivider,
        move |_| {
            if let Ok(mut st) = st_rw.try_borrow_mut() {
                select_random_preset(&mut st, Some(PresetKind::Wgsl));
            }
        },
    );

    for (idx, cat, name) in wgsl_items {
        let path = format!("&Presets/WGSL Shaders/{cat}/{name}");
        let st_pick = state.clone();
        menu.add(&path, Shortcut::None, MenuFlag::Normal, move |_| {
            if let Ok(mut st) = st_pick.try_borrow_mut() {
                activate_preset(&mut st, idx);
            }
        });
    }

    let st_imp2 = state.clone();
    menu.add(
        "&Presets/Import MilkDrop Folder...",
        Shortcut::None,
        MenuFlag::Normal,
        move |_| handle_deferred_action(&st_imp2, DeferredUiAction::ImportFolder),
    );

    let st_rescan2 = state.clone();
    menu.add(
        "&Presets/Rescan Presets",
        Shortcut::None,
        MenuFlag::Normal,
        move |_| handle_deferred_action(&st_rescan2, DeferredUiAction::RescanAll),
    );

    // ---- Help Menu ----
    let st_help = state.clone();
    menu.add(
        "&Help/Hotkeys && Usage",
        Shortcut::None,
        MenuFlag::Normal,
        move |_| {
            let (shader_dir, milk_dirs, wgsl_n, milk_n) = {
                let st = st_help.borrow();
                let (w, m) = st.presets.counts();
                let md = st
                    .presets
                    .milkdrop_dirs()
                    .iter()
                    .map(|p| p.display().to_string())
                    .collect::<Vec<_>>()
                    .join("\n  ");
                (st.presets.shader_dir().display().to_string(), md, w, m)
            };
            dialog::message_default(&format!(
                "Vulkan System-Audio Visualizer ({wgsl_n} WGSL + {milk_n} MilkDrop Presets)\n\n\
                Audio Input:\n\
                - Listens to system output automatically via PipeWire / PulseAudio Monitor.\n\
                - Switch monitor device under File -> Monitor Source or pause in Visualizer menu.\n\n\
                Hotkeys:\n\
                - Ctrl+P or /          : Browse & Search All Presets (WGSL + .milk)\n\
                - Right / N / ]        : Next Preset\n\
                - Left / P / [         : Previous Preset\n\
                - Space / R            : Random Preset (Any)\n\
                - M                    : Random MilkDrop (.milk) Preset\n\
                - G                    : Random WGSL Shader\n\
                - I                    : Show Preset & Vulkan Watermark Overlay\n\
                - F11 / F / Dbl-Click  : Toggle Fullscreen (Esc exits)\n\
                - F5                   : Reload Active Preset from Disk\n\n\
                WGSL Directory:\n  {shader_dir}\n\n\
                MilkDrop Directories:\n  {milk_dirs}"
            ));
        },
    );

    let st_about = state.clone();
    menu.add(
        "&Help/About Vulkan Audio Visualizer...",
        Shortcut::None,
        MenuFlag::Normal,
        move |_| handle_deferred_action(&st_about, DeferredUiAction::OpenAbout),
    );
}

fn handle_deferred_action(state: &SharedState, action: DeferredUiAction) {
    match action {
        DeferredUiAction::OpenBrowser => open_preset_browser(state),
        DeferredUiAction::OpenAbout => open_about_dialog(state),
        DeferredUiAction::QuitApp => close_all_and_quit(state),
        DeferredUiAction::ImportFolder => {
            if let Some(dir) = dialog::dir_chooser("Select MilkDrop (.milk) / Textures Directory", "", false) {
                let rf = {
                    let mut st = state.borrow_mut();
                    if !st.config.milkdrop_dirs.iter().any(|d| d == &dir) {
                        st.config.milkdrop_dirs.push(dir);
                        st.config.save();
                    }
                    rescan_presets_internal(&mut st);
                    st.browser_win
                        .as_ref()
                        .filter(|w| w.shown())
                        .and_then(|_| st.browser_refresh.clone())
                };
                if let Some(rf) = rf {
                    rf();
                }
                rebuild_menu_bar(state);
                open_preset_browser(state);
            }
        }
        DeferredUiAction::RescanAll => {
            let rf = {
                let mut st = state.borrow_mut();
                rescan_presets_internal(&mut st);
                st.browser_win
                    .as_ref()
                    .filter(|w| w.shown())
                    .and_then(|_| st.browser_refresh.clone())
            };
            if let Some(rf) = rf {
                rf();
            }
            rebuild_menu_bar(state);
        }
    }
}

fn rescan_presets_internal(st: &mut AppState) {
    let extra = st.config.milkdrop_dirs.clone();
    st.presets.rescan(&extra);
    let tex_dirs = st.presets.texture_dirs().to_vec();
    if let Some(viz) = st.viz.as_mut() {
        viz.update_texture_dirs(&tex_dirs);
    }
    let idx = st.presets.active_idx();
    activate_preset(st, idx);
}

fn switch_audio_source(st: &mut AppState, source: Option<String>) {
    st.config.monitor_source = source.clone();
    st.config.save();
    st.audio_paused = false;
    let ring = st.pcm_ring.clone();
    st.audio_tap.restart(ring, source.as_deref());
}

fn set_audio_monitor_enabled(st: &mut AppState, enabled: bool) {
    st.audio_paused = !enabled;
    if enabled {
        let ring = st.pcm_ring.clone();
        let src = st.config.monitor_source.clone();
        st.audio_tap.restart(ring, src.as_deref());
    } else {
        st.audio_tap.stop();
    }
}

fn reload_active_preset(st: &mut AppState) {
    let _ = st.presets.poll_hot_reload(true);
    let idx = st.presets.active_idx();
    let total = st.presets.presets().len();
    let elapsed = (Instant::now() - st.start_time).as_secs_f32();
    let p = st.presets.active_preset().clone();
    if let Some(viz) = st.viz.as_mut() {
        load_preset_into_viz(viz, &p, idx, total, elapsed);
    }
}

fn load_preset_into_viz(
    viz: &mut VisualizerEngine,
    preset: &ShaderPreset,
    preset_idx: usize,
    total: usize,
    elapsed: f32,
) {
    let is_milk = preset.kind == PresetKind::MilkDrop;
    let res = if is_milk {
        viz.load_milkdrop(&preset.file_path, preset_idx as u32 + 1, elapsed)
    } else {
        viz.load_shader(&preset.source, &preset.name)
    };
    if let Err(err) = res {
        eprintln!("Preset '{}' load warning: {err}", preset.name);
    }
    viz.set_overlay_info(
        preset_idx + 1,
        total,
        &preset.name,
        &preset.category,
        is_milk,
    );
}

fn activate_preset(st: &mut AppState, preset_idx: usize) {
    let Some(preset) = st.presets.set_active_idx(preset_idx).cloned() else {
        return;
    };
    let total = st.presets.presets().len();
    let elapsed = (Instant::now() - st.start_time).as_secs_f32();
    st.config.active_shader_id = preset.id.clone();
    st.config.save();

    if let Some(viz) = st.viz.as_mut() {
        load_preset_into_viz(viz, &preset, preset_idx, total, elapsed);
    }
}

fn select_next_preset(st: &mut AppState) {
    let _ = st.presets.next_preset();
    let idx = st.presets.active_idx();
    activate_preset(st, idx);
}

fn select_prev_preset(st: &mut AppState) {
    let _ = st.presets.prev_preset();
    let idx = st.presets.active_idx();
    activate_preset(st, idx);
}

fn select_random_preset(st: &mut AppState, kind: Option<PresetKind>) {
    let candidates: Vec<usize> = st
        .presets
        .presets()
        .iter()
        .enumerate()
        .filter(|(_, p)| kind.map_or(true, |k| p.kind == k))
        .map(|(i, _)| i)
        .collect();
    if candidates.is_empty() {
        return;
    }
    st.rng_seq = st.rng_seq.wrapping_add(1);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u64)
        .unwrap_or(17);
    let mixed = (nanos ^ st.rng_seq.wrapping_mul(0x9E37_79B9_7F4A_7C15)) as usize;
    let mut pick = candidates[mixed % candidates.len()];
    if candidates.len() > 1 && pick == st.presets.active_idx() {
        pick = candidates[(mixed + 1) % candidates.len()];
    }
    activate_preset(st, pick);
}

fn open_preset_browser(state: &SharedState) {
    let existing = {
        let st = state.borrow();
        st.browser_win
            .as_ref()
            .map(|w| (w.clone(), st.browser_refresh.clone()))
    };
    if let Some((mut win, refresh)) = existing {
        if let Some(rf) = refresh {
            rf();
        }
        win.show();
        return;
    }

    let (wgsl_n, milk_n) = {
        let st = state.borrow();
        st.presets.counts()
    };

    let mut win = Window::default()
        .with_size(680, 560)
        .with_label(&format!("Presets & Search  ({wgsl_n} WGSL · {milk_n} MilkDrop)"));
    win.make_resizable(true);
    win.size_range(480, 360, 0, 0);
    if let Some(icon) = load_vulkan_icon() {
        win.set_icon(Some(icon));
    }

    let mut fmt_choice = Choice::new(12, 12, 145, 28, None);
    fmt_choice.add("All Formats", Shortcut::None, MenuFlag::Normal, |_| {});
    fmt_choice.add("MilkDrop (.milk)", Shortcut::None, MenuFlag::Normal, |_| {});
    fmt_choice.add("WGSL Shaders", Shortcut::None, MenuFlag::Normal, |_| {});
    fmt_choice.set_value(0);

    let mut cat_choice = Choice::new(165, 12, 200, 28, None);
    let mut cat_keys: Vec<Option<String>> = vec![None];
    cat_choice.add("All Categories", Shortcut::None, MenuFlag::Normal, |_| {});
    {
        let st = state.borrow();
        for (cat, count) in st.presets.top_categories() {
            if cat == "WGSL Shaders" {
                continue;
            }
            let lbl = format!("{} ({count})", sanitize_choice_label(&cat));
            cat_choice.add(&lbl, Shortcut::None, MenuFlag::Normal, |_| {});
            cat_keys.push(Some(cat));
        }
    }
    cat_choice.set_value(0);

    let mut search = Input::new(373, 12, 295, 28, None);
    search.set_trigger(CallbackTrigger::Changed);
    search.set_tooltip("Type to filter presets (name, author, category)...");

    let mut info_lbl = Frame::new(12, 46, 656, 22, "");
    info_lbl.set_align(Align::Left | Align::Inside);
    info_lbl.set_label_size(12);

    let mut list = HoldBrowser::new(12, 72, 656, 434, None);
    list.set_frame(FrameType::BorderBox);
    list.set_text_size(13);

    let mut btn_prev = Button::new(12, 516, 86, 32, "@< Prev");
    let mut btn_next = Button::new(104, 516, 86, 32, "Next @>");
    let mut btn_rand_md = Button::new(196, 516, 130, 32, "Rand MilkDrop");
    let mut btn_close = Button::new(582, 516, 86, 32, "Close (Esc)");

    win.end();

    let filtered_ref: Rc<RefCell<Vec<usize>>> = Rc::new(RefCell::new(Vec::new()));
    let cat_keys_rc = Rc::new(cat_keys);

    let refresh_fn: Rc<dyn Fn()> = {
        let st_r = state.clone();
        let filt_r = filtered_ref.clone();
        let cats_r = cat_keys_rc.clone();
        let fc = fmt_choice.clone();
        let cc = cat_choice.clone();
        let si = search.clone();
        let br_orig = list.clone();
        let lbl_orig = info_lbl.clone();
        Rc::new(move || {
            let mut br = br_orig.clone();
            let mut lbl = lbl_orig.clone();
            let filter = match fc.value() {
                1 => PresetFilter::MilkDropOnly,
                2 => PresetFilter::WgslOnly,
                _ => PresetFilter::All,
            };
            let cat_opt = cats_r
                .get(cc.value().max(0) as usize)
                .and_then(|c| c.as_deref());
            let query = si.value();

            let (indices, rows, active_row, active_name) = {
                let st = st_r.borrow();
                let indices = st.presets.filter_indices_with(&query, filter, cat_opt);
                let active_idx = st.presets.active_idx();
                let mut rows = Vec::with_capacity(indices.len());
                let mut active_row = None;
                for (row_i, &p_idx) in indices.iter().enumerate() {
                    let p = &st.presets.presets()[p_idx];
                    let tag = match p.kind {
                        PresetKind::MilkDrop => "MILK",
                        PresetKind::Wgsl => "WGSL",
                    };
                    rows.push(format!("[{tag}]  {}   ({})", p.name, p.category));
                    if p_idx == active_idx {
                        active_row = Some((row_i + 1) as i32);
                    }
                }
                let active_name = st.presets.active_preset().name.clone();
                (indices, rows, active_row, active_name)
            };

            *filt_r.borrow_mut() = indices;
            br.clear();
            for line in &rows {
                br.add(line);
            }
            if let Some(r) = active_row {
                br.select(r);
                br.middle_line(r);
            }
            lbl.set_label(&format!(
                "Showing {} presets  ·  Active: {active_name}  ·  Click/Enter to apply",
                rows.len()
            ));
        })
    };

    {
        let mut st = state.borrow_mut();
        st.browser_refresh = Some(refresh_fn.clone());
        st.browser_win = Some(win.clone());
    }

    refresh_fn();

    {
        let rf = refresh_fn.clone();
        fmt_choice.set_callback(move |_| rf());
    }
    {
        let rf = refresh_fn.clone();
        cat_choice.set_callback(move |_| rf());
    }
    {
        let rf = refresh_fn.clone();
        search.set_callback(move |_| rf());
    }

    // Activate selected row on click in HoldBrowser
    {
        let st_pick = state.clone();
        let filt_r = filtered_ref.clone();
        let mut lbl = info_lbl.clone();
        list.set_callback(move |b| {
            let row = b.value();
            if row <= 0 {
                return;
            }
            if let Some(&preset_idx) = filt_r.borrow().get((row - 1) as usize) {
                if let Ok(mut st) = st_pick.try_borrow_mut() {
                    activate_preset(&mut st, preset_idx);
                    let active_name = st.presets.active_preset().name.clone();
                    lbl.set_label(&format!(
                        "Showing {} presets  ·  Active: {active_name}  ·  Click/Enter to apply",
                        filt_r.borrow().len()
                    ));
                }
            }
        });
    }

    // Navigation buttons inside preset browser
    {
        let st_p = state.clone();
        let rf = refresh_fn.clone();
        btn_prev.set_callback(move |_| {
            if let Ok(mut st) = st_p.try_borrow_mut() {
                select_prev_preset(&mut st);
            }
            rf();
        });
    }
    {
        let st_n = state.clone();
        let rf = refresh_fn.clone();
        btn_next.set_callback(move |_| {
            if let Ok(mut st) = st_n.try_borrow_mut() {
                select_next_preset(&mut st);
            }
            rf();
        });
    }
    {
        let st_m = state.clone();
        let rf = refresh_fn.clone();
        btn_rand_md.set_callback(move |_| {
            if let Ok(mut st) = st_m.try_borrow_mut() {
                select_random_preset(&mut st, Some(PresetKind::MilkDrop));
            }
            rf();
        });
    }
    {
        let mut w_close = win.clone();
        btn_close.set_callback(move |_| w_close.hide());
    }

    // Keep child layout proportional on window resize
    {
        let mut fc = fmt_choice.clone();
        let mut cc = cat_choice.clone();
        let mut si = search.clone();
        let mut lbl = info_lbl.clone();
        let mut br = list.clone();
        let mut bp = btn_prev.clone();
        let mut bn = btn_next.clone();
        let mut brm = btn_rand_md.clone();
        let mut bc = btn_close.clone();
        win.resize_callback(move |_, _, _, w, h| {
            let w = w.max(480);
            let h = h.max(360);
            let sw = (w - 385).max(120);
            fc.resize(12, 12, 145, 28);
            cc.resize(165, 12, 200, 28);
            si.resize(373, 12, sw, 28);
            lbl.resize(12, 46, w - 24, 22);
            br.resize(12, 72, w - 24, h - 126);
            let by = h - 44;
            bp.resize(12, by, 86, 32);
            bn.resize(104, by, 86, 32);
            brm.resize(196, by, 130, 32);
            bc.resize(w - 98, by, 86, 32);
        });
    }

    // Enter activates top/selected preset, Esc hides browser
    {
        let st_k = state.clone();
        let filt_r = filtered_ref.clone();
        let mut br = list.clone();
        let mut lbl = info_lbl.clone();
        win.handle(move |w, ev| {
            if ev == Event::KeyDown {
                match app::event_key() {
                    k if k == Key::Escape => {
                        w.hide();
                        return true;
                    }
                    k if k == Key::Enter => {
                        let row = br.value().max(1);
                        if let Some(&preset_idx) = filt_r.borrow().get((row - 1) as usize) {
                            if let Ok(mut st) = st_k.try_borrow_mut() {
                                activate_preset(&mut st, preset_idx);
                                let active_name = st.presets.active_preset().name.clone();
                                lbl.set_label(&format!(
                                    "Showing {} presets  ·  Active: {active_name}  ·  Click/Enter to apply",
                                    filt_r.borrow().len()
                                ));
                            }
                            br.select(row);
                        }
                        return true;
                    }
                    _ => {}
                }
            }
            false
        });
    }

    win.set_callback(|w| w.hide());
    win.show();
    let _ = search.take_focus();
}

fn open_about_dialog(state: &SharedState) {
    {
        let st = state.borrow();
        if let Some(ref w) = st.about_win {
            if w.shown() {
                let mut wc = w.clone();
                drop(st);
                wc.show();
                return;
            }
        }
    }

    let (gpu, backend, wgsl_n, milk_n, tex_n, audio_src, err_note) = {
        let st = state.borrow();
        let (w, m) = st.presets.counts();
        let tex_n = st.presets.texture_dirs().len();
        let audio_src = if st.audio_paused {
            "Paused".to_string()
        } else if let Some(err) = st.audio_tap.last_error() {
            format!("Error ({err})")
        } else {
            let s = st.audio_tap.active_source();
            if s == "@DEFAULT_MONITOR@" {
                "Auto-Detect (System Monitor)".to_string()
            } else {
                s.replace('&', "&&")
            }
        };
        match st.viz.as_ref() {
            Some(v) => (
                v.adapter_name().to_string(),
                v.backend_name().to_string(),
                w,
                m,
                tex_n,
                audio_src,
                v.last_error().map(str::to_string),
            ),
            None => (
                "Unavailable".to_string(),
                "None".to_string(),
                w,
                m,
                tex_n,
                audio_src,
                None,
            ),
        }
    };

    let mut win = Window::default()
        .with_size(520, 440)
        .with_label("About Vulkan Audio Visualizer");
    win.make_resizable(false);
    if let Some(icon) = load_vulkan_icon() {
        win.set_icon(Some(icon));
    }

    // Crisp light pill card for the Khronos Vulkan logo SVG render
    let mut logo_box = Frame::new(20, 18, 480, 112, None);
    logo_box.set_frame(FrameType::RoundedBox);
    logo_box.set_color(Color::from_rgb(245, 247, 251));
    logo_box.set_align(Align::Center | Align::Inside);
    if let Ok(mut img) = PngImage::from_data(VULKAN_ABOUT_PNG) {
        let iw = img.width().max(1);
        let ih = img.height().max(1);
        let scale = (440.0 / iw as f32).min(92.0 / ih as f32).min(1.0);
        img.scale((iw as f32 * scale) as i32, (ih as f32 * scale) as i32, true, true);
        logo_box.set_image(Some(img));
    } else {
        logo_box.set_label("VULKAN");
    }

    let mut khronos_lbl = Frame::new(
        20,
        136,
        480,
        22,
        "Vulkan™/® Logo: Khronos Group (Public Domain via Wikimedia Commons)",
    );
    khronos_lbl.set_label_size(11);
    khronos_lbl.set_label_color(Color::from_rgb(155, 165, 184));

    let mut title_lbl = Frame::new(20, 164, 480, 28, "Vulkan Audio Visualizer v0.2.0");
    title_lbl.set_label_size(17);

    let err_line = err_note
        .map(|e| {
            let clean: String = e.chars().take(54).collect();
            format!("\n  Last Fallback Note  : {}", clean.replace('&', "&&"))
        })
        .unwrap_or_default();
    let mut details = Frame::new(20, 198, 480, 182, None);
    details.set_frame(FrameType::BorderBox);
    details.set_color(Color::from_rgb(24, 28, 38));
    details.set_align(Align::Left | Align::Inside);
    details.set_label_size(12);
    details.set_label(&format!(
        "  WGSL Shaders        : {wgsl_n} built-in + custom hot-reloadable\n  \
           MilkDrop (.milk)    : {milk_n} presets  (EEL2 + HLSL->GLSL native on wgpu)\n  \
           Texture Packs       : {tex_n} directory roots\n  \
           GPU Adapter         : {gpu}  ({backend})\n  \
           Audio Monitor       : {audio_src}\n  \
           Audio FFT           : PipeWire/PulseAudio · 2048-pt Hann · 64 bands{err_line}\n\n  \
           Ctrl+P or / : Browse Presets   ·   M : Random MilkDrop   ·   I : Overlay"
    ));

    let mut close_btn = Button::new(210, 392, 100, 32, "Close");
    win.end();

    {
        let mut wc = win.clone();
        close_btn.set_callback(move |_| wc.hide());
    }
    win.handle(|w, ev| {
        if ev == Event::KeyDown && app::event_key() == Key::Escape {
            w.hide();
            return true;
        }
        false
    });
    win.set_callback(|w| w.hide());

    state.borrow_mut().about_win = Some(win.clone());
    win.show();
}
