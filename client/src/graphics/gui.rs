use std::{
    fs,
    net::SocketAddr,
    path::{Path, PathBuf},
    str::FromStr,
    sync::Arc,
};

use rand::random;

use yakui::{
    Alignment, Color, align, button, colored_box, colored_box_container, column, label, pad, row,
    textbox, widgets::Pad,
};

use common::worldgen::sample_enviro_at;
use common::{SimConfig, worldgen::WorldgenPreset};
use save::Save;

use crate::{Sim, config::RawConfig};

pub struct GuiState {
    show_gui: bool,
    config_panel: ConfigPanel,
}

#[derive(Clone)]
pub struct SessionOptions {
    pub player_name: Arc<str>,
    pub chunk_parallelism: u32,
    pub server: Option<SocketAddr>,
    pub sim_config: SimConfig,
    pub start_in_freecam: bool,
}

pub enum GuiAction {
    StartSession {
        session: SessionOptions,
        raw_config: RawConfig,
    },
}

impl GuiState {
    pub fn new(raw_config: RawConfig, save_dir: PathBuf) -> Self {
        GuiState {
            show_gui: true,
            config_panel: ConfigPanel::new(raw_config, save_dir),
        }
    }

    /// Toggles whether the GUI is shown. The startup panel stays visible until a session starts.
    pub fn toggle_gui(&mut self) {
        if self.config_panel.awaiting_start {
            return;
        }
        self.show_gui = !self.show_gui;
    }

    /// Prepare the GUI for rendering. This should be called between
    /// Yakui::start and Yakui::finish.
    pub fn run(&mut self, sim: Option<&Sim>) -> Option<GuiAction> {
        let mut action = None;

        if self.show_gui {
            self.render_crosshair();
        }

        if self.config_panel.should_render(self.show_gui) {
            if let Some(result) = self.config_panel.draw() {
                action = Some(result);
            }
        }

        if let Some(sim) = sim {
            self.render_material_panel(sim);
            // Debug info overlay
            self.render_debug_panel(sim);
        }

        action
    }

    pub fn mark_session_started(&mut self) {
        self.config_panel.mark_started();
        self.show_gui = false;
    }

    pub fn set_error(&mut self, message: impl Into<String>) {
        self.config_panel
            .set_message(PanelMessage::Error(message.into()));
        self.show_gui = true;
    }

    pub fn set_info(&mut self, message: impl Into<String>) {
        self.config_panel
            .set_message(PanelMessage::Info(message.into()));
        self.show_gui = true;
    }

    fn render_material_panel(&self, sim: &Sim) {
        if !self.show_gui {
            return;
        }

        align(Alignment::TOP_LEFT, || {
            pad(Pad::all(8.0), || {
                colored_box_container(Color::BLACK.with_alpha(0.7), || {
                    let material_count_string = if sim.cfg.gameplay_enabled {
                        sim.count_inventory_entities_matching_tile(sim.selected_tile())
                            .to_string()
                    } else {
                        "∞".to_string()
                    };
                    label(format!(
                        "Selected material: {:?} (×{})",
                        sim.selected_tile(),
                        material_count_string
                    ));
                });
            });
        });
    }

    fn render_debug_panel(&self, sim: &Sim) {
        if !self.show_gui {
            return;
        }

        // Try to get a view position and sample enviro. If local character not ready,
        // skip rendering.
        let view_pos = sim.view();

        // `sample_enviro_at` expects the graph to have the node populated; if not, skip.
        let sample = match std::panic::catch_unwind(|| {
            sample_enviro_at(&sim.graph, &view_pos, sim.cfg.world_seed)
        }) {
            Ok(s) => s,
            Err(_) => return,
        };

        align(Alignment::TOP_RIGHT, || {
            pad(Pad::all(8.0), || {
                colored_box_container(Color::BLACK.with_alpha(0.7), || {
                    label(format!("Biome: {}", Self::biome_name(sample.biome)));
                });
            });
        });
    }

    fn biome_name(id: u8) -> &'static str {
        match id {
            0 => "Rainforest",
            1 => "Swampland",
            2 => "Seasonal Forest",
            3 => "Forest",
            4 => "Savanna",
            5 => "Shrubland",
            6 => "Taiga",
            7 => "Desert",
            8 => "Plains",
            9 => "Ice Desert",
            10 => "Tundra",
            _ => "Unknown",
        }
    }

    fn render_crosshair(&self) {
        align(Alignment::CENTER, || {
            colored_box(Color::BLACK.with_alpha(0.9), [5.0, 5.0]);
        });
    }
}

struct ConfigPanel {
    awaiting_start: bool,
    form: ConfigForm,
    message: Option<PanelMessage>,
    raw: RawConfig,
    save_dir: PathBuf,
    available_saves: Vec<SaveSummary>,
    selected_save: Option<usize>,
}

struct ConfigForm {
    name: String,
    chunk_parallelism: String,
    server: String,
    worldgen: WorldgenDescriptor,
    save_name: String,
    seed: String,
    start_in_freecam: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum WorldgenDescriptor {
    Hyperbolic,
    Flat,
}

enum PanelMessage {
    Info(String),
    Error(String),
}

#[derive(Clone)]
struct SaveSummary {
    name: String,
    seed: Option<u64>,
}

impl ConfigPanel {
    fn new(raw: RawConfig, save_dir: PathBuf) -> Self {
        let name = raw
            .name
            .clone()
            .unwrap_or_else(|| Arc::<str>::from(whoami::username()));
        let chunk_parallelism = raw.chunk_load_parallelism.unwrap_or(256).to_string();
        let server = raw.server.map(|addr| addr.to_string()).unwrap_or_default();
        let worldgen = match raw.local_simulation.worldgen {
            WorldgenPreset::Hyperbolic => WorldgenDescriptor::Hyperbolic,
            WorldgenPreset::Flat => WorldgenDescriptor::Flat,
        };
        let save_name = raw
            .save
            .clone()
            .unwrap_or_else(|| "default.save".into())
            .to_string_lossy()
            .to_string();
        let seed = raw
            .local_simulation
            .world_seed
            .map(|s| s.to_string())
            .unwrap_or_default();

        let mut panel = ConfigPanel {
            awaiting_start: true,
            form: ConfigForm {
                name: name.to_string(),
                chunk_parallelism,
                server,
                worldgen,
                save_name,
                seed,
                start_in_freecam: true,
            },
            message: None,
            raw,
            save_dir,
            available_saves: Vec::new(),
            selected_save: None,
        };
        panel.refresh_save_list();
        panel
    }

    fn should_render(&self, show_gui: bool) -> bool {
        self.awaiting_start || show_gui
    }

    fn draw(&mut self) -> Option<GuiAction> {
        let mut action = None;

        align(Alignment::CENTER, || {
            pad(Pad::all(16.0), || {
                colored_box_container(Color::BLACK.with_alpha(0.85), || {
                    column(|| {
                        label("Client Settings");
                        pad(Pad::all(8.0), || {
                            Self::draw_text_field("Display name", &mut self.form.name);
                        });

                        pad(Pad::all(6.0), || {
                            self.draw_worldgen_row();
                        });

                        pad(Pad::all(6.0), || {
                            self.draw_world_list();
                        });

                        pad(Pad::all(6.0), || {
                            self.draw_save_name_field();
                        });

                        pad(Pad::all(6.0), || {
                            column(|| {
                                label("World seed (leave blank for random)");
                                row(|| {
                                    let response = textbox(self.form.seed.clone());
                                    if let Some(text) = response.text.clone() {
                                        self.form.seed = text;
                                    }
                                    if button("Randomize").clicked {
                                        self.form.seed = random::<u64>().to_string();
                                    }
                                });
                            });
                        });

                        pad(Pad::all(6.0), || {
                            self.draw_movement_mode_row();
                        });

                        pad(Pad::all(6.0), || {
                            Self::draw_text_field(
                                "Chunk load parallelism",
                                &mut self.form.chunk_parallelism,
                            );
                        });

                        pad(Pad::all(6.0), || {
                            Self::draw_text_field(
                                "Server address (leave blank for local simulation)",
                                &mut self.form.server,
                            );
                        });

                        if let Some(message) = &self.message {
                            pad(Pad::all(6.0), || {
                                label(match message {
                                    PanelMessage::Info(msg) => format!("Info: {}", msg),
                                    PanelMessage::Error(msg) => format!("Error: {}", msg),
                                });
                            });
                        }

                        pad(Pad::all(10.0), || {
                            let button_label = if self.awaiting_start {
                                if self.form.server.trim().is_empty() {
                                    "Start Local Simulation"
                                } else {
                                    "Connect to Server"
                                }
                            } else {
                                "Save Changes"
                            };

                            if button(button_label).clicked {
                                match self.build_session_options() {
                                    Ok(result) => {
                                        action = Some(result);
                                        if self.awaiting_start {
                                            self.message = Some(PanelMessage::Info(
                                                "Launching session...".to_string(),
                                            ));
                                        } else {
                                            self.message = Some(PanelMessage::Info(
                                                "Configuration saved.".to_string(),
                                            ));
                                        }
                                    }
                                    Err(err) => {
                                        self.message = Some(PanelMessage::Error(err));
                                    }
                                }
                            }
                        });
                    });
                });
            });
        });

        action
    }

    fn draw_text_field(label_text: &str, value: &mut String) {
        column(|| {
            label(label_text.to_string());
            let response = textbox(value.clone());
            if let Some(text) = response.text.clone() {
                *value = text;
            }
        });
    }

    fn draw_save_name_field(&mut self) {
        column(|| {
            label("World save name".to_string());
            let response = textbox(self.form.save_name.clone());
            if let Some(text) = response.text.clone() {
                if text != self.form.save_name {
                    self.form.save_name = text;
                    self.sync_selection_with_form();
                }
            }
        });
    }

    fn draw_world_list(&mut self) {
        column(|| {
            label("Worlds".to_string());
            if self.available_saves.is_empty() {
                label("No worlds found in save directory".to_string());
            } else {
                for idx in 0..self.available_saves.len() {
                    let save_name = self.available_saves[idx].name.clone();
                    let save_seed = self.available_saves[idx].seed;
                    let selected = self.selected_save == Some(idx);
                    row(|| {
                        let button_label = if selected {
                            format!("[x] {}", save_name)
                        } else {
                            format!("[ ] {}", save_name)
                        };
                        if button(button_label).clicked {
                            self.select_save(idx);
                        }
                        let seed_label = match save_seed {
                            Some(seed) => format!("Seed: {}", seed),
                            None => "Seed: <unknown>".to_string(),
                        };
                        label(seed_label);
                    });
                }
            }

            if button("Refresh world list").clicked {
                self.refresh_save_list();
            }
        });
    }

    fn draw_worldgen_row(&mut self) {
        column(|| {
            label("World generation preset");
            row(|| {
                let hyper_label = if self.form.worldgen == WorldgenDescriptor::Hyperbolic {
                    "[x] Hyperbolic terrain"
                } else {
                    "[ ] Hyperbolic terrain"
                };
                if button(hyper_label).clicked {
                    self.form.worldgen = WorldgenDescriptor::Hyperbolic;
                }
                let flat_label = if self.form.worldgen == WorldgenDescriptor::Flat {
                    "[x] Flat terrain"
                } else {
                    "[ ] Flat terrain"
                };
                if button(flat_label).clicked {
                    self.form.worldgen = WorldgenDescriptor::Flat;
                }
            });
        });
    }

    fn draw_movement_mode_row(&mut self) {
        column(|| {
            label("Starting movement mode");
            row(|| {
                let freecam_label = if self.form.start_in_freecam {
                    "[x] Freecam (no-clip)"
                } else {
                    "[ ] Freecam (no-clip)"
                };
                if button(freecam_label).clicked {
                    self.form.start_in_freecam = true;
                }
                let player_label = if !self.form.start_in_freecam {
                    "[x] Player controller"
                } else {
                    "[ ] Player controller"
                };
                if button(player_label).clicked {
                    self.form.start_in_freecam = false;
                }
            });
        });
    }

    fn refresh_save_list(&mut self) {
        let mut saves = Vec::new();
        let chunk_size_hint = self.chunk_size_hint();
        if let Ok(entries) = fs::read_dir(&self.save_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if !path.is_file() {
                    continue;
                }
                let name = entry.file_name().to_string_lossy().to_string();
                let seed = self.read_save_seed(&path, chunk_size_hint);
                saves.push(SaveSummary { name, seed });
            }
        }
        saves.sort_by(|a, b| a.name.cmp(&b.name));
        self.available_saves = saves;
        self.sync_selection_with_form();
    }

    fn select_save(&mut self, idx: usize) {
        if let Some(save) = self.available_saves.get(idx) {
            self.selected_save = Some(idx);
            self.form.save_name = save.name.clone();
            self.form.seed = save.seed.map(|s| s.to_string()).unwrap_or_default();
        }
    }

    fn sync_selection_with_form(&mut self) {
        let trimmed = self.form.save_name.trim();
        if trimmed.is_empty() {
            self.selected_save = None;
            return;
        }
        self.selected_save = self
            .available_saves
            .iter()
            .position(|save| save.name == trimmed);
        if let Some(idx) = self.selected_save {
            if let Some(save) = self.available_saves.get(idx)
                && self.form.seed.trim().is_empty()
                && let Some(seed) = save.seed
            {
                self.form.seed = seed.to_string();
            }
        }
    }

    fn read_save_seed(&self, path: &Path, chunk_size_hint: u8) -> Option<u64> {
        Save::open(path, chunk_size_hint, 0)
            .ok()
            .and_then(|save| match save.meta().seed {
                0 => None,
                seed => Some(seed),
            })
    }

    fn chunk_size_hint(&self) -> u8 {
        self.raw.local_simulation.chunk_size.unwrap_or(16)
    }

    fn build_session_options(&mut self) -> Result<GuiAction, String> {
        let name = self.form.name.trim().to_string();
        if name.is_empty() {
            return Err("Display name cannot be empty".into());
        }

        let chunk_parallelism: u32 = self
            .form
            .chunk_parallelism
            .trim()
            .parse()
            .map_err(|_| "Chunk load parallelism must be a positive integer".to_string())?;
        if chunk_parallelism == 0 {
            return Err("Chunk load parallelism must be greater than zero".into());
        }

        let save_name = self.form.save_name.trim();
        if save_name.is_empty() {
            return Err("World save name cannot be empty".into());
        }

        let seed = if self.form.seed.trim().is_empty() {
            random::<u64>()
        } else {
            self.form
                .seed
                .trim()
                .parse::<u64>()
                .map_err(|_| "World seed must be a valid unsigned integer".to_string())?
        };

        let server = self.parse_server()?;

        let mut raw = self.raw.clone();
        raw.name = Some(Arc::<str>::from(name.clone()));
        raw.chunk_load_parallelism = Some(chunk_parallelism);
        raw.server = server;
        raw.local_simulation.worldgen = match self.form.worldgen {
            WorldgenDescriptor::Hyperbolic => WorldgenPreset::Hyperbolic,
            WorldgenDescriptor::Flat => WorldgenPreset::Flat,
        };
        raw.local_simulation.world_seed = Some(seed);
        raw.save = Some(save_name.into());

        let sim_config = SimConfig::from_raw(&raw.local_simulation);
        let session = SessionOptions {
            player_name: Arc::<str>::from(name),
            chunk_parallelism,
            server,
            sim_config,
            start_in_freecam: self.form.start_in_freecam,
        };

        self.raw = raw.clone();

        Ok(GuiAction::StartSession {
            session,
            raw_config: raw,
        })
    }

    fn parse_server(&self) -> Result<Option<SocketAddr>, String> {
        let trimmed = self.form.server.trim();
        if trimmed.is_empty() {
            return Ok(None);
        }
        SocketAddr::from_str(trimmed)
            .map(Some)
            .map_err(|err| format!("Invalid server address: {err}"))
    }

    fn mark_started(&mut self) {
        self.awaiting_start = false;
        self.message = None;
    }

    fn set_message(&mut self, message: PanelMessage) {
        self.message = Some(message);
    }
}
