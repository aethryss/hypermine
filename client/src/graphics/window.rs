use std::sync::Arc;
use std::time::Instant;
use std::{f32, os::raw::c_char, path::PathBuf, thread};

use ash::{khr, vk};
use directories::ProjectDirs;
use lahar::DedicatedImage;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use save::Save;
use tokio::runtime::Builder;
use tracing::{debug, error, error_span, info, Instrument};
use winit::event::KeyEvent;
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::{
    dpi::PhysicalSize,
    event::{DeviceEvent, ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    window::{CursorGrabMode, Window as WinitWindow},
};

use super::gui::{GuiAction, GuiState, SessionOptions};
use super::{Base, Core, Draw, Frustum};
use crate::{config::RawConfig, net, Config, Sim};
use common::proto;
use server::{self, Message, Server};

/// OS window
pub struct EarlyWindow {
    window: WinitWindow,
    required_extensions: &'static [*const c_char],
}

impl EarlyWindow {
    pub fn new(event_loop: &ActiveEventLoop) -> Self {
        let mut attrs = WinitWindow::default_attributes();
        attrs.title = "hypermine".into();
        let window = event_loop.create_window(attrs).unwrap();
        Self {
            window,
            required_extensions: ash_window::enumerate_required_extensions(
                event_loop.display_handle().unwrap().as_raw(),
            )
            .expect("unsupported platform"),
        }
    }

    /// Identify the Vulkan extension needed to render to this window
    pub fn required_extensions(&self) -> &'static [*const c_char] {
        self.required_extensions
    }
}

/// OS window + rendering handles
pub struct Window {
    _core: Arc<Core>,
    pub window: WinitWindow,
    config: Arc<Config>,
    dirs: Arc<ProjectDirs>,
    config_path: PathBuf,
    raw_config: RawConfig,
    surface_fn: khr::surface::Instance,
    surface: vk::SurfaceKHR,
    swapchain: Option<SwapchainMgr>,
    swapchain_needs_update: bool,
    draw: Option<Draw>,
    sim: Option<Sim>,
    gui_state: GuiState,
    yak: yakui::Yakui,
    yak_winit: yakui_winit::YakuiWinit,
    net: Option<server::Handle>,
    active_session: Option<SessionOptions>,
    input: InputState,
    last_frame: Option<Instant>,
}

impl Window {
    /// Finish constructing a window
    pub fn new(
        early: EarlyWindow,
        core: Arc<Core>,
        config: Arc<Config>,
        dirs: Arc<ProjectDirs>,
        raw_config: RawConfig,
    ) -> Self {
        let surface = unsafe {
            ash_window::create_surface(
                &core.entry,
                &core.instance,
                early.window.display_handle().unwrap().as_raw(),
                early.window.window_handle().unwrap().as_raw(),
                None,
            )
            .unwrap()
        };
        let surface_fn = khr::surface::Instance::new(&core.entry, &core.instance);
        let config_path = Config::config_path(&dirs);
        let gui_state = GuiState::new(raw_config.clone());
        let yak = yakui::Yakui::new();
        let yak_winit = yakui_winit::YakuiWinit::new(&early.window);

        Self {
            _core: core,
            window: early.window,
            config,
            dirs,
            config_path,
            raw_config,
            surface,
            surface_fn,
            swapchain: None,
            swapchain_needs_update: false,
            draw: None,
            sim: None,
            gui_state,
            yak,
            yak_winit,
            net: None,
            active_session: None,
            input: InputState::default(),
            last_frame: None,
        }
    }

    /// Determine whether this window can be rendered to from a particular device and queue family
    pub fn supports(&self, physical: vk::PhysicalDevice, queue_family_index: u32) -> bool {
        unsafe {
            self.surface_fn
                .get_physical_device_surface_support(physical, queue_family_index, self.surface)
                .unwrap()
        }
    }

    pub fn init_rendering(&mut self, gfx: Arc<Base>) {
        // Allocate the presentable images we'll be rendering to
        self.swapchain = Some(SwapchainMgr::new(
            self,
            gfx.clone(),
            self.window.inner_size(),
        ));
        // Construct the core rendering object
        self.draw = Some(Draw::new(gfx, self.config.clone()));
    }

    pub fn handle_device_event(&mut self, event: DeviceEvent) {
        match event {
            DeviceEvent::MouseMotion { delta } if self.input.mouse_captured => {
                if let Some(sim) = self.sim.as_mut() {
                    const SENSITIVITY: f32 = 2e-3;
                    sim.look(
                        -delta.0 as f32 * SENSITIVITY,
                        -delta.1 as f32 * SENSITIVITY,
                        0.0,
                    );
                }
            }
            _ => {}
        }
    }

    pub fn handle_event(&mut self, event: WindowEvent, event_loop: &ActiveEventLoop) {
        // Forward events to yakui first for GUI handling
        self.yak_winit.handle_window_event(&mut self.yak, &event);
        
        match event {
            WindowEvent::RedrawRequested => {
                let mut pending_messages = Vec::new();
                if let Some(net) = self.net.as_mut() {
                    while let Ok(msg) = net.incoming.try_recv() {
                        pending_messages.push(msg);
                    }
                }
                for msg in pending_messages {
                    self.handle_net(msg);
                }

                if let (Some(sim), Some(net)) = (self.sim.as_mut(), self.net.as_mut()) {
                    let this_frame = Instant::now();
                    let dt = this_frame - self.last_frame.unwrap_or(this_frame);
                    sim.set_movement_input(self.input.movement());
                    sim.set_jump_held(self.input.jump);

                    sim.look(0.0, 0.0, 2.0 * self.input.roll() * dt.as_secs_f32());

                    sim.step(dt, net);
                    self.last_frame = Some(this_frame);
                }

                self.draw();
            }
            WindowEvent::Resized(_) => {
                // Some environments may not emit the vulkan signals that recommend or
                // require surface reconstruction, so we need to check for messages from the
                // windowing system too. We defer actually performing the update until
                // drawing to avoid doing unnecessary work between frames.
                self.swapchain_needs_update = true;
            }
            WindowEvent::CloseRequested => {
                info!("exiting due to closed window");
                event_loop.exit();
            }
            WindowEvent::MouseInput {
                button: MouseButton::Left,
                state: ElementState::Pressed,
                ..
            } => {
                // Only capture mouse when a sim is active (game has started)
                if let Some(sim) = self.sim.as_mut() {
                    if self.input.mouse_captured {
                        sim.set_break_block_pressed_true();
                    }
                    let _ = self
                        .window
                        .set_cursor_grab(CursorGrabMode::Confined)
                        .or_else(|_e| self.window.set_cursor_grab(CursorGrabMode::Locked));
                    self.window.set_cursor_visible(false);
                    self.input.mouse_captured = true;
                }
            }
            WindowEvent::MouseInput {
                button: MouseButton::Right,
                state: ElementState::Pressed,
                ..
            } => {
                if self.input.mouse_captured
                    && let Some(sim) = self.sim.as_mut()
                {
                    sim.set_place_block_pressed_true();
                }
            }
            WindowEvent::MouseWheel {
                delta,
                ..
            } => {
                // Scroll wheel for block selection
                let scroll_direction = match delta {
                    MouseScrollDelta::LineDelta(_, y) => {
                        if y > 0.0 { 1 } else { -1 }
                    }
                    MouseScrollDelta::PixelDelta(pos) => {
                        if pos.y > 0.0 { 1 } else { -1 }
                    }
                };
                
                let max_tile_id = 70u16; // Total number of tile types (0-69)
                let current = self.input.selected_tile_id as i32;
                let new_id = (current + scroll_direction) % max_tile_id as i32;
                let new_id = if new_id < 0 { 
                    (max_tile_id as i32 + new_id) as u16 
                } else { 
                    new_id as u16 
                };
                self.input.selected_tile_id = new_id;
                
                // Update the sim with the selected tile ID
                if let Some(sim) = self.sim.as_mut() {
                    sim.select_tile_by_id(new_id);
                    debug!("Selected tile ID: {}", new_id);
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state,
                        physical_key: PhysicalKey::Code(key),
                        ..
                    },
                ..
            } => match key {
                KeyCode::KeyW => {
                    self.input.forward = state == ElementState::Pressed;
                }
                KeyCode::KeyA => {
                    self.input.left = state == ElementState::Pressed;
                }
                KeyCode::KeyS => {
                    self.input.back = state == ElementState::Pressed;
                }
                KeyCode::KeyD => {
                    self.input.right = state == ElementState::Pressed;
                }
                KeyCode::KeyQ => {
                    self.input.anticlockwise = state == ElementState::Pressed;
                }
                KeyCode::KeyE => {
                    self.input.clockwise = state == ElementState::Pressed;
                }
                KeyCode::KeyR => {
                    self.input.up = state == ElementState::Pressed;
                }
                KeyCode::KeyF => {
                    self.input.down = state == ElementState::Pressed;
                }
                KeyCode::Space => {
                    if let Some(sim) = self.sim.as_mut() {
                        if !self.input.jump && state == ElementState::Pressed {
                            sim.set_jump_pressed_true();
                        }
                        self.input.jump = state == ElementState::Pressed;
                    }
                }
                KeyCode::KeyV if state == ElementState::Pressed => {
                    if let Some(sim) = self.sim.as_mut() {
                        sim.toggle_no_clip();
                    }
                }
                KeyCode::F1 if state == ElementState::Pressed => {
                    self.gui_state.toggle_gui();
                }
                KeyCode::Escape => {
                    let _ = self.window.set_cursor_grab(CursorGrabMode::None);
                    self.window.set_cursor_visible(true);
                    self.input.mouse_captured = false;
                }
                _ => {
                    if let Some(material_idx) = number_key_to_index(key)
                        && state == ElementState::Pressed
                        && let Some(sim) = self.sim.as_mut()
                    {
                        sim.select_tile(material_idx);
                    }
                }
            },
            WindowEvent::Focused(focused) => {
                if !focused {
                    let _ = self.window.set_cursor_grab(CursorGrabMode::None);
                    self.window.set_cursor_visible(true);
                    self.input.mouse_captured = false;
                }
            }
            _ => {}
        }
    }

    fn handle_net(&mut self, msg: Message) {
        match msg {
            Message::ConnectionLost(e) => {
                error!("connection lost: {}", e);
                self.net = None;
                self.sim = None;
                self.active_session = None;
                self.gui_state = GuiState::new(self.raw_config.clone());
                self.gui_state
                    .set_error(format!("Connection lost: {}", e));
            }
            Message::Hello(msg) => {
                let chunk_parallelism = self
                    .active_session
                    .as_ref()
                    .map(|session| session.chunk_parallelism as usize)
                    .unwrap_or(self.config.chunk_load_parallelism as usize);
                let start_in_freecam = self
                    .active_session
                    .as_ref()
                    .map(|session| session.start_in_freecam)
                    .unwrap_or(true);
                let sim = Sim::new(msg.sim_config, chunk_parallelism, msg.character, start_in_freecam);
                if let Some(draw) = self.draw.as_mut() {
                    draw.configure(sim.cfg());
                }
                self.sim = Some(sim);
            }
            msg => {
                if let Some(sim) = self.sim.as_mut() {
                    sim.handle_net(msg);
                } else {
                    error!("Received game data before ServerHello");
                }
            }
        }
    }

    /// Draw a new frame
    fn draw(&mut self) {
        struct FrameRenderData {
            frame_id: u32,
            extent: vk::Extent2D,
            buffer: vk::Framebuffer,
            depth_view: vk::ImageView,
            present: vk::Semaphore,
            frustum: Frustum,
        }

        let (frame_data, paint, gui_action) = {
            let swapchain = self.swapchain.as_mut().unwrap();
            let draw = self.draw.as_mut().unwrap();
            unsafe {
                draw.wait();
                let frame_id = loop {
                    if self.swapchain_needs_update {
                        draw.wait_idle();
                        swapchain.update(&self.surface_fn, self.surface, self.window.inner_size());
                        self.swapchain_needs_update = false;
                    }
                    match swapchain.acquire_next_image(draw.image_acquired()) {
                        Ok((idx, suboptimal)) => {
                            self.swapchain_needs_update = suboptimal;
                            break idx;
                        }
                        Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                            self.swapchain_needs_update = true;
                        }
                        Err(e) => panic!("acquire_next_image: {e}"),
                    }
                };
                let extent = swapchain.state.extent;
                let aspect_ratio = extent.width as f32 / extent.height as f32;
                let frame = &swapchain.state.frames[frame_id as usize];
                let frustum = Frustum::from_vfov(f32::consts::FRAC_PI_4 * 1.2, aspect_ratio);
                self.yak
                    .set_surface_size([extent.width as f32, extent.height as f32].into());
                self.yak
                    .set_unscaled_viewport(yakui::geometry::Rect::from_pos_size(
                        Default::default(),
                        [extent.width as f32, extent.height as f32].into(),
                    ));
                self.yak.start();
                let action = self.gui_state.run(self.sim.as_ref());
                self.yak.finish();
                let paint = self.yak.paint();
                (
                    FrameRenderData {
                        frame_id,
                        extent,
                        buffer: frame.buffer,
                        depth_view: frame.depth_view,
                        present: frame.present,
                        frustum,
                    },
                    paint,
                    action,
                )
            }
        };

        let draw = self.draw.as_mut().unwrap();
        unsafe {
            draw.draw(
                self.sim.as_mut(),
                paint,
                frame_data.buffer,
                frame_data.depth_view,
                frame_data.extent,
                frame_data.present,
                &frame_data.frustum,
            );
        }

        let swapchain = self.swapchain.as_mut().unwrap();
        unsafe {
            match swapchain.queue_present(frame_data.frame_id) {
                Ok(false) => {}
                Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.swapchain_needs_update = true;
                }
                Err(e) => panic!("queue_present: {e}"),
            }
        }

        if let Some(action) = gui_action {
            self.handle_gui_action(action);
        }
    }

    fn handle_gui_action(&mut self, action: GuiAction) {
        match action {
            GuiAction::StartSession { session, raw_config } => {
                if self.active_session.is_some() {
                    match Config::save_raw_config(&self.config_path, &raw_config) {
                        Ok(()) => {
                            self.raw_config = raw_config;
                            self.gui_state
                                .set_info("Configuration saved for next launch.");
                        }
                        Err(err) => {
                            self.gui_state.set_error(format!(
                                "Failed to save configuration: {}",
                                err
                            ));
                        }
                    }
                    return;
                }

                match self.start_session(session, raw_config) {
                    Ok(()) => {
                        self.gui_state.mark_session_started();
                    }
                    Err(err) => {
                        self.gui_state.set_error(err);
                    }
                }
            }
        }
    }

    fn start_session(&mut self, session: SessionOptions, raw_config: RawConfig) -> Result<(), String> {
        Config::save_raw_config(&self.config_path, &raw_config)
            .map_err(|err| format!("Failed to save configuration: {}", err))?;

        let handle = match session.server {
            Some(server_addr) => net::spawn(net::Options {
                name: session.player_name.clone(),
                server: server_addr,
            }),
            None => self.start_local_session(&session)?,
        };

        self.net = Some(handle);
        self.active_session = Some(session);
        self.raw_config = raw_config;
        self.sim = None;
        self.input = InputState::default();
        self.last_frame = None;
        Ok(())
    }

    fn start_local_session(&self, session: &SessionOptions) -> Result<server::Handle, String> {
        let save_path = self.dirs.data_local_dir().join(&self.config.save);
        if let Some(parent) = save_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|err| format!("Failed to create save directory: {}", err))?;
        }
        let save = Save::open(&save_path, session.sim_config.chunk_size)
            .map_err(|err| format!("Couldn't open save: {}", err))?;

        info!("using save file {}", save_path.display());

        let mut server = Server::new(None, session.sim_config.clone(), save)
            .map_err(|err| format!("Failed to start server: {:#}", err))?;
        let (handle, backend) = server::Handle::loopback();
        let name = (*session.player_name).into();

        thread::spawn(move || {
            let runtime = Builder::new_current_thread()
                .enable_time()
                .build()
                .unwrap();
            let _guard = runtime.enter();
            if let Err(err) = server.connect(proto::ClientHello { name }, backend) {
                error!("failed to connect to local server: {}", err);
                return;
            }
            runtime.block_on(server.run().instrument(error_span!("server")));
            debug!("server thread terminated");
        });

        Ok(handle)
    }

    pub fn raw_config(&self) -> &RawConfig {
        &self.raw_config
    }
}

fn number_key_to_index(key: KeyCode) -> Option<usize> {
    match key {
        KeyCode::Digit1 => Some(0),
        KeyCode::Digit2 => Some(1),
        KeyCode::Digit3 => Some(2),
        KeyCode::Digit4 => Some(3),
        KeyCode::Digit5 => Some(4),
        KeyCode::Digit6 => Some(5),
        KeyCode::Digit7 => Some(6),
        KeyCode::Digit8 => Some(7),
        KeyCode::Digit9 => Some(8),
        KeyCode::Digit0 => Some(9),
        _ => None,
    }
}

impl Drop for Window {
    fn drop(&mut self) {
        self.draw.take();
        self.swapchain.take();
        unsafe {
            self.surface_fn.destroy_surface(self.surface, None);
        }
    }
}

struct SwapchainMgr {
    state: SwapchainState,
    format: vk::SurfaceFormatKHR,
}

impl SwapchainMgr {
    /// Construct a swapchain manager for a certain window
    fn new(window: &Window, gfx: Arc<Base>, fallback_size: PhysicalSize<u32>) -> Self {
        let device = &*gfx.device;
        let swapchain_fn = khr::swapchain::Device::new(&gfx.core.instance, device);
        let surface_formats = unsafe {
            window
                .surface_fn
                .get_physical_device_surface_formats(gfx.physical, window.surface)
                .unwrap()
        };
        let desired_format = vk::SurfaceFormatKHR {
            format: super::base::COLOR_FORMAT,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        };

        let desirable_format = |x: &vk::SurfaceFormatKHR| -> bool {
            x.format == desired_format.format && x.color_space == desired_format.color_space
        };

        if (surface_formats.len() != 1
            || (surface_formats[0].format != vk::Format::UNDEFINED
                || surface_formats[0].color_space != desired_format.color_space))
            && !surface_formats.iter().any(desirable_format)
        {
            panic!("no suitable surface format: {surface_formats:?}");
        }

        Self {
            state: unsafe {
                SwapchainState::new(
                    &window.surface_fn,
                    swapchain_fn,
                    gfx,
                    window.surface,
                    desired_format,
                    vk::SwapchainKHR::null(),
                    fallback_size,
                )
            },
            format: desired_format,
        }
    }

    /// Recreate the swapchain based on the window's current capabilities
    ///
    /// # Safety
    /// - There must be no operations scheduled that access the current swapchain
    unsafe fn update(
        &mut self,
        surface_fn: &khr::surface::Instance,
        surface: vk::SurfaceKHR,
        fallback_size: PhysicalSize<u32>,
    ) {
        unsafe {
            self.state = SwapchainState::new(
                surface_fn,
                self.state.swapchain_fn.clone(),
                self.state.gfx.clone(),
                surface,
                self.format,
                self.state.handle,
                fallback_size,
            );
        }
    }

    /// Get the index of the next frame to use
    unsafe fn acquire_next_image(&self, signal: vk::Semaphore) -> Result<(u32, bool), vk::Result> {
        unsafe {
            self.state.swapchain_fn.acquire_next_image(
                self.state.handle,
                u64::MAX,
                signal,
                vk::Fence::null(),
            )
        }
    }

    /// Present a frame on the window
    unsafe fn queue_present(&self, index: u32) -> Result<bool, vk::Result> {
        unsafe {
            self.state.swapchain_fn.queue_present(
                self.state.gfx.queue,
                &vk::PresentInfoKHR::default()
                    .wait_semaphores(&[self.state.frames[index as usize].present])
                    .swapchains(&[self.state.handle])
                    .image_indices(&[index]),
            )
        }
    }
}

/// Data that's replaced when the swapchain is updated
struct SwapchainState {
    gfx: Arc<Base>,
    swapchain_fn: khr::swapchain::Device,
    extent: vk::Extent2D,
    handle: vk::SwapchainKHR,
    frames: Vec<Frame>,
}

impl SwapchainState {
    unsafe fn new(
        surface_fn: &khr::surface::Instance,
        swapchain_fn: khr::swapchain::Device,
        gfx: Arc<Base>,
        surface: vk::SurfaceKHR,
        format: vk::SurfaceFormatKHR,
        old: vk::SwapchainKHR,
        fallback_size: PhysicalSize<u32>,
    ) -> Self {
        unsafe {
            let device = &*gfx.device;

            let surface_capabilities = surface_fn
                .get_physical_device_surface_capabilities(gfx.physical, surface)
                .unwrap();
            let extent = match surface_capabilities.current_extent.width {
                // If Vulkan doesn't know, winit probably does. Known to apply at least to Wayland.
                std::u32::MAX => vk::Extent2D {
                    width: fallback_size.width,
                    height: fallback_size.height,
                },
                _ => surface_capabilities.current_extent,
            };
            let pre_transform = if surface_capabilities
                .supported_transforms
                .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
            {
                vk::SurfaceTransformFlagsKHR::IDENTITY
            } else {
                surface_capabilities.current_transform
            };
            let present_modes = surface_fn
                .get_physical_device_surface_present_modes(gfx.physical, surface)
                .unwrap();
            let present_mode = present_modes
                .iter()
                .cloned()
                .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
                .unwrap_or(vk::PresentModeKHR::FIFO);

            let image_count = if surface_capabilities.max_image_count > 0 {
                surface_capabilities
                    .max_image_count
                    .min(surface_capabilities.min_image_count + 1)
            } else {
                surface_capabilities.min_image_count + 1
            };

            let handle = swapchain_fn
                .create_swapchain(
                    &vk::SwapchainCreateInfoKHR::default()
                        .surface(surface)
                        .min_image_count(image_count)
                        .image_color_space(format.color_space)
                        .image_format(format.format)
                        .image_extent(extent)
                        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .pre_transform(pre_transform)
                        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                        .present_mode(present_mode)
                        .clipped(true)
                        .image_array_layers(1)
                        .old_swapchain(old),
                    None,
                )
                .unwrap();

            let frames = swapchain_fn
                .get_swapchain_images(handle)
                .unwrap()
                .into_iter()
                .map(|image| {
                    let view = device
                        .create_image_view(
                            &vk::ImageViewCreateInfo::default()
                                .view_type(vk::ImageViewType::TYPE_2D)
                                .format(format.format)
                                .subresource_range(vk::ImageSubresourceRange {
                                    aspect_mask: vk::ImageAspectFlags::COLOR,
                                    base_mip_level: 0,
                                    level_count: 1,
                                    base_array_layer: 0,
                                    layer_count: 1,
                                })
                                .image(image),
                            None,
                        )
                        .unwrap();
                    gfx.set_name(view, cstr!("swapchain"));
                    let depth = DedicatedImage::new(
                        device,
                        &gfx.memory_properties,
                        &vk::ImageCreateInfo::default()
                            .image_type(vk::ImageType::TYPE_2D)
                            .format(vk::Format::D32_SFLOAT)
                            .extent(vk::Extent3D {
                                width: extent.width,
                                height: extent.height,
                                depth: 1,
                            })
                            .mip_levels(1)
                            .array_layers(1)
                            .samples(vk::SampleCountFlags::TYPE_1)
                            .usage(
                                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                                    | vk::ImageUsageFlags::INPUT_ATTACHMENT,
                            ),
                    );
                    gfx.set_name(depth.handle, cstr!("depth"));
                    gfx.set_name(depth.memory, cstr!("depth"));
                    let depth_view = device
                        .create_image_view(
                            &vk::ImageViewCreateInfo::default()
                                .image(depth.handle)
                                .view_type(vk::ImageViewType::TYPE_2D)
                                .format(vk::Format::D32_SFLOAT)
                                .subresource_range(vk::ImageSubresourceRange {
                                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                                    base_mip_level: 0,
                                    level_count: 1,
                                    base_array_layer: 0,
                                    layer_count: 1,
                                }),
                            None,
                        )
                        .unwrap();
                    gfx.set_name(depth_view, cstr!("depth"));
                    let present = device.create_semaphore(&Default::default(), None).unwrap();
                    gfx.set_name(present, cstr!("present"));
                    Frame {
                        view,
                        depth,
                        depth_view,
                        buffer: device
                            .create_framebuffer(
                                &vk::FramebufferCreateInfo::default()
                                    .render_pass(gfx.render_pass)
                                    .attachments(&[view, depth_view])
                                    .width(extent.width)
                                    .height(extent.height)
                                    .layers(1),
                                None,
                            )
                            .unwrap(),
                        present,
                    }
                })
                .collect();

            Self {
                swapchain_fn,
                gfx,
                extent,
                handle,
                frames,
            }
        }
    }
}

impl Drop for SwapchainState {
    fn drop(&mut self) {
        let device = &*self.gfx.device;
        unsafe {
            for frame in &mut self.frames {
                device.destroy_framebuffer(frame.buffer, None);
                device.destroy_image_view(frame.depth_view, None);
                device.destroy_image_view(frame.view, None);
                frame.depth.destroy(device);
                device.destroy_semaphore(frame.present, None);
            }
            self.swapchain_fn.destroy_swapchain(self.handle, None);
        }
    }
}

struct Frame {
    /// Image view for an entire swapchain image
    view: vk::ImageView,
    /// Depth buffer to use when rendering to this image
    depth: DedicatedImage,
    /// View thereof
    depth_view: vk::ImageView,
    /// Framebuffer referencing `view` and `depth_view`
    buffer: vk::Framebuffer,
    /// Semaphore used to ensure the frame isn't presented until rendering completes
    present: vk::Semaphore,
}

#[derive(Default)]
struct InputState {
    forward: bool,
    back: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
    jump: bool,
    clockwise: bool,
    anticlockwise: bool,
    mouse_captured: bool,
    // DEBUG: selected tile ID for placement (cycled with scroll wheel)
    selected_tile_id: u16,
}

impl InputState {
    fn movement(&self) -> na::Vector3<f32> {
        na::Vector3::new(
            self.right as u8 as f32 - self.left as u8 as f32,
            self.up as u8 as f32 - self.down as u8 as f32,
            self.back as u8 as f32 - self.forward as u8 as f32,
        )
    }

    fn roll(&self) -> f32 {
        self.anticlockwise as u8 as f32 - self.clockwise as u8 as f32
    }
}
