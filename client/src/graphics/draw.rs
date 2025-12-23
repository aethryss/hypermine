use std::sync::Arc;
use std::time::Instant;

use ash::vk;
use common::traversal;
use common::{
    dodeca,
    graph::NodeId,
    math::{MDirection, MIsometry, MPoint, MVector},
};
use lahar::Staged;
use metrics::histogram;

use super::{Base, Fog, Frustum, GltfScene, Meshes, TransparencyMap, Voxels, fog, voxels};
use crate::{Asset, Config, Loader, Sim};
use common::SimConfig;
use common::node::ChunkId;
use common::proto::{Character, Position};

/// Manages rendering, independent of what is being rendered to
pub struct Draw {
    gfx: Arc<Base>,
    cfg: Arc<Config>,
    /// Used to allocate the command buffers we render with
    cmd_pool: vk::CommandPool,
    /// Allows accurate frame timing information to be recorded
    timestamp_pool: vk::QueryPool,
    /// State that varies per frame in flight
    states: Vec<State>,
    /// The index of the next element of `states` to use
    next_state: usize,
    /// A reference time
    epoch: Instant,
    /// The lowest common denominator between the interfaces of our graphics pipelines
    ///
    /// Represents e.g. the binding for common uniforms
    common_pipeline_layout: vk::PipelineLayout,
    /// Descriptor pool from which descriptor sets shared between many pipelines are allocated
    common_descriptor_pool: vk::DescriptorPool,

    /// Drives async asset loading
    loader: Loader,

    //
    // Rendering pipelines
    //
    /// Populated after connect, once the voxel configuration is known
    voxels: Option<Voxels>,
    meshes: Meshes,
    fog: Fog,

    /// Reusable storage for barriers that prevent races between image upload and read
    image_barriers: Vec<vk::ImageMemoryBarrier<'static>>,
    /// Reusable storage for barriers that prevent races between buffer upload and read
    buffer_barriers: Vec<vk::BufferMemoryBarrier<'static>>,

    /// Yakui Vulkan context
    yakui_vulkan: yakui_vulkan::YakuiVulkan,

    /// Miscellany
    character_model: Asset<GltfScene>,

    /// Cached skylight uniforms, stabilized by camera chunk.
    skylight_cache: Option<SkylightCache>,
}

#[derive(Copy, Clone)]
struct SkylightCache {
    anchor_chunk: ChunkId,
    skylight_view_projection: na::Matrix4<f32>,
    skylight_params: na::Vector4<f32>,
    skylight_bounds: na::Vector4<f32>,
}

/// Maximum number of simultaneous frames in flight
const PIPELINE_DEPTH: u32 = 2;
const TIMESTAMPS_PER_FRAME: u32 = 3;

impl Draw {
    pub fn new(gfx: Arc<Base>, cfg: Arc<Config>) -> Self {
        let device = &*gfx.device;
        unsafe {
            // Allocate a command buffer for each frame state
            let cmd_pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(gfx.queue_family)
                        .flags(
                            vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER
                                | vk::CommandPoolCreateFlags::TRANSIENT,
                        ),
                    None,
                )
                .unwrap();
            let cmds = device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(cmd_pool)
                        .command_buffer_count(2 * PIPELINE_DEPTH),
                )
                .unwrap();

            let timestamp_pool = device
                .create_query_pool(
                    &vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::TIMESTAMP)
                        .query_count(TIMESTAMPS_PER_FRAME * PIPELINE_DEPTH),
                    None,
                )
                .unwrap();
            gfx.set_name(timestamp_pool, cstr!("timestamp pool"));

            let common_pipeline_layout = device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().set_layouts(&[gfx.common_layout]),
                    None,
                )
                .unwrap();

            // Allocate descriptor sets for data used by all graphics pipelines (e.g. common
            // uniforms)
            let common_descriptor_pool = device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(PIPELINE_DEPTH)
                        .pool_sizes(&[
                            vk::DescriptorPoolSize {
                                ty: vk::DescriptorType::UNIFORM_BUFFER,
                                descriptor_count: PIPELINE_DEPTH,
                            },
                            vk::DescriptorPoolSize {
                                ty: vk::DescriptorType::INPUT_ATTACHMENT,
                                // 2 input attachments per frame: scene color + depth
                                descriptor_count: PIPELINE_DEPTH * 2,
                            },
                        ]),
                    None,
                )
                .unwrap();
            let common_ds = device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(common_descriptor_pool)
                        .set_layouts(&vec![gfx.common_layout; PIPELINE_DEPTH as usize]),
                )
                .unwrap();

            let mut loader = Loader::new(cfg.clone(), gfx.clone());

            // Construct the per-frame states
            let states = cmds
                .chunks(2)
                .zip(common_ds)
                .map(|(cmds, common_ds)| {
                    let uniforms = Staged::new(
                        device,
                        &gfx.memory_properties,
                        vk::BufferUsageFlags::UNIFORM_BUFFER,
                    );
                    device.update_descriptor_sets(
                        &[vk::WriteDescriptorSet::default()
                            .dst_set(common_ds)
                            .dst_binding(0)
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                            .buffer_info(&[vk::DescriptorBufferInfo {
                                buffer: uniforms.buffer(),
                                offset: 0,
                                range: vk::WHOLE_SIZE,
                            }])],
                        &[],
                    );
                    let x = State {
                        cmd: cmds[0],
                        post_cmd: cmds[1],
                        common_ds,
                        image_acquired: device.create_semaphore(&Default::default(), None).unwrap(),
                        fence: device
                            .create_fence(
                                &vk::FenceCreateInfo::default()
                                    .flags(vk::FenceCreateFlags::SIGNALED),
                                None,
                            )
                            .unwrap(),
                        uniforms,
                        used: false,
                        in_flight: false,

                        voxels: None,
                    };
                    gfx.set_name(x.cmd, cstr!("frame"));
                    gfx.set_name(x.post_cmd, cstr!("post-frame"));
                    gfx.set_name(x.image_acquired, cstr!("image acquired"));
                    gfx.set_name(x.fence, cstr!("render complete"));
                    gfx.set_name(x.uniforms.buffer(), cstr!("uniforms"));
                    x
                })
                .collect();

            let meshes = Meshes::new(&gfx, loader.ctx().mesh_ds_layout);

            let fog = Fog::new(&gfx);

            gfx.save_pipeline_cache();

            let mut yakui_vulkan_options = yakui_vulkan::Options::default();
            yakui_vulkan_options.render_pass = gfx.render_pass;
            yakui_vulkan_options.subpass = 1;
            let mut yakui_vulkan = yakui_vulkan::YakuiVulkan::new(
                &yakui_vulkan::VulkanContext::new(device, gfx.queue, gfx.memory_properties),
                yakui_vulkan_options,
            );
            for _ in 0..PIPELINE_DEPTH {
                yakui_vulkan.transfers_submitted();
            }

            let character_model = loader.load(
                "character model",
                super::GlbFile {
                    path: "character.glb".into(),
                },
            );

            Self {
                gfx,
                cfg,
                cmd_pool,
                timestamp_pool,
                states,
                next_state: 0,
                epoch: Instant::now(),
                common_pipeline_layout,
                common_descriptor_pool,

                loader,

                voxels: None,
                meshes,
                fog,

                buffer_barriers: Vec::new(),
                image_barriers: Vec::new(),

                yakui_vulkan,

                character_model,

                skylight_cache: None,
            }
        }
    }

    /// Called with server-defined world parameters once they're known
    pub fn configure(&mut self, cfg: &SimConfig) {
        // Load transparency map synchronously - we need it before creating voxel buffers
        let transparency_map = TransparencyMap::from_atlas(std::path::Path::new(
            "assets/materials/terrain.png",
        ))
        .unwrap_or_else(|e| {
            tracing::warn!("Failed to load transparency map: {}, using default", e);
            TransparencyMap::default()
        });

        let voxels = Voxels::new(
            &self.gfx,
            self.cfg.clone(),
            &mut self.loader,
            u32::from(cfg.chunk_size),
            PIPELINE_DEPTH,
            &transparency_map,
        );
        for state in &mut self.states {
            state.voxels = Some(voxels::Frame::new(&self.gfx, &voxels));
        }
        self.voxels = Some(voxels);
    }

    /// Waits for a frame's worth of resources to become available for use in rendering a new frame
    ///
    /// Call before signaling the image_acquired semaphore or invoking `draw`.
    pub unsafe fn wait(&mut self) {
        unsafe {
            let device = &*self.gfx.device;
            let state = &mut self.states[self.next_state];
            device.wait_for_fences(&[state.fence], true, !0).unwrap();
            self.yakui_vulkan
                .transfers_finished(&yakui_vulkan::VulkanContext::new(
                    device,
                    self.gfx.queue,
                    self.gfx.memory_properties,
                ));
            state.in_flight = false;
        }
    }

    /// Semaphore that must be signaled when an output framebuffer can be rendered to
    ///
    /// Don't signal until after `wait`ing; call before `draw`
    pub fn image_acquired(&self) -> vk::Semaphore {
        self.states[self.next_state].image_acquired
    }

    /// Submit commands to the GPU to draw a frame
    ///
    /// `framebuffer` must have swapchain color, scene color, and depth buffers attached and have
    /// the dimensions specified in `extent`. The `present` semaphore is signaled when rendering
    /// is complete and the color image can be presented.
    ///
    /// Submits commands that wait on `image_acquired` before writing to `framebuffer`'s color
    /// attachment.
    #[allow(clippy::too_many_arguments)] // Every argument is of a different type, making this less of a problem.
    pub unsafe fn draw(
        &mut self,
        mut sim: Option<&mut Sim>,
        yakui_paint_dom: &yakui::paint::PaintDom,
        framebuffer: vk::Framebuffer,
        scene_color_view: vk::ImageView,
        depth_view: vk::ImageView,
        extent: vk::Extent2D,
        present: vk::Semaphore,
        frustum: &Frustum,
    ) {
        unsafe {
            let draw_started = Instant::now();
            let view = sim.as_ref().map_or_else(Position::origin, |sim| sim.view());
            let projection = frustum.projection(1.0e-4);
            let view_projection = projection.matrix() * na::Matrix4::from(view.local.inverse());
            self.loader.drive();

            let device = &*self.gfx.device;
            let state_index = self.next_state;
            let state = &mut self.states[self.next_state];
            let cmd = state.cmd;

            let yakui_vulkan_context = yakui_vulkan::VulkanContext::new(
                device,
                self.gfx.queue,
                self.gfx.memory_properties,
            );

            // We're using this state again, so put the fence back in the unsignaled state and compute
            // the next frame to use
            device.reset_fences(&[state.fence]).unwrap();
            self.next_state = (self.next_state + 1) % PIPELINE_DEPTH as usize;

            // Set up framebuffer attachments for the composite pass (subpass 1)
            // Binding 1: scene color, Binding 2: depth
            device.update_descriptor_sets(
                &[
                    vk::WriteDescriptorSet::default()
                        .dst_set(state.common_ds)
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::INPUT_ATTACHMENT)
                        .image_info(&[vk::DescriptorImageInfo {
                            sampler: vk::Sampler::null(),
                            image_view: scene_color_view,
                            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        }]),
                    vk::WriteDescriptorSet::default()
                        .dst_set(state.common_ds)
                        .dst_binding(2)
                        .descriptor_type(vk::DescriptorType::INPUT_ATTACHMENT)
                        .image_info(&[vk::DescriptorImageInfo {
                            sampler: vk::Sampler::null(),
                            image_view: depth_view,
                            image_layout: vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                        }]),
                ],
                &[],
            );

            // Handle completed queries
            let first_query = state_index as u32 * TIMESTAMPS_PER_FRAME;
            if state.used {
                // Collect timestamps from the last time we drew this frame
                let mut queries = [0u64; TIMESTAMPS_PER_FRAME as usize];
                // `WAIT` is guaranteed not to block here because `Self::draw` is only called after
                // `Self::wait` ensures that the prior instance of this frame is complete.
                device
                    .get_query_pool_results(
                        self.timestamp_pool,
                        first_query,
                        &mut queries,
                        vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                    )
                    .unwrap();
                let draw_seconds = self.gfx.limits.timestamp_period as f64
                    * 1e-9
                    * (queries[1] - queries[0]) as f64;
                let after_seconds = self.gfx.limits.timestamp_period as f64
                    * 1e-9
                    * (queries[2] - queries[1]) as f64;
                histogram!("frame.gpu.draw").record(draw_seconds);
                histogram!("frame.gpu.after_draw").record(after_seconds);
            }

            device
                .begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .unwrap();
            device
                .begin_command_buffer(
                    state.post_cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .unwrap();

            device.cmd_reset_query_pool(
                cmd,
                self.timestamp_pool,
                first_query,
                TIMESTAMPS_PER_FRAME,
            );
            let mut timestamp_index = first_query;
            device.cmd_write_timestamp(
                cmd,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.timestamp_pool,
                timestamp_index,
            );
            timestamp_index += 1;

            self.yakui_vulkan
                .transfer(yakui_paint_dom, &yakui_vulkan_context, cmd);

            // Schedule transfer of uniform data. Note that we defer actually preparing the data to just
            // before submitting the command buffer so time-sensitive values can be set with minimum
            // latency.
            state.uniforms.record_transfer(device, cmd);
            self.buffer_barriers.push(
                vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::UNIFORM_READ)
                    .buffer(state.uniforms.buffer())
                    .size(vk::WHOLE_SIZE),
            );

            let nearby_nodes_started = Instant::now();
            let nearby_nodes = if let Some(sim) = sim.as_deref() {
                traversal::nearby_nodes(&sim.graph, &view, self.cfg.local_simulation.view_distance)
            } else {
                vec![]
            };
            histogram!("frame.cpu.nearby_nodes").record(nearby_nodes_started.elapsed());

            // Stabilize skylight projection by the camera's current chunk (node + closest vertex).
            // This prevents shadowmap jitter during movement within a chunk.
            // Additionally, compute skylight uniforms from an unoriented view so camera rotation
            // doesn't rotate the shadow projection.
            let skylight_view = sim
                .as_deref()
                .map_or(view, |s| s.view_unoriented());

            let view_vertex = traversal::closest_vertex_to_position_local(skylight_view.local);
            let anchor_chunk = ChunkId::new(skylight_view.node, view_vertex);
            let (skylight_view_projection, skylight_params, skylight_bounds) =
                if let Some(cache) = self.skylight_cache
                    && cache.anchor_chunk == anchor_chunk
                {
                    (
                        cache.skylight_view_projection,
                        cache.skylight_params,
                        cache.skylight_bounds,
                    )
                } else {
                    let (svp, sp, sb) =
                        compute_skylight_uniforms(sim.as_deref(), &skylight_view, &nearby_nodes);
                    self.skylight_cache = Some(SkylightCache {
                        anchor_chunk,
                        skylight_view_projection: svp,
                        skylight_params: sp,
                        skylight_bounds: sb,
                    });
                    (svp, sp, sb)
                };

            if let (Some(voxels), Some(sim)) = (self.voxels.as_mut(), sim.as_mut()) {
                voxels.prepare(
                    device,
                    state.voxels.as_mut().unwrap(),
                    sim,
                    &nearby_nodes,
                    state.post_cmd,
                    frustum,
                );
            }

            // Ensure reads of just-transferred memory wait until it's ready
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::VERTEX_SHADER | vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::default(),
                &[],
                &self.buffer_barriers,
                &self.image_barriers,
            );
            self.buffer_barriers.clear();
            self.image_barriers.clear();

            // Skylight shadow-map prepass (depth-only) so voxel shading can sample it.
            if let Some(ref mut voxels) = self.voxels {
                voxels.draw_skylight_shadow(
                    device,
                    state.common_ds,
                    state.voxels.as_ref().unwrap(),
                    cmd,
                );
            }

            device.cmd_begin_render_pass(
                cmd,
                &vk::RenderPassBeginInfo::default()
                    .render_pass(self.gfx.render_pass)
                    .framebuffer(framebuffer)
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D::default(),
                        extent,
                    })
                    // Clear values for: [swapchain, scene_color, depth]
                    .clear_values(&[
                        // Attachment 0: Swapchain - don't care, will be overwritten by composite
                        vk::ClearValue {
                            color: vk::ClearColorValue {
                                float32: [0.0, 0.0, 0.0, 1.0],
                            },
                        },
                        // Attachment 1: Scene color - clear to transparent black
                        // Alpha=0 means "no geometry here, show sky"
                        vk::ClearValue {
                            color: vk::ClearColorValue {
                                float32: [0.0, 0.0, 0.0, 0.0],
                            },
                        },
                        // Attachment 2: Depth - clear to 0 (reverse-Z, far plane)
                        vk::ClearValue {
                            depth_stencil: vk::ClearDepthStencilValue {
                                depth: 0.0,
                                stencil: 0,
                            },
                        },
                    ]),
                vk::SubpassContents::INLINE,
            );

            // Set up common dynamic state
            let viewports = [vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: extent.width as f32,
                height: extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            }];
            let scissors = [vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: extent.width,
                    height: extent.height,
                },
            }];
            device.cmd_set_viewport(cmd, 0, &viewports);
            device.cmd_set_scissor(cmd, 0, &scissors);

            // Record the actual rendering commands
            if let Some(ref mut voxels) = self.voxels {
                voxels.draw(
                    device,
                    &self.loader,
                    state.common_ds,
                    state.voxels.as_ref().unwrap(),
                    cmd,
                );
            }

            if let Some(sim) = sim.as_deref() {
                for (node, transform) in nearby_nodes {
                    for &entity in sim.graph_entities.get(node) {
                        if sim.local_character == Some(entity) {
                            // Don't draw ourself
                            continue;
                        }
                        let pos = sim
                            .world
                            .get::<&Position>(entity)
                            .expect("positionless entity in graph");
                        if let Some(character_model) = self.loader.get(self.character_model)
                            && let Ok(ch) = sim.world.get::<&Character>(entity)
                        {
                            let transform = na::Matrix4::from(transform * pos.local)
                                * na::Matrix4::new_scaling(sim.cfg().meters_to_absolute)
                                * ch.state.orientation.to_homogeneous();
                            for mesh in &character_model.0 {
                                self.meshes
                                    .draw(device, state.common_ds, cmd, mesh, &transform);
                            }
                        }
                    }
                }
            }

            device.cmd_next_subpass(cmd, vk::SubpassContents::INLINE);

            self.fog.draw(device, state.common_ds, cmd);

            self.yakui_vulkan
                .paint(yakui_paint_dom, &yakui_vulkan_context, cmd, extent);

            // Finish up
            device.cmd_end_render_pass(cmd);
            device.cmd_write_timestamp(
                cmd,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.timestamp_pool,
                timestamp_index,
            );
            timestamp_index += 1;
            device.end_command_buffer(cmd).unwrap();

            device.cmd_write_timestamp(
                state.post_cmd,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.timestamp_pool,
                timestamp_index,
            );
            device.end_command_buffer(state.post_cmd).unwrap();

            // Specify the uniform data before actually submitting the command to transfer it
            // inverse_view transforms from view space to node-local space (the camera's orientation)
            let inverse_view = na::Matrix4::from(view.local);
            // Get up and north directions for the sky, in VIEW SPACE.
            // Up: from the node's surface normal
            // North: from the "compass_forward" which experiences holonomy but doesn't rotate with camera
            let (world_up, world_north) = sim
                .as_ref()
                .and_then(|s| {
                    let node_state = s.graph[view.node].state.as_ref()?;
                    let view_from_node = view.local.inverse();

                    // Up from the node's surface normal
                    let local_up = node_state.up_direction();
                    let view_up = &view_from_node * local_up;
                    let up3: na::Vector3<f32> = view_up.as_ref().xyz().normalize();

                    // North from the compass forward direction (position-local space)
                    // compass_forward is in position-local space (before orientation)
                    // To get to view space, we apply the inverse of the camera orientation
                    let compass_fwd = s.compass_forward();
                    let orientation = s.camera_orientation();
                    // orientation maps position-local to view, so inverse maps view to position-local
                    // Therefore: view_dir = orientation * position_local_dir
                    let north3 = (orientation.inverse() * compass_fwd.as_ref()).normalize();

                    Some((
                        na::Vector4::new(up3.x, up3.y, up3.z, 0.0),
                        na::Vector4::new(north3.x, north3.y, north3.z, 0.0),
                    ))
                })
                .unwrap_or((
                    na::Vector4::new(0.0, 1.0, 0.0, 0.0),
                    na::Vector4::new(0.0, 0.0, -1.0, 0.0),
                ));
            state.uniforms.write(Uniforms {
                view_projection,
                inverse_projection: *projection.inverse().matrix(),
                inverse_view,
                skylight_view_projection,
                skylight_params,
                skylight_bounds,
                world_up,
                world_north,
                fog_density: fog::density(self.cfg.local_simulation.fog_distance, 1e-3, 5.0),
                time: self.epoch.elapsed().as_secs_f32(),
            });

            // Submit the commands to the GPU
            device
                .queue_submit(
                    self.gfx.queue,
                    &[
                        vk::SubmitInfo::default()
                            .command_buffers(&[cmd])
                            .wait_semaphores(&[state.image_acquired])
                            .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
                            .signal_semaphores(&[present]),
                        vk::SubmitInfo::default().command_buffers(&[state.post_cmd]),
                    ],
                    state.fence,
                )
                .unwrap();
            self.yakui_vulkan.transfers_submitted();
            state.used = true;
            state.in_flight = true;
            histogram!("frame.cpu").record(draw_started.elapsed());
        }
    }

    /// Wait for all drawing to complete
    ///
    /// Useful to e.g. ensure it's safe to deallocate an image that's being rendered to
    pub fn wait_idle(&self) {
        let device = &*self.gfx.device;
        for state in &self.states {
            unsafe {
                device.wait_for_fences(&[state.fence], true, !0).unwrap();
            }
        }
    }
}

fn compute_skylight_uniforms(
    sim: Option<&Sim>,
    view: &Position,
    nearby_nodes: &[(NodeId, MIsometry<f32>)],
) -> (na::Matrix4<f32>, na::Vector4<f32>, na::Vector4<f32>) {
    // Default: disabled skylight
    let mut skylight_params = na::Vector4::new(0.0, 0.0, 0.001, 0.0);
    let mut skylight_view_projection = na::Matrix4::identity();
    let mut skylight_bounds = na::Vector4::new(0.25, 0.25, 0.0, 0.0);

    let Some(sim) = sim else {
        return (skylight_view_projection, skylight_params, skylight_bounds);
    };
    if !sim.graph.contains(view.node) {
        return (skylight_view_projection, skylight_params, skylight_bounds);
    }
    let node_state = match sim.graph[view.node].state.as_ref() {
        Some(s) => s,
        None => return (skylight_view_projection, skylight_params, skylight_bounds),
    };

    // View-from-node (camera transform). This matches how `view_projection` is built.
    let view_from_node = view.local.inverse();

    // Down direction is the surface normal pointing into terrain.
    // Prefer the actual Minkowski-space plane normal so the Fermi frame stays consistent with
    // the hyperbolic horizon geometry.
    let down_in_node: MVector<f32> = -node_state.up_direction();
    let down_in_view: MVector<f32> = view_from_node * down_in_node;

    // The plane normal should be spacelike (direction-like), but in rare numerical/edge cases it
    // can become non-direction-like. Guard to avoid panicking.
    let down_dir: MDirection<f32> = if down_in_view.mip(&down_in_view) > 1.0e-6 {
        down_in_view.normalized_direction()
    } else {
        // Fallback: use the engine's robust relative-up (Euclidean XYZ normalization).
        // This is only used when the Minkowski normal becomes degenerate.
        let Some(up_view) = sim.graph.get_relative_up(view) else {
            return (skylight_view_projection, skylight_params, skylight_bounds);
        };
        let down_view = na::UnitVector3::new_unchecked(-up_view.into_inner());
        down_view.into()
    };

    // Build a Minkowski isometry that maps `down_dir` -> +X.
    // Using two reflections: R_b * R_(a+b) where a maps to b.
    // If `a` is almost `-b`, then `a+b` is near zero and cannot be normalized; handle that case
    // by reflecting about `b` directly (which maps -b to b).
    let target = MDirection::x();
    let sum: MVector<f32> = down_dir.as_ref() + target.as_ref();
    let fermi_from_view = if sum.mip(&sum) <= 1.0e-6 {
        common::math::MIsometry::reflection(&target)
    } else {
        let sum_dir = sum.normalized_direction();
        common::math::MIsometry::reflection(&target) * common::math::MIsometry::reflection(&sum_dir)
    };

    // Determine ortho bounds in Klein coords by scanning the node centers we will render.
    // Inflate by the node bounding sphere radius converted to Klein radius.
    let mut max_u = 0.25f32;
    let mut max_v = 0.25f32;
    for &(_node, node_to_viewnode) in nearby_nodes {
        let p_view = view_from_node * node_to_viewnode * MPoint::origin();
        let p_fermi = fermi_from_view * p_view;
        // Homogeneous sign is arbitrary: use a positive w for stable Klein projections.
        let mut v: na::Vector4<f32> = p_fermi.into();
        if v.w < 0.0 {
            v = -v;
        }
        let w = v.w.max(1.0e-6);
        max_u = max_u.max((v.y / w).abs());
        max_v = max_v.max((v.z / w).abs());
    }
    let inflate = libm::tanhf(dodeca::BOUNDING_SPHERE_RADIUS);
    // NOTE: (y/w, z/w) are Klein-model coordinates and approach 1.0 at infinity.
    // Clamping too aggressively here causes the shadow-map to stop covering distant
    // (but still rendered) chunks. We keep a tiny safety margin below 1.0 to avoid
    // numerical blow-ups near the ideal boundary.
    const KLEIN_MAX: f32 = 0.9995;
    max_u = (max_u + inflate).min(KLEIN_MAX).max(0.05);
    max_v = (max_v + inflate).min(KLEIN_MAX).max(0.05);

    skylight_bounds.x = max_u;
    skylight_bounds.y = max_v;

    // Store just the node->Fermi transform. Shaders will compute Klein ratios explicitly,
    // which avoids artifacts from using a projective matrix with varying clip.w.
    skylight_view_projection =
        na::Matrix4::from(fermi_from_view) * na::Matrix4::from(view_from_node);

    // Defaults: moderately bright skylight, strong shadows.
    skylight_params.x = 0.85; // intensity
    skylight_params.y = 0.85; // shadow strength
    skylight_params.z = 0.0015; // bias

    (skylight_view_projection, skylight_params, skylight_bounds)
}

impl Drop for Draw {
    fn drop(&mut self) {
        let device = &*self.gfx.device;
        unsafe {
            for state in &mut self.states {
                if state.in_flight {
                    device.wait_for_fences(&[state.fence], true, !0).unwrap();
                    state.in_flight = false;
                }
                device.destroy_semaphore(state.image_acquired, None);
                device.destroy_fence(state.fence, None);
                state.uniforms.destroy(device);
                if let Some(mut voxels) = state.voxels.take() {
                    voxels.destroy(device);
                }
            }
            self.yakui_vulkan.cleanup(&self.gfx.device);
            device.destroy_command_pool(self.cmd_pool, None);
            device.destroy_query_pool(self.timestamp_pool, None);
            device.destroy_descriptor_pool(self.common_descriptor_pool, None);
            device.destroy_pipeline_layout(self.common_pipeline_layout, None);
            self.fog.destroy(device);
            self.meshes.destroy(device);
            if let Some(mut voxels) = self.voxels.take() {
                voxels.destroy(device);
            }
        }
    }
}

struct State {
    /// Semaphore signaled by someone else to indicate that output to the framebuffer can begin
    image_acquired: vk::Semaphore,
    /// Fence signaled when this state is no longer in use
    fence: vk::Fence,
    /// Command buffer we record the frame's rendering onto
    cmd: vk::CommandBuffer,
    /// Work performed after rendering, overlapping with the next frame's CPU work
    post_cmd: vk::CommandBuffer,
    /// Descriptor set for graphics-pipeline-independent data
    common_ds: vk::DescriptorSet,
    /// The common uniform buffer
    uniforms: Staged<Uniforms>,
    /// Whether this state has been previously used
    ///
    /// Indicates that e.g. valid timestamps are associated with this query
    used: bool,
    /// Whether this state is currently being accessed by the GPU
    ///
    /// True for the period between `cmd` being submitted and `fence` being waited.
    in_flight: bool,

    // Per-pipeline states
    voxels: Option<voxels::Frame>,
}

/// Data stored in the common uniform buffer
///
/// Alignment and padding must be manually managed to match the std140 ABI as expected by the
/// shaders.
#[repr(C)]
#[derive(Copy, Clone)]
struct Uniforms {
    /// Camera projection matrix
    view_projection: na::Matrix4<f32>,
    inverse_projection: na::Matrix4<f32>,
    /// Maps view space to world space (camera orientation)
    inverse_view: na::Matrix4<f32>,

    /// Skylight shadow-map projection + parameters
    skylight_view_projection: na::Matrix4<f32>,
    skylight_params: na::Vector4<f32>,
    skylight_bounds: na::Vector4<f32>,
    /// World up direction in view space (xyz, w unused for alignment)
    world_up: na::Vector4<f32>,
    /// World north direction in view space (xyz, w unused for alignment)
    world_north: na::Vector4<f32>,
    fog_density: f32,
    /// Cycles through [0,1) once per second for simple animation effects
    time: f32,
}
