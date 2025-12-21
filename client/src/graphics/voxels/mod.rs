mod surface;
pub mod surface_extraction;
mod skylight;

#[cfg(test)]
mod tests;

use std::{sync::Arc, time::Instant};

use ash::{Device, vk};
use metrics::histogram;
use tracing::warn;

use crate::{
    Config, Loader, Sim,
    graphics::{Base, Frustum, TransparencyMap},
};
use common::{
    LruSlab,
    dodeca::{self, Vertex},
    graph::NodeId,
    lru_slab::SlotId,
    math::{MIsometry, MPoint},
    node::{Chunk, ChunkId, VoxelData},
};

use surface::Surface;
use surface_extraction::{DrawBuffer, ExtractTask, ScratchBuffer, SurfaceExtraction};
use skylight::SkylightShadow;

pub const SKYLIGHT_SOFT_EDGE_BLOCKS: f32 = 8.0;
pub const SKYLIGHT_TEXELS_PER_BLOCK: f32 = 2.0;

fn absolute_voxel_size(chunk_size: u8) -> f32 {
    // This matches the `common::sim_config::meters_to_absolute` derivation, but we only need the
    // absolute-unit size of a voxel for scaling the shadow map.
    let a = common::math::MVector::from(
        dodeca::Vertex::A.chunk_to_node() * na::Vector4::new(1.0, 0.5, 0.5, 1.0),
    )
    .normalized_point();
    let b = common::math::MVector::from(
        dodeca::Vertex::A.chunk_to_node() * na::Vector4::new(0.0, 0.5, 0.5, 1.0),
    )
    .normalized_point();
    let minimum_chunk_face_separation = a.distance(&b);
    minimum_chunk_face_separation / f32::from(chunk_size)
}

fn next_power_of_two_clamped(x: u32, min: u32, max: u32) -> u32 {
    let x = x.clamp(min, max);
    x.next_power_of_two().clamp(min, max)
}

pub struct Voxels {
    config: Arc<Config>,
    surface_extraction: SurfaceExtraction,
    extraction_scratch: ScratchBuffer,
    surfaces: DrawBuffer,
    states: LruSlab<SurfaceState>,
    draw: Surface,
    skylight: SkylightShadow,
    max_chunks: u32,
}

impl Voxels {
    pub fn new(
        gfx: &Base,
        config: Arc<Config>,
        loader: &mut Loader,
        dimension: u32,
        frames: u32,
        transparency_map: &TransparencyMap,
    ) -> Self {
        let max_faces = 3 * (dimension.pow(3) + dimension.pow(2));
        let max_supported_chunks = gfx.limits.max_storage_buffer_range / (8 * max_faces);
        let max_chunks = if MAX_CHUNKS > max_supported_chunks {
            warn!(
                "clamping max chunks to {} due to SSBO size limit",
                max_supported_chunks
            );
            max_supported_chunks
        } else {
            MAX_CHUNKS
        };
        let surfaces = DrawBuffer::new(gfx, max_chunks, dimension);
        // Skylight shadow-map resolution is derived from view distance so that texels-per-block is
        // roughly constant across configs.
        let view_distance_blocks = config.local_simulation.view_distance
            / absolute_voxel_size(config.local_simulation.chunk_size);
        let diameter_blocks = 2.0 * view_distance_blocks;
        let desired_res = (diameter_blocks * SKYLIGHT_TEXELS_PER_BLOCK).ceil() as u32;
        let res = next_power_of_two_clamped(desired_res, 512, 4096);
        let skylight = SkylightShadow::new(gfx, &surfaces, vk::Extent2D { width: res, height: res });

        let mut draw = Surface::new(gfx, loader, &surfaces);
        unsafe {
            draw.set_skylight_shadow_map(&*gfx.device, skylight.depth_view(), skylight.sampler());
        }
        let surface_extraction = SurfaceExtraction::new(gfx);
        let extraction_scratch = surface_extraction::ScratchBuffer::new(
            gfx,
            &surface_extraction,
            config.chunk_load_parallelism * frames,
            dimension,
            transparency_map,
        );
        Self {
            config,
            surface_extraction,
            extraction_scratch,
            surfaces,
            states: LruSlab::with_capacity(max_chunks),
            draw,
            skylight,
            max_chunks,
        }
    }

    /// Render the skylight shadow map (depth-only) for the currently drawn chunks.
    ///
    /// Must run before the main voxel pass, in the same command buffer, so the voxel
    /// fragment shader can sample the resulting depth texture.
    pub unsafe fn draw_skylight_shadow(
        &mut self,
        device: &Device,
        common_ds: vk::DescriptorSet,
        frame: &Frame,
        cmd: vk::CommandBuffer,
    ) {
        unsafe {
            self.skylight.draw(
                device,
                common_ds,
                self.surfaces.dimension(),
                cmd,
                &frame.surface,
                &self.surfaces,
                &frame.shadow_drawn,
            );
        }
    }

    /// Determine what to render and stage chunk transforms
    ///
    /// Surface extraction commands are written to `cmd`, and will be presumed complete for the next
    /// (not current) frame.
    pub unsafe fn prepare(
        &mut self,
        device: &Device,
        frame: &mut Frame,
        sim: &mut Sim,
        nearby_nodes: &[(NodeId, MIsometry<f32>)],
        cmd: vk::CommandBuffer,
        frustum: &Frustum,
    ) {
        // Clean up after previous frame
        for i in frame.extracted.drain(..) {
            self.extraction_scratch.free(i);
        }
        for chunk in frame.drawn.drain(..) {
            self.states.peek_mut(chunk).refcount -= 1;
        }
        for chunk in frame.shadow_drawn.drain(..) {
            self.states.peek_mut(chunk).refcount -= 1;
        }

        // Determine what to load/render
        let view = sim.view();
        if !sim.graph.contains(view.node) {
            // Graph is temporarily out of sync with the server; we don't know where we are, so
            // there's no point trying to draw.
            return;
        }
        let node_scan_started = Instant::now();
        let frustum_planes = frustum.planes();
        let local_to_view = view.local.inverse();
        let mut extractions = Vec::new();
        for &(node, ref node_transform) in nearby_nodes {
            let node_to_view = local_to_view * node_transform;
            let origin = node_to_view * MPoint::origin();
            let node_visible = frustum_planes.contain(&origin, dodeca::BOUNDING_SPHERE_RADIUS);

            use Chunk::*;
            for vertex in Vertex::iter() {
                let chunk = ChunkId::new(node, vertex);

                // Fetch existing chunk, or extract surface of new chunk
                let &mut Populated {
                    ref mut surface,
                    ref mut old_surface,
                    ref voxels,
                    ref light,
                    ref mut light_dirty,
                } = &mut sim.graph[chunk]
                else {
                    continue;
                };

                if let Some(slot) = surface.or(*old_surface) {
                    // Always include nearby opaque surfaces in the skylight occlusion pass.
                    // This avoids chunk-boundary shadow discontinuities when occluders are just
                    // outside the camera frustum.
                    self.states.get_mut(slot).refcount += 1;
                    frame.shadow_drawn.push(slot);
                    frame.surface.transforms_mut()[slot.0 as usize] =
                        na::Matrix4::from(*node_transform) * vertex.chunk_to_node();

                    // Render the surface in the main pass only if it's in the camera frustum.
                    if node_visible {
                        self.states.get_mut(slot).refcount += 1;
                        frame.drawn.push(slot);
                    }
                }
                
                // Check if we need to extract: either no surface, or light changed
                let needs_extraction = surface.is_none() || *light_dirty;
                
                if let (true, &VoxelData::Dense(ref data)) = (needs_extraction, voxels) {
                    // Extract a surface so it can be drawn in future frames
                    if frame.extracted.len() == self.config.chunk_load_parallelism as usize {
                        continue;
                    }
                    let removed = if self.states.len() == self.max_chunks {
                        let slot = self.states.lru().expect("full LRU table is nonempty");
                        if self.states.peek(slot).refcount != 0 {
                            warn!("MAX_CHUNKS is too small");
                            break;
                        }
                        Some((slot, self.states.remove(slot)))
                    } else {
                        None
                    };
                    let scratch_slot = self.extraction_scratch.alloc().expect(
                        "there are at least chunks_loaded_per_frame scratch slots per frame",
                    );
                    frame.extracted.push(scratch_slot);
                    let slot = self.states.insert(SurfaceState {
                        node,
                        chunk: vertex,
                        refcount: 0,
                    });
                    *surface = Some(slot);
                    *light_dirty = false;  // Mark light as processed
                    let storage = self.extraction_scratch.storage(scratch_slot);
                    storage.copy_from_slice(&data[..]);
                    // Copy light data to GPU staging buffer
                    let light_storage = self.extraction_scratch.light_storage(scratch_slot);
                    match light {
                        common::light::LightData::Uniform(value) => {
                            // Fill entire storage with the same light value
                            light_storage.fill(value.to_u16());
                        }
                        common::light::LightData::Dense(light_data) => {
                            // Copy dense light data
                            for (dst, src) in light_storage.iter_mut().zip(light_data.iter()) {
                                *dst = src.to_u16();
                            }
                        }
                    }
                    if let Some((lru_slot, lru)) = removed
                        && let Populated {
                            ref mut surface,
                            ref mut old_surface,
                            ..
                        } = sim.graph[lru.node].chunks[lru.chunk]
                    {
                        // Remove references to released slot IDs
                        if *surface == Some(lru_slot) {
                            *surface = None;
                        }
                        if *old_surface == Some(lru_slot) {
                            *old_surface = None;
                        }
                    }
                    let node_is_odd = sim.graph.depth(node) & 1 != 0;
                    extractions.push(ExtractTask {
                        index: scratch_slot,
                        indirect_offsets: self.surfaces.indirect_offsets(slot.0),
                        face_offsets: self.surfaces.face_offsets(slot.0),
                        draw_id: slot.0,
                        reverse_winding: vertex.parity() ^ node_is_odd,
                    });
                }
            }
        }
        unsafe {
            self.extraction_scratch.extract(
                device,
                &self.surface_extraction,
                self.surfaces.indirect_buffers(),
                self.surfaces.face_buffers(),
                cmd,
                &extractions,
            );
        }
        histogram!("frame.cpu.voxels.node_scan").record(node_scan_started.elapsed());
    }

    pub unsafe fn draw(
        &mut self,
        device: &Device,
        loader: &Loader,
        common_ds: vk::DescriptorSet,
        frame: &Frame,
        cmd: vk::CommandBuffer,
    ) {
        unsafe {
            let started = Instant::now();
            if !self.draw.bind(
                device,
                loader,
                self.surfaces.dimension(),
                common_ds,
                &frame.surface,
                cmd,
            ) {
                return;
            }

            // Draw in order: Opaque -> Cutout -> Translucent
            // This ensures proper depth handling and blending
            for class_idx in 0..surface_extraction::TRANSPARENCY_CLASS_COUNT {
                self.draw
                    .bind_transparency_class(device, common_ds, cmd, class_idx);
                for chunk in &frame.drawn {
                    self.draw
                        .draw(device, cmd, &self.surfaces, chunk.0, class_idx);
                }
            }

            histogram!("frame.cpu.voxels.draw").record(started.elapsed());
        }
    }

    pub unsafe fn destroy(&mut self, device: &Device) {
        unsafe {
            self.surface_extraction.destroy(device);
            self.extraction_scratch.destroy(device);
            self.surfaces.destroy(device);
            self.draw.destroy(device);
            self.skylight.destroy(device);
        }
    }
}

pub struct Frame {
    surface: surface::Frame,
    /// Scratch slots completed in this frame
    extracted: Vec<u32>,
    drawn: Vec<SlotId>,
    /// Chunks drawn into the skylight shadow map (broader than frustum-culling).
    shadow_drawn: Vec<SlotId>,
}

impl Frame {
    pub unsafe fn destroy(&mut self, device: &Device) {
        unsafe {
            self.surface.destroy(device);
        }
    }
}

impl Frame {
    pub fn new(gfx: &Base, ctx: &Voxels) -> Self {
        Self {
            surface: surface::Frame::new(gfx, ctx.states.capacity()),
            extracted: Vec::new(),
            drawn: Vec::new(),
            shadow_drawn: Vec::new(),
        }
    }
}

/// Maximum number of concurrently drawn voxel chunks
const MAX_CHUNKS: u32 = 8192;

struct SurfaceState {
    node: NodeId,
    chunk: common::dodeca::Vertex,
    refcount: u32,
}
