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
        // The voxel surface buffers can get *very* large for higher chunk dimensions.
        // We must cap the number of resident chunks based on:
        // 1) `max_storage_buffer_range` (we bind the whole SSBO as a descriptor range), and
        // 2) a conservative fraction of device-local memory to avoid VK_ERROR_OUT_OF_DEVICE_MEMORY.
        //
        // Keep these constants in sync with `surface_extraction.rs`.
        const FACE_SIZE: u64 = 12; // bytes per Surface
        const INDIRECT_SIZE: u64 = 16; // bytes per VkDrawIndirectCommand

        fn round_up(value: u64, alignment: u64) -> u64 {
            value.div_ceil(alignment) * alignment
        }

        fn device_local_heap_size(props: &vk::PhysicalDeviceMemoryProperties) -> u64 {
            let mut max_heap = 0u64;
            for i in 0..(props.memory_type_count as usize) {
                let mt = props.memory_types[i];
                if mt
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                {
                    let heap = props.memory_heaps[mt.heap_index as usize];
                    max_heap = max_heap.max(heap.size);
                }
            }
            max_heap
        }

        let max_faces = 3u64 * (u64::from(dimension).pow(3) + u64::from(dimension).pow(2));
        let face_buffer_unit = round_up(
            max_faces * FACE_SIZE,
            gfx.limits.min_storage_buffer_offset_alignment as u64,
        );

        // SSBO range limit (buffer size must be <= maxStorageBufferRange because we bind WHOLE_SIZE).
        let max_by_range = (gfx.limits.max_storage_buffer_range as u64 / face_buffer_unit).max(1);

        // Conservative VRAM budget: keep voxel buffers to <= 25% of the largest device-local heap.
        // This avoids hard failures on GPUs with smaller VRAM or when other apps are using VRAM.
        let heap_bytes = device_local_heap_size(&gfx.memory_properties);
        let vram_budget = heap_bytes / 4;
        let classes = surface_extraction::TRANSPARENCY_CLASS_COUNT as u64;
        let bytes_per_chunk = classes * (face_buffer_unit + INDIRECT_SIZE)
            + (surface::TRANSFORM_SIZE as u64) /* transforms buffer is separate but scales with count */;
        let max_by_vram = if bytes_per_chunk > 0 {
            (vram_budget / bytes_per_chunk).max(1)
        } else {
            1
        };

        let mut max_chunks = MAX_CHUNKS as u64;
        max_chunks = max_chunks.min(max_by_range).min(max_by_vram);
        let max_chunks = max_chunks.max(1) as u32;

        if max_chunks < MAX_CHUNKS {
            warn!(
                max_chunks,
                max_by_range,
                max_by_vram,
                heap_bytes,
                vram_budget,
                face_buffer_unit,
                "clamping max chunks to avoid GPU OOM"
            );
        }

        let surfaces = DrawBuffer::new(gfx, max_chunks, dimension);
        // Skylight shadow-map uses a fixed resolution; keep it constant regardless of window size.
        let skylight = SkylightShadow::new(
            gfx,
            &surfaces,
            vk::Extent2D {
                width: 1024,
                height: 1024,
            },
        );

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
