//! Light propagation system for Hyperbolicraft.
//!
//! This module implements fluid-like light propagation where light spreads outward
//! from emitting blocks at a constant tick rate, decaying as it travels through
//! different block types.
//!
//! # Algorithm Overview
//!
//! Light propagates using a BFS-like algorithm:
//! 1. Initialize light sources from blocks with emission
//! 2. Each tick, light spreads one block in each cardinal direction
//! 3. Light decays based on block type: 1 for air/cutout, 3 for translucent, blocked by opaque
//! 4. Light updates cross chunk boundaries when needed
//!
//! # Performance Considerations
//!
//! - Only chunks with light sources or pending updates are processed
//! - A dirty flag tracks chunks that need surface re-extraction
//! - Light margin data is synchronized like voxel margins

use fxhash::FxHashSet;
use tracing::trace;

use crate::dodeca::Vertex;
use crate::graph::Graph;
use crate::margins::fix_light_margins;
use crate::light::{coords_to_index, LightData, LightValue};
use crate::node::{Chunk, ChunkId};
use crate::voxel_math::{ChunkAxisPermutation, ChunkDirection, CoordAxis, CoordSign, Coords};
use crate::world::BlockRegistry;

/// Tracks which chunks have pending light updates.
#[derive(Default)]
pub struct LightUpdateQueue {
    /// Chunks that need light recalculation this tick
    pending: FxHashSet<ChunkId>,
    /// Chunks that were updated and may propagate to neighbors next tick
    propagating: FxHashSet<ChunkId>,
}

impl LightUpdateQueue {
    pub fn new() -> Self {
        Self::default()
    }

    /// Marks a chunk for light update (e.g., when a block changes).
    pub fn mark_dirty(&mut self, chunk: ChunkId) {
        self.pending.insert(chunk);
    }

    /// Returns true if there are pending updates.
    pub fn has_pending(&self) -> bool {
        !self.pending.is_empty() || !self.propagating.is_empty()
    }
}

/// Result of a single light propagation step.
pub struct LightUpdateResult {
    /// Chunks whose light data changed and need surface re-extraction
    pub dirty_chunks: Vec<ChunkId>,
}

/// Performs one tick of light propagation across all pending chunks.
///
/// This function:
/// 1. Processes all chunks marked for update
/// 2. Spreads light from sources and existing light values
/// 3. Marks neighboring chunks for next-tick updates if light reached their boundary
///
/// # Returns
///
/// A list of chunks whose light data changed, for surface re-extraction.
pub fn propagate_light_tick(graph: &mut Graph, queue: &mut LightUpdateQueue) -> LightUpdateResult {
    let dimension = graph.layout().dimension();
    let mut dirty_chunks = Vec::new();

    // Merge propagating chunks into pending for processing
    let propagating_count = queue.propagating.len();
    for chunk in queue.propagating.drain() {
        queue.pending.insert(chunk);
    }

    // Process each pending chunk
    let pending: Vec<ChunkId> = queue.pending.drain().collect();
    let pending_count = pending.len();

    for chunk_id in pending {
        if !matches!(graph[chunk_id], Chunk::Populated { .. }) {
            continue;
        }

        // Run multiple iterations within the chunk to propagate light faster
        let mut chunk_changed = false;
        for _ in 0..dimension {
            let changed = update_chunk_light(graph, chunk_id, dimension);
            if changed {
                chunk_changed = true;
            } else {
                break; // Light has stabilized within this chunk
            }
        }

        if chunk_changed {
            dirty_chunks.push(chunk_id);

            // Check if light reached any boundary - if so, queue neighbors
            let has_boundary_light = check_boundary_light(graph, chunk_id, dimension);
            
            if has_boundary_light {
                // Mark neighboring chunks for potential updates next tick
                for direction in ChunkDirection::iter() {
                    if let Some(neighbor) = graph.get_chunk_neighbor(chunk_id, direction.axis, direction.sign) {
                        if matches!(graph[neighbor], Chunk::Populated { .. }) {
                            queue.propagating.insert(neighbor);
                        }
                    }
                }
            }
            
            // Also re-queue self if still changing (light still propagating internally)
            queue.propagating.insert(chunk_id);
        }
    }

    // Synchronize light margins between dirty chunks and their neighbors
    // This is critical for light to visually cross chunk boundaries in the GPU
    let mut margin_updated_chunks = FxHashSet::default();
    for &chunk_id in &dirty_chunks {
        let neighbors = sync_light_margins_for_chunk(graph, chunk_id, dimension);
        for neighbor_id in neighbors {
            margin_updated_chunks.insert(neighbor_id);
        }
    }
    
    // Mark any neighbors whose margins were updated as needing re-extraction
    for neighbor_id in margin_updated_chunks {
        if !dirty_chunks.contains(&neighbor_id) {
            // Mark the neighbor's light as dirty so it gets re-extracted
            if let Chunk::Populated { ref mut light_dirty, .. } = graph[neighbor_id] {
                *light_dirty = true;
            }
            dirty_chunks.push(neighbor_id);
        }
    }

    if !dirty_chunks.is_empty() || pending_count > 0 {
        trace!(
            pending = pending_count,
            from_propagating = propagating_count,
            dirty = dirty_chunks.len(),
            next_queue = queue.propagating.len(),
            "light propagation tick"
        );
    }

    LightUpdateResult { dirty_chunks }
}

/// Checks if any boundary voxel has non-zero light.
fn check_boundary_light(graph: &Graph, chunk_id: ChunkId, dimension: u8) -> bool {
    let Chunk::Populated { ref light, .. } = graph[chunk_id] else {
        return false;
    };

    // Check all 6 faces of the chunk
    for axis in 0..3u8 {
        for sign in [0u8, dimension - 1] {
            for u in 0..dimension {
                for v in 0..dimension {
                    let (x, y, z) = match axis {
                        0 => (sign, u, v),      // X faces
                        1 => (u, sign, v),      // Y faces
                        _ => (u, v, sign),      // Z faces
                    };
                    let index = coords_to_index(x, y, z, dimension);
                    if !light.get(index).is_zero() {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// Synchronizes light margins between a chunk and all its face neighbors.
/// This copies boundary light values to neighbor margins and vice versa.
/// Returns the list of neighbor chunks whose margins were updated.
fn sync_light_margins_for_chunk(graph: &mut Graph, chunk_id: ChunkId, dimension: u8) -> Vec<ChunkId> {
    let mut updated_neighbors = Vec::new();
    
    for direction in ChunkDirection::iter() {
        let Some(neighbor_id) = graph.get_chunk_neighbor(chunk_id, direction.axis, direction.sign) else {
            continue;
        };
        
        // Get mutable references to both chunks' light data
        // We need to be careful with the borrow checker here
        let (chunk_vertex, chunk_light_ptr, neighbor_light_ptr) = {
            let Chunk::Populated { ref mut light, .. } = graph[chunk_id] else {
                continue;
            };
            let chunk_light_ptr = light as *mut _;
            let chunk_vertex = chunk_id.vertex;
            
            let Chunk::Populated { light: ref mut neighbor_light, .. } = graph[neighbor_id] else {
                continue;
            };
            let neighbor_light_ptr = neighbor_light as *mut _;
            
            (chunk_vertex, chunk_light_ptr, neighbor_light_ptr)
        };
        
        // SAFETY: We have exclusive access to the graph, and chunk_id != neighbor_id
        // because get_chunk_neighbor returns a different chunk.
        unsafe {
            fix_light_margins(
                dimension,
                chunk_vertex,
                &mut *chunk_light_ptr,
                direction,
                &mut *neighbor_light_ptr,
            );
        }
        
        updated_neighbors.push(neighbor_id);
    }
    
    updated_neighbors
}

/// Updates light values for a single chunk.
///
/// Returns true if any light values changed.
fn update_chunk_light(graph: &mut Graph, chunk_id: ChunkId, dimension: u8) -> bool {
    // First pass: calculate new light values
    let new_light = calculate_chunk_light(graph, chunk_id, dimension);

    // Second pass: apply and check for changes
    let Chunk::Populated {
        ref mut light,
        ref mut light_dirty,
        ..
    } = graph[chunk_id]
    else {
        return false;
    };

    let changed = apply_light_changes(light, &new_light, dimension);

    if changed {
        *light_dirty = true;
    }

    changed
}

/// Calculates new light values for a chunk based on sources and neighbor light.
fn calculate_chunk_light(graph: &Graph, chunk_id: ChunkId, dimension: u8) -> LightData {
    let Chunk::Populated {
        ref voxels,
        ref light,
        ..
    } = graph[chunk_id]
    else {
        return LightData::default();
    };

    let mut new_light = LightData::new_dense(dimension);
    let new_light_data = new_light.data_mut(dimension);

    // Process each voxel in the chunk
    for z in 0..dimension {
        for y in 0..dimension {
            for x in 0..dimension {
                let index = coords_to_index(x, y, z, dimension);
                let block_id = voxels.get(index);
                let block_info = BlockRegistry::get_light_info(block_id);

                // Start with the block's own emission
                let mut max_light = block_info.emission;

                // Check each neighbor for incoming light
                for (dx, dy, dz) in NEIGHBOR_OFFSETS {
                    let nx = x as i8 + dx;
                    let ny = y as i8 + dy;
                    let nz = z as i8 + dz;

                    // Only process neighbors within this chunk
                    // Cross-chunk light will propagate over multiple ticks
                    let neighbor_light = if nx >= 0
                        && nx < dimension as i8
                        && ny >= 0
                        && ny < dimension as i8
                        && nz >= 0
                        && nz < dimension as i8
                    {
                        // Neighbor is in the same chunk
                        let neighbor_index = coords_to_index(nx as u8, ny as u8, nz as u8, dimension);
                        light.get(neighbor_index)
                    } else {
                        // Cross-chunk neighbor - get from neighbor chunk's boundary
                        get_neighbor_chunk_light(graph, chunk_id, dimension, x, y, z, dx, dy, dz)
                    };

                    // Decay light based on the current block's behavior
                    if let Some(decay) = block_info.behavior.decay_amount() {
                        let decayed = neighbor_light.decay(decay);
                        max_light = max_light.max(decayed);
                    }
                    // If opaque, light doesn't propagate through
                }

                new_light_data[index] = max_light;
            }
        }
    }

    new_light
}

/// Gets light from a neighbor chunk at the boundary.
/// 
/// This reads the light value from the neighbor chunk's boundary voxel that
/// corresponds to the margin position we're sampling from.
/// 
/// In hyperbolic space, when crossing chunk boundaries, coordinate axes may be
/// permuted. This follows the same logic as voxel margin synchronization in margins.rs.
fn get_neighbor_chunk_light(
    graph: &Graph,
    chunk_id: ChunkId,
    dimension: u8,
    x: u8,
    y: u8,
    z: u8,
    dx: i8,
    dy: i8,
    dz: i8,
) -> LightValue {
    // Determine direction to neighbor chunk
    let direction = if dx != 0 {
        ChunkDirection { 
            axis: CoordAxis::X, 
            sign: if dx < 0 { CoordSign::Minus } else { CoordSign::Plus }
        }
    } else if dy != 0 {
        ChunkDirection { 
            axis: CoordAxis::Y, 
            sign: if dy < 0 { CoordSign::Minus } else { CoordSign::Plus }
        }
    } else {
        ChunkDirection { 
            axis: CoordAxis::Z, 
            sign: if dz < 0 { CoordSign::Minus } else { CoordSign::Plus }
        }
    };

    // Get the neighbor chunk
    let Some(neighbor_id) = graph.get_chunk_neighbor(chunk_id, direction.axis, direction.sign) else {
        return LightValue::ZERO;
    };

    let Chunk::Populated { ref light, .. } = graph[neighbor_id] else {
        return LightValue::ZERO;
    };

    // Get the axis permutation for crossing this boundary
    // This is the same logic used in margins.rs for voxel margin synchronization
    let neighbor_axis_permutation = neighbor_axis_permutation(chunk_id.vertex, direction);

    // We want to read from the neighbor chunk at the position adjacent to us.
    // 
    // In margins.rs, the pattern is:
    //   neighbor_voxel_data[(neighbor_axis_permutation * coords_of_boundary_voxel).to_index(dimension)]
    // 
    // Where coords_of_boundary_voxel is in OUR coordinate system, and we apply
    // the permutation to convert it to the neighbor's coordinate system.
    //
    // Our boundary coordinate (where we are looking FROM) should be the face
    // of our chunk in the direction we're looking. The neighbor's corresponding
    // voxel is at THEIR boundary on the opposite side.
    
    // Our boundary position in our coordinate system
    let our_boundary = match direction.sign {
        CoordSign::Plus => dimension - 1,  // We're at the + edge of our chunk
        CoordSign::Minus => 0,              // We're at the - edge of our chunk
    };
    
    // Build our boundary coordinates
    let mut our_boundary_coords = [0u8; 3];
    for axis in CoordAxis::iter() {
        if axis == direction.axis {
            our_boundary_coords[axis as usize] = our_boundary;
        } else {
            our_boundary_coords[axis as usize] = match axis {
                CoordAxis::X => x,
                CoordAxis::Y => y,
                CoordAxis::Z => z,
            };
        }
    }
    let our_boundary_coords = Coords(our_boundary_coords);
    
    // Apply the axis permutation to get coordinates in the neighbor's frame
    // This converts from our coordinate system to the neighbor's
    let neighbor_coords = neighbor_axis_permutation * our_boundary_coords;

    let index = coords_to_index(
        neighbor_coords[CoordAxis::X],
        neighbor_coords[CoordAxis::Y],
        neighbor_coords[CoordAxis::Z],
        dimension,
    );
    light.get(index)
}

/// Gets the axis permutation for crossing a chunk boundary.
/// This is the same logic used in margins.rs.
fn neighbor_axis_permutation(vertex: Vertex, direction: ChunkDirection) -> ChunkAxisPermutation {
    match direction.sign {
        CoordSign::Plus => vertex.chunk_axis_permutations()[direction.axis as usize],
        CoordSign::Minus => ChunkAxisPermutation::IDENTITY,
    }
}

/// Applies new light values and returns true if any changed.
fn apply_light_changes(light: &mut LightData, new_light: &LightData, dimension: u8) -> bool {
    let mut changed = false;

    let old_data = light.data_mut(dimension);
    let LightData::Dense(new_data) = new_light else {
        return false;
    };

    for z in 0..dimension {
        for y in 0..dimension {
            for x in 0..dimension {
                let index = coords_to_index(x, y, z, dimension);
                if old_data[index] != new_data[index] {
                    old_data[index] = new_data[index];
                    changed = true;
                }
            }
        }
    }

    changed
}

/// The 6 cardinal neighbor offsets.
const NEIGHBOR_OFFSETS: [(i8, i8, i8); 6] = [
    (-1, 0, 0), // -X
    (1, 0, 0),  // +X
    (0, -1, 0), // -Y
    (0, 1, 0),  // +Y
    (0, 0, -1), // -Z
    (0, 0, 1),  // +Z
];

/// Initializes light for a newly generated chunk based on its light-emitting blocks.
///
/// This should be called after a chunk is populated with voxel data.
pub fn initialize_chunk_light(graph: &mut Graph, chunk_id: ChunkId) {
    let dimension = graph.layout().dimension();

    let Chunk::Populated {
        ref voxels,
        ref mut light,
        ref mut light_dirty,
        ..
    } = graph[chunk_id]
    else {
        return;
    };

    let mut has_emitters = false;

    // Check if any blocks emit light
    for z in 0..dimension {
        for y in 0..dimension {
            for x in 0..dimension {
                let index = coords_to_index(x, y, z, dimension);
                let block_id = voxels.get(index);
                let emission = BlockRegistry::get_light_emission(block_id);

                if !emission.is_zero() {
                    has_emitters = true;
                    let light_data = light.data_mut(dimension);
                    light_data[index] = emission;
                }
            }
        }
    }

    if has_emitters {
        *light_dirty = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::BlockKind;

    #[test]
    fn test_light_decay() {
        let light = LightValue::new(15, 10, 5);
        assert_eq!(light.decay(1), LightValue::new(14, 9, 4));
        assert_eq!(light.decay(3), LightValue::new(12, 7, 2));
        assert_eq!(light.decay(10), LightValue::new(5, 0, 0));
    }

    #[test]
    fn test_torch_emission() {
        let torch_light = BlockKind::Torch.light_emission();
        assert!(!torch_light.is_zero());
        assert!(torch_light.r() >= 10); // Torch should emit significant light
    }

    #[test]
    fn test_glowstone_emission() {
        let glow_light = BlockKind::Glowstone.light_emission();
        assert_eq!(glow_light.brightness(), 15); // Glowstone should be max brightness
    }
}
