use serde::{Deserialize, Serialize};

use crate::{
    dodeca::{Side, Vertex},
    graph::{Graph, NodeId},
    margins,
    math::MVector,
    node::{ChunkId, VoxelData},
    plane::Plane,
};
use crate::proto::Position as ProtoPosition;
use crate::math::MPoint;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorldgenPreset {
    Hyperbolic,
    Flat,
}

impl Default for WorldgenPreset {
    fn default() -> Self {
        WorldgenPreset::Hyperbolic
    }
}

/// Contains all information about a node used for world generation.
/// Simplified to only essential data for flat geodesic plane generation.
pub struct NodeState {
    surface: Plane,
}

impl NodeState {
    pub fn root() -> Self {
        Self {
            surface: Plane::from(Side::A),
        }
    }

    pub fn child(&self, _graph: &Graph, _node: NodeId, side: Side) -> Self {
        Self {
            surface: side * self.surface,
        }
    }

    pub fn up_direction(&self) -> MVector<f32> {
        *self.surface.scaled_normal()
    }
}

/// Lightweight sample of environment at a given position for debugging / UI.
#[derive(Debug, Clone, Copy)]
pub struct EnviroSample {
    pub biome: u8,
    pub temp_normalized: f32,
    pub rain_normalized: f32,
    /// Signed distance from the node's surface plane (positive = below surface/solid)
    pub signed_distance: f32,
    /// Projected X coordinate (Beltrami-like: x/w) for quick visualization
    pub proj_x: f32,
    /// Projected Z coordinate (Beltrami-like: z/w) for quick visualization
    pub proj_z: f32,
}

/// Sample coarse environmental values at a position.
pub fn sample_enviro_at(graph: &Graph, pos: &ProtoPosition) -> EnviroSample {
    let node_state = graph.node_state(pos.node);

    let temp_normalized = 0.5;
    let rain_normalized = 0.5;
    let biome = 0;

    // Signed distance from this node's surface plane to the point described by pos.local
    let mpoint = pos.local * MPoint::origin();
    let signed_distance = node_state.surface.distance_to(&mpoint);

    // Quick projection for on-plane visualization
    let v4: na::Vector4<f32> = mpoint.into();
    let proj_x = if v4.w.abs() > 1e-6 { v4.x / v4.w } else { v4.x };
    let proj_z = if v4.w.abs() > 1e-6 { v4.z / v4.w } else { v4.z };

    EnviroSample {
        biome,
        temp_normalized,
        rain_normalized,
        signed_distance,
        proj_x,
        proj_z,
    }
}

struct VoxelCoords {
    counter: u32,
    dimension: u8,
}

impl VoxelCoords {
    fn new(dimension: u8) -> Self {
        VoxelCoords {
            counter: 0,
            dimension,
        }
    }
}

impl Iterator for VoxelCoords {
    type Item = (u8, u8, u8);

    fn next(&mut self) -> Option<Self::Item> {
        let dim = u32::from(self.dimension);

        if self.counter == dim.pow(3) {
            return None;
        }

        let result = (
            (self.counter / dim.pow(2)) as u8,
            ((self.counter / dim) % dim) as u8,
            (self.counter % dim) as u8,
        );

        self.counter += 1;
        Some(result)
    }
}

/// Data needed to generate a chunk
pub struct ChunkParams {
    /// Number of voxels along an edge
    dimension: u8,
    /// Which vertex of the containing node this chunk lies against
    chunk: Vertex,
    /// Reference plane for the terrain surface
    surface: Plane,
}

impl ChunkParams {
    /// Extract data necessary to generate a chunk
    pub fn new(graph: &mut Graph, chunk: ChunkId) -> Self {
        graph.ensure_node_state(chunk.node);
        let state = graph.node_state(chunk.node);
        Self {
            dimension: graph.layout().dimension(),
            chunk: chunk.vertex,
            surface: state.surface,
        }
    }

    pub fn chunk(&self) -> Vertex {
        self.chunk
    }

    /// Generate voxels making up a flat geodesic plane chunk
    /// Simple rule: stone below surface (negative elevation), air above
    pub fn generate_voxels(&self) -> VoxelData {
        let mut voxels = VoxelData::Solid(0); // Block ID 0 is Air
        
        for (x, y, z) in VoxelCoords::new(self.dimension) {
            let coords = na::Vector3::new(x, y, z);
            let center = voxel_center(self.dimension, coords);
            let voxel_elevation = self.surface.distance_to_chunk(self.chunk, &center);
            
            // Simple flat plane: stone below surface (negative elevation), air above
            if voxel_elevation < 0.0 {
                voxels.data_mut(self.dimension)[index(self.dimension, coords)] = 1; // Block ID 1 is Stone
            }
        }
        
        margins::initialize_margins(self.dimension, &mut voxels);
        voxels
    }
}

/// Location of the center of a voxel in a unit chunk
fn voxel_center(dimension: u8, voxel: na::Vector3<u8>) -> na::Vector3<f32> {
    voxel.map(|x| f32::from(x) + 0.5) / f32::from(dimension)
}

fn index(dimension: u8, v: na::Vector3<u8>) -> usize {
    let v = v.map(|x| usize::from(x) + 1);

    // LWM = Length (of cube sides) With Margins
    let lwm = usize::from(dimension) + 2;
    v.x + v.y * lwm + v.z * lwm.pow(2)
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::*;

    const CHUNK_SIZE: u8 = 12;

    #[test]
    fn chunk_indexing_origin() {
        // (0, 0, 0) in localized coords
        let origin_index = 1 + (usize::from(CHUNK_SIZE) + 2) + (usize::from(CHUNK_SIZE) + 2).pow(2);

        // simple sanity check
        assert_eq!(index(CHUNK_SIZE, na::Vector3::repeat(0)), origin_index);
    }

    #[test]
    fn chunk_indexing_absolute() {
        let origin_index = 1 + (usize::from(CHUNK_SIZE) + 2) + (usize::from(CHUNK_SIZE) + 2).pow(2);
        // (0.5, 0.5, 0.5) in localized coords
        let center_index = index(CHUNK_SIZE, na::Vector3::repeat(CHUNK_SIZE / 2));
        // the point farthest from the origin, (1, 1, 1) in localized coords
        let anti_index = index(CHUNK_SIZE, na::Vector3::repeat(CHUNK_SIZE));

        assert_eq!(index(CHUNK_SIZE, na::Vector3::new(0, 0, 0)), origin_index);

        // biggest possible index in subchunk closest to origin still isn't the center
        assert!(
            index(
                CHUNK_SIZE,
                na::Vector3::new(CHUNK_SIZE / 2 - 1, CHUNK_SIZE / 2 - 1, CHUNK_SIZE / 2 - 1,)
            ) < center_index
        );
        // but the first chunk in the subchunk across from that is
        assert_eq!(
            index(
                CHUNK_SIZE,
                na::Vector3::new(CHUNK_SIZE / 2, CHUNK_SIZE / 2, CHUNK_SIZE / 2)
            ),
            center_index
        );

        // biggest possible index in subchunk closest to anti_origin is still not quite
        // the anti_origin
        assert!(
            index(
                CHUNK_SIZE,
                na::Vector3::new(CHUNK_SIZE - 1, CHUNK_SIZE - 1, CHUNK_SIZE - 1,)
            ) < anti_index
        );

        // one is added in the chunk indexing so this works out fine, the
        // domain is still CHUNK_SIZE because 0 is included.
        assert_eq!(
            index(
                CHUNK_SIZE,
                na::Vector3::new(CHUNK_SIZE - 1, CHUNK_SIZE - 1, CHUNK_SIZE - 1,)
            ),
            index(CHUNK_SIZE, na::Vector3::repeat(CHUNK_SIZE - 1))
        );
    }

    #[test]
    fn check_voxel_iterable() {
        let dimension = 12;

        for (counter, (x, y, z)) in (VoxelCoords::new(dimension as u8)).enumerate() {
            let index = z as usize + y as usize * dimension + x as usize * dimension.pow(2);
            assert!(counter == index);
        }
    }
}
