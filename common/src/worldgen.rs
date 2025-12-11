//! World generation unit conventions
//!
//! This module primarily operates in normalized, unit-chunk coordinates:
//!
//! - Voxel positions: `voxel_center` returns each axis in [0, 1]. One voxel spans `1 / dimension` of a chunk side.
//! - Distances to planes: `Plane::distance_to_chunk(chunk, &point)` yields a signed distance in the same unit-chunk scale.
//!   Interpreting values: `1.0` ≈ one full chunk edge length along the plane normal; positive is below/inside terrain
//!   for the guiding surface, negative is above/outside (void), unless otherwise noted.
//! - Elevation vs. distance: Environmental `max_elevation` is a height-like quantity produced by noise (see
//!   `EnviroFactors::varied_from`). For solid/void decisions it is mapped into unit-chunk distance by dividing by
//!   `TERRAIN_SMOOTHNESS = 10.0`, e.g. comparisons use `max_elevation / TERRAIN_SMOOTHNESS` vs. `voxel_elevation`.
//! - Roads: Road placement thresholds (e.g., `horizontal_distance > 0.3`, `elevation < 0.075`) are in unit-chunk
//!   distances, so `0.3` is ~30% of the chunk width.
//! - Noise scaling: Feature sizes are controlled by sampling unit-chunk coordinates scaled by constants
//!   (e.g. `world_pos = center * 10.0`, frequencies ~0.012–0.08). These are dimensionless but effectively tied to the
//!   unit-chunk coordinate system.
//!
//! Rule of thumb: `1.0` equals one chunk side length; a single voxel is `1 / dimension` of that. Terrain “height” from
//! enviro is converted to this distance space via division by `TERRAIN_SMOOTHNESS`.
//! MOST OF THE FEATURES AND CONVENTIONS LISTED HERE ARE JUST FOR REFERENCE,
//! AND ARE TO-BE-DEPRECATED ONCE BETTER TERRAIN GENERATION IS IMPLEMENTED.

use rand::{Rng, SeedableRng, distr::Uniform};
use rand_distr::Normal;
use serde::{Deserialize, Serialize};

use crate::{
    dodeca::{Side, Vertex},
    graph::{Graph, NodeId},
    margins,
    math::{MIsometry, MPoint, MVector},
    node::{ChunkId, VoxelData},
    plane::Plane,
    proto::Position as ProtoPosition,
    terraingen::VoronoiInfo,
    voxel_math::CoordAxis,
    world::{BlockID, BlockKind},
    hyper_noise::{HyperbolicNoise, FbmConfig},
};

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
#[derive(Clone, Copy, PartialEq, Debug)]
enum NodeStateKind {
    Sky,
    DeepSky,
    Land,
    DeepLand,
}
use NodeStateKind::*;

impl NodeStateKind {
    const ROOT: Self = Land;

    /// What state comes after this state, from a given side?
    fn child(self, side: Side) -> Self {
        match (self, side) {
            (Sky, Side::A) => Land,
            (Land, Side::A) => Sky,
            (Sky, _) if !side.adjacent_to(Side::A) => DeepSky,
            (Land, _) if !side.adjacent_to(Side::A) => DeepLand,
            _ => self,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum NodeStateRoad {
    East,
    DeepEast,
    West,
    DeepWest,
}
use NodeStateRoad::*;

use rand_pcg::Pcg64Mcg;

impl NodeStateRoad {
    const ROOT: Self = West;

    /// What state comes after this state, from a given side?
    fn child(self, side: Side) -> Self {
        match (self, side) {
            (East, Side::B) => West,
            (West, Side::B) => East,
            (East, _) if !side.adjacent_to(Side::B) => DeepEast,
            (West, _) if !side.adjacent_to(Side::B) => DeepWest,
            _ => self,
        }
    }
}

/// Contains all information about a node used for world generation. Most world
/// generation logic uses this information as a starting point.
pub struct NodeState {
    kind: NodeStateKind,
    surface: Plane,
    road_state: NodeStateRoad,
    enviro: EnviroFactors,
    world_from_node: MIsometry<f32>,
}
impl NodeState {
    pub fn root(graph: &Graph, node: NodeId) -> Self {
        let world_from_node = MIsometry::identity();
        let enviro = EnviroFactors::sample(graph.hash_of(node) as u64, &world_from_node);
        Self {
            kind: NodeStateKind::ROOT,
            surface: Plane::from(Side::A),
            road_state: NodeStateRoad::ROOT,
            enviro,
            world_from_node,
        }
    }

    pub fn child(&self, graph: &Graph, node: NodeId, side: Side) -> Self {
        let child_kind = self.kind.child(side);
        let child_road = self.road_state.child(side);
        let world_from_node = &self.world_from_node * side.reflection();
        let enviro = EnviroFactors::sample(graph.hash_of(node) as u64, &world_from_node);

        Self {
            kind: child_kind,
            surface: match child_kind {
                Land => Plane::from(Side::A),
                Sky => -Plane::from(Side::A),
                _ => side * self.surface,
            },
            road_state: child_road,
            enviro,
            world_from_node,
        }
    }

    pub fn up_direction(&self) -> MVector<f32> {
        *self.surface.scaled_normal()
    }

    pub fn world_from_node(&self) -> &MIsometry<f32> {
        &self.world_from_node
    }
}

/// Lightweight sample of environment at a given position for debugging / UI overlays.
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

/// Sample coarse environmental values at a position for UI debugging.
pub fn sample_enviro_at(graph: &Graph, pos: &ProtoPosition) -> EnviroSample {
    let node_state = graph.node_state(pos.node);
    let enviro = node_state.enviro;

    let mpoint = pos.local * MPoint::origin();
    let signed_distance = node_state.surface.distance_to(&mpoint);

    let v4: na::Vector4<f32> = mpoint.into();
    let proj_x = if v4.w.abs() > 1e-6 { v4.x / v4.w } else { v4.x };
    let proj_z = if v4.w.abs() > 1e-6 { v4.z / v4.w } else { v4.z };

    EnviroSample {
        biome: 0,
        temp_normalized: enviro.temperature,
        rain_normalized: enviro.rainfall,
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
    preset: WorldgenPreset,
    /// Number of voxels along an edge
    dimension: u8,
    /// Which vertex of the containing node this chunk lies against
    chunk: Vertex,
    /// Random quantities stored at the eight adjacent nodes, used for terrain generation
    env: ChunkIncidentEnviroFactors,
    /// Reference plane for the terrain surface
    surface: Plane,
    /// Whether this chunk contains a segment of the road
    is_road: bool,
    /// Whether this chunk contains a section of the road's supports
    is_road_support: bool,
    /// Random quantity used to seed terrain gen
    node_spice: u64,
    /// Transform from node-local space to global hyperbolic coordinates
    world_from_node: MIsometry<f32>,
    /// Orientation of the chunk's up direction in voxel space
    orientation: ChunkOrientation,
    /// Conversion from hyperbolic noise distances to approximate "block" units in the horizontal plane
    hyper_block_scale: f64,
    /// Conversion from hyperbolic noise distances to "block" units along the vertical axis
    hyper_vertical_block_scale: f64,
}

#[derive(Clone, Copy)]
struct ChunkOrientation {
    up_axis: CoordAxis,
    up_sign: i8,
    horizontal_axes: [CoordAxis; 2],
}

impl ChunkOrientation {
    fn from_surface(surface: Plane, chunk: Vertex, dimension: u8) -> Self {
        let mut best_axis = CoordAxis::X;
        let mut best_magnitude = 0.0f32;
        let mut best_sign = 1i8;
        let center = na::Vector3::repeat(0.5);
        let step = (1.0 / f32::from(dimension)).min(0.25);

        for axis in CoordAxis::iter() {
            let mut forward = center;
            forward[axis as usize] = (forward[axis as usize] + step).min(0.999);
            let mut backward = center;
            backward[axis as usize] = (backward[axis as usize] - step).max(0.001);
            let forward_dist = surface.distance_to_chunk(chunk, &forward);
            let backward_dist = surface.distance_to_chunk(chunk, &backward);
            let slope = forward_dist - backward_dist;
            let magnitude = slope.abs();
            if magnitude > best_magnitude {
                best_magnitude = magnitude;
                best_axis = axis;
                best_sign = if slope < 0.0 { 1 } else { -1 };
            }
        }

        Self {
            up_axis: best_axis,
            up_sign: best_sign,
            horizontal_axes: best_axis.other_axes(),
        }
    }

    fn up_axis(self) -> CoordAxis {
        self.up_axis
    }

    fn up_sign(self) -> i8 {
        self.up_sign
    }

    fn horizontal_axes(self) -> [CoordAxis; 2] {
        self.horizontal_axes
    }

    fn column_index(&self, dimension: u8, coords: na::Vector3<u8>) -> usize {
        let [axis0, axis1] = self.horizontal_axes;
        let a = coords[axis0 as usize] as usize;
        let b = coords[axis1 as usize] as usize;
        a + b * dimension as usize
    }

    fn column_index_components(&self, dimension: u8, axis0: u8, axis1: u8) -> usize {
        (axis0 as usize) + (axis1 as usize) * dimension as usize
    }
}

impl ChunkParams {
    /// Extract data necessary to generate a chunk, generating new graph nodes if necessary
    pub fn new(graph: &mut Graph, chunk: ChunkId, preset: WorldgenPreset) -> Self {
        graph.ensure_node_state(chunk.node);
        let env = chunk_incident_enviro_factors(graph, chunk);
        let state = graph.node_state(chunk.node);
        let dimension = graph.layout().dimension();
        let world_from_node = state.world_from_node().clone();
        let orientation = ChunkOrientation::from_surface(state.surface, chunk.vertex, dimension);
        let mut params = Self {
            preset,
            dimension,
            chunk: chunk.vertex,
            env,
            surface: state.surface,
            is_road: state.kind == Sky
                && ((state.road_state == East) || (state.road_state == West)),
            is_road_support: ((state.kind == Land) || (state.kind == DeepLand))
                && ((state.road_state == East) || (state.road_state == West)),
            node_spice: graph.hash_of(chunk.node) as u64,
            world_from_node,
            orientation,
            hyper_block_scale: 1.0,
            hyper_vertical_block_scale: 1.0,
        };

        params.hyper_block_scale = params.estimate_horizontal_block_scale();
        params.hyper_vertical_block_scale = params.estimate_vertical_block_scale();
        params
    }

    pub fn chunk(&self) -> Vertex {
        self.chunk
    }

    /// Compute the world-space MPoint for a local chunk coordinate.
    /// This gives a true hyperbolic point that is consistent across chunk boundaries.
    fn world_mpoint(&self, local: na::Vector3<f32>) -> MPoint<f32> {
        let node_space = self.chunk.chunk_to_node() * local.push(1.0);
        let point = MVector::from(node_space).normalized_point();
        &self.world_from_node * point
    }

    /// Compute globally consistent hyperbolic coordinates for noise sampling.
    /// Uses signed hyperbolic distance from fixed reference planes through the origin.
    /// Returns (x, z) coordinates that are continuous across chunk boundaries.
    fn hyperbolic_noise_coords(&self, local: na::Vector3<f32>) -> na::Vector2<f64> {
        let point = self.world_mpoint(local);
        // Use signed distance from planes x=0 and z=0 in hyperbolic space.
        // A plane through the origin with normal n has the property that
        // the signed distance from point p is asinh(n · p) where · is mip.
        // For x=0 plane, normal is (1,0,0,0), so distance = asinh(p.x)
        // For z=0 plane, normal is (0,0,1,0), so distance = asinh(p.z)
        let v: na::Vector4<f32> = point.into();
        na::Vector2::new(v.x.asinh() as f64, v.z.asinh() as f64)
    }

    /// Compute 3D globally consistent hyperbolic coordinates for noise sampling.
    /// Returns (x, y, z) coordinates that are continuous across chunk boundaries.
    fn hyperbolic_noise_coords_3d(&self, local: na::Vector3<f32>) -> na::Vector3<f64> {
        let point = self.world_mpoint(local);
        let v: na::Vector4<f32> = point.into();
        na::Vector3::new(v.x.asinh() as f64, v.y.asinh() as f64, v.z.asinh() as f64)
    }

    /// Hyperbolic noise coordinates scaled to approximate "block" units horizontally.
    fn hyperbolic_block_coords(&self, local: na::Vector3<f32>) -> na::Vector2<f64> {
        self.hyperbolic_noise_coords(local) * self.hyper_block_scale
    }

    /// Hyperbolic noise coordinates scaled to approximate "block" units in 3D.
    fn hyperbolic_block_coords_3d(&self, local: na::Vector3<f32>) -> na::Vector3<f64> {
        let coords = self.hyperbolic_noise_coords_3d(local);
        na::Vector3::new(
            coords.x * self.hyper_block_scale,
            coords.y * self.hyper_vertical_block_scale,
            coords.z * self.hyper_block_scale,
        )
    }

    fn estimate_horizontal_block_scale(&self) -> f64 {
        let step = 1.0 / f32::from(self.dimension);
        let center = na::Vector3::repeat(0.5);
        let base = self.hyperbolic_noise_coords(center);
        let axes = self.orientation.horizontal_axes();
        let axis_count = axes.len() as f64;
        let mut total = 0.0f64;
        for axis in axes {
            let mut offset = center;
            offset[axis as usize] = (offset[axis as usize] + step).clamp(0.0, 1.0);
            let delta = self.hyperbolic_noise_coords(offset) - base;
            total += delta.norm();
        }
        let avg = total / axis_count.max(1.0);
        if avg <= std::f64::EPSILON {
            1.0
        } else {
            1.0 / avg
        }
    }

    fn estimate_vertical_block_scale(&self) -> f64 {
        let step = 1.0 / f32::from(self.dimension);
        let center = na::Vector3::repeat(0.5);
        let base = self.hyperbolic_noise_coords_3d(center);
        let axis = self.orientation.up_axis() as usize;
        let direction = if self.orientation.up_sign() >= 0 {
            1.0
        } else {
            -1.0
        };
        let mut offset = center;
        offset[axis] = (offset[axis] + step * direction).clamp(0.0, 1.0);
        let delta = self.hyperbolic_noise_coords_3d(offset) - base;
        let dist = delta.norm();
        if dist <= std::f64::EPSILON {
            1.0
        } else {
            1.0 / dist
        }
    }

    fn vertical_density_scale(&self) -> f32 {
        f32::from(self.dimension) / 128.0
    }

    fn orientation(&self) -> ChunkOrientation {
        self.orientation
    }

    /// Generate voxels making up the chunk
    pub fn generate_voxels(&self) -> VoxelData {
        match self.preset {
            WorldgenPreset::Hyperbolic => remcpe::generate_chunk(self),
            WorldgenPreset::Flat => self.generate_legacy_voxels(),
        }
    }

    fn generate_legacy_voxels(&self) -> VoxelData {
        // Maximum difference between elevations at the center of a chunk and any other point in the chunk
        // TODO: Compute what this actually is, current value is a guess! Real one must be > 0.6
        // empirically.
        const ELEVATION_MARGIN: f32 = 0.7;
        let center_elevation = self
            .surface
            .distance_to_chunk(self.chunk, &na::Vector3::repeat(0.5));
        let chunk_top = center_elevation - ELEVATION_MARGIN;
        let chunk_bottom = center_elevation + ELEVATION_MARGIN;
        if (chunk_bottom <= FLAT_SURFACE_HEIGHT) && !(self.is_road || self.is_road_support) {
            // The whole chunk lies above the guiding plane
            return VoxelData::Solid(BlockKind::Air.id());
        }

        if (chunk_top >= FLAT_SURFACE_HEIGHT) && !self.is_road {
            // The whole chunk is underground
            return VoxelData::Solid(BlockKind::Dirt.id());
        }

        let mut voxels = VoxelData::Solid(BlockKind::Air.id());
        let mut rng = rand_pcg::Pcg64Mcg::seed_from_u64(hash(self.node_spice, self.chunk as u64));

        self.generate_terrain(&mut voxels, &mut rng);

        if self.is_road {
            self.generate_road(&mut voxels);
        } else if self.is_road_support {
            self.generate_road_support(&mut voxels);
        }

        // TODO: Don't generate detailed data for solid chunks with no neighboring voids

        if self.dimension > 4 && matches!(voxels, VoxelData::Dense(_)) {
            self.generate_trees(&mut voxels, &mut rng);
        }

        margins::initialize_margins(self.dimension, &mut voxels);
        voxels
    }

    /// Performs all terrain generation that can be done one voxel at a time and with
    /// only the containing chunk's surrounding nodes' envirofactors.
    fn generate_terrain(&self, voxels: &mut VoxelData, rng: &mut Pcg64Mcg) {
        let normal = Normal::new(0.0, 0.03).unwrap();
        for (x, y, z) in VoxelCoords::new(self.dimension) {
            let coords = na::Vector3::new(x, y, z);
            let center = voxel_center(self.dimension, coords);
            let trilerp_coords = center.map(|x| (1.0 - x) * 0.5);

            let rain = trilerp(&self.env.rainfalls, trilerp_coords) + rng.sample(normal);
            let temp = trilerp(&self.env.temperatures, trilerp_coords) + rng.sample(normal);
            let voxel_elevation = self.surface.distance_to_chunk(self.chunk, &center);

            if voxel_elevation < FLAT_SURFACE_HEIGHT {
                continue;
            }

            let dist = voxel_elevation - FLAT_SURFACE_HEIGHT;
            let voxel_mat = VoronoiInfo::terraingen_voronoi(
                f64::from(FLAT_BASE_ELEVATION),
                f64::from(rain),
                f64::from(temp),
                f64::from(dist),
            );

            voxels.data_mut(self.dimension)[index(self.dimension, coords)] = voxel_mat.into();
        }
    }

    /// Places a road along the guiding plane.
    fn generate_road(&self, voxels: &mut VoxelData) {
        let plane = -Plane::from(Side::B);

        for (x, y, z) in VoxelCoords::new(self.dimension) {
            let coords = na::Vector3::new(x, y, z);
            let center = voxel_center(self.dimension, coords);
            let horizontal_distance = plane.distance_to_chunk(self.chunk, &center);
            let elevation = self.surface.distance_to_chunk(self.chunk, &center);

            if horizontal_distance > 0.3 || elevation > 0.9 {
                continue;
            }

            let mut block_id = BlockKind::Air.id();

            if elevation < 0.075 {
                block_id = if horizontal_distance < 0.15 {
                    // Inner
                    BlockKind::Brick.id()
                } else {
                    // Outer
                    BlockKind::Cobblestone.id()
                };
            }

            voxels.data_mut(self.dimension)[index(self.dimension, coords)] = block_id;
        }
    }

    /// Fills the half-plane below the road with wooden supports.
    fn generate_road_support(&self, voxels: &mut VoxelData) {
        let plane = -Plane::from(Side::B);

        for (x, y, z) in VoxelCoords::new(self.dimension) {
            let coords = na::Vector3::new(x, y, z);
            let center = voxel_center(self.dimension, coords);
            let horizontal_distance = plane.distance_to_chunk(self.chunk, &center);

            if horizontal_distance > 0.3 {
                continue;
            }

            let block_id = if self.trussing_at(coords) {
                BlockKind::WoodPlanks.id()
            } else {
                BlockKind::Air.id()
            };

            if block_id != BlockKind::Air.id() {
                voxels.data_mut(self.dimension)[index(self.dimension, coords)] = block_id;
            }
        }
    }

    /// Make a truss-shaped template
    fn trussing_at(&self, coords: na::Vector3<u8>) -> bool {
        // Generates planar diagonals, but corner is offset
        let mut criteria_met = 0_u32;
        let x = coords[0];
        let y = coords[1];
        let z = coords[2];
        let offset = self.dimension / 3;

        // straight lines.
        criteria_met += u32::from(x == offset);
        criteria_met += u32::from(y == offset);
        criteria_met += u32::from(z == offset);

        // main diagonal
        criteria_met += u32::from(x == y);
        criteria_met += u32::from(y == z);
        criteria_met += u32::from(x == z);

        criteria_met >= 2
    }

    /// Plants trees on dirt and grass. Trees consist of a block of wood
    /// and a block of leaves. The leaf block is on the opposite face of the
    /// wood block as the ground block.
    fn generate_trees(&self, voxels: &mut VoxelData, rng: &mut Pcg64Mcg) {
        // margins are added to keep voxels outside the chunk from being read/written
        let random_position = Uniform::new(1, self.dimension - 1).unwrap();

        let rain = self.env.rainfalls[0];
        let tree_candidate_count =
            (u32::from(self.dimension - 2).pow(3) as f32 * (rain / 100.0).clamp(0.0, 0.5)) as usize;
        for _ in 0..tree_candidate_count {
            let loc = na::Vector3::from_fn(|_, _| rng.sample(random_position));
            let voxel_of_interest_index = index(self.dimension, loc);
            let neighbor_data = self.voxel_neighbors(loc, voxels);

            let num_void_neighbors = neighbor_data
                .iter()
                .filter(|n| n.block_id == BlockKind::Air.id())
                .count();

            // Only plant a tree if there is exactly one adjacent block of dirt or grass
            if num_void_neighbors == 5 {
                for i in neighbor_data.iter() {
                    if (i.block_id == BlockKind::Dirt.id()) || (i.block_id == BlockKind::Grass.id())
                    {
                        voxels.data_mut(self.dimension)[voxel_of_interest_index] =
                            BlockKind::Log.id();
                        let leaf_location = index(self.dimension, i.coords_opposing);
                        voxels.data_mut(self.dimension)[leaf_location] = BlockKind::Leaves.id();
                    }
                }
            }
        }
    }

    /// Provides information on the type of material in a voxel's six neighbours
    fn voxel_neighbors(&self, coords: na::Vector3<u8>, voxels: &VoxelData) -> [NeighborData; 6] {
        [
            self.neighbor(coords, -1, 0, 0, voxels),
            self.neighbor(coords, 1, 0, 0, voxels),
            self.neighbor(coords, 0, -1, 0, voxels),
            self.neighbor(coords, 0, 1, 0, voxels),
            self.neighbor(coords, 0, 0, -1, voxels),
            self.neighbor(coords, 0, 0, 1, voxels),
        ]
    }

    fn neighbor(
        &self,
        w: na::Vector3<u8>,
        x: i8,
        y: i8,
        z: i8,
        voxels: &VoxelData,
    ) -> NeighborData {
        let coords = na::Vector3::new(
            (w.x as i8 + x) as u8,
            (w.y as i8 + y) as u8,
            (w.z as i8 + z) as u8,
        );
        let coords_opposing = na::Vector3::new(
            (w.x as i8 - x) as u8,
            (w.y as i8 - y) as u8,
            (w.z as i8 - z) as u8,
        );
        let block_id = voxels.get(index(self.dimension, coords));

        NeighborData {
            coords_opposing,
            block_id,
        }
    }
}

#[allow(dead_code)]
const TERRAIN_SMOOTHNESS: f32 = 10.0;
#[allow(dead_code)]
const FLAT_MAX_ELEVATION: f32 = TERRAIN_SMOOTHNESS * 2.0;
#[allow(dead_code)]
const FLAT_BASE_ELEVATION: f32 = 0.0;
#[allow(dead_code)]
const FLAT_SURFACE_HEIGHT: f32 = 0.0;

struct NeighborData {
    coords_opposing: na::Vector3<u8>,
    block_id: BlockID,
}

const ENVIRO_TEMP_FREQ: f64 = 0.02;
const ENVIRO_RAIN_FREQ: f64 = 0.03;
const ENVIRO_HEIGHT_FREQ: f64 = 0.01;
const ENVIRO_BLOCK_FREQ: f64 = 0.05;
const ENVIRO_HEIGHT_VARIANCE: f32 = 48.0;
const ENVIRO_FBM: FbmConfig = FbmConfig {
    octaves: 5,
    lacunarity: 2.0,
    gain: 0.5,
};
const ENVIRO_SEED_TEMP: u64 = 0x5eed_5eed_a5a5_0001;
const ENVIRO_SEED_RAIN: u64 = 0xface_b00c_baad_f00d;
const ENVIRO_SEED_HEIGHT: u64 = 0x91ce_d9a1_1234_5678;
const ENVIRO_SEED_BLOCK: u64 = 0xfeed_babe_0dd5_1dea;

#[derive(Copy, Clone)]
struct EnviroFactors {
    max_elevation: f32,
    temperature: f32,
    rainfall: f32,
    blockiness: f32,
}
impl EnviroFactors {
    fn sample(spice: u64, world_from_node: &MIsometry<f32>) -> Self {
        let coords3 = hyperbolic_coords_from_transform(world_from_node);
        let coords2 = na::Vector2::new(coords3.x, coords3.z);

        let temp_noise = HyperbolicNoise::new(ENVIRO_SEED_TEMP ^ spice)
            .fbm2(coords2 * ENVIRO_TEMP_FREQ, ENVIRO_FBM);
        let rain_noise = HyperbolicNoise::new(ENVIRO_SEED_RAIN ^ spice)
            .fbm2(coords2 * ENVIRO_RAIN_FREQ, ENVIRO_FBM);
        let height_noise = HyperbolicNoise::new(ENVIRO_SEED_HEIGHT ^ spice)
            .fbm2(coords2 * ENVIRO_HEIGHT_FREQ, ENVIRO_FBM);
        let block_noise = HyperbolicNoise::new(ENVIRO_SEED_BLOCK ^ spice)
            .fbm3(coords3 * ENVIRO_BLOCK_FREQ, ENVIRO_FBM);

        Self {
            max_elevation: (height_noise as f32) * ENVIRO_HEIGHT_VARIANCE,
            temperature: normalize_noise(temp_noise),
            rainfall: normalize_noise(rain_noise),
            blockiness: normalize_noise(block_noise),
        }
    }
}
impl From<EnviroFactors> for (f32, f32, f32, f32) {
    fn from(envirofactors: EnviroFactors) -> Self {
        (
            envirofactors.max_elevation,
            envirofactors.temperature,
            envirofactors.rainfall,
            envirofactors.blockiness,
        )
    }
}
struct ChunkIncidentEnviroFactors {
    max_elevations: [f32; 8],
    temperatures: [f32; 8],
    rainfalls: [f32; 8],
    #[allow(dead_code)]
    blockinesses: [f32; 8],
}

fn hyperbolic_coords_from_transform(transform: &MIsometry<f32>) -> na::Vector3<f64> {
    let point = transform * MPoint::origin();
    let v4: na::Vector4<f32> = point.into();
    na::Vector3::new(v4.x.asinh() as f64, v4.y.asinh() as f64, v4.z.asinh() as f64)
}

fn normalize_noise(value: f64) -> f32 {
    ((value as f32) * 0.5 + 0.5).clamp(0.0, 1.0)
}

/// Returns the max_elevation values for the nodes that are incident to this chunk,
/// sorted and converted to f32 for use in functions like trilerp.
///
/// Returns `None` if not all incident nodes are populated.
fn chunk_incident_enviro_factors(graph: &mut Graph, chunk: ChunkId) -> ChunkIncidentEnviroFactors {
    let mut i = chunk.vertex.dual_vertices().map(|(_, path)| {
        let node = path.fold(chunk.node, |node, side| graph.ensure_neighbor(node, side));
        graph.ensure_node_state(node);
        graph.node_state(node).enviro
    });

    // this is a bit cursed, but I don't want to collect into a vec because perf,
    // and I can't just return an iterator because then something still references graph.
    let (e1, t1, r1, b1) = i.next().unwrap().into();
    let (e2, t2, r2, b2) = i.next().unwrap().into();
    let (e3, t3, r3, b3) = i.next().unwrap().into();
    let (e4, t4, r4, b4) = i.next().unwrap().into();
    let (e5, t5, r5, b5) = i.next().unwrap().into();
    let (e6, t6, r6, b6) = i.next().unwrap().into();
    let (e7, t7, r7, b7) = i.next().unwrap().into();
    let (e8, t8, r8, b8) = i.next().unwrap().into();

    ChunkIncidentEnviroFactors {
        max_elevations: [e1, e2, e3, e4, e5, e6, e7, e8],
        temperatures: [t1, t2, t3, t4, t5, t6, t7, t8],
        rainfalls: [r1, r2, r3, r4, r5, r6, r7, r8],
        blockinesses: [b1, b2, b3, b4, b5, b6, b7, b8],
    }
}

/// Linearly interpolate at interior and boundary of a cube given values at the eight corners.
fn trilerp<N: na::RealField + Copy>(
    &[v000, v001, v010, v011, v100, v101, v110, v111]: &[N; 8],
    t: na::Vector3<N>,
) -> N {
    fn lerp<N: na::RealField + Copy>(v0: N, v1: N, t: N) -> N {
        v0 * (N::one() - t) + v1 * t
    }
    fn bilerp<N: na::RealField + Copy>(v00: N, v01: N, v10: N, v11: N, t: na::Vector2<N>) -> N {
        lerp(lerp(v00, v01, t.x), lerp(v10, v11, t.x), t.y)
    }

    lerp(
        bilerp(v000, v100, v010, v110, t.xy()),
        bilerp(v001, v101, v011, v111, t.xy()),
        t.z,
    )
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

fn hash(a: u64, b: u64) -> u64 {
    use std::ops::BitXor;
    a.rotate_left(5)
        .bitxor(b)
        .wrapping_mul(0x517c_c1b7_2722_0a95)
}

mod remcpe {
    use super::*;
    use crate::hyper_noise::{FbmConfig, HyperbolicNoise};
    use rand::{Rng, SeedableRng};
    use rand_pcg::Pcg64Mcg;

    const WORLD_SEED: u32 = 0x5eed5eed;
    const SEA_LEVEL: f32 = 0.0;
    const MC_MIN_HEIGHT: f32 = -96.0;
    const MC_MAX_HEIGHT: f32 = 96.0;
    const MC_BEDROCK_FLOOR: f32 = -96.0;
    const HEIGHT_MULTIPLIER: f32 = 1.25;
    #[allow(dead_code)]
    const FLAT_COLUMN_HEIGHT: f32 = 0.0;
    const DEFAULT_SURFACE_DEPTH_MIN: u8 = 1;
    const DEFAULT_SURFACE_DEPTH_MAX: u8 = 3;
    const CAVE_PROBABILITY: f32 = 0.0125;
    const HEIGHT_NOISE_BASE_FREQ: f64 = 1.0 / 64.0;
    const HEIGHT_NOISE_AMPLITUDE: f32 = 24.0;
    const HEIGHT_NOISE_FBM: FbmConfig = FbmConfig {
        octaves: 4,
        lacunarity: 2.0,
        gain: 0.5,
    };

    fn chunk_height_span(params: &ChunkParams) -> f32 {
        f32::from(params.dimension) * HEIGHT_MULTIPLIER
    }

    fn scale_height_to_chunk(params: &ChunkParams, value: f32) -> f32 {
        (value / MC_MAX_HEIGHT) * chunk_height_span(params)
    }

    pub fn generate_chunk(params: &ChunkParams) -> VoxelData {
        let mut voxels = VoxelData::Solid(BlockKind::Air.id());
        let mut columns = build_columns(params);
        {
            let data = voxels.data_mut(params.dimension);
            fill_base_layers(params, data, &mut columns);
            carve_caves(params, data, &columns);
            populate_ores(params, data, &columns);
            populate_surface_features(params, data, &mut columns);
        }
        margins::initialize_margins(params.dimension, &mut voxels);
        voxels
    }

    #[derive(Clone)]
    struct ColumnData {
        horizontal: na::Vector2<f32>,
        height: f32,
        biome: BiomeKind,
        top_block: BlockID,
        filler_block: BlockID,
        surface_depth: u8,
        freeze_water: bool,
        vegetation_bias: f32,
        top_voxel: Option<na::Vector3<u8>>,
        top_depth: f32,
    }

    impl ColumnData {
        fn new() -> Self {
            Self {
                horizontal: na::Vector2::zeros(),
                height: 0.0,
                biome: BiomeKind::Plains,
                top_block: BlockKind::Grass.id(),
                filler_block: BlockKind::Dirt.id(),
                surface_depth: 1,
                freeze_water: false,
                vegetation_bias: 0.0,
                top_voxel: None,
                top_depth: f32::INFINITY,
            }
        }

        fn idx(params: &ChunkParams, coords: na::Vector3<u8>) -> usize {
            params.orientation().column_index(params.dimension, coords)
        }

        fn idx_components(params: &ChunkParams, axis0: u8, axis1: u8) -> usize {
            params
                .orientation()
                .column_index_components(params.dimension, axis0, axis1)
        }
    }

    #[derive(Clone, Copy, Debug)]
    enum BiomeKind {
        Plains,
        Desert,
        Forest,
        RainForest,
        SeasonalForest,
        Taiga,
        Tundra,
    }

    impl BiomeKind {
        fn classify(temp: f32, rain: f32) -> Self {
            if temp < 0.25 {
                return BiomeKind::Tundra;
            }
            if temp < 0.4 {
                if rain > 0.5 {
                    BiomeKind::Taiga
                } else {
                    BiomeKind::Plains
                }
            } else if temp > 0.8 && rain < 0.2 {
                BiomeKind::Desert
            } else if rain > 0.8 && temp > 0.7 {
                BiomeKind::RainForest
            } else if rain > 0.6 {
                BiomeKind::Forest
            } else if rain > 0.3 {
                BiomeKind::SeasonalForest
            } else {
                BiomeKind::Plains
            }
        }

        fn height_offset(self) -> f32 {
            match self {
                BiomeKind::Desert => -6.0,
                BiomeKind::Plains => -2.0,
                BiomeKind::Forest => 4.0,
                BiomeKind::RainForest => 6.0,
                BiomeKind::SeasonalForest => 3.0,
                BiomeKind::Taiga => 2.0,
                BiomeKind::Tundra => -4.0,
            }
        }

        fn top_block(self) -> BlockID {
            match self {
                BiomeKind::Desert => BlockKind::Sand.id(),
                BiomeKind::Tundra => BlockKind::SnowBlock.id(),
                _ => BlockKind::Grass.id(),
            }
        }

        fn filler_block(self) -> BlockID {
            match self {
                BiomeKind::Desert => BlockKind::Sandstone.id(),
                BiomeKind::Tundra => BlockKind::Dirt.id(),
                _ => BlockKind::Dirt.id(),
            }
        }

        fn freeze_water(self) -> bool {
            matches!(self, BiomeKind::Taiga | BiomeKind::Tundra)
        }

        fn tree_weight(self) -> i32 {
            match self {
                BiomeKind::RainForest => 8,
                BiomeKind::Forest => 6,
                BiomeKind::SeasonalForest => 5,
                BiomeKind::Taiga => 4,
                BiomeKind::Plains => 1,
                _ => 0,
            }
        }
    }

    fn build_columns(params: &ChunkParams) -> Vec<ColumnData> {
        let dimension = params.dimension as usize;
        let mut columns = vec![ColumnData::new(); dimension * dimension];
        let dim_f = f32::from(params.dimension);
        let orientation = params.orientation();
        let [axis0, axis1] = orientation.horizontal_axes();
        let noise = HyperbolicNoise::new(WORLD_SEED as u64);

        for a in 0..params.dimension {
            for b in 0..params.dimension {
                let idx = ColumnData::idx_components(params, a, b);
                let mut local = na::Vector3::repeat(0.5);
                local[axis0 as usize] = (f32::from(a) + 0.5) / dim_f;
                local[axis1 as usize] = (f32::from(b) + 0.5) / dim_f;
                let trilerp_coords = local.map(|x| (1.0 - x) * 0.5);
                let hyper_coords = params.hyperbolic_block_coords(local);
                let hx = hyper_coords.x as f32;
                let hz = hyper_coords.y as f32;
                let noise_coords = hyper_coords * HEIGHT_NOISE_BASE_FREQ;
                let height_variation = noise.fbm2(noise_coords, HEIGHT_NOISE_FBM) as f32;
                let env_temp = trilerp(&params.env.temperatures, trilerp_coords).clamp(0.0, 1.0);
                let env_rain = trilerp(&params.env.rainfalls, trilerp_coords).clamp(0.0, 1.0);
                let env_height = trilerp(&params.env.max_elevations, trilerp_coords);

                let seed = hash(
                    params.node_spice,
                    hash(WORLD_SEED as u64, hash(a as u64, b as u64)),
                );
                let mut column_rng = Pcg64Mcg::seed_from_u64(seed);
                let temp = (env_temp + column_rng.random_range(-0.025..0.025)).clamp(0.0, 1.0);
                let rain = (env_rain + column_rng.random_range(-0.05..0.05)).clamp(0.0, 1.0);
                let biome = BiomeKind::classify(temp, rain);
                let mut height = env_height + biome.height_offset();
                height += height_variation * HEIGHT_NOISE_AMPLITUDE;
                height = height.clamp(MC_MIN_HEIGHT, MC_MAX_HEIGHT);
                let scaled_height = scale_height_to_chunk(params, height);
                let surface_depth = column_rng
                    .random_range(DEFAULT_SURFACE_DEPTH_MIN..=DEFAULT_SURFACE_DEPTH_MAX)
                    .max(1);

                let mut column = ColumnData::new();
                column.horizontal = na::Vector2::new(hx, hz);
                column.height = scaled_height;
                column.biome = biome;
                column.top_block = biome.top_block();
                column.filler_block = biome.filler_block();
                column.surface_depth = surface_depth;
                column.freeze_water = biome.freeze_water();
                column.vegetation_bias = rain;
                columns[idx] = column;
            }
        }

        columns
    }

    fn fill_base_layers(params: &ChunkParams, voxels: &mut [BlockID], columns: &mut [ColumnData]) {
        let mut rng = Pcg64Mcg::seed_from_u64(params.node_spice);
        let bedrock_floor = scale_height_to_chunk(params, MC_BEDROCK_FLOOR);
        let bedrock_variance = scale_height_to_chunk(params, 4.0);
        let sea_level = scale_height_to_chunk(params, SEA_LEVEL);
        for (coords_x, coords_y, coords_z) in VoxelCoords::new(params.dimension) {
            let coords = na::Vector3::new(coords_x, coords_y, coords_z);
            let column_idx = ColumnData::idx(params, coords);
            let column = &mut columns[column_idx];
            let center = voxel_center(params.dimension, coords);
            let block_height = -params.surface.distance_to_chunk(params.chunk, &center)
                * f32::from(params.dimension);
            let mut block = BlockKind::Air.id();

            if block_height <= bedrock_floor + rng.random_range(0.0..bedrock_variance) {
                block = BlockKind::Bedrock.id();
            } else if block_height <= column.height {
                let depth = column.height - block_height;
                if depth < column.surface_depth as f32 {
                    block = column.top_block;
                    if depth < column.top_depth {
                        column.top_voxel = Some(coords);
                        column.top_depth = depth;
                    }
                } else if depth < column.surface_depth as f32 + 3.0 {
                    block = column.filler_block;
                } else {
                    block = BlockKind::Stone.id();
                }
            } else if block_height <= sea_level {
                block = if column.freeze_water && block_height >= sea_level - 1.0 {
                    BlockKind::Ice.id()
                } else {
                    BlockKind::Water.id()
                };
            }

            if block != BlockKind::Air.id() {
                voxels[index(params.dimension, coords)] = block;
            }
        }
    }

    fn carve_caves(params: &ChunkParams, voxels: &mut [BlockID], columns: &[ColumnData]) {
        let min_cave_height = scale_height_to_chunk(params, MC_MIN_HEIGHT + 4.0);
        for (coords_x, coords_y, coords_z) in VoxelCoords::new(params.dimension) {
            let coords = na::Vector3::new(coords_x, coords_y, coords_z);
            let column_idx = ColumnData::idx(params, coords);
            let column = &columns[column_idx];
            let center = voxel_center(params.dimension, coords);
            let block_height = -params.surface.distance_to_chunk(params.chunk, &center)
                * f32::from(params.dimension);
            if block_height > column.height || block_height < min_cave_height {
                continue;
            }

            let hyper_coords = params.hyperbolic_block_coords_3d(center);
            let hx_bits = hyper_coords.x.to_bits() as u64;
            let hy_bits = hyper_coords.y.to_bits() as u64;
            let hz_bits = hyper_coords.z.to_bits() as u64;
            let coord_hash = hash(
                hx_bits ^ hy_bits,
                hz_bits ^ hash(coords_x as u64, coords_y as u64),
            );
            let sample_seed = hash(params.node_spice, hash(WORLD_SEED as u64, coord_hash));
            let carve_value = ((sample_seed >> 16) & 0xffff_ffff) as f32 / u32::MAX as f32;
            if carve_value < CAVE_PROBABILITY {
                let idx = index(params.dimension, coords);
                match voxels[idx] {
                    0 => {}
                    block if block == BlockKind::Bedrock.id() => {}
                    block => {
                        if block != BlockKind::Water.id() && block != BlockKind::Ice.id() {
                            voxels[idx] = BlockKind::Air.id();
                        }
                    }
                }
            }
        }
    }

    struct OreConfig {
        block: BlockID,
        attempts: u32,
        cluster: u8,
        min_height: f32,
        max_height: f32,
    }

    fn populate_ores(params: &ChunkParams, voxels: &mut [BlockID], _columns: &[ColumnData]) {
        let configs = [
            OreConfig {
                block: BlockKind::Dirt.id(),
                attempts: 20,
                cluster: 32,
                min_height: MC_MIN_HEIGHT,
                max_height: MC_MAX_HEIGHT,
            },
            OreConfig {
                block: BlockKind::Gravel.id(),
                attempts: 12,
                cluster: 24,
                min_height: MC_MIN_HEIGHT,
                max_height: MC_MAX_HEIGHT,
            },
            OreConfig {
                block: BlockKind::CoalOre.id(),
                attempts: 16,
                cluster: 16,
                min_height: -64.0,
                max_height: 64.0,
            },
            OreConfig {
                block: BlockKind::IronOre.id(),
                attempts: 20,
                cluster: 8,
                min_height: -32.0,
                max_height: 32.0,
            },
            OreConfig {
                block: BlockKind::GoldOre.id(),
                attempts: 2,
                cluster: 8,
                min_height: -16.0,
                max_height: 16.0,
            },
            OreConfig {
                block: BlockKind::RedstoneOre.id(),
                attempts: 8,
                cluster: 7,
                min_height: -24.0,
                max_height: 0.0,
            },
            OreConfig {
                block: BlockKind::DiamondOre.id(),
                attempts: 1,
                cluster: 6,
                min_height: -16.0,
                max_height: -4.0,
            },
        ];

        let mut rng = Pcg64Mcg::seed_from_u64(hash(params.node_spice, params.chunk as u64));
        let density_scale = params.vertical_density_scale().clamp(0.01, 8.0);
        for cfg in configs {
            let min_height = scale_height_to_chunk(params, cfg.min_height);
            let max_height = scale_height_to_chunk(params, cfg.max_height);
            let scaled_attempts = cfg.attempts as f32 * density_scale;
            let mut attempts = scaled_attempts.floor() as u32;
            let fractional = scaled_attempts - attempts as f32;
            if fractional > 0.0 && rng.random_bool(fractional as f64) {
                attempts += 1;
            }
            for _ in 0..attempts {
                let x = rng.random_range(0..params.dimension);
                let z = rng.random_range(0..params.dimension);
                let y = rng.random_range(0..params.dimension);
                let coords = na::Vector3::new(x, y, z);
                let height = column_height(params, coords);
                if height < min_height || height > max_height {
                    continue;
                }
                place_ore_cluster(params, voxels, coords, cfg.cluster, cfg.block, &mut rng);
            }
        }
    }

    fn place_ore_cluster(
        params: &ChunkParams,
        voxels: &mut [BlockID],
        start: na::Vector3<u8>,
        cluster: u8,
        block: BlockID,
        rng: &mut Pcg64Mcg,
    ) {
        let dimension = params.dimension;
        for _ in 0..cluster {
            let offset = [
                rng.random_range(-1..=1),
                rng.random_range(-1..=1),
                rng.random_range(-1..=1),
            ];
            let mut coords = start;
            for axis in 0..3 {
                let value = coords[axis] as i16 + offset[axis] as i16;
                if value < 0 || value >= dimension as i16 {
                    continue;
                }
                coords[axis] = value as u8;
            }
            let idx = index(dimension, coords);
            if matches!(voxels[idx], b if b == BlockKind::Stone.id()) {
                voxels[idx] = block;
            }
        }
    }

    fn column_height(params: &ChunkParams, coords: na::Vector3<u8>) -> f32 {
        let center = voxel_center(params.dimension, coords);
        -params.surface.distance_to_chunk(params.chunk, &center) * f32::from(params.dimension)
    }

    fn populate_surface_features(
        params: &ChunkParams,
        voxels: &mut [BlockID],
        columns: &mut [ColumnData],
    ) {
        let mut rng =
            Pcg64Mcg::seed_from_u64(hash(params.node_spice, params.chunk as u64 ^ 0xfeed));
        let density_scale = params.vertical_density_scale().clamp(0.01, 1.0);
        for a in 0..params.dimension {
            for b in 0..params.dimension {
                let idx = ColumnData::idx_components(params, a, b);
                let column = &columns[idx];
                if column.top_voxel.is_none() {
                    continue;
                }
                let top = column.top_voxel.unwrap();
                match column.biome {
                    BiomeKind::Desert => {
                        let cactus_prob = (0.05 * density_scale).min(1.0);
                        if rng.random_bool(cactus_prob as f64) {
                            place_cactus(params, voxels, top, rng.random_range(2..5));
                        }
                    }
                    _ => {
                        if column.biome.tree_weight() > 0 {
                            let base_prob = column.biome.tree_weight() as f32 / 10.0;
                            let tree_prob = (base_prob * density_scale).clamp(0.0, 1.0);
                            if rng.random_bool(tree_prob as f64) {
                                let _ = place_tree(params, voxels, top, &mut rng);
                                continue;
                            }
                        }
                        let flower_prob = (0.05 * density_scale).min(1.0);
                        if rng.random_bool(flower_prob as f64) {
                            place_flower(params, voxels, top);
                            continue;
                        }
                        let mushroom_prob = (0.03 * density_scale).min(1.0);
                        if rng.random_bool(mushroom_prob as f64) {
                            place_mushroom(params, voxels, top, rng.random_bool(0.5));
                        }
                    }
                }
            }
        }
    }

    fn place_tree(
        params: &ChunkParams,
        voxels: &mut [BlockID],
        base: na::Vector3<u8>,
        rng: &mut Pcg64Mcg,
    ) -> bool {
        let height = rng.random_range(4..7);
        let orientation = params.orientation();
        if !can_extend_along_up(params.dimension, base, &orientation, height as i32 + 2) {
            return false;
        }
        let idx = index(params.dimension, base);
        let soil = voxels[idx];
        if soil != BlockKind::Grass.id() && soil != BlockKind::Dirt.id() {
            return false;
        }
        let mut pos = base.map(|c| i32::from(c));
        for _ in 0..height {
            pos = offset_along_axis(pos, orientation.up_axis(), orientation.up_sign() as i32);
            if !in_bounds(pos, params.dimension) {
                return false;
            }
            let coords = vec_i32_to_u8(pos);
            let idx = index(params.dimension, coords);
            voxels[idx] = BlockKind::Log.id();
        }
        let mut leaf_base = base.map(|c| i32::from(c));
        leaf_base = offset_along_axis(
            leaf_base,
            orientation.up_axis(),
            orientation.up_sign() as i32 * height as i32,
        );
        let [axis_a, axis_b] = orientation.horizontal_axes();
        for dx in -2i32..=2i32 {
            for dz in -2i32..=2i32 {
                for dy in 0i32..=2i32 {
                    if dx.abs() + dz.abs() + dy > 4 {
                        continue;
                    }
                    let mut coords = leaf_base;
                    coords[axis_a as usize] += dx;
                    coords[axis_b as usize] += dz;
                    coords[orientation.up_axis() as usize] += dy * orientation.up_sign() as i32;
                    if !in_bounds(coords, params.dimension) {
                        continue;
                    }
                    let coords_u8 = vec_i32_to_u8(coords);
                    let idx = index(params.dimension, coords_u8);
                    if voxels[idx] == BlockKind::Air.id() {
                        voxels[idx] = BlockKind::Leaves.id();
                    }
                }
            }
        }
        true
    }

    fn place_cactus(
        params: &ChunkParams,
        voxels: &mut [BlockID],
        base: na::Vector3<u8>,
        height: u8,
    ) {
        let orientation = params.orientation();
        if !can_extend_along_up(params.dimension, base, &orientation, height as i32 + 1) {
            return;
        }
        let mut pos = base.map(|c| i32::from(c));
        for _ in 0..height {
            pos = offset_along_axis(pos, orientation.up_axis(), orientation.up_sign() as i32);
            if !in_bounds(pos, params.dimension) {
                break;
            }
            let coords = vec_i32_to_u8(pos);
            let idx = index(params.dimension, coords);
            voxels[idx] = BlockKind::Cactus.id();
        }
    }

    fn place_flower(params: &ChunkParams, voxels: &mut [BlockID], base: na::Vector3<u8>) {
        if let Some(coords) = neighbor_up(params.dimension, base, params.orientation()) {
            let idx = index(params.dimension, coords);
            if voxels[idx] == BlockKind::Air.id() {
                voxels[idx] = BlockKind::Flower.id();
            }
        }
    }

    fn place_mushroom(
        params: &ChunkParams,
        voxels: &mut [BlockID],
        base: na::Vector3<u8>,
        brown: bool,
    ) {
        if let Some(coords) = neighbor_up(params.dimension, base, params.orientation()) {
            let idx = index(params.dimension, coords);
            if voxels[idx] == BlockKind::Air.id() {
                voxels[idx] = if brown {
                    BlockKind::BrownMushroom.id()
                } else {
                    BlockKind::RedMushroom.id()
                };
            }
        }
    }

    fn neighbor_up(
        dimension: u8,
        base: na::Vector3<u8>,
        orientation: ChunkOrientation,
    ) -> Option<na::Vector3<u8>> {
        let next = offset_along_axis(
            base.map(|c| i32::from(c)),
            orientation.up_axis(),
            orientation.up_sign() as i32,
        );
        if !in_bounds(next, dimension) {
            None
        } else {
            Some(vec_i32_to_u8(next))
        }
    }

    fn can_extend_along_up(
        dimension: u8,
        base: na::Vector3<u8>,
        orientation: &ChunkOrientation,
        needed: i32,
    ) -> bool {
        let coord = i32::from(base[orientation.up_axis() as usize]);
        let room = if orientation.up_sign() > 0 {
            (i32::from(dimension) - 1) - coord
        } else {
            coord
        };
        room >= needed
    }

    fn offset_along_axis(
        mut coords: na::Vector3<i32>,
        axis: CoordAxis,
        delta: i32,
    ) -> na::Vector3<i32> {
        coords[axis as usize] += delta;
        coords
    }

    fn in_bounds(coords: na::Vector3<i32>, dimension: u8) -> bool {
        coords.x >= 0
            && coords.y >= 0
            && coords.z >= 0
            && coords.x < i32::from(dimension)
            && coords.y < i32::from(dimension)
            && coords.z < i32::from(dimension)
    }

    fn vec_i32_to_u8(coords: na::Vector3<i32>) -> na::Vector3<u8> {
        na::Vector3::new(coords.x as u8, coords.y as u8, coords.z as u8)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::*;
    use crate::{
        hyper_noise::HyperbolicNoise,
        voxel_math::{CoordAxis, CoordSign},
    };

    const CHUNK_SIZE: u8 = 12;
    const HEIGHT_BASE_FREQ: f64 = 1.0 / 64.0;

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
    fn check_chunk_incident_max_elevations() {
        let mut g = Graph::new(1);
        for (i, path) in Vertex::A.dual_vertices().map(|(_, p)| p).enumerate() {
            let new_node = path.fold(NodeId::ROOT, |node, side| g.ensure_neighbor(node, side));

            // assigning state
            g.ensure_node_state(new_node);
            g[new_node].state.as_mut().unwrap().enviro.max_elevation = i as f32 + 1.0;
        }

        let enviros = chunk_incident_enviro_factors(&mut g, ChunkId::new(NodeId::ROOT, Vertex::A));
        for (i, max_elevation) in enviros.max_elevations.into_iter().enumerate() {
            println!("{i}, {max_elevation}");
            assert_abs_diff_eq!(max_elevation, (i + 1) as f32, epsilon = 1e-8);
        }

        // see corresponding test for trilerp
        let center_max_elevation = trilerp(&enviros.max_elevations, na::Vector3::repeat(0.5));
        assert_abs_diff_eq!(center_max_elevation, 4.5, epsilon = 1e-8);

        let mut checked_center = false;
        let center = na::Vector3::repeat(CHUNK_SIZE / 2);
        'top: for z in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    let a = na::Vector3::new(x, y, z);
                    if a == center {
                        checked_center = true;
                        let c = center.map(|x| x as f32) / CHUNK_SIZE as f32;
                        let center_max_elevation = trilerp(&enviros.max_elevations, c);
                        assert_abs_diff_eq!(center_max_elevation, 4.5, epsilon = 1e-8);
                        break 'top;
                    }
                }
            }
        }

        if !checked_center {
            panic!("Never checked trilerping center max_elevation!");
        }
    }

    #[test]
    fn check_trilerp() {
        assert_abs_diff_eq!(
            1.0,
            trilerp(
                &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                na::Vector3::new(0.0, 0.0, 0.0),
            ),
            epsilon = 1e-8,
        );
        assert_abs_diff_eq!(
            1.0,
            trilerp(
                &[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                na::Vector3::new(1.0, 0.0, 0.0),
            ),
            epsilon = 1e-8,
        );
        assert_abs_diff_eq!(
            1.0,
            trilerp(
                &[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                na::Vector3::new(0.0, 1.0, 0.0),
            ),
            epsilon = 1e-8,
        );
        assert_abs_diff_eq!(
            1.0,
            trilerp(
                &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                na::Vector3::new(1.0, 1.0, 0.0),
            ),
            epsilon = 1e-8,
        );
        assert_abs_diff_eq!(
            1.0,
            trilerp(
                &[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                na::Vector3::new(0.0, 0.0, 1.0),
            ),
            epsilon = 1e-8,
        );
        assert_abs_diff_eq!(
            1.0,
            trilerp(
                &[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                na::Vector3::new(1.0, 0.0, 1.0),
            ),
            epsilon = 1e-8,
        );
        assert_abs_diff_eq!(
            1.0,
            trilerp(
                &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                na::Vector3::new(0.0, 1.0, 1.0),
            ),
            epsilon = 1e-8,
        );
        assert_abs_diff_eq!(
            1.0,
            trilerp(
                &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                na::Vector3::new(1.0, 1.0, 1.0),
            ),
            epsilon = 1e-8,
        );

        assert_abs_diff_eq!(
            0.5,
            trilerp(
                &[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                na::Vector3::new(0.5, 0.5, 0.5),
            ),
            epsilon = 1e-8,
        );
        assert_abs_diff_eq!(
            0.5,
            trilerp(
                &[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                na::Vector3::new(0.5, 0.5, 0.5),
            ),
            epsilon = 1e-8,
        );

        assert_abs_diff_eq!(
            4.5,
            trilerp(
                &[1.0, 5.0, 3.0, 7.0, 2.0, 6.0, 4.0, 8.0],
                na::Vector3::new(0.5, 0.5, 0.5),
            ),
            epsilon = 1e-8,
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

    fn convert_local_between_vertices(
        from: Vertex,
        to: Vertex,
        local: na::Vector3<f32>,
    ) -> na::Vector3<f32> {
        let mut point = from.chunk_to_node_f64()
            * na::Vector4::new(local.x as f64, local.y as f64, local.z as f64, 1.0);
        point /= point.w;
        let mut converted = to.node_to_chunk_f64() * point;
        converted /= converted.w;
        na::Vector3::new(converted.x as f32, converted.y as f32, converted.z as f32)
    }

    #[test]
    fn hyperbolic_noise_continuity_across_chunks() {
        let mut graph = Graph::new(CHUNK_SIZE);
        let chunk_a = ChunkId::new(NodeId::ROOT, Vertex::A);
        let chunk_b = graph
            .get_chunk_neighbor(chunk_a, CoordAxis::X, CoordSign::Plus)
            .expect("expected adjacent chunk");
        let params_a = ChunkParams::new(&mut graph, chunk_a, WorldgenPreset::Hyperbolic);
        let params_b = ChunkParams::new(&mut graph, chunk_b, WorldgenPreset::Hyperbolic);

        // Center of the shared face between the two chunks.
        let local_a = na::Vector3::new(1.0, 0.5, 0.5);
        let local_b = convert_local_between_vertices(chunk_a.vertex, chunk_b.vertex, local_a);
        assert!(
            local_b.iter().all(|c| (-1e-4..=1.0 + 1e-4).contains(c)),
            "converted local coords outside chunk: {:?}",
            local_b
        );

        let hyper_a = params_a.hyperbolic_noise_coords(local_a);
        let hyper_b = params_b.hyperbolic_noise_coords(local_b);
        let noise = HyperbolicNoise::new(0x5eed_5eed);
        let sample_a = noise.sample2(hyper_a);
        let sample_b = noise.sample2(hyper_b);
        assert!(
            (sample_a - sample_b).abs() < 1e-6,
            "noise discontinuity detected: {sample_a} vs {sample_b}"
        );
    }

    fn assert_face_continuity(
        params_a: &ChunkParams,
        params_b: &ChunkParams,
        chunk_a: ChunkId,
        chunk_b: ChunkId,
        axis: CoordAxis,
        sign: CoordSign,
    ) {
        const GRID_STEPS: usize = 6;
        const EPS: f64 = 1e-6;
        let mut base = [0.5f32; 3];
        base[axis as usize] = if matches!(sign, CoordSign::Plus) { 1.0 } else { 0.0 };
        let [axis_u, axis_v] = axis.other_axes();
        let noise = HyperbolicNoise::new(0x5eed_5eed);

        for i in 0..=GRID_STEPS {
            for j in 0..=GRID_STEPS {
                let mut local_a = na::Vector3::new(base[0], base[1], base[2]);
                local_a[axis_u as usize] = i as f32 / GRID_STEPS as f32;
                local_a[axis_v as usize] = j as f32 / GRID_STEPS as f32;

                let mut local_b = convert_local_between_vertices(chunk_a.vertex, chunk_b.vertex, local_a);
                assert!(
                    local_b
                        .iter()
                        .all(|c| (-1e-3..=1.0 + 1e-3).contains(c)),
                    "converted local coords outside chunk: {:?}",
                    local_b
                );
                local_b = local_b.map(|c| c.clamp(0.0, 1.0));

                let hyper2_a = params_a.hyperbolic_noise_coords(local_a);
                let hyper2_b = params_b.hyperbolic_noise_coords(local_b);
                assert!(
                    (hyper2_a - hyper2_b).norm() < 1e-6,
                    "2D noise coords diverged across face: {:?} vs {:?}",
                    hyper2_a,
                    hyper2_b
                );
                let hyper3_a = params_a.hyperbolic_noise_coords_3d(local_a);
                let hyper3_b = params_b.hyperbolic_noise_coords_3d(local_b);
                assert!(
                    (hyper3_a - hyper3_b).norm() < 1e-6,
                    "3D noise coords diverged across face"
                );

                let block2_a = params_a.hyperbolic_block_coords(local_a);
                let block2_b = params_b.hyperbolic_block_coords(local_b);
                assert!(
                    (block2_a - block2_b).norm() < 1e-5,
                    "block coords diverged across face"
                );
                let block3_a = params_a.hyperbolic_block_coords_3d(local_a);
                let block3_b = params_b.hyperbolic_block_coords_3d(local_b);
                assert!(
                    (block3_a - block3_b).norm() < 1e-5,
                    "block 3d coords diverged across face"
                );

                let sample2_a = noise.sample2(hyper2_a);
                let sample2_b = noise.sample2(hyper2_b);
                assert!(
                    (sample2_a - sample2_b).abs() < EPS,
                    "2D noise sample discontinuity: {sample2_a} vs {sample2_b}"
                );

                let sample3_a = noise.sample3(hyper3_a);
                let sample3_b = noise.sample3(hyper3_b);
                assert!(
                    (sample3_a - sample3_b).abs() < EPS,
                    "3D noise sample discontinuity: {sample3_a} vs {sample3_b}"
                );

                let height_a = noise.sample2(block2_a * HEIGHT_BASE_FREQ);
                let height_b = noise.sample2(block2_b * HEIGHT_BASE_FREQ);
                assert!(
                    (height_a - height_b).abs() < EPS,
                    "height noise discontinuity: {height_a} vs {height_b}"
                );
            }
        }
    }

    #[test]
    #[ignore = "fails until hyperbolic noise continuity across chunk faces is fixed"]
    fn hyperbolic_noise_face_stress_test() {
        const STRIP_STEPS: usize = 5;
        let mut graph = Graph::new(CHUNK_SIZE);

        for axis in CoordAxis::iter() {
            for sign in CoordSign::iter() {
                let mut current = ChunkId::new(NodeId::ROOT, Vertex::A);
                for _ in 0..STRIP_STEPS {
                    if matches!(sign, CoordSign::Minus) {
                        let side = current.vertex.canonical_sides()[axis as usize];
                        graph.ensure_neighbor(current.node, side);
                    }
                    let Some(neighbor) = graph.get_chunk_neighbor(current, axis, sign) else {
                        break;
                    };
                    let params_a = ChunkParams::new(&mut graph, current, WorldgenPreset::Hyperbolic);
                    let params_b = ChunkParams::new(&mut graph, neighbor, WorldgenPreset::Hyperbolic);
                    assert_face_continuity(&params_a, &params_b, current, neighbor, axis, sign);
                    current = neighbor;
                }
            }
        }
    }

    #[test]
    #[ignore]
    fn debug_hyper_noise_scale() {
        let mut graph = Graph::new(CHUNK_SIZE);
        let chunk = ChunkId::new(NodeId::ROOT, Vertex::A);
        let params = ChunkParams::new(&mut graph, chunk, WorldgenPreset::Hyperbolic);
        let center = na::Vector3::repeat(0.5);
        let step = 1.0 / f32::from(params.dimension);
        let x_offset = center + na::Vector3::new(step, 0.0, 0.0);
        let c0 = params.hyperbolic_noise_coords(center);
        let c1 = params.hyperbolic_noise_coords(x_offset);
        let axes = params.orientation().horizontal_axes();
        for axis in axes {
            let mut axis_offset = center;
            axis_offset[axis as usize] = (axis_offset[axis as usize] + step).clamp(0.0, 1.0);
            let delta = params.hyperbolic_noise_coords(axis_offset) - c0;
            eprintln!("axis {:?} delta {:?} norm {:.6}", axis, delta, delta.norm());
        }
        panic!(
            "center {:?} offset {:?} delta {:?} horizontal_scale {:.3} vertical_scale {:.3}",
            c0,
            c1,
            c1 - c0,
            params.hyper_block_scale,
            params.hyper_vertical_block_scale,
        );
    }
}
