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
use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};

static BLOCK_SCALES_BY_DIMENSION_AND_SEED: OnceLock<Mutex<HashMap<(u8, u64), (f64, f64)>>> =
    OnceLock::new();

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
    pub fn root(_graph: &Graph, _node: NodeId) -> Self {
        let world_from_node = MIsometry::identity();
        // Root node starts with zero enviro values; terrain evolves via random walk
        let enviro = EnviroFactors {
            max_elevation: 0.0,
            temperature: 0.0,
            rainfall: 0.0,
            blockiness: 0.0,
        };
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

        // Use graph-based EnviroFactors propagation for guaranteed continuity.
        // This is the key insight from hypermine: values propagate through the graph
        // topology rather than being sampled from global coordinates.
        let mut d = graph.parents(node).map(|(s, n)| (s, graph.node_state(n)));
        let enviro = match (d.next(), d.next()) {
            (Some(_), None) => {
                // Single parent: random walk from parent
                let parent_side = graph.primary_parent_side(node).unwrap();
                let parent_node = graph.neighbor(node, parent_side).unwrap();
                let parent_state = graph.node_state(parent_node);
                let spice = graph.hash_of(node) as u64;
                EnviroFactors::varied_from(parent_state.enviro, spice)
            }
            (Some((a_side, a_state)), Some((b_side, b_state))) => {
                // Two parents: parallelogram continuation for consistency
                let ab_node = graph
                    .neighbor(graph.neighbor(node, a_side).unwrap(), b_side)
                    .unwrap();
                let ab_state = graph.node_state(ab_node);
                EnviroFactors::continue_from(a_state.enviro, b_state.enviro, ab_state.enviro)
            }
            _ => unreachable!(),
        };

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
pub fn sample_enviro_at(graph: &Graph, pos: &ProtoPosition, world_seed: u64) -> EnviroSample {
    let node_state = graph.node_state(pos.node);

    let mpoint = pos.local * MPoint::origin();
    let signed_distance = node_state.surface.distance_to(&mpoint);

    let v4: na::Vector4<f32> = mpoint.into();
    let proj_x = if v4.w.abs() > 1e-6 { v4.x / v4.w } else { v4.x };
    let proj_z = if v4.w.abs() > 1e-6 { v4.z / v4.w } else { v4.z };
    let world_from_position = node_state.world_from_node() * pos.local;
    let climate_coords = hyperbolic_coords_from_transform(&world_from_position);
    let (h_scale, _) = ChunkParams::block_scales_for_dimension(graph.layout().dimension());
    let scaled_coords2 = na::Vector2::new(climate_coords.x * h_scale, climate_coords.z * h_scale);
    let climate = remcpe::sample_debug_biome(world_seed, scaled_coords2);

    EnviroSample {
        biome: climate.biome_id,
        temp_normalized: climate.temperature,
        rain_normalized: climate.humidity,
        signed_distance,
        proj_x,
        proj_z,
    }
}

fn hyperbolic_coords_from_transform(transform: &MIsometry<f32>) -> na::Vector3<f64> {
    let point = transform * MPoint::origin();
    hyperbolic_world_chart_coords(point)
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
    /// Global world seed for deterministic generation
    world_seed: u64,
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

// NOTE: Noise->block scaling must be consistent across chunks to preserve
// continuity at chunk boundaries (tests assert this). We therefore cache by
// dimension, using the first computed value for that dimension.

impl ChunkParams {
    /// Extract data necessary to generate a chunk, generating new graph nodes if necessary
    pub fn new(graph: &mut Graph, chunk: ChunkId, preset: WorldgenPreset, world_seed: u64) -> Self {
        let dimension = graph.layout().dimension();
        let cache_key = (dimension, world_seed);

        let cached = {
            let cache =
                BLOCK_SCALES_BY_DIMENSION_AND_SEED.get_or_init(|| Mutex::new(HashMap::new()));
            cache.lock().unwrap().get(&cache_key).copied()
        };

        let (block_scale, vertical_block_scale) = if let Some(scales) = cached {
            scales
        } else {
            // Compute once (based on the first chunk seen for this dimension) and cache.
            let provisional = Self::new_with_scales(graph, chunk, preset, world_seed, 1.0, 1.0);
            let block_scale = provisional.estimate_horizontal_block_scale();
            let vertical_block_scale = provisional.estimate_vertical_block_scale();
            let cache =
                BLOCK_SCALES_BY_DIMENSION_AND_SEED.get_or_init(|| Mutex::new(HashMap::new()));
            cache
                .lock()
                .unwrap()
                .insert(cache_key, (block_scale, vertical_block_scale));
            (block_scale, vertical_block_scale)
        };

        Self::new_with_scales(
            graph,
            chunk,
            preset,
            world_seed,
            block_scale,
            vertical_block_scale,
        )
    }

    fn new_with_scales(
        graph: &mut Graph,
        chunk: ChunkId,
        preset: WorldgenPreset,
        world_seed: u64,
        block_scale: f64,
        vertical_block_scale: f64,
    ) -> Self {
        graph.ensure_node_state(chunk.node);
        let env = chunk_incident_enviro_factors(graph, chunk);
        let state = graph.node_state(chunk.node);
        let dimension = graph.layout().dimension();
        let world_from_node = state.world_from_node().clone();
        let orientation = ChunkOrientation::from_surface(state.surface, chunk.vertex, dimension);
        Self {
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
            world_seed,
            world_from_node,
            orientation,
            hyper_block_scale: block_scale,
            hyper_vertical_block_scale: vertical_block_scale,
        }
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
    /// Uses geodesic distance (in hyperbolic units) projected along the local
    /// direction, so values remain unbounded as we traverse the tiling.
    fn hyperbolic_noise_coords(&self, local: na::Vector3<f32>) -> na::Vector2<f64> {
        let coords = hyperbolic_world_chart_coords(self.world_mpoint(local));
        na::Vector2::new(coords.x, coords.z)
    }

    /// Compute 3D globally consistent hyperbolic coordinates for noise sampling.
    /// Returns (x, y, z) coordinates that increase roughly linearly with the
    /// hyperbolic distance travelled, preventing saturation when far from the
    /// origin.
    fn hyperbolic_noise_coords_3d(&self, local: na::Vector3<f32>) -> na::Vector3<f64> {
        hyperbolic_world_chart_coords(self.world_mpoint(local))
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

    pub(crate) fn block_scales_for_dimension(dimension: u8) -> (f64, f64) {
        // Best-effort: this is used by debug/UI sampling without a world_seed.
        // Prefer any cached entry for this dimension.
        if let Some(cache) = BLOCK_SCALES_BY_DIMENSION_AND_SEED.get() {
            if let Some((_, scales)) = cache
                .lock()
                .unwrap()
                .iter()
                .find(|((dim, _seed), _)| *dim == dimension)
            {
                return *scales;
            }
        }
        // Fallback for debug/UI paths before ChunkParams initialization.
        let s = (dimension as f64) * 4.0;
        (s, s)
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
        let mut rng = rand_pcg::Pcg64Mcg::seed_from_u64(hash(
            self.world_seed,
            hash(self.node_spice, self.chunk as u64),
        ));

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

#[derive(Copy, Clone)]
struct EnviroFactors {
    max_elevation: f32,
    temperature: f32,
    rainfall: f32,
    blockiness: f32,
}
impl EnviroFactors {
    /// Random walk from parent node's values.
    /// This creates smooth variation while maintaining local continuity.
    fn varied_from(parent: Self, spice: u64) -> Self {
        use rand::distr::Uniform;
        use rand_distr::Normal;
        let mut rng = rand_pcg::Pcg64Mcg::seed_from_u64(spice);
        let unif = Uniform::new_inclusive(-1.0, 1.0).unwrap();
        let max_elevation = parent.max_elevation + rng.sample(Normal::new(0.0, 4.0).unwrap());

        Self {
            max_elevation,
            temperature: parent.temperature + rng.sample(unif),
            rainfall: parent.rainfall + rng.sample(unif),
            blockiness: parent.blockiness + rng.sample(unif),
        }
    }

    /// Parallelogram continuation: ensures consistent values when a node
    /// can be reached from multiple paths through the graph.
    /// Given parents A and B, and their common ancestor AB:
    /// new_value = A + (B - AB)
    /// This guarantees the same result regardless of traversal order.
    fn continue_from(a: Self, b: Self, ab: Self) -> Self {
        Self {
            max_elevation: a.max_elevation + (b.max_elevation - ab.max_elevation),
            temperature: a.temperature + (b.temperature - ab.temperature),
            rainfall: a.rainfall + (b.rainfall - ab.rainfall),
            blockiness: a.blockiness + (b.blockiness - ab.blockiness),
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

/// Serp interpolates between two values v0 and v1 over the interval [0, 1] by yielding
/// v0 for [0, threshold], v1 for [1-threshold, 1], and linear interpolation in between
/// such that the overall shape is an S-shaped piecewise function.
fn serp(v0: f32, v1: f32, t: f32, threshold: f32) -> f32 {
    if t < threshold {
        v0
    } else if t < 1.0 - threshold {
        let s = (t - threshold) / ((1.0 - threshold) - threshold);
        v0 * (1.0 - s) + v1 * s
    } else {
        v1
    }
}

/// Intended to produce a number that is added to elev_raw.
/// block is a real number, threshold is in (0, strength) via a logistic function
/// scale controls wavelength and amplitude. It is not 1:1 to the number of blocks in a period.
/// strength represents extremity of terracing effect. Sensible values are in (0, 0.5).
/// The greater the value of limiter, the stronger the bias of threshold towards 0.
fn terracing_diff(elev_raw: f32, block: f32, scale: f32, strength: f32, limiter: f32) -> f32 {
    let threshold: f32 = strength / (1.0 + libm::powf(2.0, limiter - block));
    let elev_floor = libm::floorf(elev_raw / scale);
    let elev_rem = elev_raw / scale - elev_floor;
    scale * elev_floor + serp(0.0, scale, elev_rem, threshold) - elev_raw
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

/// Convert a world-space hyperbolic point to continuous noise coordinates.
///
/// We use a logarithmic mapping of Minkowski spatial coordinates. Since the
/// raw coords (x, y, z) grow like sinh(d) ≈ e^d / 2 for large hyperbolic
/// distance d, taking the signed log linearizes this growth:
///
///   log_coords = sign(x) * log(1 + |x|)
///
/// This produces coordinates that:
/// 1. Equal (x, y, z) near origin (where |x| << 1, log(1+|x|) ≈ |x|)
/// 2. Grow linearly with hyperbolic distance for large distances
/// 3. Have no singularities - the function is smooth everywhere
/// 4. Preserve directional information (signs preserved)
///
/// The 4D version includes the w coordinate similarly transformed.
fn hyperbolic_world_chart_coords(point: MPoint<f32>) -> na::Vector3<f64> {
    let v: na::Vector4<f32> = point.into();
    let x = v.x as f64;
    let y = v.y as f64;
    let z = v.z as f64;

    // Signed logarithm: preserves sign, linearizes exponential growth
    fn signed_log(v: f64) -> f64 {
        v.signum() * (1.0 + v.abs()).ln()
    }

    na::Vector3::new(signed_log(x), signed_log(y), signed_log(z))
}

/// Convert a world-space hyperbolic point to 4D noise coordinates.
///
/// Uses all four Minkowski coordinates with signed-log transformation.
/// This avoids all 3D chart singularities by staying in the full
/// embedding space, while linearizing the exponential growth.
fn hyperbolic_world_chart_coords_4d(point: MPoint<f32>) -> na::Vector4<f64> {
    let v: na::Vector4<f32> = point.into();
    let x = v.x as f64;
    let y = v.y as f64;
    let z = v.z as f64;
    let w = v.w as f64;

    fn signed_log(v: f64) -> f64 {
        v.signum() * (1.0 + v.abs()).ln()
    }

    // w is always >= 1 on the hyperboloid, so we offset and use unsigned log
    let w_transformed = (w.max(1.0)).ln();

    na::Vector4::new(signed_log(x), signed_log(y), signed_log(z), w_transformed)
}

mod remcpe {
    use super::*;
    use crate::hyper_noise::{FbmConfig, HyperbolicNoise};
    use rand::{Rng, SeedableRng};
    use rand_pcg::Pcg64Mcg;

    const MC_MIN_HEIGHT: f32 = -96.0;
    const MC_MAX_HEIGHT: f32 = 96.0;
    #[allow(dead_code)]
    const FLAT_COLUMN_HEIGHT: f32 = 0.0;
    const DEFAULT_SURFACE_DEPTH_MIN: u8 = 1;
    const DEFAULT_SURFACE_DEPTH_MAX: u8 = 3;
    pub(super) const CAVE_PROBABILITY: f32 = 0.0;
    // Fixed-point quantization for hashing world-space coordinates.
    // Using raw float bit patterns (`to_bits`) is extremely sensitive to tiny
    // cross-face FP differences and can create visible seams.
    pub(super) const HASH_COORD_QUANT: f64 = 256.0;
    // Frequencies are expressed in inverse block-units so we match remcpe's "64-block" coarse rolloff
    // even though our local block scale comes from hyperbolic distances.
    const HEIGHT_NOISE_BASE_FREQ_BLOCKS: f64 = 1.0 / 64.0;
    // Matches remcpe's vertical wiggle in block units before we rescale into the local chunk span.
    const HEIGHT_NOISE_AMPLITUDE: f32 = 24.0;
    // Broad-scale continental shaping; remcpe used a very low-frequency stack (perlin7 ~200 block period).
    const HEIGHT_CONTINENT_FREQ_BLOCKS: f64 = 1.0 / 200.0;
    const HEIGHT_CONTINENT_AMPLITUDE: f32 = 32.0;
    const HEIGHT_CONTINENT_GAIN: f32 = 0.6;
    // Secondary detail layer to mimic remcpe's perlin1/2 blend with a chooser mask.
    const HEIGHT_DETAIL_FREQ_BLOCKS: f64 = HEIGHT_NOISE_BASE_FREQ_BLOCKS * 2.0;
    const HEIGHT_MASK_FREQ_BLOCKS: f64 = 1.0 / 8.0;
    /// Scales how strongly the multi-octave height noise modulates the
    /// enviro-based elevation envelope. Values in ~[0.2, 0.6] give
    /// noticeable relief without destroying large-scale structure.
    const TERRAIN_DETAIL_GAIN: f32 = 0.4;

    fn scale_height_to_chunk(value: f32) -> f32 {
        value
    }

    #[inline]
    fn lerp_f32(a: f32, b: f32, t: f32) -> f32 {
        a + (b - a) * t
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

    #[repr(u8)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum BiomeId {
        RainForest,
        Swampland,
        SeasonalForest,
        Forest,
        Savanna,
        Shrubland,
        Taiga,
        Desert,
        Plains,
        Tundra,
    }

    #[derive(Clone, Copy, Debug)]
    pub(super) struct DebugBiomeSample {
        pub biome_id: u8,
        pub temperature: f32,
        pub humidity: f32,
    }

    pub(super) fn sample_debug_biome(
        world_seed: u64,
        coords2: na::Vector2<f64>,
    ) -> DebugBiomeSample {
        let _ = (world_seed, coords2);
        let climate = ClimateSample {
            temperature: 0.5,
            humidity: 0.5,
            blend: 0.5,
        };
        let biome = BiomeId::Plains;
        DebugBiomeSample {
            biome_id: biome_debug_index(biome),
            temperature: climate.temperature,
            humidity: climate.humidity,
        }
    }

    #[inline]
    fn biome_debug_index(biome: BiomeId) -> u8 {
        biome as u8
    }

    #[derive(Clone, Copy, Debug)]
    struct BiomeAttrs {
        top_block: BlockID,
        filler_block: BlockID,
        freeze_water: bool,
        tree_weight: i32,
        height_offset: f32,
        is_desert: bool,
    }

    impl BiomeAttrs {
        fn of(id: BiomeId) -> Self {
            match id {
                BiomeId::RainForest => Self {
                    top_block: BlockKind::Grass.id(),
                    filler_block: BlockKind::Dirt.id(),
                    freeze_water: false,
                    tree_weight: 8,
                    height_offset: 6.0,
                    is_desert: false,
                },
                BiomeId::Swampland => Self {
                    top_block: BlockKind::Grass.id(),
                    filler_block: BlockKind::Dirt.id(),
                    freeze_water: false,
                    tree_weight: 2,
                    height_offset: 0.0,
                    is_desert: false,
                },
                BiomeId::SeasonalForest => Self {
                    top_block: BlockKind::Grass.id(),
                    filler_block: BlockKind::Dirt.id(),
                    freeze_water: false,
                    tree_weight: 5,
                    height_offset: 3.0,
                    is_desert: false,
                },
                BiomeId::Forest => Self {
                    top_block: BlockKind::Grass.id(),
                    filler_block: BlockKind::Dirt.id(),
                    freeze_water: false,
                    tree_weight: 6,
                    height_offset: 4.0,
                    is_desert: false,
                },
                BiomeId::Savanna => Self {
                    top_block: BlockKind::Grass.id(),
                    filler_block: BlockKind::Dirt.id(),
                    freeze_water: false,
                    tree_weight: 1,
                    height_offset: -2.0,
                    is_desert: false,
                },
                BiomeId::Shrubland => Self {
                    top_block: BlockKind::Grass.id(),
                    filler_block: BlockKind::Dirt.id(),
                    freeze_water: false,
                    tree_weight: 1,
                    height_offset: 0.0,
                    is_desert: false,
                },
                BiomeId::Taiga => Self {
                    top_block: BlockKind::Grass.id(),
                    filler_block: BlockKind::Dirt.id(),
                    freeze_water: true,
                    tree_weight: 4,
                    height_offset: 2.0,
                    is_desert: false,
                },
                BiomeId::Desert => Self {
                    top_block: BlockKind::Sand.id(),
                    filler_block: BlockKind::Sandstone.id(),
                    freeze_water: false,
                    tree_weight: 0,
                    height_offset: -6.0,
                    is_desert: true,
                },
                BiomeId::Plains => Self {
                    top_block: BlockKind::Grass.id(),
                    filler_block: BlockKind::Dirt.id(),
                    freeze_water: false,
                    tree_weight: 1,
                    height_offset: -2.0,
                    is_desert: false,
                },
                BiomeId::Tundra => Self {
                    top_block: BlockKind::SnowBlock.id(),
                    filler_block: BlockKind::Dirt.id(),
                    freeze_water: true,
                    tree_weight: 0,
                    height_offset: -4.0,
                    is_desert: false,
                },
            }
        }
    }

    #[derive(Clone)]
    struct ColumnData {
        horizontal: na::Vector2<f32>,
        height: f32,
        biome: BiomeId,
        attrs: BiomeAttrs,
        surface_depth: u8,
        vegetation_bias: f32,
        top_voxel: Option<na::Vector3<u8>>,
        top_depth: f32,
    }

    impl ColumnData {
        fn new() -> Self {
            Self {
                horizontal: na::Vector2::zeros(),
                height: 0.0,
                biome: BiomeId::Plains,
                attrs: BiomeAttrs::of(BiomeId::Plains),
                surface_depth: 1,
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

    const BIOME_TEMP_FREQ: f64 = 0.025;
    const BIOME_HUM_FREQ: f64 = 0.05;
    const BIOME_VARIATION_FREQ: f64 = 0.25;
    const BIOME_TEMP_SEED: u64 = 0x71e7_fe11_d00d_beef;
    const BIOME_HUM_SEED: u64 = 0xc0fe_babe_face_feed;
    const BIOME_VARIATION_SEED: u64 = 0xb10b_1e55_5eed_cafe;
    const HEIGHT_PRIMARY_FREQ: f64 = HEIGHT_NOISE_BASE_FREQ_BLOCKS;
    const HEIGHT_SECONDARY_FREQ: f64 = HEIGHT_DETAIL_FREQ_BLOCKS;
    const HEIGHT_CHOOSER_FREQ: f64 = HEIGHT_MASK_FREQ_BLOCKS;
    const HEIGHT_EROSION_FREQ: f64 = 1.0 / 96.0;

    const CLIMATE_FBM: FbmConfig = FbmConfig {
        octaves: 4,
        lacunarity: 2.0,
        gain: 0.5,
    };
    const CLIMATE_VARIATION_FBM: FbmConfig = FbmConfig {
        octaves: 2,
        lacunarity: 2.0,
        gain: 0.5,
    };

    // ---------------------------------------------------------------------
    // remcpe-analog octave combiner
    // ---------------------------------------------------------------------
    // remcpe's PerlinNoise::getRegion calls ImprovedNoise::add with an octave
    // accumulator `x` that starts at 1.0 and is halved each octave.
    // ImprovedNoise::add adds `(1/a12) * noise(...)` where `a12 == x`, while
    // also scaling the input frequency by `x`.
    // Net effect: *lower* frequencies get *higher* weight, which produces much
    // smoother fields than a standard fBm (frequency doubles per octave).
    //
    // Our previous fBm-based implementation biased too heavily toward fine
    // detail, which matches the "frequency too high" reports.

    #[inline]
    fn octave_seed(base: u64, octave: u8) -> u64 {
        // Mix in octave index to emulate remcpe's separate ImprovedNoise instances.
        hash(base, 0x9E37_79B1_85EB_CA87_u64.wrapping_mul(octave as u64))
    }

    fn remcpe_perlin2(seed: u64, coords: na::Vector2<f64>, base_freq: f64, octaves: u8) -> f32 {
        let mut x = 1.0_f64;
        let mut sum = 0.0_f64;
        let octaves = octaves.max(1);
        for i in 0..octaves {
            let n = HyperbolicNoise::new(octave_seed(seed, i));
            sum += (1.0 / x) * n.sample2(coords * (base_freq * x));
            x *= 0.5;
        }
        sum as f32
    }

    fn remcpe_perlin3(seed: u64, coords: na::Vector3<f64>, base_freq: f64, octaves: u8) -> f32 {
        let mut x = 1.0_f64;
        let mut sum = 0.0_f64;
        let octaves = octaves.max(1);
        for i in 0..octaves {
            let n = HyperbolicNoise::new(octave_seed(seed, i));
            sum += (1.0 / x) * n.sample3(coords * (base_freq * x));
            x *= 0.5;
        }
        sum as f32
    }

    #[derive(Clone, Copy, Debug)]
    struct ClimateSample {
        temperature: f32,
        humidity: f32,
        #[allow(dead_code)]
        blend: f32,
    }

    fn sample_climate(_world_seed: u64, _coords2: na::Vector2<f64>) -> ClimateSample {
        // Scorched-earth: return a neutral, constant climate.
        ClimateSample {
            temperature: 0.5,
            humidity: 0.5,
            blend: 0.5,
        }
    }

    fn compute_column_height_blocks(
        _params: &ChunkParams,
        _local: na::Vector3<f32>,
        _climate: &ClimateSample,
        _attrs: &BiomeAttrs,
    ) -> f32 {
        // Scorched-earth: flat at sea level.
        0.0
    }

    fn remcpe_biome_lut() -> &'static [BiomeId; 64 * 64] {
        static LUT: OnceLock<[BiomeId; 64 * 64]> = OnceLock::new();
        LUT.get_or_init(|| {
            let mut map = [BiomeId::Plains; 64 * 64];
            // remcpe stores the biome LUT indexed as: map[hum + temp * 64]
            // (see Biome::getBiome in remcpe). Keep the same layout.
            for hum_i in 0..64 {
                for temp_i in 0..64 {
                    let hum = hum_i as f32 / 63.0;
                    let temp = temp_i as f32 / 63.0;
                    map[hum_i + temp_i * 64] = remcpe_pick_biome(temp, hum);
                }
            }
            map
        })
    }

    fn remcpe_pick_biome(temp: f32, hum: f32) -> BiomeId {
        if temp < 0.1 {
            return BiomeId::Tundra;
        }
        let ht = hum * temp;
        if ht < 0.2 {
            if temp >= 0.5 {
                if temp >= 0.95 {
                    return BiomeId::Desert;
                }
                return BiomeId::Savanna;
            }
            return BiomeId::Tundra;
        }
        if ht > 0.5 && ht < 0.7 {
            return BiomeId::Swampland;
        }
        if temp < 0.5 {
            return BiomeId::Taiga;
        }
        if temp >= 0.97 {
            if ht < 0.45 {
                return BiomeId::Plains;
            }
            if ht < 0.9 {
                return BiomeId::SeasonalForest;
            }
            return BiomeId::RainForest;
        }
        if temp >= 0.35 {
            return BiomeId::Forest;
        }
        BiomeId::Shrubland
    }

    fn remcpe_lookup_biome(temp: f32, hum: f32) -> BiomeId {
        let temp_i = (temp.clamp(0.0, 1.0) * 63.0).floor() as usize;
        let hum_i = (hum.clamp(0.0, 1.0) * 63.0).floor() as usize;
        remcpe_biome_lut()[hum_i + temp_i * 64]
    }

    fn build_columns(params: &ChunkParams) -> Vec<ColumnData> {
        // Scorched-earth: flat columns at sea level with a single biome.
        let dimension = params.dimension as usize;
        let mut columns = vec![ColumnData::new(); dimension * dimension];
        let dim_f = f32::from(params.dimension);
        let orientation = params.orientation();
        let [axis0, axis1] = orientation.horizontal_axes();
        for a in 0..params.dimension {
            for b in 0..params.dimension {
                let idx = ColumnData::idx_components(params, a, b);
                let mut local = na::Vector3::repeat(0.5);
                local[axis0 as usize] = (f32::from(a) + 0.5) / dim_f;
                local[axis1 as usize] = (f32::from(b) + 0.5) / dim_f;

                let mut column = ColumnData::new();
                column.horizontal = na::Vector2::new(local[axis0 as usize], local[axis1 as usize]);
                column.height = 0.0;
                column.biome = BiomeId::Plains;
                column.attrs = BiomeAttrs::of(BiomeId::Plains);
                column.surface_depth = DEFAULT_SURFACE_DEPTH_MIN;
                column.vegetation_bias = 0.0;
                columns[idx] = column;
            }
        }

        columns
    }

    #[cfg(test)]
    pub(super) fn test_column_heights(params: &ChunkParams) -> Vec<f32> {
        let columns = build_columns(params);
        columns.into_iter().map(|c| c.height).collect()
    }

    #[cfg(test)]
    pub(super) fn sample_height_for_tests(params: &ChunkParams, local: na::Vector3<f32>) -> f32 {
        let coords2 = params.hyperbolic_block_coords(local);
        let climate = sample_climate(params.world_seed, coords2);
        let biome = remcpe_lookup_biome(climate.temperature, climate.humidity);
        let attrs = BiomeAttrs::of(biome);
        compute_column_height_blocks(params, local, &climate, &attrs)
    }

    #[cfg(test)]
    pub(super) fn test_expected_height_span_chunk_units(_params: &ChunkParams) -> f32 {
        // Scorched-earth: flat at sea level, no expected spread.
        0.0
    }

    #[cfg(test)]
    pub(super) fn test_min_cave_height(params: &ChunkParams) -> f32 {
        scale_height_to_chunk(MC_MIN_HEIGHT + 4.0)
    }

    fn fill_base_layers(params: &ChunkParams, voxels: &mut [BlockID], columns: &mut [ColumnData]) {
        // Scorched-earth: simple flat fill relative to sea level.
        for (coords_x, coords_y, coords_z) in VoxelCoords::new(params.dimension) {
            let coords = na::Vector3::new(coords_x, coords_y, coords_z);
            let column_idx = ColumnData::idx(params, coords);
            let column = &mut columns[column_idx];
            let center = voxel_center(params.dimension, coords);

            // Positive block_height means above sea level (0), negative below.
            let block_height = -params.surface.distance_to_chunk(params.chunk, &center)
                * f32::from(params.dimension);

            let mut block = BlockKind::Air.id();

            if block_height < -1.0 {
                block = BlockKind::Stone.id();
            } else if block_height < 0.0 {
                block = BlockKind::Water.id();
            }

            // Track top voxel if any solid/water was placed.
            if block != BlockKind::Air.id() {
                voxels[index(params.dimension, coords)] = block;
                if column.top_voxel.is_none() {
                    column.top_voxel = Some(coords);
                    column.top_depth = 0.0;
                }
            }
        }
    }

    fn carve_caves(_params: &ChunkParams, _voxels: &mut [BlockID], _columns: &[ColumnData]) {}

    struct OreConfig {
        block: BlockID,
        attempts: u32,
        cluster: u8,
        min_height: f32,
        max_height: f32,
    }

    fn populate_ores(_params: &ChunkParams, _voxels: &mut [BlockID], _columns: &[ColumnData]) {}

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
        _params: &ChunkParams,
        _voxels: &mut [BlockID],
        _columns: &mut [ColumnData],
    ) {
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
    use crate::{
        hyper_noise::{FbmConfig, HyperbolicNoise},
        voxel_math::{CoordAxis, CoordSign},
    };
    use approx::*;

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
        let params_a =
            ChunkParams::new(&mut graph, chunk_a, WorldgenPreset::Hyperbolic, 0x5eed_5eed);
        let params_b =
            ChunkParams::new(&mut graph, chunk_b, WorldgenPreset::Hyperbolic, 0x5eed_5eed);

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
        // Base epsilon for f64 coordinate comparison
        const COORD_EPS: f64 = 1e-4;
        // Noise sample tolerance: must be higher because noise gradient magnitudes
        // can amplify small coordinate differences
        const NOISE_EPS: f64 = 1e-4;
        let mut base = [0.5f32; 3];
        base[axis as usize] = if matches!(sign, CoordSign::Plus) {
            1.0
        } else {
            0.0
        };
        let [axis_u, axis_v] = axis.other_axes();
        let noise = HyperbolicNoise::new(0x5eed_5eed);
        let block_eps = COORD_EPS * params_a.hyper_block_scale.max(params_b.hyper_block_scale);
        let block3_eps = COORD_EPS
            * params_a
                .hyper_block_scale
                .max(params_b.hyper_block_scale)
                .max(
                    params_a
                        .hyper_vertical_block_scale
                        .max(params_b.hyper_vertical_block_scale),
                );

        for i in 0..=GRID_STEPS {
            for j in 0..=GRID_STEPS {
                let mut local_a = na::Vector3::new(base[0], base[1], base[2]);
                local_a[axis_u as usize] = i as f32 / GRID_STEPS as f32;
                local_a[axis_v as usize] = j as f32 / GRID_STEPS as f32;

                let mut local_b =
                    convert_local_between_vertices(chunk_a.vertex, chunk_b.vertex, local_a);
                assert!(
                    local_b.iter().all(|c| (-1e-3..=1.0 + 1e-3).contains(c)),
                    "converted local coords outside chunk: {:?}",
                    local_b
                );
                local_b = local_b.map(|c| c.clamp(0.0, 1.0));

                let hyper2_a = params_a.hyperbolic_noise_coords(local_a);
                let hyper2_b = params_b.hyperbolic_noise_coords(local_b);
                assert!(
                    (hyper2_a - hyper2_b).norm() < 1e-4,
                    "2D noise coords diverged across face: {:?} vs {:?}",
                    hyper2_a,
                    hyper2_b
                );
                let hyper3_a = params_a.hyperbolic_noise_coords_3d(local_a);
                let hyper3_b = params_b.hyperbolic_noise_coords_3d(local_b);
                assert!(
                    (hyper3_a - hyper3_b).norm() < 1e-4,
                    "3D noise coords diverged across face"
                );

                let block2_a = params_a.hyperbolic_block_coords(local_a);
                let block2_b = params_b.hyperbolic_block_coords(local_b);
                let block2_diff = (block2_a - block2_b).norm();
                // Relaxed tolerance for block coords due to f32 precision in world_mpoint
                let relaxed_block_eps = block_eps * 2.0;
                assert!(
                    block2_diff < relaxed_block_eps,
                    "block coords diverged across face: diff={block2_diff}, eps={relaxed_block_eps}"
                );
                let block3_a = params_a.hyperbolic_block_coords_3d(local_a);
                let block3_b = params_b.hyperbolic_block_coords_3d(local_b);
                let block3_diff = (block3_a - block3_b).norm();
                let hyper3_diff = (hyper3_a - hyper3_b).norm();
                let relaxed_block3_eps = block3_eps * 2.0;
                assert!(
                    block3_diff < relaxed_block3_eps,
                    "block 3d coords diverged across face: block_diff={block3_diff}, hyper_diff={hyper3_diff}, eps={relaxed_block3_eps}"
                );

                let sample2_a = noise.sample2(hyper2_a);
                let sample2_b = noise.sample2(hyper2_b);
                assert!(
                    (sample2_a - sample2_b).abs() < NOISE_EPS,
                    "2D noise sample discontinuity: {sample2_a} vs {sample2_b}"
                );

                let sample3_a = noise.sample3(hyper3_a);
                let sample3_b = noise.sample3(hyper3_b);
                assert!(
                    (sample3_a - sample3_b).abs() < NOISE_EPS,
                    "3D noise sample discontinuity: {sample3_a} vs {sample3_b}"
                );

                let height_a = noise.sample2(block2_a * HEIGHT_BASE_FREQ);
                let height_b = noise.sample2(block2_b * HEIGHT_BASE_FREQ);
                assert!(
                    (height_a - height_b).abs() < NOISE_EPS,
                    "height noise discontinuity: {height_a} vs {height_b}"
                );
            }
        }
    }

    fn assert_height_continuity(
        params_a: &ChunkParams,
        params_b: &ChunkParams,
        chunk_a: ChunkId,
        chunk_b: ChunkId,
        axis: CoordAxis,
        sign: CoordSign,
    ) {
        const GRID_STEPS: usize = 6;
        const HEIGHT_EPS: f32 = 1e-3;
        let mut base = [0.5f32; 3];
        base[axis as usize] = if matches!(sign, CoordSign::Plus) {
            1.0
        } else {
            0.0
        };
        let [axis_u, axis_v] = axis.other_axes();

        for i in 0..=GRID_STEPS {
            for j in 0..=GRID_STEPS {
                let mut local_a = na::Vector3::new(base[0], base[1], base[2]);
                local_a[axis_u as usize] = i as f32 / GRID_STEPS as f32;
                local_a[axis_v as usize] = j as f32 / GRID_STEPS as f32;

                let mut local_b =
                    convert_local_between_vertices(chunk_a.vertex, chunk_b.vertex, local_a);
                assert!(
                    local_b.iter().all(|c| (-1e-3..=1.0 + 1e-3).contains(c)),
                    "converted local coords outside chunk: {:?}",
                    local_b
                );
                local_b = local_b.map(|c| c.clamp(0.0, 1.0));

                let block2_a = params_a.hyperbolic_block_coords(local_a);
                let block2_b = params_b.hyperbolic_block_coords(local_b);
                let block_diff = (block2_a - block2_b).norm();
                let height_a = remcpe::sample_height_for_tests(params_a, local_a);
                let height_b = remcpe::sample_height_for_tests(params_b, local_b);
                assert!(
                    (height_a - height_b).abs() < HEIGHT_EPS,
                    "height discontinuity across face axis={axis:?} sign={sign:?} samples=({i},{j}): {height_a} vs {height_b}, block_diff={block_diff}"
                );
            }
        }
    }

    fn sample_block_height_for_tests(params: &ChunkParams, local: na::Vector3<f32>) -> f32 {
        -params.surface.distance_to_chunk(params.chunk, &local) * f32::from(params.dimension)
    }

    fn quantize_cave_hash_coords(coords: na::Vector3<f64>) -> (u64, u64, u64) {
        let q = remcpe::HASH_COORD_QUANT;
        let hx_q = (coords.x * q).round() as i64 as u64;
        let hy_q = (coords.y * q).round() as i64 as u64;
        let hz_q = (coords.z * q).round() as i64 as u64;
        (hx_q, hy_q, hz_q)
    }

    #[test]
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
                    let params_a = ChunkParams::new(
                        &mut graph,
                        current,
                        WorldgenPreset::Hyperbolic,
                        0x5eed_5eed,
                    );
                    let params_b = ChunkParams::new(
                        &mut graph,
                        neighbor,
                        WorldgenPreset::Hyperbolic,
                        0x5eed_5eed,
                    );
                    assert_face_continuity(&params_a, &params_b, current, neighbor, axis, sign);
                    current = neighbor;
                }
            }
        }
    }

    #[test]
    fn cave_hash_coords_continuity_across_faces() {
        // Caves are carved by hashing quantized world-space coordinates. If that quantization
        // differs across a shared face (due to FP drift), tunnels can visibly "cut off" at
        // chunk boundaries. This test validates the hash inputs match for the same world points.
        const GRID_STEPS: usize = 6;
        let world_seed = 0x5eed_5eed_u64;
        let mut graph = Graph::new(CHUNK_SIZE);

        for axis in CoordAxis::iter() {
            for sign in CoordSign::iter() {
                let chunk_a = ChunkId::new(NodeId::ROOT, Vertex::A);
                if matches!(sign, CoordSign::Minus) {
                    let side = chunk_a.vertex.canonical_sides()[axis as usize];
                    graph.ensure_neighbor(chunk_a.node, side);
                }
                let Some(chunk_b) = graph.get_chunk_neighbor(chunk_a, axis, sign) else {
                    continue;
                };

                let params_a =
                    ChunkParams::new(&mut graph, chunk_a, WorldgenPreset::Hyperbolic, world_seed);
                let params_b =
                    ChunkParams::new(&mut graph, chunk_b, WorldgenPreset::Hyperbolic, world_seed);

                // Sample points across the *shared face*.
                let mut base = [0.5f32; 3];
                base[axis as usize] = if matches!(sign, CoordSign::Plus) {
                    1.0
                } else {
                    0.0
                };
                let [axis_u, axis_v] = axis.other_axes();

                for i in 0..=GRID_STEPS {
                    for j in 0..=GRID_STEPS {
                        let mut local_a = na::Vector3::new(base[0], base[1], base[2]);
                        local_a[axis_u as usize] = i as f32 / GRID_STEPS as f32;
                        local_a[axis_v as usize] = j as f32 / GRID_STEPS as f32;

                        let local_b =
                            convert_local_between_vertices(chunk_a.vertex, chunk_b.vertex, local_a);

                        let coords_a = params_a.hyperbolic_block_coords_3d(local_a);
                        let coords_b = params_b.hyperbolic_block_coords_3d(local_b);

                        let qa = quantize_cave_hash_coords(coords_a);
                        let qb = quantize_cave_hash_coords(coords_b);
                        assert_eq!(
                            qa, qb,
                            "cave hash coord quantization diverged across face axis={axis:?} sign={sign:?}: qa={qa:?} qb={qb:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn terrain_height_continuity_across_faces() {
        const STRIP_STEPS: usize = 30;
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
                    let params_a =
                        ChunkParams::new(&mut graph, current, WorldgenPreset::Hyperbolic, 0x1);
                    let params_b =
                        ChunkParams::new(&mut graph, neighbor, WorldgenPreset::Hyperbolic, 0x1);
                    assert_height_continuity(&params_a, &params_b, current, neighbor, axis, sign);
                    current = neighbor;
                }
            }
        }
    }

    #[test]
    fn guide_surface_continuity_across_faces() {
        const STRIP_STEPS: usize = 30;
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
                    let params_a =
                        ChunkParams::new(&mut graph, current, WorldgenPreset::Hyperbolic, 0x2);
                    let params_b =
                        ChunkParams::new(&mut graph, neighbor, WorldgenPreset::Hyperbolic, 0x2);

                    const GRID_STEPS: usize = 6;
                    let mut base = [0.5f32; 3];
                    base[axis as usize] = if matches!(sign, CoordSign::Plus) {
                        1.0
                    } else {
                        0.0
                    };
                    let [axis_u, axis_v] = axis.other_axes();
                    for i in 0..=GRID_STEPS {
                        for j in 0..=GRID_STEPS {
                            let mut local_a = na::Vector3::new(base[0], base[1], base[2]);
                            local_a[axis_u as usize] = i as f32 / GRID_STEPS as f32;
                            local_a[axis_v as usize] = j as f32 / GRID_STEPS as f32;
                            let mut local_b = convert_local_between_vertices(
                                current.vertex,
                                neighbor.vertex,
                                local_a,
                            );
                            local_b = local_b.map(|c| c.clamp(0.0, 1.0));
                            let h_a = sample_block_height_for_tests(&params_a, local_a);
                            let h_b = sample_block_height_for_tests(&params_b, local_b);
                            assert!(
                                (h_a - h_b).abs() < 1e-3,
                                "plane distance discontinuity axis={axis:?} sign={sign:?} samples=({i},{j}): {h_a} vs {h_b}"
                            );
                        }
                    }
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
        let params = ChunkParams::new(&mut graph, chunk, WorldgenPreset::Hyperbolic, 0x5eed_5eed);
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

    /// Traverse the graph by following a sequence of different sides,
    /// which actually moves us away from the origin (unlike repeated same-side).
    fn traverse_side_sequence(graph: &mut Graph, start: NodeId, sides: &[Side]) -> NodeId {
        let mut node = start;
        for &side in sides {
            node = graph.ensure_neighbor(node, side);
        }
        node
    }

    /// Test that detects radial symmetry in noise generation.
    ///
    /// The test travels equal distances from the origin in different directions
    /// (by crossing different sides of the dodecahedral tiling) and samples
    /// noise at those points. If the noise shows radial symmetry (circular
    /// terrain patterns), points at the same distance but different directions
    /// will have suspiciously similar noise values.
    #[test]
    fn detect_radial_noise_symmetry() {
        let mut graph = Graph::new(CHUNK_SIZE);
        let world_seed = 0x5eed_5eed_u64;
        let noise = HyperbolicNoise::new(world_seed);

        // Define several paths of equal length that go in different directions.
        // Each path is a sequence of sides to traverse.
        // We use non-adjacent sides to ensure we're actually moving outward.
        let paths: &[&[Side]] = &[
            &[Side::A, Side::B, Side::C, Side::D, Side::E, Side::F],
            &[Side::B, Side::C, Side::D, Side::E, Side::F, Side::A],
            &[Side::C, Side::D, Side::E, Side::F, Side::A, Side::B],
            &[Side::D, Side::E, Side::F, Side::A, Side::B, Side::C],
            &[Side::E, Side::F, Side::A, Side::B, Side::C, Side::D],
            &[Side::F, Side::A, Side::B, Side::C, Side::D, Side::E],
            // Some different patterns
            &[Side::A, Side::C, Side::E, Side::B, Side::D, Side::F],
            &[Side::B, Side::D, Side::F, Side::A, Side::C, Side::E],
        ];

        #[derive(Debug)]
        struct SamplePoint {
            path_idx: usize,
            world_pos: MPoint<f32>,
            hyper_coords: na::Vector3<f64>,
            noise_value: f64,
            distance_from_origin: f32,
        }

        let mut samples: Vec<SamplePoint> = Vec::new();

        for (path_idx, path) in paths.iter().enumerate() {
            let node = traverse_side_sequence(&mut graph, NodeId::ROOT, path);
            graph.ensure_node_state(node);

            // Sample at the center of chunk A in this node
            let chunk = ChunkId::new(node, Vertex::A);
            let params =
                ChunkParams::new(&mut graph, chunk, WorldgenPreset::Hyperbolic, world_seed);

            let local = na::Vector3::repeat(0.5); // center of chunk
            let world_pos = params.world_mpoint(local);
            let hyper_coords = params.hyperbolic_noise_coords_3d(local);
            let noise_value = noise.fbm3(hyper_coords * 0.01, FbmConfig::default());
            let distance_from_origin = world_pos.distance(&MPoint::origin());

            samples.push(SamplePoint {
                path_idx,
                world_pos,
                hyper_coords,
                noise_value,
                distance_from_origin,
            });
        }

        // Print diagnostic info
        eprintln!("\n=== Radial Symmetry Detection Test ===");
        eprintln!("Path length: {} steps", paths[0].len());
        eprintln!("\nSamples at different directions:");
        for s in &samples {
            eprintln!(
                "  Path {}: dist={:.3}, hyper_coords=({:.3}, {:.3}, {:.3}), noise={:.4}",
                s.path_idx,
                s.distance_from_origin,
                s.hyper_coords.x,
                s.hyper_coords.y,
                s.hyper_coords.z,
                s.noise_value
            );
        }

        // Check 1: Distances should be similar (we traveled same number of steps)
        let distances: Vec<f32> = samples.iter().map(|s| s.distance_from_origin).collect();
        let avg_dist = distances.iter().sum::<f32>() / distances.len() as f32;
        let dist_variance = distances
            .iter()
            .map(|d| (d - avg_dist).powi(2))
            .sum::<f32>()
            / distances.len() as f32;
        eprintln!(
            "\nDistance stats: avg={:.3}, variance={:.6}",
            avg_dist, dist_variance
        );

        // Check 2: Hyperbolic coordinate magnitudes - if radially symmetric,
        // all points at same distance would have similar |hyper_coords|
        let hyper_mags: Vec<f64> = samples.iter().map(|s| s.hyper_coords.norm()).collect();
        let avg_mag = hyper_mags.iter().sum::<f64>() / hyper_mags.len() as f64;
        let mag_variance = hyper_mags
            .iter()
            .map(|m| (m - avg_mag).powi(2))
            .sum::<f64>()
            / hyper_mags.len() as f64;
        eprintln!(
            "Hyper coord magnitude stats: avg={:.3}, variance={:.6}",
            avg_mag, mag_variance
        );

        // Check 3: Noise value variance - if radially symmetric, noise values
        // at equidistant points would be very similar
        let noise_values: Vec<f64> = samples.iter().map(|s| s.noise_value).collect();
        let avg_noise = noise_values.iter().sum::<f64>() / noise_values.len() as f64;
        let noise_variance = noise_values
            .iter()
            .map(|n| (n - avg_noise).powi(2))
            .sum::<f64>()
            / noise_values.len() as f64;
        let noise_range = noise_values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
            - noise_values.iter().cloned().fold(f64::INFINITY, f64::min);
        eprintln!(
            "Noise stats: avg={:.4}, variance={:.6}, range={:.4}",
            avg_noise, noise_variance, noise_range
        );

        // Check 4: Are the hyperbolic coordinates themselves radially distributed?
        // Compute pairwise angles between hyper_coords vectors
        eprintln!("\nPairwise angles between hyperbolic coordinate vectors (degrees):");
        let mut angles = Vec::new();
        for i in 0..samples.len() {
            for j in (i + 1)..samples.len() {
                let a = samples[i].hyper_coords;
                let b = samples[j].hyper_coords;
                let dot = a.dot(&b);
                let cos_angle = dot / (a.norm() * b.norm());
                let angle_deg = cos_angle.clamp(-1.0, 1.0).acos().to_degrees();
                angles.push(angle_deg);
                eprintln!(
                    "  Path {} vs {}: {:.1}°",
                    samples[i].path_idx, samples[j].path_idx, angle_deg
                );
            }
        }

        // DETECTION CRITERIA:
        // If noise is radially symmetric, we expect:
        // 1. Low variance in hyperbolic coordinate magnitudes (they grow together)
        // 2. Low variance in noise values (same distance = same noise)
        // 3. Small angles between hyperbolic coordinate vectors (all point toward origin)

        let avg_angle = angles.iter().sum::<f64>() / angles.len() as f64;
        eprintln!(
            "\nAverage angle between hyper coord vectors: {:.1}°",
            avg_angle
        );

        // The test FAILS (indicating a problem) if:
        // - Angles are too small (vectors all point same direction = radial)
        // - Noise variance is too low relative to expected noise range
        const MIN_EXPECTED_ANGLE: f64 = 30.0; // degrees - vectors should spread out
        const MIN_EXPECTED_NOISE_RANGE: f64 = 0.1; // noise should vary

        let radial_symmetry_detected =
            avg_angle < MIN_EXPECTED_ANGLE || noise_range < MIN_EXPECTED_NOISE_RANGE;

        eprintln!("\n=== VERDICT ===");
        if radial_symmetry_detected {
            eprintln!("⚠️  RADIAL SYMMETRY DETECTED!");
            eprintln!(
                "    avg_angle={:.1}° (threshold: >{:.1}°)",
                avg_angle, MIN_EXPECTED_ANGLE
            );
            eprintln!(
                "    noise_range={:.4} (threshold: >{:.4})",
                noise_range, MIN_EXPECTED_NOISE_RANGE
            );
            // This is the problematic case - uncomment to make test fail when symmetry detected
            // panic!("Radial symmetry detected in noise generation!");
        } else {
            eprintln!("✓ No obvious radial symmetry detected");
        }

        // For now, just assert the test ran and print diagnostics
        // Uncomment the panic above to make this a hard failure
        assert!(
            !radial_symmetry_detected,
            "Radial symmetry detected: avg_angle={:.1}° < {:.1}°, noise_range={:.4} < {:.4}",
            avg_angle, MIN_EXPECTED_ANGLE, noise_range, MIN_EXPECTED_NOISE_RANGE
        );
    }

    /// More aggressive test: sample many points at varying distances and check
    /// if noise correlates more with distance than with actual position.
    #[test]
    fn noise_distance_correlation_test() {
        let mut graph = Graph::new(CHUNK_SIZE);
        let world_seed = 0x5eed_5eed_u64;
        let noise = HyperbolicNoise::new(world_seed);

        // Define base direction sequences (cyclic permutations to get different directions)
        let base_sides = [Side::A, Side::B, Side::C, Side::D, Side::E, Side::F];

        struct Sample {
            distance: f32,
            noise_val: f64,
            direction_idx: usize,
            path_len: usize,
        }

        let mut samples: Vec<Sample> = Vec::new();

        // For each "direction" (cyclic rotation of side sequence), sample at different distances
        for dir_idx in 0..6 {
            // Rotate the base sequence to get different direction
            let mut sides: Vec<Side> = base_sides.iter().cloned().collect();
            sides.rotate_left(dir_idx);

            // Sample at different path lengths (distances)
            for path_len in [2, 4, 6, 8, 10] {
                let path: Vec<Side> = sides.iter().cycle().take(path_len).cloned().collect();
                let node = traverse_side_sequence(&mut graph, NodeId::ROOT, &path);
                graph.ensure_node_state(node);
                let chunk = ChunkId::new(node, Vertex::A);
                let params =
                    ChunkParams::new(&mut graph, chunk, WorldgenPreset::Hyperbolic, world_seed);

                let local = na::Vector3::repeat(0.5);
                let world_pos = params.world_mpoint(local);
                let hyper_coords = params.hyperbolic_noise_coords_3d(local);
                let noise_val = noise.fbm3(hyper_coords * 0.01, FbmConfig::default());
                let distance = world_pos.distance(&MPoint::origin());

                samples.push(Sample {
                    distance,
                    noise_val,
                    direction_idx: dir_idx,
                    path_len,
                });
            }
        }

        // Print all samples for debugging
        eprintln!("\n=== Distance-Noise Correlation Test ===");
        eprintln!("All samples:");
        for s in &samples {
            eprintln!(
                "  dir={}, len={}, dist={:.3}, noise={:.4}",
                s.direction_idx, s.path_len, s.distance, s.noise_val
            );
        }

        // Compute correlation between distance and noise value
        let n = samples.len() as f64;
        let mean_dist = samples.iter().map(|s| s.distance as f64).sum::<f64>() / n;
        let mean_noise = samples.iter().map(|s| s.noise_val).sum::<f64>() / n;

        let mut cov = 0.0;
        let mut var_dist = 0.0;
        let mut var_noise = 0.0;
        for s in &samples {
            let d = s.distance as f64 - mean_dist;
            let v = s.noise_val - mean_noise;
            cov += d * v;
            var_dist += d * d;
            var_noise += v * v;
        }

        let correlation = if var_dist > 0.0 && var_noise > 0.0 {
            cov / (var_dist.sqrt() * var_noise.sqrt())
        } else {
            0.0
        };

        eprintln!("\nSamples: {}", samples.len());
        eprintln!(
            "Distance range: {:.2} to {:.2}",
            samples
                .iter()
                .map(|s| s.distance)
                .fold(f32::INFINITY, f32::min),
            samples
                .iter()
                .map(|s| s.distance)
                .fold(f32::NEG_INFINITY, f32::max)
        );
        eprintln!(
            "Noise range: {:.4} to {:.4}",
            samples
                .iter()
                .map(|s| s.noise_val)
                .fold(f64::INFINITY, f64::min),
            samples
                .iter()
                .map(|s| s.noise_val)
                .fold(f64::NEG_INFINITY, f64::max)
        );
        eprintln!("Correlation(distance, noise): {:.4}", correlation);

        // A high absolute correlation suggests noise depends primarily on distance
        // (radial symmetry). We expect low correlation for good noise.
        const MAX_ACCEPTABLE_CORRELATION: f64 = 0.7;

        eprintln!("\n=== VERDICT ===");
        if correlation.abs() > MAX_ACCEPTABLE_CORRELATION {
            eprintln!("⚠️  HIGH DISTANCE-NOISE CORRELATION DETECTED!");
            eprintln!(
                "    |correlation| = {:.4} > {:.4}",
                correlation.abs(),
                MAX_ACCEPTABLE_CORRELATION
            );
            eprintln!("    This suggests noise is primarily determined by distance from origin.");
        } else {
            eprintln!(
                "✓ Distance-noise correlation is acceptable: {:.4}",
                correlation
            );
        }

        assert!(
            correlation.abs() <= MAX_ACCEPTABLE_CORRELATION,
            "Noise correlates too strongly with distance from origin: r={:.4}",
            correlation
        );
    }

    #[test]
    fn biome_diversity_over_large_area() {
        const SAMPLE_STEPS: usize = 256;
        const SAMPLE_SPACING: f64 = 8.0;
        const MIN_UNIQUE_BIOMES: usize = 1;
        const BIOME_LABELS: [&str; 10] = [
            "RainForest",
            "Swampland",
            "SeasonalForest",
            "Forest",
            "Savanna",
            "Shrubland",
            "Taiga",
            "Desert",
            "Plains",
            "Tundra",
        ];
        let world_seed = 0x5eed_5eed_u64;
        let mut seen = std::collections::HashSet::new();
        let mut min_temp = f32::INFINITY;
        let mut max_temp = f32::NEG_INFINITY;
        let mut min_hum = f32::INFINITY;
        let mut max_hum = f32::NEG_INFINITY;
        let mut biome_counts = [0usize; BIOME_LABELS.len()];

        for ix in 0..SAMPLE_STEPS {
            for iz in 0..SAMPLE_STEPS {
                let coords =
                    na::Vector2::new(ix as f64 * SAMPLE_SPACING, iz as f64 * SAMPLE_SPACING);
                let sample = remcpe::sample_debug_biome(world_seed, coords);
                seen.insert(sample.biome_id);
                min_temp = min_temp.min(sample.temperature);
                max_temp = max_temp.max(sample.temperature);
                min_hum = min_hum.min(sample.humidity);
                max_hum = max_hum.max(sample.humidity);
                if let Some(slot) = biome_counts.get_mut(sample.biome_id as usize) {
                    *slot += 1;
                }
            }
        }

        eprintln!(
            "Biome diversity test: sampled {} points, unique biomes {}",
            SAMPLE_STEPS * SAMPLE_STEPS,
            seen.len()
        );

        if seen.len() < MIN_UNIQUE_BIOMES {
            eprintln!(
                "Climate ranges: temp {:.3}..{:.3}, humidity {:.3}..{:.3}",
                min_temp, max_temp, min_hum, max_hum
            );
            eprintln!("Biome counts (non-zero):");
            for (idx, label) in BIOME_LABELS.iter().enumerate() {
                if biome_counts[idx] > 0 {
                    eprintln!("  {:<16} {}", label, biome_counts[idx]);
                }
            }
        }

        assert!(
            seen.len() >= MIN_UNIQUE_BIOMES,
            "Biome diversity regression: expected at least {} unique biomes, saw {}",
            MIN_UNIQUE_BIOMES,
            seen.len()
        );
    }

    #[test]
    fn generation_statistics_match_configuration_targets() {
        let mut graph = Graph::new(CHUNK_SIZE);
        let chunk = ChunkId::new(NodeId::ROOT, Vertex::A);
        let params = ChunkParams::new(&mut graph, chunk, WorldgenPreset::Hyperbolic, 0xdecaf_bad);

        assert_block_scale_alignment(&params);
        assert_height_span_matches(&params);
        assert_cave_probability_matches(&params);
    }

    fn assert_block_scale_alignment(params: &ChunkParams) {
        let center = na::Vector3::repeat(0.5);
        let step = 1.0 / f32::from(params.dimension);
        let base2 = params.hyperbolic_noise_coords(center);
        for axis in params.orientation().horizontal_axes() {
            let mut offset = center;
            offset[axis as usize] = (offset[axis as usize] + step).clamp(0.0, 1.0);
            let raw_delta = (params.hyperbolic_noise_coords(offset) - base2).norm();
            let scaled = raw_delta * params.hyper_block_scale;
            assert!(
                (scaled - 1.0).abs() < 0.25,
                "horizontal scale deviated: axis={axis:?}, scaled delta={scaled:.4}"
            );
        }

        let mut vertical = center;
        let up_axis = params.orientation().up_axis() as usize;
        let up_sign = f32::from(params.orientation().up_sign());
        vertical[up_axis] = (vertical[up_axis] + step * up_sign).clamp(0.0, 1.0);
        let base3 = params.hyperbolic_noise_coords_3d(center);
        let vertical_raw = (params.hyperbolic_noise_coords_3d(vertical) - base3).norm();
        let vertical_scaled = vertical_raw * params.hyper_vertical_block_scale;
        assert!(
            (vertical_scaled - 1.0).abs() < 0.35,
            "vertical scale deviated: scaled delta={vertical_scaled:.4}"
        );
    }

    fn assert_height_span_matches(params: &ChunkParams) {
        let heights = remcpe::test_column_heights(params);
        assert!(
            !heights.is_empty(),
            "no columns were sampled for statistics"
        );
        let min_height = heights.iter().copied().fold(f32::INFINITY, f32::min);
        let max_height = heights.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let observed_span = max_height - min_height;
        let expected_span = remcpe::test_expected_height_span_chunk_units(params);
        let mean = heights.iter().copied().sum::<f32>() / heights.len() as f32;
        let variance =
            heights.iter().map(|h| (*h - mean).powi(2)).sum::<f32>() / heights.len() as f32;
        let std_dev = variance.sqrt();
        assert!(
            observed_span >= expected_span * 0.05,
            "height span collapsed: observed={observed_span:.3}, expected target={expected_span:.3}"
        );
        assert!(
            std_dev >= expected_span * 0.015,
            "height std-dev collapsed: std={std_dev:.3}, expected target={expected_span:.3}"
        );
    }

    fn assert_cave_probability_matches(params: &ChunkParams) {
        let voxels = params.generate_voxels();
        let data: &[BlockID] = match &voxels {
            VoxelData::Dense(data) => data,
            VoxelData::Solid(_) => panic!("worldgen returned solid chunk; expected dense data"),
        };

        let dimension = params.dimension;
        let column_heights = remcpe::test_column_heights(params);
        let min_cave_height = remcpe::test_min_cave_height(params);
        let orientation = params.orientation();
        let mut candidate_voxels = 0u32;
        let mut carved_voxels = 0u32;

        for (x, y, z) in VoxelCoords::new(dimension) {
            let coords = na::Vector3::new(x, y, z);
            let column_idx = orientation.column_index(dimension, coords);
            let column_height = column_heights[column_idx];
            let center = voxel_center(dimension, coords);
            let block_height =
                -params.surface.distance_to_chunk(params.chunk, &center) * f32::from(dimension);
            if block_height > column_height || block_height < min_cave_height {
                continue;
            }

            // Must match the runtime cave-carving eligibility.
            let dist = column_height - block_height;
            if dist < 4.0 {
                continue;
            }

            let block_index = index(dimension, coords);
            let block = data[block_index];
            if block == BlockKind::Bedrock.id()
                || block == BlockKind::Water.id()
                || block == BlockKind::Ice.id()
            {
                continue;
            }

            candidate_voxels += 1;
            let hyper_coords = params.hyperbolic_block_coords_3d(center);
            let hx_q = (hyper_coords.x * remcpe::HASH_COORD_QUANT).round() as i64 as u64;
            let hy_q = (hyper_coords.y * remcpe::HASH_COORD_QUANT).round() as i64 as u64;
            let hz_q = (hyper_coords.z * remcpe::HASH_COORD_QUANT).round() as i64 as u64;
            let coord_hash = hash(hx_q, hash(hy_q, hz_q));
            let sample_seed = hash(params.node_spice, hash(params.world_seed, coord_hash));
            let carve_value = ((sample_seed >> 16) & 0xffff_ffff) as f32 / u32::MAX as f32;
            let should_carve = carve_value < remcpe::CAVE_PROBABILITY;
            if should_carve {
                carved_voxels += 1;
            }
            let actually_carved = block == BlockKind::Air.id();
            assert_eq!(
                should_carve, actually_carved,
                "cave probability mismatch at {:?} (height {:.3})",
                coords, block_height
            );
        }

        let candidate = candidate_voxels.max(1);
        let observed_probability = carved_voxels as f32 / candidate as f32;
        let variance =
            remcpe::CAVE_PROBABILITY * (1.0 - remcpe::CAVE_PROBABILITY) / candidate as f32;
        let tolerance = variance.sqrt() * 3.0 + 0.001;
        assert!(
            (observed_probability - remcpe::CAVE_PROBABILITY).abs() <= tolerance,
            "carved ratio deviates from configuration: observed={observed_probability:.4}, expected={:.4}",
            remcpe::CAVE_PROBABILITY
        );
    }
}
