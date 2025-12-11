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
use noise::{NoiseFn, Perlin};
use serde::{Deserialize, Serialize};

use crate::{
    dodeca::{Side, Vertex},
    graph::{Graph, NodeId},
    margins,
    math::{self, MIsometry, MPoint, MVector},
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
    pub fn root() -> Self {
        Self {
            kind: NodeStateKind::ROOT,
            surface: Plane::from(Side::A),
            road_state: NodeStateRoad::ROOT,
            enviro: EnviroFactors {
                max_elevation: 0.0,
                temperature: 0.0,
                rainfall: 0.0,
                blockiness: 0.0,
            },
            world_from_node: MIsometry::identity(),
        }
    }

    pub fn child(&self, graph: &Graph, node: NodeId, side: Side) -> Self {
        let mut d = graph.parents(node).map(|(s, n)| (s, graph.node_state(n)));
        let enviro = match (d.next(), d.next()) {
            (Some(_), None) => {
                let parent_side = graph.primary_parent_side(node).unwrap();
                let parent_node = graph.neighbor(node, parent_side).unwrap();
                let parent_state = graph.node_state(parent_node);
                let spice = graph.hash_of(node) as u64;
                EnviroFactors::varied_from(parent_state.enviro, spice)
            }
            (Some((a_side, a_state)), Some((b_side, b_state))) => {
                let ab_node = graph
                    .neighbor(graph.neighbor(node, a_side).unwrap(), b_side)
                    .unwrap();
                let ab_state = graph.node_state(ab_node);
                EnviroFactors::continue_from(a_state.enviro, b_state.enviro, ab_state.enviro)
            }
            _ => unreachable!(),
        };

        let child_kind = self.kind.child(side);
        let child_road = self.road_state.child(side);
        let world_from_node = &self.world_from_node * side.reflection();

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
    /// Center of this chunk in global coordinates (Beltrami projection)
    chunk_center: na::Vector3<f32>,
    /// Basis spanning the local horizontal plane (east, north, up)
    horizontal_basis: [na::Vector3<f32>; 3],
    /// Approximate scale mapping one chunk edge along the plane
    horizontal_scale: f32,
    /// Orientation of the chunk's up direction in voxel space
    orientation: ChunkOrientation,
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
        let chunk_center = chunk_point_world(
            chunk.vertex,
            &world_from_node,
            na::Vector3::repeat(0.5),
        );
        let horizontal_basis = plane_basis(state.surface);
        let horizontal_scale = {
            let along_x = chunk_point_world(
                chunk.vertex,
                &world_from_node,
                na::Vector3::new(1.0, 0.5, 0.5),
            ) - chunk_point_world(
                chunk.vertex,
                &world_from_node,
                na::Vector3::new(0.0, 0.5, 0.5),
            );
            let scale = along_x.dot(&horizontal_basis[0]).abs();
            if scale <= f32::EPSILON {
                f32::from(dimension)
            } else {
                scale
            }
        };
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
            world_from_node,
            chunk_center,
            horizontal_basis,
            horizontal_scale,
            orientation,
        }
    }

    pub fn chunk(&self) -> Vertex {
        self.chunk
    }

    pub fn world_point(&self, local: na::Vector3<f32>) -> na::Vector3<f32> {
        chunk_point_world(self.chunk, &self.world_from_node, local)
    }

    fn horizontal_coords(&self, local: na::Vector3<f32>) -> na::Vector2<f32> {
        let world = self.world_point(local);
        let relative = world - self.chunk_center;
        na::Vector2::new(
            relative.dot(&self.horizontal_basis[0]) / self.horizontal_scale,
            relative.dot(&self.horizontal_basis[1]) / self.horizontal_scale,
        )
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
        // Determine whether this chunk might contain a boundary between solid and void
        let mut me_min = self.env.max_elevations[0];
        let mut me_max = self.env.max_elevations[0];
        for &me in &self.env.max_elevations[1..] {
            me_min = me_min.min(me);
            me_max = me_max.max(me);
        }
        // Maximum difference between elevations at the center of a chunk and any other point in the chunk
        // TODO: Compute what this actually is, current value is a guess! Real one must be > 0.6
        // empirically.
        const ELEVATION_MARGIN: f32 = 0.7;
        let center_elevation = self
            .surface
            .distance_to_chunk(self.chunk, &na::Vector3::repeat(0.5));
        if (center_elevation - ELEVATION_MARGIN > me_max / TERRAIN_SMOOTHNESS)
            && !(self.is_road || self.is_road_support)
        {
            // The whole chunk is above ground and not part of the road
            return VoxelData::Solid(BlockKind::Air.id());
        }

        if (center_elevation + ELEVATION_MARGIN < me_min / TERRAIN_SMOOTHNESS) && !self.is_road {
            // The whole chunk is underground
            // TODO: More accurate VoxelData
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
        use noise::{NoiseFn, Perlin};
        let normal = Normal::new(0.0, 0.03).unwrap();
        
        // Minecraft-style 3D noise for caves and overhangs
        let density_noise = Perlin::new(1337);
        let cave_noise = Perlin::new(420);

        for (x, y, z) in VoxelCoords::new(self.dimension) {
            let coords = na::Vector3::new(x, y, z);
            let center = voxel_center(self.dimension, coords);
            let trilerp_coords = center.map(|x| (1.0 - x) * 0.5);

            let rain = trilerp(&self.env.rainfalls, trilerp_coords) + rng.sample(normal);
            let temp = trilerp(&self.env.temperatures, trilerp_coords) + rng.sample(normal);

            // elev is calculated in multiple steps. The initial value elev_pre_terracing
            // is used to calculate elev_pre_noise which is used to calculate elev.
            let elev_pre_terracing = trilerp(&self.env.max_elevations, trilerp_coords);
            let block = trilerp(&self.env.blockinesses, trilerp_coords);
            let voxel_elevation = self.surface.distance_to_chunk(self.chunk, &center);
            let strength = 0.4 / (1.0 + math::sqr(voxel_elevation));
            let terracing_small = terracing_diff(elev_pre_terracing, block, 5.0, strength, 2.0);
            let terracing_big = terracing_diff(elev_pre_terracing, block, 15.0, strength, -1.0);
            // Small and big terracing effects must not sum to more than 1,
            // otherwise the terracing fails to be (nonstrictly) monotonic
            // and the terrain gets trenches ringing around its cliffs.
            let elev_pre_noise = elev_pre_terracing + 0.6 * terracing_small + 0.4 * terracing_big;

            // initial value dist_pre_noise is the difference between the voxel's distance
            // from the guiding plane and the voxel's calculated elev value. It represents
            // how far from the terrain surface a voxel is.
            let dist_pre_noise = elev_pre_noise / TERRAIN_SMOOTHNESS - voxel_elevation;

            // adding noise allows interfaces between strata to be rough
            let elev = elev_pre_noise + TERRAIN_SMOOTHNESS * rng.sample(normal);

            // Final value of dist is calculated in this roundabout way for greater control
            // over how noise in elev affects dist.
            let mut dist = if dist_pre_noise > 0.0 {
                // The .max(0.0) keeps the top of the ground smooth
                // while still allowing the surface/general terrain interface to be rough
                (elev / TERRAIN_SMOOTHNESS - voxel_elevation).max(0.0)
            } else {
                // Distance not updated for updated elevation if distance was originally
                // negative. This ensures that no voxels that would have otherwise
                // been void are changed to a material---so no floating dirt blocks.
                dist_pre_noise
            };
            
            // MINECRAFT-STYLE HYBRID: 2D base terrain + 3D density for caves/overhangs
            // Minecraft uses 2D noise for the overall terrain shape, then 3D noise
            // to add/remove material for caves, overhangs, and terrain features
            
            // Create world-space coordinates for noise sampling
            // MUCH larger scale than before
            let world_pos = center * 10.0;
            
            // 3D density noise - Minecraft style with multiple frequencies
            // Lower frequencies = larger features
            let density_3d = 
                // Large-scale terrain deformation (overhangs, natural arches)
                density_noise.get([world_pos.x as f64 * 0.015, world_pos.y as f64 * 0.015, world_pos.z as f64 * 0.015]) * 1.2 +
                // Medium-scale detail
                density_noise.get([world_pos.x as f64 * 0.04, world_pos.y as f64 * 0.04, world_pos.z as f64 * 0.04]) * 0.5 +
                // Fine detail
                density_noise.get([world_pos.x as f64 * 0.08, world_pos.y as f64 * 0.08, world_pos.z as f64 * 0.08]) * 0.25;
            
            // Cave noise - much larger scale for proper cave systems
            let cave_density = cave_noise.get([
                world_pos.x as f64 * 0.012,
                world_pos.y as f64 * 0.012,
                world_pos.z as f64 * 0.012
            ]);
            
            // Apply 3D density modulation near the surface for overhangs and cliffs
            // Minecraft applies 3D noise strongest near y=64 (mid-height)
            if dist_pre_noise > -3.0 && dist_pre_noise < 8.0 {
                // Surface zone: 3D noise creates overhangs, cliffs, and interesting terrain
                // Strength increases near the surface
                let surface_factor = 1.0 - (dist_pre_noise / 8.0).abs().min(1.0);
                dist += density_3d as f32 * surface_factor * 3.5;
            }
            
            // Cave carving: Minecraft-style 3D cave systems
            // Carve caves underground when noise is in a specific range
            if dist_pre_noise < -1.0 && cave_density.abs() < 0.2 {
                // Creates winding cave tunnels
                dist = (cave_density.abs() as f32 - 0.2) * 5.0;
            }

            if dist >= 0.0 {
                let voxel_mat = VoronoiInfo::terraingen_voronoi(
                    f64::from(elev),
                    f64::from(rain),
                    f64::from(temp),
                    f64::from(dist),
                );
                voxels.data_mut(self.dimension)[index(self.dimension, coords)] =
                    voxel_mat.into();
            }
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
                    if (i.block_id == BlockKind::Dirt.id()) || (i.block_id == BlockKind::Grass.id()) {
                        voxels.data_mut(self.dimension)[voxel_of_interest_index] = BlockKind::Log.id();
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

const TERRAIN_SMOOTHNESS: f32 = 10.0;

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
    fn varied_from(parent: Self, spice: u64) -> Self {
        let mut rng = rand_pcg::Pcg64Mcg::seed_from_u64(spice);
        let unif = Uniform::new_inclusive(-1.0, 1.0).unwrap();
        
        // Minecraft uses 2D noise for base terrain, 3D noise for caves/overhangs
        // This generates the base elevation using 2D-style noise
        let perlin1 = Perlin::new(42);
        let perlin2 = Perlin::new(43);
        
        // Create pseudo-3D coordinates from the node hash
        // MUCH larger scale for continent-sized features
        let base_scale = 0.002; // Even lower frequency = bigger features
        let x = (spice % 10000) as f64 * base_scale;
        let y = ((spice / 10000) % 10000) as f64 * base_scale;
        let z = (spice / 100000000) as f64 * base_scale;
        
        // Base terrain elevation (Minecraft's "continentalness" equivalent)
        let octaves = 5;
        let lacunarity = 2.0;
        let persistence = 0.5;
        
        let mut elevation = 0.0;
        let mut amplitude = 1.0;
        let mut frequency = 1.0;
        let mut max_value = 0.0;
        
        for _octave in 0..octaves {
            elevation += perlin1.get([x * frequency, y * frequency, z * frequency]) * amplitude;
            max_value += amplitude;
            amplitude *= persistence;
            frequency *= lacunarity;
        }
        
        elevation /= max_value;
        
        // Add "erosion" - controls whether terrain is mountainous or flat
        let erosion = perlin2.get([x * 0.8, y * 0.8, z * 0.8]);
        
        // Combine elevation and erosion (Minecraft's approach)
        let base_height = elevation * 0.6 + erosion * 0.4;
        
        // Shape terrain: create distinct biomes
        let shaped_height = if base_height > 0.3 {
            // High mountains
            0.3 + (base_height - 0.3).powf(1.3) * 3.0
        } else if base_height > 0.0 {
            // Hills and plains
            base_height * 0.8
        } else if base_height > -0.3 {
            // Low areas
            base_height * 0.4
        } else {
            // Deep valleys
            -0.12 + (base_height + 0.3) * 0.2
        };
        
        // Add variation
        let random_variation = rng.sample(Normal::new(0.0, 1.5).unwrap());
        
        // Scale to block heights - larger range for more dramatic terrain
        let max_elevation = shaped_height as f32 * 55.0 + random_variation;

        Self {
            max_elevation,
            temperature: parent.temperature + rng.sample(unif),
            rainfall: parent.rainfall + rng.sample(unif),
            blockiness: parent.blockiness + rng.sample(unif),
        }
    }
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

/// serp interpolates between two values v0 and v1 over the interval [0, 1] by yielding
/// v0 for [0, threshold], v1 for [1-threshold, 1], and linear interpolation in between
/// such that the overall shape is an S-shaped piecewise function.
/// threshold should be between 0 and 0.5.
fn serp<N: na::RealField + Copy>(v0: N, v1: N, t: N, threshold: N) -> N {
    if t < threshold {
        v0
    } else if t < (N::one() - threshold) {
        let s = (t - threshold) / ((N::one() - threshold) - threshold);
        v0 * (N::one() - s) + v1 * s
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

fn chunk_point_world(
    vertex: Vertex,
    world_from_node: &MIsometry<f32>,
    local: na::Vector3<f32>,
) -> na::Vector3<f32> {
    let node_space = vertex.chunk_to_node() * local.push(1.0);
    let point = MVector::from(node_space).normalized_point();
    minkowski_to_vec3(world_from_node * point)
}

fn minkowski_to_vec3(point: MPoint<f32>) -> na::Vector3<f32> {
    let v: na::Vector4<f32> = point.into();
    let inv_w = if v.w.abs() > f32::EPSILON {
        v.w.recip()
    } else {
        1.0
    };
    na::Vector3::new(v.x * inv_w, v.y * inv_w, v.z * inv_w)
}

fn plane_basis(surface: Plane) -> [na::Vector3<f32>; 3] {
    let normal_vec: na::Vector4<f32> = (*surface.scaled_normal()).into();
    let mut up = na::Vector3::new(normal_vec.x, normal_vec.y, normal_vec.z);
    if up.norm_squared() < 1.0e-6 {
        up = na::Vector3::new(0.0, 1.0, 0.0);
    }
    up = up.normalize();
    let mut east = up.cross(&na::Vector3::new(0.0, 0.0, 1.0));
    if east.norm_squared() < 1.0e-6 {
        east = up.cross(&na::Vector3::new(1.0, 0.0, 0.0));
    }
    east = east.normalize();
    let north = up.cross(&east);
    [east, north, up]
}

mod remcpe {
    use super::*;
    use noise::Perlin;
    use rand::{Rng, SeedableRng};
    use rand_pcg::Pcg64Mcg;

    const WORLD_SEED: u32 = 0x5eed5eed;
    const SEA_LEVEL: f32 = 0.0;
    const MC_MIN_HEIGHT: f32 = -96.0;
    const MC_MAX_HEIGHT: f32 = 96.0;
    const MC_BEDROCK_FLOOR: f32 = -96.0;
    const HEIGHT_MULTIPLIER: f32 = 1.25;
    const CONTINENTAL_FREQ: f64 = 1.0 / 550.0;
    const DETAIL_FREQ: f64 = 1.0 / 85.0;
    const RIDGE_FREQ: f64 = 1.0 / 220.0;
    const TEMP_FREQ: f64 = 1.0 / 700.0;
    const RAIN_FREQ: f64 = 1.0 / 640.0;
    const SURF_FREQ: f64 = 1.0 / 18.0;
    const CAVE_HORIZ_FREQ: f64 = 1.0 / 28.0;
    const CAVE_VERT_FREQ: f64 = 1.0 / 32.0;

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
            params
                .orientation()
                .column_index(params.dimension, coords)
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
        let continental = Perlin::new(WORLD_SEED);
        let detail = Perlin::new(WORLD_SEED ^ 0x9e37);
        let ridge = Perlin::new(WORLD_SEED ^ 0x4d2);
        let temp_noise = Perlin::new(WORLD_SEED ^ 0x6ac1);
        let rain_noise = Perlin::new(WORLD_SEED ^ 0x35a7);
        let surf_noise = Perlin::new(WORLD_SEED ^ 0x8e21);

        for a in 0..params.dimension {
            for b in 0..params.dimension {
                let idx = ColumnData::idx_components(params, a, b);
                let mut local = na::Vector3::repeat(0.5);
                local[axis0 as usize] = (f32::from(a) + 0.5) / dim_f;
                local[axis1 as usize] = (f32::from(b) + 0.5) / dim_f;
                let horizontal = params.horizontal_coords(local);
                let hx = horizontal.x as f64;
                let hz = horizontal.y as f64;
                let continental_val = continental.get([hx * CONTINENTAL_FREQ, hz * CONTINENTAL_FREQ]) as f32;
                let detail_val = detail.get([hx * DETAIL_FREQ, hz * DETAIL_FREQ]) as f32;
                let ridge_val = ridge.get([hx * RIDGE_FREQ, hz * RIDGE_FREQ]) as f32;
                let temp = ((temp_noise.get([hx * TEMP_FREQ, hz * TEMP_FREQ]) as f32) * 0.4 + 0.6)
                    .clamp(0.0, 1.0);
                let rain = ((rain_noise.get([hx * RAIN_FREQ, hz * RAIN_FREQ]) as f32) * 0.5 + 0.5)
                    .clamp(0.0, 1.0);
                let biome = BiomeKind::classify(temp, rain);
                let mut height = continental_val * 48.0 + detail_val * 8.0
                    + ridge_val.abs() * 18.0
                    + biome.height_offset();
                height = height.clamp(MC_MIN_HEIGHT, MC_MAX_HEIGHT);
                let scaled_height = scale_height_to_chunk(params, height);
                let surface_depth = ((surf_noise.get([hx * SURF_FREQ, hz * SURF_FREQ]) as f32 + 1.0)
                    * 1.5
                    + 1.0) as u8;
                let mut column = ColumnData::new();
                column.horizontal = horizontal;
                column.height = scaled_height;
                column.biome = biome;
                column.top_block = biome.top_block();
                column.filler_block = biome.filler_block();
                column.surface_depth = surface_depth.max(1);
                column.freeze_water = biome.freeze_water();
                column.vegetation_bias = rain;
                columns[idx] = column;
            }
        }

        columns
    }

    fn fill_base_layers(
        params: &ChunkParams,
        voxels: &mut [BlockID],
        columns: &mut [ColumnData],
    ) {
        let mut rng = Pcg64Mcg::seed_from_u64(params.node_spice);
        let bedrock_floor = scale_height_to_chunk(params, MC_BEDROCK_FLOOR);
        let bedrock_variance = scale_height_to_chunk(params, 4.0);
        let sea_level = scale_height_to_chunk(params, SEA_LEVEL);
        for (coords_x, coords_y, coords_z) in VoxelCoords::new(params.dimension) {
            let coords = na::Vector3::new(coords_x, coords_y, coords_z);
            let column_idx = ColumnData::idx(params, coords);
            let column = &mut columns[column_idx];
            let center = voxel_center(params.dimension, coords);
            let block_height = -params
                .surface
                .distance_to_chunk(params.chunk, &center)
                * f32::from(params.dimension);
            let mut block = BlockKind::Air.id();

            if block_height <= bedrock_floor + rng.gen_range(0.0..bedrock_variance) {
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
        let cave_noise = Perlin::new(WORLD_SEED ^ 0xacab);
        let min_cave_height = scale_height_to_chunk(params, MC_MIN_HEIGHT + 4.0);
        for (coords_x, coords_y, coords_z) in VoxelCoords::new(params.dimension) {
            let coords = na::Vector3::new(coords_x, coords_y, coords_z);
            let column_idx = ColumnData::idx(params, coords);
            let column = &columns[column_idx];
            let center = voxel_center(params.dimension, coords);
            let block_height = -params
                .surface
                .distance_to_chunk(params.chunk, &center)
                * f32::from(params.dimension);
            if block_height > column.height || block_height < min_cave_height {
                continue;
            }
            let horizontal = column.horizontal;
            let noise_val = cave_noise.get([
                horizontal.x as f64 * CAVE_HORIZ_FREQ,
                block_height as f64 * CAVE_VERT_FREQ,
                horizontal.y as f64 * CAVE_HORIZ_FREQ,
            ]) as f32;
            if noise_val > 0.55 {
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
            if fractional > 0.0 && rng.gen_bool(fractional as f64) {
                attempts += 1;
            }
            for _ in 0..attempts {
                let x = rng.gen_range(0..params.dimension);
                let z = rng.gen_range(0..params.dimension);
                let y = rng.gen_range(0..params.dimension);
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
                rng.gen_range(-1..=1),
                rng.gen_range(-1..=1),
                rng.gen_range(-1..=1),
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
        -params
            .surface
            .distance_to_chunk(params.chunk, &center)
            * f32::from(params.dimension)
    }

    fn populate_surface_features(
        params: &ChunkParams,
        voxels: &mut [BlockID],
        columns: &mut [ColumnData],
    ) {
        let mut rng = Pcg64Mcg::seed_from_u64(hash(params.node_spice, params.chunk as u64 ^ 0xfeed));
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
                        if rng.gen_bool(cactus_prob as f64) {
                            place_cactus(params, voxels, top, rng.gen_range(2..5));
                        }
                    }
                    _ => {
                        if column.biome.tree_weight() > 0 {
                            let base_prob = column.biome.tree_weight() as f32 / 10.0;
                            let tree_prob = (base_prob * density_scale).clamp(0.0, 1.0);
                            if rng.gen_bool(tree_prob as f64) {
                                let _ = place_tree(params, voxels, top, &mut rng);
                                continue;
                            }
                        }
                        let flower_prob = (0.05 * density_scale).min(1.0);
                        if rng.gen_bool(flower_prob as f64) {
                            place_flower(params, voxels, top);
                            continue;
                        }
                        let mushroom_prob = (0.03 * density_scale).min(1.0);
                        if rng.gen_bool(mushroom_prob as f64) {
                            place_mushroom(params, voxels, top, rng.gen_bool(0.5));
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
        let height = rng.gen_range(4..7);
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

    fn place_cactus(params: &ChunkParams, voxels: &mut [BlockID], base: na::Vector3<u8>, height: u8) {
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
}
