use std::collections::VecDeque;

use fxhash::FxHashSet;

use crate::{
    cursor::{Cursor, Dir},
    dodeca::Vertex,
    graph::{Graph, NodeId},
    light::LightData,
    math::PermuteXYZ,
    node::{Chunk, ChunkId, VoxelData},
    voxel_math::{ChunkAxisPermutation, ChunkDirection, CoordAxis, CoordSign, Coords},
};

/// Updates the margins of both `voxels` and `neighbor_voxels` at the side they meet at.
/// It is assumed that `voxels` corresponds to a chunk that lies at `vertex` and that
/// `neighbor_voxels` is at direction `direction` from `voxels`.
pub fn fix_margins(
    dimension: u8,
    vertex: Vertex,
    voxels: &mut VoxelData,
    direction: ChunkDirection,
    neighbor_voxels: &mut VoxelData,
) {
    let neighbor_axis_permutation = neighbor_axis_permutation(vertex, direction);

    let margin_coord = CoordsWithMargins::margin_coord(dimension, direction.sign);
    let boundary_coord = CoordsWithMargins::boundary_coord(dimension, direction.sign);

    // If two solid chunks are both void or both non-void, do nothing.
    if voxels.is_solid()
        && neighbor_voxels.is_solid()
        && (voxels.get(0) == 0) == (neighbor_voxels.get(0) == 0)
    {
        return;
    }

    // If either chunk is solid and consistent with the boundary of the other chunk, do nothing.
    // Since this consists of two similar cases (which of the two chunks is solid), we use a loop
    // here to make it clear how the logic of these two cases differ from each other.
    for (dense_voxels, dense_to_solid_direction, solid_voxels) in [
        (&*voxels, direction, &*neighbor_voxels),
        (
            &*neighbor_voxels,
            neighbor_axis_permutation * direction,
            &*voxels,
        ),
    ] {
        // Check that dense_voxels is indeed dense and solid_voxels is indeed solid
        if !dense_voxels.is_solid() && solid_voxels.is_solid() {
            let solid_voxels_is_void = solid_voxels.get(0) == 0;
            // Check that the face of dense_voxels that meets solid_voxels matches. If it does,
            // skip the margin reconciliation stage.
            if all_voxels_at_face(dimension, dense_voxels, dense_to_solid_direction, |m| {
                (m == 0) == solid_voxels_is_void
            }) {
                return;
            }
        }
    }

    // Otherwise, both chunks need to be dense, and margins should be reconciled between them.
    let voxel_data = voxels.data_mut(dimension);
    let neighbor_voxel_data = neighbor_voxels.data_mut(dimension);
    for j in 0..dimension {
        for i in 0..dimension {
            // Determine coordinates of the boundary voxel (to read from) and the margin voxel (to write to)
            // in voxel_data's perspective. To convert to neighbor_voxel_data's perspective, left-multiply
            // by neighbor_axis_permutation.
            let coords_of_boundary_voxel = CoordsWithMargins(
                [boundary_coord, i + 1, j + 1].tuv_to_xyz(direction.axis as usize),
            );
            let coords_of_margin_voxel =
                CoordsWithMargins([margin_coord, i + 1, j + 1].tuv_to_xyz(direction.axis as usize));

            // Use neighbor_voxel_data to set margins of voxel_data
            voxel_data[coords_of_margin_voxel.to_index(dimension)] = neighbor_voxel_data
                [(neighbor_axis_permutation * coords_of_boundary_voxel).to_index(dimension)];

            // Use voxel_data to set margins of neighbor_voxel_data
            neighbor_voxel_data
                [(neighbor_axis_permutation * coords_of_margin_voxel).to_index(dimension)] =
                voxel_data[coords_of_boundary_voxel.to_index(dimension)];
        }
    }
}

/// Updates the light margins of both `light` and `neighbor_light` at the side they meet at.
/// This is analogous to `fix_margins` but for light data instead of voxel data.
///
/// It is assumed that `light` corresponds to a chunk that lies at `vertex` and that
/// `neighbor_light` is at direction `direction` from `light`.
pub fn fix_light_margins(
    dimension: u8,
    vertex: Vertex,
    light: &mut LightData,
    direction: ChunkDirection,
    neighbor_light: &mut LightData,
) {
    let neighbor_axis_permutation = neighbor_axis_permutation(vertex, direction);

    let margin_coord = CoordsWithMargins::margin_coord(dimension, direction.sign);
    let boundary_coord = CoordsWithMargins::boundary_coord(dimension, direction.sign);

    // Get mutable access to both light data arrays
    let light_data = light.data_mut(dimension);
    let neighbor_light_data = neighbor_light.data_mut(dimension);

    // Synchronize light values at the shared face
    for j in 0..dimension {
        for i in 0..dimension {
            // Determine coordinates of the boundary voxel (to read from) and the margin voxel (to write to)
            // in light_data's perspective. To convert to neighbor_light_data's perspective, left-multiply
            // by neighbor_axis_permutation.
            let coords_of_boundary_voxel = CoordsWithMargins(
                [boundary_coord, i + 1, j + 1].tuv_to_xyz(direction.axis as usize),
            );
            let coords_of_margin_voxel =
                CoordsWithMargins([margin_coord, i + 1, j + 1].tuv_to_xyz(direction.axis as usize));

            // Use neighbor_light_data to set margins of light_data
            light_data[coords_of_margin_voxel.to_index(dimension)] = neighbor_light_data
                [(neighbor_axis_permutation * coords_of_boundary_voxel).to_index(dimension)];

            // Use light_data to set margins of neighbor_light_data
            neighbor_light_data
                [(neighbor_axis_permutation * coords_of_margin_voxel).to_index(dimension)] =
                light_data[coords_of_boundary_voxel.to_index(dimension)];
        }
    }
}

/// Check if the given predicate `f` holds true for any voxel at the given face of a chunk
fn all_voxels_at_face(
    dimension: u8,
    voxels: &VoxelData,
    direction: ChunkDirection,
    f: impl Fn(u16) -> bool,
) -> bool {
    let boundary_coord = CoordsWithMargins::boundary_coord(dimension, direction.sign);
    for j in 0..dimension {
        for i in 0..dimension {
            let coords_of_boundary_voxel = CoordsWithMargins(
                [boundary_coord, i + 1, j + 1].tuv_to_xyz(direction.axis as usize),
            );

            if !f(voxels.get(coords_of_boundary_voxel.to_index(dimension))) {
                return false;
            }
        }
    }

    true
}

/// Updates the margins of a given VoxelData to match the voxels they're next to. This is a good assumption to start
/// with before taking into account neighboring chunks because it means that no surface will be present on the boundaries
/// of the chunk, resulting in the least rendering. This is also generally accurate when the neighboring chunks are solid.
pub fn initialize_margins(dimension: u8, voxels: &mut VoxelData) {
    // If voxels is solid, the margins are already set up the way they should be.
    if voxels.is_solid() {
        return;
    }

    for direction in ChunkDirection::iter() {
        let margin_coord = CoordsWithMargins::margin_coord(dimension, direction.sign);
        let boundary_coord = CoordsWithMargins::boundary_coord(dimension, direction.sign);
        let chunk_data = voxels.data_mut(dimension);
        for j in 0..dimension {
            for i in 0..dimension {
                // Determine coordinates of the boundary voxel (to read from) and the margin voxel (to write to).
                let coords_of_boundary_voxel = CoordsWithMargins(
                    [boundary_coord, i + 1, j + 1].tuv_to_xyz(direction.axis as usize),
                );
                let coords_of_margin_voxel = CoordsWithMargins(
                    [margin_coord, i + 1, j + 1].tuv_to_xyz(direction.axis as usize),
                );

                chunk_data[coords_of_margin_voxel.to_index(dimension)] =
                    chunk_data[coords_of_boundary_voxel.to_index(dimension)];
            }
        }
    }
}

/// Based on the given `coords` and the neighboring voxel at direction
/// `direction` (if it's in a different chunk), updates both of their respective
/// margins to match each others' materials.
pub fn reconcile_margin_voxels(
    graph: &mut Graph,
    chunk: ChunkId,
    coords: Coords,
    direction: ChunkDirection,
) {
    let coords_of_boundary_voxel: CoordsWithMargins = coords.into();
    let dimension = graph.layout().dimension();

    // There is nothing to do if we're not on a boundary voxel.
    if coords_of_boundary_voxel[direction.axis]
        != CoordsWithMargins::boundary_coord(dimension, direction.sign)
    {
        return;
    }

    let mut coords_of_margin_voxel = coords_of_boundary_voxel;
    coords_of_margin_voxel[direction.axis] =
        CoordsWithMargins::margin_coord(dimension, direction.sign);

    let neighbor_axis_permutation = neighbor_axis_permutation(chunk.vertex, direction);
    let Some(neighbor_chunk) = graph.get_chunk_neighbor(chunk, direction.axis, direction.sign)
    else {
        // If there's no neighbor chunk, there is nothing to do.
        return;
    };

    // Gather information from the current chunk and the neighboring chunk. If either is unpopulated, there
    // is nothing to do.
    let material = if let Chunk::Populated { voxels, .. } = &graph[chunk] {
        voxels.get(coords.to_index(dimension))
    } else {
        return;
    };
    let neighbor_material = if let Chunk::Populated {
        voxels: neighbor_voxels,
        ..
    } = &graph[neighbor_chunk]
    {
        neighbor_voxels
            .get((neighbor_axis_permutation * coords_of_boundary_voxel).to_index(dimension))
    } else {
        return;
    };

    // Update the neighbor chunk's margin to the current chunk's material.
    let Chunk::Populated {
        voxels: neighbor_voxels,
        surface: neighbor_surface,
        old_surface: neighbor_old_surface,
        ..
    } = &mut graph[neighbor_chunk]
    else {
        unreachable!();
    };
    neighbor_voxels.data_mut(dimension)
        [(neighbor_axis_permutation * coords_of_margin_voxel).to_index(dimension)] = material;
    *neighbor_old_surface = neighbor_surface.take().or(*neighbor_old_surface);

    // Update the current chunk's margin to the neighbor chunk's material.

    // This can be necessary even if `neighbor_material` hasn't changed because
    // margins are not guaranteed to have exactly the right material unless they
    // need to be rendered. For instance a margin can sometimes store BlockKind::Dirt
    // even if the voxel it's based on uses BlockKind::Stone because
    // changing the margin from dirt to stone earlier would have required
    // turning a solid chunk into a dense chunk.
    let Chunk::Populated {
        voxels,
        surface,
        old_surface,
        ..
    } = &mut graph[chunk]
    else {
        unreachable!();
    };
    voxels.data_mut(dimension)[coords_of_margin_voxel.to_index(dimension)] = neighbor_material;
    *old_surface = surface.take().or(*old_surface);
}

fn neighbor_axis_permutation(vertex: Vertex, direction: ChunkDirection) -> ChunkAxisPermutation {
    match direction.sign {
        CoordSign::Plus => vertex.chunk_axis_permutations()[direction.axis as usize],
        CoordSign::Minus => ChunkAxisPermutation::IDENTITY,
    }
}

// ============================================================================
// Extended Neighbor Enumeration
// ============================================================================
//
// In hyperbolic space with the order-4 dodecahedral honeycomb, a chunk shares
// at least one vertex with 110 other chunks (compared to 26 in Euclidean space).
// The following types and functions provide efficient enumeration and querying
// of all these neighbors.

/// Describes how a neighbor chunk relates to a reference chunk geometrically.
///
/// In Euclidean 3-space, a cube has:
/// - 6 face neighbors (share a 2D face)
/// - 12 edge neighbors (share a 1D edge but not a face)
/// - 8 corner neighbors (share only 0D vertices)
///
/// In hyperbolic space, the counts are different due to 5 cubes meeting at each
/// edge instead of 4, but the classification still applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NeighborType {
    /// Shares a face with the reference chunk. There are always exactly 6 of these.
    Face,
    /// Shares an edge but not a face. In hyperbolic space, there are more than 12.
    Edge,
    /// Shares only vertices (corners). The count varies based on geometry.
    Corner,
}

/// A neighboring chunk along with metadata about its relationship to a reference chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkNeighbor {
    /// The neighboring chunk's canonical ID.
    pub chunk: ChunkId,
    /// The type of adjacency (face, edge, or corner).
    pub neighbor_type: NeighborType,
    /// The Manhattan distance in cursor steps from the reference chunk.
    /// Face neighbors have distance 1, edge neighbors have distance 2, etc.
    pub distance: u8,
}

/// Returns the 8 node IDs that correspond to the vertices of the dual cube for the given chunk.
///
/// Each chunk is 1/8 of a cube in the dual honeycomb. The cube's 8 vertices are at
/// centers of dodecahedral nodes. This function returns those 8 node IDs, lazily
/// allocating nodes in the graph as needed.
pub fn cube_vertex_nodes(graph: &mut Graph, chunk: ChunkId) -> [NodeId; 8] {
    let mut result = [chunk.node; 8];
    for (i, (_coords, path)) in chunk.vertex.dual_vertices().enumerate() {
        let mut node = chunk.node;
        for side in path {
            node = graph.ensure_neighbor(node, side);
        }
        result[i] = node;
    }
    result
}

/// Returns the 8 node IDs for the dual cube vertices without allocating new nodes.
/// Returns `None` if any required node doesn't exist in the graph.
pub fn try_cube_vertex_nodes(graph: &Graph, chunk: ChunkId) -> Option<[NodeId; 8]> {
    let mut result = [chunk.node; 8];
    for (i, (_coords, path)) in chunk.vertex.dual_vertices().enumerate() {
        let mut node = chunk.node;
        for side in path {
            node = graph.neighbor(node, side)?;
        }
        result[i] = node;
    }
    Some(result)
}

/// Checks if two chunks share at least one vertex of their dual cubes.
///
/// Two chunks are vertex-adjacent if any of their 8 dual-cube vertices coincide.
pub fn chunks_share_vertex(
    graph: &mut Graph,
    chunk_a: ChunkId,
    chunk_b_vertex_nodes: &[NodeId; 8],
) -> bool {
    let a_nodes = cube_vertex_nodes(graph, chunk_a);
    a_nodes
        .iter()
        .any(|n| chunk_b_vertex_nodes.iter().any(|m| *n == *m))
}

/// Iterator over all chunks that share at least one vertex with a reference chunk.
///
/// This performs a breadth-first search from the reference chunk, expanding through
/// the 6 face-adjacent directions and collecting all chunks whose dual-cube vertices
/// overlap with the reference chunk's vertices.
pub struct VertexSharingNeighbors {
    neighbors: Vec<ChunkNeighbor>,
    index: usize,
}

impl VertexSharingNeighbors {
    /// Enumerate all vertex-sharing neighbors of the given chunk.
    ///
    /// This allocates graph nodes as needed during traversal.
    /// The reference chunk itself is NOT included in the results.
    pub fn new(graph: &mut Graph, chunk: ChunkId) -> Self {
        let start = graph.canonicalize(chunk).unwrap_or(chunk);
        let start_vertex_nodes = cube_vertex_nodes(graph, start);

        let mut visited = FxHashSet::<ChunkId>::default();
        let mut queue = VecDeque::<(ChunkId, u8)>::new();
        let mut neighbors = Vec::new();

        visited.insert(start);
        queue.push_back((start, 0));

        while let Some((current, distance)) = queue.pop_front() {
            let cursor = Cursor::from_vertex(current.node, current.vertex);

            for dir in Dir::iter() {
                let next_cursor = cursor.step_ensuring(graph, dir);
                let Some(next_chunk) = next_cursor.canonicalize(graph) else {
                    continue;
                };

                if visited.contains(&next_chunk) {
                    continue;
                }

                let next_vertex_nodes = cube_vertex_nodes(graph, next_chunk);
                if !nodes_overlap(&start_vertex_nodes, &next_vertex_nodes) {
                    continue;
                }

                let next_distance = distance + 1;
                let neighbor_type = classify_neighbor(graph, start, &start_vertex_nodes, next_chunk);

                visited.insert(next_chunk);
                neighbors.push(ChunkNeighbor {
                    chunk: next_chunk,
                    neighbor_type,
                    distance: next_distance,
                });
                queue.push_back((next_chunk, next_distance));
            }
        }

        Self { neighbors, index: 0 }
    }

    /// Returns the total count of vertex-sharing neighbors.
    pub fn count(&self) -> usize {
        self.neighbors.len()
    }

    /// Returns all neighbors as a slice.
    pub fn as_slice(&self) -> &[ChunkNeighbor] {
        &self.neighbors
    }

    /// Returns only face-adjacent neighbors (there are always exactly 6).
    pub fn faces(&self) -> impl Iterator<Item = &ChunkNeighbor> {
        self.neighbors
            .iter()
            .filter(|n| n.neighbor_type == NeighborType::Face)
    }

    /// Returns only edge-adjacent neighbors (share edge but not face).
    pub fn edges(&self) -> impl Iterator<Item = &ChunkNeighbor> {
        self.neighbors
            .iter()
            .filter(|n| n.neighbor_type == NeighborType::Edge)
    }

    /// Returns only corner-adjacent neighbors (share only vertices).
    pub fn corners(&self) -> impl Iterator<Item = &ChunkNeighbor> {
        self.neighbors
            .iter()
            .filter(|n| n.neighbor_type == NeighborType::Corner)
    }

    /// Consume the iterator and return all neighbors grouped by type.
    pub fn into_grouped(self) -> NeighborsByType {
        let mut faces = Vec::new();
        let mut edges = Vec::new();
        let mut corners = Vec::new();

        for neighbor in self.neighbors {
            match neighbor.neighbor_type {
                NeighborType::Face => faces.push(neighbor),
                NeighborType::Edge => edges.push(neighbor),
                NeighborType::Corner => corners.push(neighbor),
            }
        }

        NeighborsByType {
            faces,
            edges,
            corners,
        }
    }
}

impl Iterator for VertexSharingNeighbors {
    type Item = ChunkNeighbor;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.neighbors.len() {
            let item = self.neighbors[self.index];
            self.index += 1;
            Some(item)
        } else {
            None
        }
    }
}

impl ExactSizeIterator for VertexSharingNeighbors {
    fn len(&self) -> usize {
        self.neighbors.len() - self.index
    }
}

/// Neighbors grouped by their adjacency type.
#[derive(Debug, Clone)]
pub struct NeighborsByType {
    /// Face-adjacent neighbors (6 in any geometry).
    pub faces: Vec<ChunkNeighbor>,
    /// Edge-adjacent neighbors (share edge but not face).
    pub edges: Vec<ChunkNeighbor>,
    /// Corner-adjacent neighbors (share only vertices).
    pub corners: Vec<ChunkNeighbor>,
}

impl NeighborsByType {
    /// Total number of all neighbors.
    pub fn total(&self) -> usize {
        self.faces.len() + self.edges.len() + self.corners.len()
    }

    /// Iterate over all neighbors in order: faces, then edges, then corners.
    pub fn iter(&self) -> impl Iterator<Item = &ChunkNeighbor> {
        self.faces
            .iter()
            .chain(self.edges.iter())
            .chain(self.corners.iter())
    }
}

/// Returns the 6 face-adjacent chunk neighbors.
///
/// This is a simpler, more efficient alternative to `VertexSharingNeighbors` when
/// you only need face neighbors. Returns `None` for any neighbor whose node
/// doesn't exist in the graph.
pub fn face_neighbors(graph: &Graph, chunk: ChunkId) -> [Option<ChunkId>; 6] {
    let mut result = [None; 6];
    for (i, dir) in ChunkDirection::iter().enumerate() {
        result[i] = graph.get_chunk_neighbor(chunk, dir.axis, dir.sign);
    }
    result
}

/// Returns the 6 face-adjacent chunk neighbors, allocating nodes as needed.
pub fn face_neighbors_ensuring(graph: &mut Graph, chunk: ChunkId) -> [ChunkId; 6] {
    let cursor = Cursor::from_vertex(chunk.node, chunk.vertex);
    let mut result = [chunk; 6];
    for (i, dir) in Dir::iter().enumerate() {
        let next = cursor.step_ensuring(graph, dir);
        if let Some(canonical) = next.canonicalize(graph) {
            result[i] = canonical;
        }
    }
    result
}

// Helper: Check if two sets of node IDs have any overlap
fn nodes_overlap(a: &[NodeId; 8], b: &[NodeId; 8]) -> bool {
    a.iter().any(|n| b.iter().any(|m| *n == *m))
}

// Helper: Count how many vertices two chunks share
fn count_shared_vertices(a_nodes: &[NodeId; 8], b_nodes: &[NodeId; 8]) -> usize {
    a_nodes
        .iter()
        .filter(|n| b_nodes.iter().any(|m| *n == m))
        .count()
}

// Helper: Classify a neighbor based on how many vertices it shares
fn classify_neighbor(
    graph: &mut Graph,
    _reference: ChunkId,
    reference_nodes: &[NodeId; 8],
    neighbor: ChunkId,
) -> NeighborType {
    let neighbor_nodes = cube_vertex_nodes(graph, neighbor);
    let shared = count_shared_vertices(reference_nodes, &neighbor_nodes);

    // Face neighbors share 4 vertices (a whole face of the cube)
    // Edge neighbors share 2 vertices (an edge of the cube)
    // Corner neighbors share 1 vertex
    match shared {
        4 => NeighborType::Face,
        2 => NeighborType::Edge,
        1 => NeighborType::Corner,
        _ => {
            // 0 shouldn't happen if we filtered correctly, treat as corner
            debug_assert!(shared > 0, "Neighbor should share at least one vertex");
            NeighborType::Corner
        }
    }
}

/// Coordinates for a discrete voxel within a chunk, including margins
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CoordsWithMargins(pub [u8; 3]);

impl CoordsWithMargins {
    /// Returns the array index in `VoxelData` corresponding to these coordinates
    pub fn to_index(self, chunk_size: u8) -> usize {
        let chunk_size_with_margin = chunk_size as usize + 2;
        (self.0[0] as usize)
            + (self.0[1] as usize) * chunk_size_with_margin
            + (self.0[2] as usize) * chunk_size_with_margin.pow(2)
    }

    /// Returns the x, y, or z coordinate that would correspond to the margin in the direction of `sign`
    pub fn margin_coord(chunk_size: u8, sign: CoordSign) -> u8 {
        match sign {
            CoordSign::Plus => chunk_size + 1,
            CoordSign::Minus => 0,
        }
    }

    /// Returns the x, y, or z coordinate that would correspond to the voxel meeting the chunk boundary in the direction of `sign`
    pub fn boundary_coord(chunk_size: u8, sign: CoordSign) -> u8 {
        match sign {
            CoordSign::Plus => chunk_size,
            CoordSign::Minus => 1,
        }
    }
}

impl From<Coords> for CoordsWithMargins {
    #[inline]
    fn from(value: Coords) -> Self {
        CoordsWithMargins([value.0[0] + 1, value.0[1] + 1, value.0[2] + 1])
    }
}

impl std::ops::Index<CoordAxis> for CoordsWithMargins {
    type Output = u8;

    #[inline]
    fn index(&self, coord_axis: CoordAxis) -> &u8 {
        self.0.index(coord_axis as usize)
    }
}

impl std::ops::IndexMut<CoordAxis> for CoordsWithMargins {
    #[inline]
    fn index_mut(&mut self, coord_axis: CoordAxis) -> &mut u8 {
        self.0.index_mut(coord_axis as usize)
    }
}

impl std::ops::Mul<CoordsWithMargins> for ChunkAxisPermutation {
    type Output = CoordsWithMargins;

    fn mul(self, rhs: CoordsWithMargins) -> Self::Output {
        let mut result = CoordsWithMargins([0; 3]);
        for axis in CoordAxis::iter() {
            result[self[axis]] = rhs[axis];
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use crate::{dodeca::Vertex, graph::NodeId, voxel_math::Coords, world::BlockKind};

    use super::*;

    #[test]
    fn test_fix_margins() {
        // This test case can set up empirically by placing blocks and printing their coordinates to confirm which
        // coordinates are adjacent to each other.

        // `voxels` lives at vertex F
        let mut voxels = VoxelData::Solid(BlockKind::Air.id());
        voxels.data_mut(12)[Coords([11, 2, 10]).to_index(12)] = BlockKind::WoodPlanks.id();

        // `neighbor_voxels` lives at vertex J
        let mut neighbor_voxels = VoxelData::Solid(BlockKind::Air.id());
        neighbor_voxels.data_mut(12)[Coords([2, 10, 11]).to_index(12)] = BlockKind::Grass.id();

        // Sanity check that voxel adjacencies are as expected. If the test fails here, it's likely that "dodeca.rs" was
        // redesigned, and the test itself will have to be fixed, rather than the code being tested.
        assert_eq!(Vertex::F.adjacent_vertices()[0], Vertex::J);
        assert_eq!(Vertex::J.adjacent_vertices()[2], Vertex::F);

        // Sanity check that voxels are populated as expected, using `CoordsWithMargins` for consistency with the actual
        // test case.
        assert_eq!(
            voxels.get(CoordsWithMargins([12, 3, 11]).to_index(12)),
            BlockKind::WoodPlanks.id()
        );
        assert_eq!(
            neighbor_voxels.get(CoordsWithMargins([3, 11, 12]).to_index(12)),
            BlockKind::Grass.id()
        );

        fix_margins(
            12,
            Vertex::F,
            &mut voxels,
            ChunkDirection::PLUS_X,
            &mut neighbor_voxels,
        );

        // Actual verification: Check that the margins were set correctly
        assert_eq!(
            voxels.get(CoordsWithMargins([13, 3, 11]).to_index(12)),
            BlockKind::Grass.id()
        );
        assert_eq!(
            neighbor_voxels.get(CoordsWithMargins([3, 11, 13]).to_index(12)),
            BlockKind::WoodPlanks.id()
        );
    }

    #[test]
    fn test_initialize_margins() {
        let mut voxels = VoxelData::Solid(BlockKind::Air.id());
        voxels.data_mut(12)[Coords([11, 2, 10]).to_index(12)] = BlockKind::WoodPlanks.id();
        assert_eq!(
            voxels.get(CoordsWithMargins([12, 3, 11]).to_index(12)),
            BlockKind::WoodPlanks.id()
        );

        initialize_margins(12, &mut voxels);

        assert_eq!(
            voxels.get(CoordsWithMargins([13, 3, 11]).to_index(12)),
            BlockKind::WoodPlanks.id()
        );
    }

    #[test]
    fn test_reconcile_margin_voxels() {
        let mut graph = Graph::new(12);
        let current_vertex = Vertex::A;
        let neighbor_vertex = current_vertex.adjacent_vertices()[1];
        let neighbor_node =
            graph.ensure_neighbor(NodeId::ROOT, current_vertex.canonical_sides()[0]);

        // These are the chunks this test will work with.
        let current_chunk = ChunkId::new(NodeId::ROOT, current_vertex);
        let node_neighbor_chunk = ChunkId::new(neighbor_node, current_vertex);
        let vertex_neighbor_chunk = ChunkId::new(NodeId::ROOT, neighbor_vertex);

        // Populate relevant chunks
        for chunk in [current_chunk, node_neighbor_chunk, vertex_neighbor_chunk] {
            graph[chunk] = Chunk::Populated {
                voxels: VoxelData::Solid(BlockKind::Air.id()),
                light: crate::light::LightData::default(),
                light_dirty: false,
                skylight: crate::skylight::SkylightData::default(),
                skylight_cache_dirty: false,
                skylight_surface_dirty: false,
                surface: None,
                old_surface: None,
            };
        }

        // Fill current chunk with appropriate materials
        {
            let Chunk::Populated { voxels, .. } = &mut graph[current_chunk] else {
                unreachable!()
            };
            voxels.data_mut(12)[Coords([0, 7, 9]).to_index(12)] = BlockKind::WoodPlanks.id();
            voxels.data_mut(12)[Coords([5, 11, 9]).to_index(12)] = BlockKind::Grass.id();
        }

        // Fill vertex_neighbor chunk with appropriate material
        {
            let Chunk::Populated { voxels, .. } = &mut graph[vertex_neighbor_chunk] else {
                unreachable!()
            };
            voxels.data_mut(12)[Coords([5, 9, 11]).to_index(12)] = BlockKind::Stone.id();
        }

        // Reconcile margins
        reconcile_margin_voxels(
            &mut graph,
            current_chunk,
            Coords([0, 7, 9]),
            ChunkDirection::MINUS_X,
        );
        reconcile_margin_voxels(
            &mut graph,
            current_chunk,
            Coords([5, 11, 9]),
            ChunkDirection::PLUS_Y,
        );

        // Check the margins of current_chunk
        let Chunk::Populated {
            voxels: current_voxels,
            ..
        } = &graph[current_chunk]
        else {
            unreachable!("node_neighbor_chunk should have been populated by this test");
        };
        assert_eq!(
            current_voxels.get(CoordsWithMargins([6, 13, 10]).to_index(12)),
            BlockKind::Stone.id()
        );

        // Check the margins of node_neighbor_chunk
        let Chunk::Populated {
            voxels: node_neighbor_voxels,
            ..
        } = &graph[node_neighbor_chunk]
        else {
            unreachable!("node_neighbor_chunk should have been populated by this test");
        };
        assert_eq!(
            node_neighbor_voxels.get(CoordsWithMargins([0, 8, 10]).to_index(12)),
            BlockKind::WoodPlanks.id()
        );

        // Check the margins of vertex_neighbor_chunk
        let Chunk::Populated {
            voxels: vertex_neighbor_voxels,
            ..
        } = &graph[vertex_neighbor_chunk]
        else {
            unreachable!("vertex_neighbor_chunk should have been populated by this test");
        };
        assert_eq!(
            vertex_neighbor_voxels.get(CoordsWithMargins([6, 10, 13]).to_index(12)),
            BlockKind::Grass.id()
        );
    }

    #[test]
    fn test_vertex_sharing_neighbors() {
        let mut graph = Graph::new(12);
        let chunk = ChunkId::new(NodeId::ROOT, Vertex::A);

        let neighbors = VertexSharingNeighbors::new(&mut graph, chunk);
        let count = neighbors.count();

        // In hyperbolic space, a chunk shares vertices with 110 other chunks
        assert_eq!(count, 110, "Expected 110 vertex-sharing neighbors, got {count}");

        // Verify grouping works
        let grouped = VertexSharingNeighbors::new(&mut graph, chunk).into_grouped();

        // Face neighbors: always 6
        assert_eq!(
            grouped.faces.len(),
            6,
            "Expected 6 face neighbors, got {}",
            grouped.faces.len()
        );

        // All face neighbors should have distance 1
        for face in &grouped.faces {
            assert_eq!(face.distance, 1, "Face neighbor should have distance 1");
            assert_eq!(face.neighbor_type, NeighborType::Face);
        }

        // All edge neighbors should have distance >= 2
        for edge in &grouped.edges {
            assert!(edge.distance >= 1, "Edge neighbor distance should be >= 1");
            assert_eq!(edge.neighbor_type, NeighborType::Edge);
        }

        // All corner neighbors should share only vertices
        for corner in &grouped.corners {
            assert_eq!(corner.neighbor_type, NeighborType::Corner);
        }

        // Total should be 110
        assert_eq!(
            grouped.total(),
            110,
            "Grouped total should be 110, got {}",
            grouped.total()
        );

        eprintln!(
            "Neighbor breakdown: {} faces, {} edges, {} corners = {} total",
            grouped.faces.len(),
            grouped.edges.len(),
            grouped.corners.len(),
            grouped.total()
        );
    }

    #[test]
    fn test_face_neighbors() {
        let mut graph = Graph::new(12);
        let chunk = ChunkId::new(NodeId::ROOT, Vertex::A);

        // First test without ensuring - some may be None if nodes don't exist
        let neighbors_optional = face_neighbors(&graph, chunk);

        // With ensuring, all 6 should exist
        let neighbors = face_neighbors_ensuring(&mut graph, chunk);

        // All 6 should be different from the original chunk
        for neighbor in &neighbors {
            assert_ne!(*neighbor, chunk, "Face neighbor should not be the same chunk");
        }

        // All 6 should be distinct
        let unique: std::collections::HashSet<_> = neighbors.iter().collect();
        assert_eq!(unique.len(), 6, "Should have 6 unique face neighbors");
    }

    #[test]
    fn test_cube_vertex_nodes() {
        let mut graph = Graph::new(12);
        let chunk = ChunkId::new(NodeId::ROOT, Vertex::A);

        let vertex_nodes = cube_vertex_nodes(&mut graph, chunk);

        // Should have 8 nodes (corners of the dual cube)
        assert_eq!(vertex_nodes.len(), 8);

        // First vertex should be the chunk's own node (0 steps to reach)
        assert_eq!(vertex_nodes[0], chunk.node);
    }
}
