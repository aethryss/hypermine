use std::sync::LazyLock;

use crate::dodeca::{Side, Vertex};
use crate::graph::{Graph, NodeId};
use crate::node::ChunkId;

/// Navigates the cubic dual of a graph
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Cursor {
    node: NodeId,
    a: Side,
    b: Side,
    c: Side,
}

impl Cursor {
    /// Construct a canonical cursor for the cube at `vertex` of `node`
    pub fn from_vertex(node: NodeId, vertex: Vertex) -> Self {
        let [a, b, c] = vertex.canonical_sides();
        Self { node, a, b, c }
    }

    /// Get the neighbor towards `dir`
    pub fn step(self, graph: &Graph, dir: Dir) -> Option<Self> {
        // For a cube identified by three dodecahedral faces sharing a vertex, we identify its
        // cubical neighbors by taking each vertex incident to exactly two of the faces and the face
        // of the three it's not incident to, and selecting the cube represented by the new vertex
        // in both the dodecahedron sharing the face unique to the new vertex and that sharing the
        // face that the new vertex isn't incident to.
        let (a, b, c) = (self.a, self.b, self.c);
        let a_prime = NEIGHBORS[a as usize][b as usize][c as usize].unwrap();
        let b_prime = NEIGHBORS[b as usize][a as usize][c as usize].unwrap();
        let c_prime = NEIGHBORS[c as usize][b as usize][a as usize].unwrap();
        use Dir::*;
        let (sides, neighbor) = match dir {
            Left => ((a, b, c_prime), c),
            Right => ((a, b, c_prime), c_prime),
            Down => ((a, b_prime, c), b),
            Up => ((a, b_prime, c), b_prime),
            Forward => ((a_prime, b, c), a),
            Back => ((a_prime, b, c), a_prime),
        };
        let node = graph.neighbor(self.node, neighbor)?;
        Some(Self {
            node,
            a: sides.0,
            b: sides.1,
            c: sides.2,
        })
    }

    /// Node and dodecahedral vertex that contains the representation for this cube in the graph
    pub fn canonicalize(self, graph: &Graph) -> Option<ChunkId> {
        graph.canonicalize(ChunkId::new(
            self.node,
            Vertex::from_sides([self.a, self.b, self.c]).unwrap(),
        ))
    }

    /// Returns the node of this cursor.
    pub fn node(&self) -> NodeId {
        self.node
    }

    /// Returns the first side of this cursor.
    pub fn a(&self) -> Side {
        self.a
    }

    /// Returns the second side of this cursor.
    pub fn b(&self) -> Side {
        self.b
    }

    /// Returns the third side of this cursor.
    pub fn c(&self) -> Side {
        self.c
    }

    /// Get the neighbor towards `dir`, allocating the neighbor node if needed.
    pub fn step_ensuring(self, graph: &mut Graph, dir: Dir) -> Self {
        let (a, b, c) = (self.a, self.b, self.c);
        let a_prime = NEIGHBORS[a as usize][b as usize][c as usize].unwrap();
        let b_prime = NEIGHBORS[b as usize][a as usize][c as usize].unwrap();
        let c_prime = NEIGHBORS[c as usize][b as usize][a as usize].unwrap();
        use Dir::*;
        let (sides, neighbor) = match dir {
            Left => ((a, b, c_prime), c),
            Right => ((a, b, c_prime), c_prime),
            Down => ((a, b_prime, c), b),
            Up => ((a, b_prime, c), b_prime),
            Forward => ((a_prime, b, c), a),
            Back => ((a_prime, b, c), a_prime),
        };
        let node = graph.ensure_neighbor(self.node, neighbor);
        Self {
            node,
            a: sides.0,
            b: sides.1,
            c: sides.2,
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Dir {
    Left,
    Right,
    Down,
    Up,
    Forward,
    Back,
}
impl Dir {
    pub fn iter() -> impl ExactSizeIterator<Item = Self> + Clone {
        use Dir::*;
        [Left, Right, Down, Up, Forward, Back].into_iter()
    }

    /// Returns the unit vector corresponding to the direction.
    pub fn vector(self) -> na::Vector3<isize> {
        use Dir::*;
        match self {
            Up => na::Vector3::x(),
            Down => -na::Vector3::x(),
            Left => na::Vector3::y(),
            Right => -na::Vector3::y(),
            Forward => na::Vector3::z(),
            Back => -na::Vector3::z(),
        }
    }
}

/// Returns a direction's opposite direction.
impl std::ops::Neg for Dir {
    type Output = Self;
    fn neg(self) -> Self::Output {
        use Dir::*;
        match self {
            Left => Right,
            Right => Left,
            Down => Up,
            Up => Down,
            Forward => Back,
            Back => Forward,
        }
    }
}

/// Maps every (A, B, C) sharing a vertex to A', the side that shares edges with B and C but not A
static NEIGHBORS: LazyLock<
    [[[Option<Side>; Side::VALUES.len()]; Side::VALUES.len()]; Side::VALUES.len()],
> = LazyLock::new(|| {
    let mut result = [[[None; Side::VALUES.len()]; Side::VALUES.len()]; Side::VALUES.len()];
    for a in Side::iter() {
        for b in Side::iter() {
            for c in Side::iter() {
                for s in Side::iter() {
                    if s == a || s == b || s == c {
                        continue;
                    }
                    let (opposite, shared) =
                        match (s.adjacent_to(a), s.adjacent_to(b), s.adjacent_to(c)) {
                            (false, true, true) => (a, (b, c)),
                            (true, false, true) => (b, (a, c)),
                            (true, true, false) => (c, (a, b)),
                            _ => continue,
                        };
                    result[opposite as usize][shared.0 as usize][shared.1 as usize] = Some(s);
                }
            }
        }
    }
    result
});

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{proto::Position, traversal::ensure_nearby};

    fn cube_vertex_nodes_ensuring(graph: &mut Graph, chunk: ChunkId) -> Vec<NodeId> {
        // Each chunk corresponds to a cube in the dual honeycomb.
        // The cube's 8 vertices are at centers of (dodecahedral) nodes.
        // `Vertex::dual_vertices()` encodes how to reach each of those 8 nodes from `chunk.node`.
        let mut out = Vec::with_capacity(8);
        for (_coords, path) in chunk.vertex.dual_vertices() {
            let mut node = chunk.node;
            for side in path {
                // Lazily allocate the node graph so cube vertices are always resolvable.
                node = graph.ensure_neighbor(node, side);
            }
            out.push(node);
        }
        out
    }

    fn shares_any_cube_vertex_ensuring(
        graph: &mut Graph,
        a: ChunkId,
        b_vertex_nodes: &[NodeId],
    ) -> bool {
        let a_nodes = cube_vertex_nodes_ensuring(graph, a);
        a_nodes
            .into_iter()
            .any(|n| b_vertex_nodes.iter().any(|&m| m == n))
    }

    fn step_ensuring(graph: &mut Graph, cursor: Cursor, dir: Dir) -> Cursor {
        // Mirror Cursor::step, but allocate missing neighbor nodes on demand.
        let (a, b, c) = (cursor.a, cursor.b, cursor.c);
        let a_prime = NEIGHBORS[a as usize][b as usize][c as usize].unwrap();
        let b_prime = NEIGHBORS[b as usize][a as usize][c as usize].unwrap();
        let c_prime = NEIGHBORS[c as usize][b as usize][a as usize].unwrap();
        use Dir::*;
        let (sides, neighbor) = match dir {
            Left => ((a, b, c_prime), c),
            Right => ((a, b, c_prime), c_prime),
            Down => ((a, b_prime, c), b),
            Up => ((a, b_prime, c), b_prime),
            Forward => ((a_prime, b, c), a),
            Back => ((a_prime, b, c), a_prime),
        };
        let node = graph.ensure_neighbor(cursor.node, neighbor);
        Cursor {
            node,
            a: sides.0,
            b: sides.1,
            c: sides.2,
        }
    }

    fn enumerate_chunks_sharing_any_vertex_with(
        graph: &mut Graph,
        start: ChunkId,
        start_vertex_nodes: &[NodeId],
    ) -> Vec<ChunkId> {
        use std::collections::VecDeque;
        let mut visited = fxhash::FxHashSet::<ChunkId>::default();
        let mut queue = VecDeque::<ChunkId>::new();

        visited.insert(start);
        queue.push_back(start);

        while let Some(current) = queue.pop_front() {
            let cursor = Cursor::from_vertex(current.node, current.vertex);
            for dir in Dir::iter() {
                let next_cursor = step_ensuring(graph, cursor, dir);
                let next_chunk = next_cursor.canonicalize(graph).unwrap();
                if visited.contains(&next_chunk) {
                    continue;
                }
                if !shares_any_cube_vertex_ensuring(graph, next_chunk, start_vertex_nodes) {
                    continue;
                }
                visited.insert(next_chunk);
                queue.push_back(next_chunk);
            }
        }

        visited.into_iter().collect()
    }

    #[test]
    fn neighbor_sanity() {
        for v in Vertex::iter() {
            let [a, b, c] = v.canonical_sides();
            assert_eq!(
                NEIGHBORS[a as usize][b as usize][c as usize],
                NEIGHBORS[a as usize][c as usize][b as usize]
            );
        }
    }

    #[test]
    fn cursor_identities() {
        let mut graph = Graph::new(1);
        ensure_nearby(&mut graph, &Position::origin(), 3.0);
        let start = Cursor::from_vertex(NodeId::ROOT, Vertex::A);
        let wiggle = |dir| {
            let x = start.step(&graph, dir).unwrap();
            assert!(x != start);
            assert_eq!(x.step(&graph, -dir).unwrap(), start);
        };
        wiggle(Dir::Left);
        wiggle(Dir::Right);
        wiggle(Dir::Down);
        wiggle(Dir::Up);
        wiggle(Dir::Forward);
        wiggle(Dir::Back);

        let vcycle = |dir| {
            // Five steps because an edge in the dual honeycomb has
            // five cubes around itself, not four as in Euclidean space.
            let looped = start
                .step(&graph, dir)
                .expect("positive")
                .step(&graph, Dir::Down)
                .expect("down")
                .step(&graph, -dir)
                .expect("negative")
                .step(&graph, Dir::Up)
                .expect("up")
                .step(&graph, dir)
                .expect("positive");
            assert_eq!(
                looped.canonicalize(&graph).unwrap(),
                ChunkId::new(NodeId::ROOT, Vertex::A),
            );
        };
        vcycle(Dir::Left);
        vcycle(Dir::Right);
        vcycle(Dir::Forward);
        vcycle(Dir::Back);
    }

    #[test]
    fn count_chunks_sharing_vertices() {
        let mut graph = Graph::new(1);

        let start = ChunkId::new(NodeId::ROOT, Vertex::A);
        let start = graph.canonicalize(start).unwrap();
        let start_vertex_nodes = cube_vertex_nodes_ensuring(&mut graph, start);

        let chunks =
            enumerate_chunks_sharing_any_vertex_with(&mut graph, start, &start_vertex_nodes);

        // Exclude `start` itself from the count.
        let count = chunks.len().saturating_sub(1);
        eprintln!("chunks sharing at least one vertex with {start:?}: {count}");

        // Informational: always pass.
        assert!(count > 0);
    }
}
