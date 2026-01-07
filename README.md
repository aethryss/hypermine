# Hyperbolicraft

A voxel sandbox game exploring hyperbolic geometry, built in Rust with Vulkan rendering and networked multiplayer support.

Instead of infinite Euclidean grids, Hyperbolicraft uses the **order-4 dodecahedral honeycomb**:
- Each dodecahedron (node) contains 20 chunks (one per vertex)
- Chunks are connected via edges and faces to create a seamless tiling
- Coordinate systems automatically handle wrapping and reflection across boundaries

## Building

### Requirements
- Rust (stable, 2024 edition)
- Vulkan SDK
- `pkg-config`, `zstd`, `protobuf` (for protocol buffer compilation)

### Quick Start
```bash
cargo build --release
cargo run --bin client
```

For server-only builds:
```bash
cargo run --bin server
```
