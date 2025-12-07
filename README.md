# Hyperbolicraft

A voxel sandbox game exploring hyperbolic geometry, built in Rust with Vulkan rendering and networked multiplayer support.

## Overview

Hyperbolicraft is an ambitious experiment in game design and mathematics, bringing the alien landscape of **hyperbolic space** to life as a playable voxel world. Instead of Euclidean (flat) geometry, the world exists on a hyperbolic plane where the sum of angles in a triangle is less than 180°, creating an endlessly expanding and visually disorienting environment.

The game features:

- **Hyperbolic Geometry**: The fundamental world structure is based on a **right dodecahedral honeycomb** tiling of hyperbolic 3-space, providing a mathematically elegant and infinite universe
- **Voxel-based Terrain**: Minecraft-like blocky environments procedurally generated across the hyperbolic manifold
- **Multiplayer Networking**: Server-client architecture using QUIC for low-latency multiplayer gameplay
- **GPU-Accelerated Rendering**: Vulkan-based graphics with advanced features including:
  - Real-time voxel surface extraction via compute shaders
  - Ambient occlusion and fog effects
  - Efficient chunk rendering across the curved space
- **Physics & Collision**: Character movement with spherecasting-based collision detection adapted for hyperbolic geometry
- **Persistent Worlds**: Save/load system with world persistence

## Architecture

Hyperbolicraft is structured as a Rust workspace with four main crates:

### `common`
Shared mathematical primitives and world simulation logic:
- **Minkowski space mathematics** for hyperbolic geometry operations (4D vectors and matrices)
- **Dodecahedral node system**: 12-sided building blocks that tile hyperbolic space
- **Chunk management**: 20 voxel chunks per node (one per dodecahedral vertex)
- **Collision math**: Ray casting and sphere casting in hyperbolic space
- **Physics**: Character controller with gravity and movement constraints

### `client`
The game client with graphics rendering and user interaction:
- **Graphics subsystem** (Vulkan via `ash` library):
  - Window management and rendering loop
  - Pipeline caching and shader compilation
  - Texture loading and atlasing
  - GUI rendering via `yakui`
- **Voxel rendering**:
  - GPU-based surface extraction from sparse voxel data
  - Mesh generation with ambient occlusion
  - Frustum culling for performance
- **Input handling**: Keyboard/mouse controls and character movement
- **Networking**: Connection to game servers

### `server`
Authoritative game server:
- **Simulation**: Runs the physics and world simulation
- **Networking**: Handles client connections, input processing, and state synchronization
- **Entity/Component system**: Uses `hecs` ECS for managing game entities
- **Save management**: Persists world state to disk

### `save`
World persistence layer:
- **Protocol buffer** definitions for serialization
- **Save file format**: Stores chunk data, entity state, and world metadata

## Technical Highlights

### Hyperbolic Geometry
The project implements sophisticated hyperbolic space mathematics:
- **Minkowski inner product** for proper distance and angle calculations
- **Reflection matrices** for coordinate transformations between adjacent nodes
- **Horospheres and ideal points** for rendering infinity and spatial optimization
- See [docs/README.md](docs/README.md) for comprehensive mathematical background

### Dodecahedral Tiling
Instead of infinite Euclidean grids, Hyperbolicraft uses the **order-4 dodecahedral honeycomb**:
- Each dodecahedron (node) contains 20 chunks (one per vertex)
- Chunks are connected via edges and faces to create a seamless tiling
- Coordinate systems automatically handle wrapping and reflection across boundaries

### GPU-Accelerated Surface Extraction
Voxel data is stored sparsely and converted to triangle meshes on the GPU:
- Compute shaders process voxel data in parallel
- Ambient occlusion is computed during extraction
- Indirect rendering commands avoid CPU-GPU synchronization

### Networking
QUIC-based multiplayer with ordered and unordered message channels:
- Uses `quinn` library for reliable, fast transport
- Separate channels for critical (ordered) and non-critical (unordered) updates
- Predictive client-side movement with server reconciliation

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

### Development with Nix
A `shell.nix` is provided for reproducible development:
```bash
nix-shell
cargo run --bin client
```

## Usage

### Single Player
Run the client and create a new world through the UI.

### Multiplayer
1. Start the server: `cargo run --bin server`
2. Connect clients to `localhost` (or configure the server address in config)
3. Multiple players can explore and build in the same world

### Configuration
Configuration is stored in the project directory under `hyperbolicraft/`:
- `config.toml`: Client settings (rendering, input, server address)
- Save data is stored per-world with automatic persistence

## Project Structure

```
hyperbolicraft/
├── common/          # Shared crates (math, physics, world)
├── client/          # Game client (graphics, UI, input)
├── server/          # Game server (simulation, networking)
├── save/            # World persistence layer
├── docs/            # Comprehensive documentation
├── client/shaders/  # GLSL compute and render shaders
└── assets/          # Game materials and textures
```

## Documentation

Extensive documentation is available in [docs/README.md](docs/README.md) covering:
- **Background math**: Linear algebra, spherical geometry, hyperbolic geometry
- **World architecture**: Nodes, chunks, dodecahedral tiling
- **World generation**: Terrain, biomes, procedural features
- **Character physics**: Movement, collision detection
- **Networking**: Synchronization and protocol design
- **Rendering**: Vulkan pipeline, voxel extraction, GUI

## Contributing

Contributions are welcome! Please ensure:
- Code follows Rust idioms and passes `cargo clippy`
- Changes preserve backward compatibility where feasible
- Documentation is updated for new features

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- Zlib license ([LICENSE-ZLIB](LICENSE-ZLIB) or https://opensource.org/licenses/Zlib)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
