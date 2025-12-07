#version 450

layout(location = 0) in vec3 texcoords;
layout(location = 1) in float occlusion;
layout(location = 2) in flat uint texture_index;
layout(location = 0) out vec4 color;

layout(set = 1, binding = 1) uniform sampler2D terrain;

void main() {
    // texcoords.xy are normalized [0,1] face coordinates
    // texture_index is 0-255, mapping to 16×16 grid in terrain.png
    
    // Calculate which tile in the 16×16 grid (each tile is 16 pixels in 256×256 atlas)
    uint tile_x = texture_index % 16u;
    uint tile_y = texture_index / 16u;
    
    // Each tile is 16 pixels in a 256-pixel atlas
    // Convert pixel coordinates to normalized [0,1] coordinates
    float pixel_u = float(tile_x) * 16.0;
    float pixel_v = float(tile_y) * 16.0;
    
    // Map texcoords [0,1] to pixel offsets within the tile [0, 15.99]
    // Use 15.99 to prevent sampling just outside the tile edge
    float u = (pixel_u + texcoords.x * 15.99) / 256.0;
    float v = (pixel_v + texcoords.y * 15.99) / 256.0;
    
    color = texture(terrain, vec2(u, v)) * occlusion;
}
