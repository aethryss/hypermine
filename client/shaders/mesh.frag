#version 450

layout(location = 0) in flat uvec2 material_info_in;
layout(location = 1) in vec4 normal;

layout(location = 0) out vec4 color_out;

layout(set = 1, binding = 0) uniform sampler2D color;

void main() {
    uint texture_id = material_info_in.x;
    uint corner_id = material_info_in.y;

    // Atlas properties, mimicking remcpe
    float atlas_size = 256.0;
    float tile_size_pixels = 16.0;
    float tiles_per_row = atlas_size / tile_size_pixels;

    // Calculate the column and row of the tile in the atlas
    float tile_col = float(texture_id % uint(tiles_per_row));
    float tile_row = float(texture_id / uint(tiles_per_row));

    // Calculate the UV offset for the top-left corner of the tile
    vec2 tile_offset = vec2(tile_col * tile_size_pixels, tile_row * tile_size_pixels) / atlas_size;

    // Determine the corner offset based on corner_id
    // 0 = top-left, 1 = top-right, 2 = bottom-left, 3 = bottom-right
    // This needs to match how the CPU builds the mesh.
    // We also use a small inset (0.01) to prevent texture bleeding, just like remcpe.
    float tile_uv_size = (tile_size_pixels - 0.01) / atlas_size;
    vec2 corner_offset;
    if (corner_id == 0) { // Top-left
        corner_offset = vec2(0.0, 0.0);
    } else if (corner_id == 1) { // Top-right
        corner_offset = vec2(tile_uv_size, 0.0);
    } else if (corner_id == 2) { // Bottom-left
        corner_offset = vec2(0.0, tile_uv_size);
    } else { // Bottom-right
        corner_offset = vec2(tile_uv_size, tile_uv_size);
    }

    vec2 final_uv = tile_offset + corner_offset;

    color_out = texture(color, final_uv);
}
