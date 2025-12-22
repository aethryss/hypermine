#version 450

#include "common.h"

layout(location = 0) in vec3 texcoords;
layout(location = 1) in float occlusion;
layout(location = 2) in flat uint texture_index;
layout(location = 3) in vec3 light_color;
layout(location = 4) in flat vec4 skylight_shadow_clip;
layout(location = 0) out vec4 color;

layout(set = 1, binding = 1) uniform sampler2D terrain;
layout(set = 1, binding = 2) uniform sampler2D skylight_shadow;

// Specialization constants for transparency handling
// enable_alpha_test: if true, discard pixels with alpha < alpha_cutoff
// alpha_cutoff: threshold for discarding (default 0.5)
layout(constant_id = 0) const bool enable_alpha_test = false;
layout(constant_id = 1) const float alpha_cutoff = 0.5;


// Minimum ambient light color (prevents completely dark areas).
//
// Note: the render targets and textures are sRGB, so shader math happens in *linear* space.
// If you want the on-screen (sRGB) “no light” multiplier to look like `#2d2c2f`, you must
// use the linear-space equivalent here.
//
// #2d2c2f (sRGB 8-bit) -> linear (IEC 61966-2-1):
//   r=45 -> 0.026241222
//   g=44 -> 0.025186860
//   b=47 -> 0.028426040
const vec3 AMBIENT_LIGHT_COLOR = vec3(0.026241222, 0.025186860, 0.028426040);

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
    
    vec4 texColor = texture(terrain, vec2(u, v));
    
    // Alpha test for cutout rendering
    // Only enabled for cutout pipeline (and opaque as a safety)
    // Translucent pipeline has enable_alpha_test = false
    if (enable_alpha_test && texColor.a < alpha_cutoff) {
        discard;
    }

    // Skylight shadowing (shadow-map style).
    // The shadow map is rendered in a Fermi-orthographic projection aligned with the local horizon plane.
    vec3 shadow_ndc = skylight_shadow_clip.xyz / skylight_shadow_clip.w;
    vec2 shadow_uv = shadow_ndc.xy * 0.5 + 0.5;
    float shadow_depth = 1.0;
    bool shadow_valid = all(greaterThanEqual(shadow_uv, vec2(0.0))) && all(lessThanEqual(shadow_uv, vec2(1.0)));
    shadow_valid = shadow_valid && (shadow_ndc.z >= 0.0) && (shadow_ndc.z <= 1.0);
    if (shadow_valid) {
        // Hard "shrink" of the shadow silhouette:
        // take the maximum depth in a tiny neighborhood and do a hard compare.
        //
        // Intuition: along a shadow edge, adjacent texels may alternate between
        // "occluder" (smaller depth) and "lit" (larger depth). Using max depth
        // biases toward the lit decision, effectively eroding the shadow by ~1 texel
        // without blending/softening the edge.
        // Resolution-aware: derive texel coordinates from the actual shadow-map extent.
        ivec2 size_i = textureSize(skylight_shadow, 0);
        vec2 size = vec2(size_i);

        // Convert UV to a base texel coordinate (clamped).
        ivec2 tc = ivec2(floor(shadow_uv * size));
        tc = clamp(tc, ivec2(0), size_i - ivec2(1));

        ivec2 tc_px = min(tc + ivec2(1, 0), size_i - ivec2(1));
        ivec2 tc_nx = max(tc + ivec2(-1, 0), ivec2(0));
        ivec2 tc_py = min(tc + ivec2(0, 1), size_i - ivec2(1));
        ivec2 tc_ny = max(tc + ivec2(0, -1), ivec2(0));

        float d0 = texelFetch(skylight_shadow, tc, 0).r;
        float d1 = texelFetch(skylight_shadow, tc_px, 0).r;
        float d2 = texelFetch(skylight_shadow, tc_nx, 0).r;
        float d3 = texelFetch(skylight_shadow, tc_py, 0).r;
        float d4 = texelFetch(skylight_shadow, tc_ny, 0).r;

        shadow_depth = max(d0, max(max(d1, d2), max(d3, d4)));
    }
    float bias = skylight_params.z;
    bool in_shadow = shadow_valid && ((shadow_ndc.z - bias) > shadow_depth);
    float shadow_factor = in_shadow ? (1.0 - skylight_params.y) : 1.0;
    vec3 skylight = vec3(skylight_params.x) * shadow_factor;
    
    // Calculate effective light: combine block light with ambient
    // Block light color is [0,1] per channel from 4-bit values
    // Add ambient to ensure areas are never completely dark
    vec3 effective_light = max(light_color + skylight, AMBIENT_LIGHT_COLOR);
    
    // Apply lighting and occlusion
    // effective_light provides the light color/intensity
    // occlusion provides corner darkening for ambient occlusion effect
    vec3 lit_color = texColor.rgb * effective_light * occlusion;
    
    // The blend equation for translucent uses SRC_ALPHA to blend colors,
    // so we output texColor.a as the alpha (which controls color blending)
    // but then the alpha blend overwrites with 1.0 for the fog pass
    // 
    // For all pipelines, output texColor.a - the blend state handles the rest:
    // - Opaque/Cutout: blend disabled, just writes RGBA (alpha will be 1.0 for opaque textures)
    // - Translucent: blends RGB using SRC_ALPHA, overwrites A with 1.0
    color = vec4(lit_color, texColor.a);
}
