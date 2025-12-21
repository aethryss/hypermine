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

float skylight_visibility_pcf(vec2 base_uv, float shadow_z, float bias) {
    // Out-of-range depth should behave like “no shadow map coverage”.
    if (shadow_z < 0.0 || shadow_z > 1.0) {
        return 1.0;
    }

    // If filter radius is ~0, fall back to a single hard compare.
    float radius = max(skylight_params.w, 0.0);
    if (radius < 0.01) {
        float d = texture(skylight_shadow, base_uv).r;
        return ((shadow_z - bias) > d) ? 0.0 : 1.0;
    }

    vec2 texel = 1.0 / vec2(textureSize(skylight_shadow, 0));
    vec2 step_uv = texel * radius;

    // Weighted 3x3 PCF (Gaussian-ish):
    //   1 2 1
    //   2 4 2
    //   1 2 1
    float vis = 0.0;
    float wsum = 0.0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            float w = (dx == 0 && dy == 0) ? 4.0 : ((dx == 0 || dy == 0) ? 2.0 : 1.0);
            vec2 uv = base_uv + vec2(float(dx), float(dy)) * step_uv;

            // Treat samples that fall off the shadow map as “lit”, to avoid dark borders.
            bool uv_ok = all(greaterThanEqual(uv, vec2(0.0))) && all(lessThanEqual(uv, vec2(1.0)));
            float sample_vis = 1.0;
            if (uv_ok) {
                float d = texture(skylight_shadow, uv).r;
                sample_vis = ((shadow_z - bias) > d) ? 0.0 : 1.0;
            }

            vis += w * sample_vis;
            wsum += w;
        }
    }
    return vis / wsum;
}

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
    bool shadow_valid = all(greaterThanEqual(shadow_uv, vec2(0.0))) && all(lessThanEqual(shadow_uv, vec2(1.0)));
    shadow_valid = shadow_valid && (shadow_ndc.z >= 0.0) && (shadow_ndc.z <= 1.0);
    float bias = skylight_params.z;
    float visibility = shadow_valid ? skylight_visibility_pcf(shadow_uv, shadow_ndc.z, bias) : 1.0;
    float shadow_factor = 1.0 - skylight_params.y * (1.0 - visibility);
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
