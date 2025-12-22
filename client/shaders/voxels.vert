#version 460

#include "common.h"
#include "surface-extraction/surface.h"

// Maps from cube space ([0..1]^3) to local node space
layout(location = 0) in mat4 transform;

layout(location = 0) out vec3 texcoords_out;
layout(location = 1) out float occlusion;
layout(location = 2) out flat uint texture_index_out;
layout(location = 3) out vec3 light_color;
layout(location = 4) out flat float skylight_factor;
layout(location = 5) out flat float skylight_soft_uv_radius;

layout(set = 1, binding = 0) readonly restrict buffer Surfaces {
    Surface surfaces[];
};

layout(set = 1, binding = 2) uniform sampler2D skylight_shadow;

layout(push_constant) uniform PushConstants {
    uint dimension;
};

const int SOFT_SHADOW_BLOCKS = 4;

// Computes the eroded hard-shadow decision for a given shadow NDC position.
// Returns true when the point is considered in shadow.
bool hard_shadow_eroded(vec3 shadow_ndc) {
    vec2 shadow_uv = shadow_ndc.xy * 0.5 + 0.5;
    bool shadow_valid = all(greaterThanEqual(shadow_uv, vec2(0.0))) && all(lessThanEqual(shadow_uv, vec2(1.0)));
    shadow_valid = shadow_valid && (shadow_ndc.z >= 0.0) && (shadow_ndc.z <= 1.0);
    if (!shadow_valid) {
        return false;
    }

    ivec2 size_i = textureSize(skylight_shadow, 0);
    vec2 size = vec2(size_i);

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
    float shadow_depth = max(d0, max(max(d1, d2), max(d3, d4)));

    float bias = skylight_params.z;
    return (shadow_ndc.z - bias) > shadow_depth;
}

// Cheaper, non-eroded shadow test used only for interior softening samples.
// Conservative rule: if outside UV bounds, treat as shadowed to avoid brightening
// due to missing coverage.
bool hard_shadow_sample(vec2 shadow_uv, float ndc_z) {
    bool shadow_valid = all(greaterThanEqual(shadow_uv, vec2(0.0))) && all(lessThanEqual(shadow_uv, vec2(1.0)));
    if (!shadow_valid) {
        return true;
    }
    ivec2 size_i = textureSize(skylight_shadow, 0);
    vec2 size = vec2(size_i);
    ivec2 tc = ivec2(floor(shadow_uv * size));
    tc = clamp(tc, ivec2(0), size_i - ivec2(1));
    float d0 = texelFetch(skylight_shadow, tc, 0).r;
    float bias = skylight_params.z;
    return (ndc_z - bias) > d0;
}

// Project a voxel-space position (in block units) to skylight shadow NDC.
vec3 skylight_ndc(vec3 pos_vox) {
    vec4 fermi = skylight_view_projection * transform * vec4(pos_vox / float(dimension), 1.0);
    if (fermi.w < 0.0) {
        fermi *= -1.0;
    }
    float invw = 1.0 / max(fermi.w, 1.0e-6);
    float ndc_x = (fermi.y * invw) / skylight_bounds.x;
    float ndc_y = (fermi.z * invw) / skylight_bounds.y;
    float ndc_z = 0.5 * (fermi.x * invw + 1.0);
    return vec3(ndc_x, ndc_y, ndc_z);
}

// Each set of 6 vertices makes a ring around the quad, with the middle and start/end vertices
// duplicated. Note that the sign only indicates the winding of the face; all faces contain the
// origin regardless.
const uvec3 vertices[12][6] = {
    {{0, 0, 0}, {0, 0, 1}, {0, 1, 1}, {0, 1, 1}, {0, 1, 0}, {0, 0, 0}}, // -X
    {{0, 0, 0}, {1, 0, 0}, {1, 0, 1}, {1, 0, 1}, {0, 0, 1}, {0, 0, 0}}, // -Y
    {{0, 0, 0}, {0, 1, 0}, {1, 1, 0}, {1, 1, 0}, {1, 0, 0}, {0, 0, 0}}, // -Z

    {{0, 0, 0}, {0, 1, 0}, {0, 1, 1}, {0, 1, 1}, {0, 0, 1}, {0, 0, 0}}, // +X
    {{0, 0, 0}, {0, 0, 1}, {1, 0, 1}, {1, 0, 1}, {1, 0, 0}, {0, 0, 0}}, // +Y
    {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 0}}, // +Z

    // Versions of the above rotated 90 degrees so the diagonal goes the other way, used to improve
    // the consistency of barycentric interpolation of ambient occlusion
    {{0, 0, 1}, {0, 1, 1}, {0, 1, 0}, {0, 1, 0}, {0, 0, 0}, {0, 0, 1}}, // -X
    {{1, 0, 0}, {1, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 0}, {1, 0, 0}}, // -Y
    {{0, 1, 0}, {1, 1, 0}, {1, 0, 0}, {1, 0, 0}, {0, 0, 0}, {0, 1, 0}}, // -Z

    {{0, 0, 1}, {0, 0, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 1}, {0, 0, 1}}, // +X
    {{1, 0, 0}, {0, 0, 0}, {0, 0, 1}, {0, 0, 1}, {1, 0, 1}, {1, 0, 0}}, // +Y
    {{0, 1, 0}, {0, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}}  // +Z
};

const uvec2 texcoords[4][6] = {
    {{0, 0}, {0, 1}, {1, 1}, {1, 1}, {1, 0}, {0, 0}},
    {{0, 0}, {1, 0}, {1, 1}, {1, 1}, {0, 1}, {0, 0}},
    // Rotated versions
    {{0, 1}, {1, 1}, {1, 0}, {1, 0}, {0, 0}, {0, 1}},
    {{0, 1}, {0, 0}, {1, 0}, {1, 0}, {1, 1}, {0, 1}},
};

void main()  {
    uint index = gl_VertexIndex / 6;
    uint vertex = gl_VertexIndex % 6;
    Surface s = surfaces[index];
    uvec3 pos = get_pos(s);
    uint axis = get_axis(s);
    uvec2 uv = texcoords[axis / 3][vertex];
    texcoords_out = vec3(uv, 0);  // Store UV coordinates, z is unused
    texture_index_out = get_mat(s);  // Pass texture_index directly (0-255)
    occlusion = get_occlusion(s, uv);
    light_color = get_light_color(s);  // Get the block light color
    vec3 relative_coords = vertices[axis][vertex] + pos;

    // Quantize skylight shadowing per face by evaluating at the face center.
    // We compute a per-face skylight multiplier here (flat varying) to avoid per-fragment
    // shadowmap work and to enable interior-only softening in block units.
    //
    // NOTE: surfaces are always emitted on the negative-side boundary for the chosen axis.
    vec3 face_center = vec3(pos) + vec3(0.5);
    uint coord_axis = axis % 3u;
    if (coord_axis == 0u) {
        face_center.x = float(pos.x);
    } else if (coord_axis == 1u) {
        face_center.y = float(pos.y);
    } else {
        face_center.z = float(pos.z);
    }

    vec3 center_ndc = skylight_ndc(face_center);
    bool center_shadowed = hard_shadow_eroded(center_ndc);

    if (!center_shadowed) {
        // Keep non-shadowed area unchanged.
        skylight_factor = 1.0;
    } else {
        // Interior-only soft shadowing: compute a small 2D convolution of the hard shadow
        // mask in block units. This improves corner/diagonal appearance vs sampling only
        // along cardinal directions.
        //
        // Important: this never affects lit faces (since we gate on center_shadowed), so it
        // cannot bleed outward beyond the existing (eroded) shadow region.
        vec2 uv_center = center_ndc.xy * 0.5 + 0.5;
        float ndc_z = center_ndc.z;

        // Compute per-block UV steps along the two face tangent axes.
        // Use voxel-space directions (1 block) and project them; then reuse ndc_z for
        // the interior softening samples (ground-aligned faces stay stable).
        vec3 du_vox = vec3(0.0);
        vec3 dv_vox = vec3(0.0);
        if (coord_axis == 0u) {
            du_vox.y = 1.0; dv_vox.z = 1.0;
        } else if (coord_axis == 1u) {
            du_vox.z = 1.0; dv_vox.x = 1.0;
        } else {
            du_vox.x = 1.0; dv_vox.y = 1.0;
        }

        vec3 ndc_u = skylight_ndc(face_center + du_vox);
        vec3 ndc_v = skylight_ndc(face_center + dv_vox);
        vec2 uv_step_u = (ndc_u.xy * 0.5 + 0.5) - uv_center;
        vec2 uv_step_v = (ndc_v.xy * 0.5 + 0.5) - uv_center;

        // Gaussian kernel parameters.
        // sigma=2 gives a nice falloff for a radius of 4 blocks.
        const float sigma = 2.0;
        const float inv_two_sigma2 = 1.0 / (2.0 * sigma * sigma);

        float sum_w = 0.0;
        float sum_shadow = 0.0;

        // Sample a square kernel in the face plane (block units), weighted by a Gaussian.
        // We use the cheaper non-eroded test for these samples; the eroded hard boundary
        // is preserved by the center_shadowed gate above.
        for (int y = -SOFT_SHADOW_BLOCKS; y <= SOFT_SHADOW_BLOCKS; ++y) {
            for (int x = -SOFT_SHADOW_BLOCKS; x <= SOFT_SHADOW_BLOCKS; ++x) {
                float d2 = float(x * x + y * y);
                float w = exp(-d2 * inv_two_sigma2);
                vec2 o = float(x) * uv_step_u + float(y) * uv_step_v;
                bool sh = hard_shadow_sample(uv_center + o, ndc_z);
                sum_shadow += w * float(sh);
                sum_w += w;
            }
        }

        float coverage = sum_shadow / max(sum_w, 1.0e-6);

        // Map coverage -> skylight multiplier.
        // - coverage=1 => full shadowing: (1 - shadow_strength)
        // - coverage->0 => near boundary: approaches 1 (but only inside shadow)
        float strength = skylight_params.y;
        float base = 1.0 - strength;
        // Slight easing to keep the edge softer without washing out deep shadows.
        float eased = coverage * coverage;
        skylight_factor = mix(1.0, base, eased);
    }

    // Estimate how large "skylight_params.w blocks" are in shadow-map UV space for this face.
    // We do this per-face (flat) to avoid per-fragment instability.
    float soft_blocks = skylight_params.w;
    if (soft_blocks <= 0.0) {
        skylight_soft_uv_radius = 0.0;
    } else {
        // Pick two tangent axes to the face.
        uint n = coord_axis;
        uint a0 = (n + 1u) % 3u;
        uint a1 = (n + 2u) % 3u;

        vec3 p0 = face_center;
        vec3 p_u = face_center;
        vec3 p_v = face_center;
        if (a0 == 0u) p_u.x += soft_blocks; else if (a0 == 1u) p_u.y += soft_blocks; else p_u.z += soft_blocks;
        if (a1 == 0u) p_v.x += soft_blocks; else if (a1 == 1u) p_v.y += soft_blocks; else p_v.z += soft_blocks;

        // Helper macro: project a point and return shadow-map UV.
        #define SHADOW_UV(pt, out_uv) do { \
            vec4 f = skylight_view_projection * transform * vec4((pt) / dimension, 1.0); \
            if (f.w < 0.0) { f *= -1.0; } \
            float iw = 1.0 / max(f.w, 1.0e-6); \
            float x = (f.y * iw) / skylight_bounds.x; \
            float y = (f.z * iw) / skylight_bounds.y; \
            (out_uv) = vec2(x, y) * 0.5 + 0.5; \
        } while(false)

        vec2 uv0;
        vec2 uvu;
        vec2 uvv;
        SHADOW_UV(p0, uv0);
        SHADOW_UV(p_u, uvu);
        SHADOW_UV(p_v, uvv);
        #undef SHADOW_UV

        float ru = length(uvu - uv0);
        float rv = length(uvv - uv0);
        skylight_soft_uv_radius = max(ru, rv);
    }

    gl_Position = view_projection * transform * vec4(relative_coords / dimension, 1);
}
