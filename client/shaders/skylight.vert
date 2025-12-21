#version 460

#include "common.h"
#include "surface-extraction/surface.h"

// Instance transform from cube space ([0..1]^3) to local node space
layout(location = 0) in mat4 transform;

layout(set = 1, binding = 0) readonly restrict buffer Surfaces {
    Surface surfaces[];
};

layout(push_constant) uniform PushConstants {
    uint dimension;
};

// Same vertex layout as the main voxel pass
const uvec3 vertices[12][6] = {
    {{0, 0, 0}, {0, 0, 1}, {0, 1, 1}, {0, 1, 1}, {0, 1, 0}, {0, 0, 0}}, // -X
    {{0, 0, 0}, {1, 0, 0}, {1, 0, 1}, {1, 0, 1}, {0, 0, 1}, {0, 0, 0}}, // -Y
    {{0, 0, 0}, {0, 1, 0}, {1, 1, 0}, {1, 1, 0}, {1, 0, 0}, {0, 0, 0}}, // -Z

    {{0, 0, 0}, {0, 1, 0}, {0, 1, 1}, {0, 1, 1}, {0, 0, 1}, {0, 0, 0}}, // +X
    {{0, 0, 0}, {0, 0, 1}, {1, 0, 1}, {1, 0, 1}, {1, 0, 0}, {0, 0, 0}}, // +Y
    {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 0}}, // +Z

    // Flipped diagonal variants (kept for matching the indirect buffers)
    {{0, 0, 1}, {0, 1, 1}, {0, 1, 0}, {0, 1, 0}, {0, 0, 0}, {0, 0, 1}}, // -X
    {{1, 0, 0}, {1, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 0}, {1, 0, 0}}, // -Y
    {{0, 1, 0}, {1, 1, 0}, {1, 0, 0}, {1, 0, 0}, {0, 0, 0}, {0, 1, 0}}, // -Z

    {{0, 0, 1}, {0, 0, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 1}, {0, 0, 1}}, // +X
    {{1, 0, 0}, {0, 0, 0}, {0, 0, 1}, {0, 0, 1}, {1, 0, 1}, {1, 0, 0}}, // +Y
    {{0, 1, 0}, {0, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}}  // +Z
};

void main() {
    uint index = gl_VertexIndex / 6;
    uint vertex = gl_VertexIndex % 6;
    Surface s = surfaces[index];
    uvec3 pos = get_pos(s);
    uint axis = get_axis(s);

    vec3 relative_coords = vertices[axis][vertex] + pos;
    vec4 fermi = skylight_view_projection * transform * vec4(relative_coords / dimension, 1.0);
    if (fermi.w < 0.0) {
        fermi *= -1.0;
    }
    float invw = 1.0 / max(fermi.w, 1.0e-6);
    float ndc_x = (fermi.y * invw) / skylight_bounds.x;
    float ndc_y = (fermi.z * invw) / skylight_bounds.y;
    float ndc_z = 0.5 * (fermi.x * invw + 1.0);
    gl_Position = vec4(ndc_x, ndc_y, ndc_z, 1.0);
}
