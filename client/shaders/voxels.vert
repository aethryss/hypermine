#version 460

#include "common.h"
#include "surface-extraction/surface.h"

// Maps from cube space ([0..1]^3) to local node space
layout(location = 0) in mat4 transform;

layout(location = 0) out vec3 texcoords_out;
layout(location = 1) out float occlusion;
layout(location = 2) out flat uint texture_index_out;
layout(location = 3) out vec3 light_color;
layout(location = 4) out flat vec4 skylight_shadow_clip;

layout(set = 1, binding = 0) readonly restrict buffer Surfaces {
    Surface surfaces[];
};

layout(push_constant) uniform PushConstants {
    uint dimension;
};

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
    // This keeps shadows block-sharp and stable.
    vec3 face_center = vec3(pos) + vec3(0.5);
    uint base_axis = axis % 6;
    if (base_axis == 0u) { // -X
        face_center.x = float(pos.x);
    } else if (base_axis == 3u) { // +X
        face_center.x = float(pos.x) + 1.0;
    } else if (base_axis == 1u) { // -Y
        face_center.y = float(pos.y);
    } else if (base_axis == 4u) { // +Y
        face_center.y = float(pos.y) + 1.0;
    } else if (base_axis == 2u) { // -Z
        face_center.z = float(pos.z);
    } else if (base_axis == 5u) { // +Z
        face_center.z = float(pos.z) + 1.0;
    }
    skylight_shadow_clip = skylight_view_projection * transform * vec4(face_center / dimension, 1.0);

    gl_Position = view_projection * transform * vec4(relative_coords / dimension, 1);
}
