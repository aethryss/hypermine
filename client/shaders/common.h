#ifndef COMMON_H
#define COMMON_H

const float PI = 3.14159265;
const float INFINITY = 1.0 / 0.0;

layout(set = 0, binding = 0) uniform Common {
    // Maps local node space to clip space
    mat4 view_projection;
    // Maps clip space to view space
    mat4 inverse_projection;
    // Maps view space to world space (camera orientation)
    mat4 inverse_view;
    // World up direction in view space (normalized, xyz only, w unused)
    vec4 world_up;
    // World north direction in view space (from compass, xyz only, w unused)
    vec4 world_north;
    float fog_density;
    float time;
};

#endif
