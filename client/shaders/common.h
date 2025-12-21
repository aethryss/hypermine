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

    // Skylight shadow-map projection.
    // Transforms local node-space positions (after instance transform) into a Fermi-orthographic
    // clip space where:
    // - xy cover the horizon hyperplane (in Klein coords) in [-1, 1]
    // - z is depth in [0, 1] along the down direction
    mat4 skylight_view_projection;

    // x: skylight intensity (added as white light)
    // y: shadow strength (0=no shadowing, 1=full shadowing)
    // z: depth bias
    // w: unused
    vec4 skylight_params;

    // Skylight projection bounds in Klein coordinates.
    // x: max_u for (y/w)
    // y: max_v for (z/w)
    // z,w: unused
    vec4 skylight_bounds;
    // World up direction in view space (normalized, xyz only, w unused)
    vec4 world_up;
    // World north direction in view space (from compass, xyz only, w unused)
    vec4 world_north;
    float fog_density;
    float time;
};

#endif
