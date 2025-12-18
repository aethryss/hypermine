#version 450

#include "common.h"

layout(location = 0) in vec2 texcoords;

layout(location = 0) out vec4 fog;

layout(input_attachment_index=0, set=0, binding=1) uniform subpassInput depth;

// Sky color constant
const vec3 SKY_COLOR = vec3(0.5, 0.65, 0.9);

void main() {
    vec4 clip_pos = vec4(texcoords * 2.0 - 1.0, subpassLoad(depth).x, 1.0);
    vec4 scaled_view_pos = inverse_projection * clip_pos;
    // Cancel out perspective, obtaining klein ball position
    vec3 view_pos = scaled_view_pos.xyz / scaled_view_pos.w;
    float view_length = length(view_pos);
    // Convert to true hyperbolic distance, taking care to respect atanh's domain
    float dist = view_length >= 1.0 ? INFINITY : atanh(view_length);
    
    // Exponential^k fog: visibility decreases with distance
    // visibility = 1.0 at distance 0 (scene fully visible)
    // visibility → 0.0 as distance increases (scene fades to sky)
    float visibility = exp(-pow(dist * fog_density, 5));
    
    // Fog alpha = how much to blend toward sky
    // Near: fog_alpha = 0 (keep scene)
    // Far: fog_alpha = 1 (show sky)
    float fog_alpha = 1.0 - visibility;
    
    fog = vec4(SKY_COLOR, fog_alpha);
}
