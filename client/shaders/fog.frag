#version 450

#include "common.h"

layout(location = 0) in vec2 texcoords;

layout(location = 0) out vec4 fog;

layout(input_attachment_index=0, set=0, binding=1) uniform subpassInput depth;

// ============================================================================
// Procedural Sky with Hyperbolic Angular Distortion
// ============================================================================

// Hash functions for procedural noise
float hash21(vec2 p) {
    float h = dot(p, vec2(127.1, 311.7));
    return fract(sin(h) * 43758.5453123);
}

float hash31(vec3 p) {
    float h = dot(p, vec3(127.1, 311.7, 74.7));
    return fract(sin(h) * 43758.5453123);
}

// Smooth noise for clouds
float noise3(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f); // smoothstep
    
    float n000 = hash31(i + vec3(0, 0, 0));
    float n100 = hash31(i + vec3(1, 0, 0));
    float n010 = hash31(i + vec3(0, 1, 0));
    float n110 = hash31(i + vec3(1, 1, 0));
    float n001 = hash31(i + vec3(0, 0, 1));
    float n101 = hash31(i + vec3(1, 0, 1));
    float n011 = hash31(i + vec3(0, 1, 1));
    float n111 = hash31(i + vec3(1, 1, 1));
    
    float nx00 = mix(n000, n100, f.x);
    float nx10 = mix(n010, n110, f.x);
    float nx01 = mix(n001, n101, f.x);
    float nx11 = mix(n011, n111, f.x);
    
    float nxy0 = mix(nx00, nx10, f.y);
    float nxy1 = mix(nx01, nx11, f.y);
    
    return mix(nxy0, nxy1, f.z);
}

// Fractal Brownian motion for cloud detail
float fbm(vec3 p) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    
    for (int i = 0; i < 5; i++) {
        value += amplitude * noise3(p * frequency);
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    return value;
}

bool cloudCellExists(ivec2 cell) {
    // Deterministic Minecraft-like cloud footprint: a coarse cluster mask and a local mask.
    // Using integer cell coordinates ensures the top/bottom faces share the exact same pattern.
    float localHash = hash21(vec2(cell));
    ivec2 clusterCell = cell / 6;
    float clusterHash = hash21(vec2(clusterCell) + vec2(42.0, 17.0));
    return (clusterHash > 0.55) && (localHash > 0.35);
}

vec3 cloudFaceColor(
    vec3 faceNormal,
    vec3 worldUp,
    vec3 east,
    vec3 north,
    vec3 sunDir,
    vec3 zenithColor,
    vec3 horizonColor,
    vec3 nadirColor
) {
    float upDot = dot(faceNormal, worldUp);
    if (upDot > 0.9) {
        // Top: brightest
        return vec3(1.0, 1.0, 1.0);
    }
    if (upDot < -0.9) {
        // Bottom: darkest
        return vec3(0.70, 0.72, 0.75);
    }

    // Sides: shaded by sky (gradient + a bit of sun), similar to block ambient light.
    float h = clamp(dot(faceNormal, worldUp), -1.0, 1.0);
    vec3 skyAmb;
    if (h >= 0.0) {
        float t = sqrt(h);
        skyAmb = mix(horizonColor, zenithColor, t);
    } else {
        float t = sqrt(-h);
        skyAmb = mix(horizonColor, nadirColor, t);
    }

    // Make side faces strongly directional: a bright sun-facing side and a noticeably darker opposite side.
    float sunLambert = max(0.0, dot(faceNormal, sunDir));
    float sunSpec = pow(sunLambert, 48.0);

    // Convert sky ambient color to a scalar intensity to keep clouds mostly white.
    float skyLum = dot(skyAmb, vec3(0.2126, 0.7152, 0.0722));
    float ambientI = 0.30 + 0.40 * skyLum;

    vec3 base = vec3(0.92, 0.94, 0.97);
    vec3 sunColor = vec3(1.0, 0.95, 0.8);

    vec3 col = base * ambientI + sunColor * (0.75 * sunLambert + 0.45 * sunSpec);
    return clamp(col, 0.0, 1.0);
}

// Apply hyperbolic angular distortion to a direction
// In hyperbolic space, directions near the horizon are "compressed"
// compared to Euclidean projection. This warps the sky to feel more
// expansive overhead and compressed at the edges.
vec3 hyperbolicWarpDirection(vec3 dir, vec3 up) {
    // Compute angle from zenith (0 = straight up, PI = straight down)
    float cosUp = dot(dir, up);
    float angle = acos(clamp(cosUp, -1.0, 1.0));
    
    // In hyperbolic geometry, the "visual" angle grows faster near horizon
    // We model this by applying a non-linear warp: pushing angles away from
    // zenith, making the sky appear more "domed" overhead.
    // The warp function: newAngle = angle + k * sin(angle) 
    // This compresses angles near horizon (angle ≈ PI/2) more than near zenith
    float k = 0.3; // Strength of hyperbolic distortion
    float warpedAngle = angle + k * sin(angle);
    warpedAngle = clamp(warpedAngle, 0.0, PI);
    
    // Reconstruct direction with warped angle
    // Find the horizontal component of the direction
    vec3 horizontal = dir - up * cosUp;
    float horizLen = length(horizontal);
    if (horizLen < 1e-6) {
        // Direction is nearly vertical, no warp needed
        return dir;
    }
    vec3 horizNorm = horizontal / horizLen;
    
    // New direction from warped angle
    return up * cos(warpedAngle) + horizNorm * sin(warpedAngle);
}

// Compute procedural sky color for a given view-space direction
// worldUp and worldNorth are stable reference directions in view space
// worldNorth comes from a "compass" that experiences holonomy naturally but doesn't rotate with camera
vec3 proceduralSky(vec3 worldDir, vec3 worldUp, vec3 worldNorth) {
    // Apply hyperbolic angular distortion
    vec3 dir = hyperbolicWarpDirection(worldDir, worldUp);
    
    // Height above horizon: 1 = zenith, 0 = horizon, -1 = nadir
    float height = dot(dir, worldUp);
    
    // Build horizontal coordinate frame from up and north
    // Project north onto horizontal plane to ensure it's perpendicular to up
    vec3 north = normalize(worldNorth - worldUp * dot(worldNorth, worldUp));
    vec3 east = cross(worldUp, north);
    
    // === Sky gradient ===
    // Zenith: deeper blue; Horizon: lighter blue/white
    vec3 zenithColor = vec3(0.25, 0.45, 0.85);
    vec3 horizonColor = vec3(0.7, 0.8, 0.95);
    vec3 nadirColor = vec3(0.3, 0.35, 0.45); // Dark below horizon
    
    vec3 skyColor;
    if (height >= 0.0) {
        // Above horizon: blend zenith to horizon
        float t = sqrt(height); // Non-linear for more horizon color
        skyColor = mix(horizonColor, zenithColor, t);
    } else {
        // Below horizon: darker
        float t = sqrt(-height);
        skyColor = mix(horizonColor, nadirColor, t);
    }
    
    // === Sun ===
    // Sun direction: fixed in world space, moving slowly with time
    float sunAzimuth = time * 0.1 * PI * 2.0; // Complete cycle every 10 seconds (for testing)
    float sunElevation = 0.4; // About 25 degrees above horizon
    // Use pre-computed east/north from stable world reference
    vec3 sunDir = normalize(
        worldUp * sin(sunElevation) + 
        (cos(sunAzimuth) * east + sin(sunAzimuth) * north) * cos(sunElevation)
    );
    
    float sunDot = dot(dir, sunDir);
    float sunIntensity = pow(max(0.0, sunDot), 256.0); // Sharp sun disc
    float sunGlow = pow(max(0.0, sunDot), 8.0) * 0.3;   // Soft glow
    vec3 sunColor = vec3(1.0, 0.95, 0.8);
    skyColor += sunColor * (sunIntensity + sunGlow);

    // === Minecraft-style 3D Pixelated Clouds ===
    // Render clouds as an extruded voxel grid (a slab with actual side faces).
    float cloudTopHeight = 128.0;   // Height above the viewer, along worldUp
    float cloudThickness = 8.0;
    float cloudBottomHeight = cloudTopHeight - cloudThickness;

    float cloudScale = 0.02;        // Smaller = larger clouds
    float pixelSize = 1.0;          // In cloud-UV space
    vec2 cloudDrift = vec2(time * 0.05, 0.0);

    bool hitCloud = false;
    float tHit = 0.0;
    vec3 hitNormal = vec3(0.0);

    // Only render when looking upward into the cloud layer.
    float dy = height;
    if (dy > 0.001) {
        float tEnter = cloudBottomHeight / dy;
        float tExit = cloudTopHeight / dy;

        if (tExit > 0.0) {
            tEnter = max(tEnter, 0.0);

            vec2 dirUV = vec2(dot(dir, east), dot(dir, north)) * cloudScale;
            vec2 uv = dirUV * tEnter + cloudDrift;
            ivec2 cell = ivec2(floor(uv / pixelSize));

            // If we enter the slab inside a filled cell, we hit the bottom face.
            if (cloudCellExists(cell)) {
                hitCloud = true;
                tHit = tEnter;
                hitNormal = -worldUp;
            } else {
                // 2D DDA through the cloud grid while we are inside the slab.
                ivec2 step = ivec2(sign(dirUV));
                float t = tEnter;

                float tMaxX;
                float tMaxY;
                float tDeltaX;
                float tDeltaY;

                if (abs(dirUV.x) < 1e-6) {
                    step.x = 0;
                    tMaxX = INFINITY;
                    tDeltaX = INFINITY;
                } else {
                    float cellMin = float(cell.x) * pixelSize;
                    float cellMax = (float(cell.x) + 1.0) * pixelSize;
                    float nextBoundary = (dirUV.x > 0.0) ? cellMax : cellMin;
                    tMaxX = t + (nextBoundary - uv.x) / dirUV.x;
                    tDeltaX = pixelSize / abs(dirUV.x);
                }

                if (abs(dirUV.y) < 1e-6) {
                    step.y = 0;
                    tMaxY = INFINITY;
                    tDeltaY = INFINITY;
                } else {
                    float cellMin = float(cell.y) * pixelSize;
                    float cellMax = (float(cell.y) + 1.0) * pixelSize;
                    float nextBoundary = (dirUV.y > 0.0) ? cellMax : cellMin;
                    tMaxY = t + (nextBoundary - uv.y) / dirUV.y;
                    tDeltaY = pixelSize / abs(dirUV.y);
                }

                const int MAX_STEPS = 96;
                for (int i = 0; i < MAX_STEPS; i++) {
                    // Step to next cell boundary.
                    if (tMaxX < tMaxY) {
                        if (step.x == 0) {
                            break;
                        }
                        t = tMaxX;
                        tMaxX += tDeltaX;
                        cell.x += step.x;
                        hitNormal = -sign(dirUV.x) * east;
                    } else {
                        if (step.y == 0) {
                            break;
                        }
                        t = tMaxY;
                        tMaxY += tDeltaY;
                        cell.y += step.y;
                        hitNormal = -sign(dirUV.y) * north;
                    }

                    if (t > tExit) {
                        break;
                    }

                    if (cloudCellExists(cell)) {
                        hitCloud = true;
                        tHit = t;
                        break;
                    }
                }
            }
        }
    }

    if (hitCloud) {
        vec3 hitPoint = dir * tHit;
        vec3 cloudColor = cloudFaceColor(
            normalize(hitNormal),
            worldUp,
            east,
            north,
            sunDir,
            zenithColor,
            horizonColor,
            nadirColor
        );

        // Distance-based fade.
        float uWorld = dot(hitPoint, east);
        float vWorld = dot(hitPoint, north);
        float cloudDistance = length(vec2(uWorld, vWorld));
        float distanceFade = 1.0 - smoothstep(3000.0, 6000.0, cloudDistance);

        // Fade near horizon (only above horizon for this layer).
        float cloudFade = smoothstep(0.01, 0.15, dy) * distanceFade;
        skyColor = mix(skyColor, cloudColor, cloudFade);
    }
    
    return skyColor;
}

void main() {
    // Reconstruct view-space position from depth
    vec4 clip_pos = vec4(texcoords * 2.0 - 1.0, subpassLoad(depth).x, 1.0);
    vec4 scaled_view_pos = inverse_projection * clip_pos;
    vec3 view_pos = scaled_view_pos.xyz / scaled_view_pos.w;
    float view_length = length(view_pos);
    
    // Convert to true hyperbolic distance
    float dist = view_length >= 1.0 ? INFINITY : atanh(view_length);
    
    // Get view-space direction (normalized)
    vec3 view_dir = view_length > 1e-6 ? view_pos / view_length : vec3(0.0, 0.0, -1.0);
    
    // world_up and world_north are in view space (computed on CPU)
    vec3 up_dir = normalize(world_up.xyz);
    vec3 north_dir = normalize(world_north.xyz);
    
    // Compute procedural sky color in view space
    vec3 sky_color = proceduralSky(view_dir, up_dir, north_dir);
    
    // Exponential^k fog: visibility decreases with distance
    float visibility = exp(-pow(dist * fog_density, 5));
    float fog_alpha = 1.0 - visibility;

    // Dither fog alpha to reduce visible banding
    float dither = (hash21(gl_FragCoord.xy + time * 60.0) - 0.5) / 255.0;
    fog_alpha = clamp(fog_alpha + dither, 0.0, 1.0);
    
    fog = vec4(sky_color, fog_alpha);
}
