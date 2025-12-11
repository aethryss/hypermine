use na::{Vector2, Vector3};
use std::f64::consts::TAU;

/// Hyperbolic-aware gradient noise sampler.
///
/// The sampler consumes globally consistent hyperbolic coordinates (e.g. the
/// `asinh` distances produced by `ChunkParams::hyperbolic_*_coords*`) so that
/// identical world positions always map to the same lattice points regardless
/// of which chunk is generating them. This guarantees C¹ continuity across
/// chunk boundaries and deterministic noise for a fixed seed.
#[derive(Clone, Copy, Debug)]
pub struct HyperbolicNoise {
    seed: u64,
}

impl HyperbolicNoise {
    /// Creates a new sampler with the provided seed.
    pub const fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Sample 2D Perlin-style gradient noise at the provided hyperbolic coords.
    /// Returns values in approximately [-1, 1].
    #[inline]
    pub fn sample2(&self, coords: Vector2<f64>) -> f64 {
        perlin2(self.seed, coords)
    }

    /// Sample 3D gradient noise.
    #[inline]
    pub fn sample3(&self, coords: Vector3<f64>) -> f64 {
        perlin3(self.seed, coords)
    }

    /// Fractional Brownian motion (fBm) helper for 2D coordinates.
    pub fn fbm2(&self, coords: Vector2<f64>, cfg: FbmConfig) -> f64 {
        let octaves = cfg.octaves.max(1);
        let mut amplitude = 1.0;
        let mut frequency = 1.0;
        let mut sum = 0.0;
        let mut norm = 0.0;

        for _ in 0..octaves {
            let scaled = Vector2::new(coords.x * frequency, coords.y * frequency);
            sum += amplitude * self.sample2(scaled);
            norm += amplitude;
            amplitude *= cfg.gain;
            frequency *= cfg.lacunarity;
        }

        if norm > 0.0 { sum / norm } else { 0.0 }
    }

    /// Fractional Brownian motion (fBm) helper for 3D coordinates.
    pub fn fbm3(&self, coords: Vector3<f64>, cfg: FbmConfig) -> f64 {
        let octaves = cfg.octaves.max(1);
        let mut amplitude = 1.0;
        let mut frequency = 1.0;
        let mut sum = 0.0;
        let mut norm = 0.0;

        for _ in 0..octaves {
            let scaled = Vector3::new(
                coords.x * frequency,
                coords.y * frequency,
                coords.z * frequency,
            );
            sum += amplitude * self.sample3(scaled);
            norm += amplitude;
            amplitude *= cfg.gain;
            frequency *= cfg.lacunarity;
        }

        if norm > 0.0 { sum / norm } else { 0.0 }
    }
}

/// Configuration for fractional Brownian motion accumulation.
#[derive(Clone, Copy, Debug)]
pub struct FbmConfig {
    pub octaves: u8,
    pub lacunarity: f64,
    pub gain: f64,
}

impl Default for FbmConfig {
    fn default() -> Self {
        Self {
            octaves: 5,
            lacunarity: 2.0,
            gain: 0.5,
        }
    }
}

#[inline]
fn perlin2(seed: u64, coords: Vector2<f64>) -> f64 {
    let cell = coords.map(|c| c.floor());
    let frac = coords - cell;

    let ix0 = cell.x as i64;
    let iy0 = cell.y as i64;
    let ix1 = ix0.wrapping_add(1);
    let iy1 = iy0.wrapping_add(1);

    let grad00 = grad2(seed, ix0, iy0);
    let grad10 = grad2(seed, ix1, iy0);
    let grad01 = grad2(seed, ix0, iy1);
    let grad11 = grad2(seed, ix1, iy1);

    let dot00 = grad00.dot(&frac);
    let dot10 = grad10.dot(&(frac - Vector2::new(1.0, 0.0)));
    let dot01 = grad01.dot(&(frac - Vector2::new(0.0, 1.0)));
    let dot11 = grad11.dot(&(frac - Vector2::new(1.0, 1.0)));

    let u = fade(frac.x);
    let v = fade(frac.y);

    let nx0 = lerp(dot00, dot10, u);
    let nx1 = lerp(dot01, dot11, u);
    lerp(nx0, nx1, v)
}

#[inline]
fn perlin3(seed: u64, coords: Vector3<f64>) -> f64 {
    let cell = coords.map(|c| c.floor());
    let frac = coords - cell;

    let ix0 = cell.x as i64;
    let iy0 = cell.y as i64;
    let iz0 = cell.z as i64;
    let ix1 = ix0.wrapping_add(1);
    let iy1 = iy0.wrapping_add(1);
    let iz1 = iz0.wrapping_add(1);

    let grad000 = grad3(seed, ix0, iy0, iz0);
    let grad100 = grad3(seed, ix1, iy0, iz0);
    let grad010 = grad3(seed, ix0, iy1, iz0);
    let grad110 = grad3(seed, ix1, iy1, iz0);
    let grad001 = grad3(seed, ix0, iy0, iz1);
    let grad101 = grad3(seed, ix1, iy0, iz1);
    let grad011 = grad3(seed, ix0, iy1, iz1);
    let grad111 = grad3(seed, ix1, iy1, iz1);

    let dot000 = grad000.dot(&frac);
    let dot100 = grad100.dot(&(frac - Vector3::new(1.0, 0.0, 0.0)));
    let dot010 = grad010.dot(&(frac - Vector3::new(0.0, 1.0, 0.0)));
    let dot110 = grad110.dot(&(frac - Vector3::new(1.0, 1.0, 0.0)));
    let dot001 = grad001.dot(&(frac - Vector3::new(0.0, 0.0, 1.0)));
    let dot101 = grad101.dot(&(frac - Vector3::new(1.0, 0.0, 1.0)));
    let dot011 = grad011.dot(&(frac - Vector3::new(0.0, 1.0, 1.0)));
    let dot111 = grad111.dot(&(frac - Vector3::new(1.0, 1.0, 1.0)));

    let u = fade(frac.x);
    let v = fade(frac.y);
    let w = fade(frac.z);

    let nx00 = lerp(dot000, dot100, u);
    let nx10 = lerp(dot010, dot110, u);
    let nx01 = lerp(dot001, dot101, u);
    let nx11 = lerp(dot011, dot111, u);

    let nxy0 = lerp(nx00, nx10, v);
    let nxy1 = lerp(nx01, nx11, v);
    lerp(nxy0, nxy1, w)
}

#[inline]
fn grad2(seed: u64, ix: i64, iy: i64) -> Vector2<f64> {
    let hash = lattice_hash(seed, ix, iy, 0);
    let angle = (hash as f64 / u64::MAX as f64) * TAU;
    Vector2::new(angle.cos(), angle.sin())
}

#[inline]
fn grad3(seed: u64, ix: i64, iy: i64, iz: i64) -> Vector3<f64> {
    let hash = lattice_hash(seed, ix, iy, iz);
    let u = ((hash >> 32) as f64) / (u32::MAX as f64);
    let v = (hash & 0xffff_ffff) as f64 / (u32::MAX as f64);
    let theta = TAU * u;
    let z = 2.0 * v - 1.0;
    let r = (1.0 - z * z).sqrt().max(1e-9);
    Vector3::new(r * theta.cos(), r * theta.sin(), z)
}

#[inline]
fn fade(t: f64) -> f64 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[inline]
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}

#[inline]
fn lattice_hash(seed: u64, ix: i64, iy: i64, iz: i64) -> u64 {
    const PRIME_X: u64 = 0x9E37_79B1_85EB_CA87;
    const PRIME_Y: u64 = 0xC2B2_AE3D_27D4_EB4F;
    const PRIME_Z: u64 = 0x1656_67B1_9E37_79F9;

    let mut h = seed.wrapping_add((ix as u64).wrapping_mul(PRIME_X));
    h ^= h >> 33;
    h = h.wrapping_mul(PRIME_Y);
    h = h.wrapping_add((iy as u64).wrapping_mul(PRIME_Z));
    h ^= h >> 29;
    h = h.wrapping_mul(PRIME_X);
    h = h.wrapping_add((iz as u64).wrapping_mul(PRIME_Y));
    splitmix64(h)
}

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn continuity_across_integer_boundary_2d() {
        let noise = HyperbolicNoise::new(0x5eed_5eed);
        let left = noise.sample2(Vector2::new(0.999_999, -12.25));
        let right = noise.sample2(Vector2::new(1.000_001, -12.25));
        assert!((left - right).abs() < 1e-3, "2D continuity broken");
    }

    #[test]
    fn continuity_across_integer_boundary_3d() {
        let noise = HyperbolicNoise::new(0x1234_5678);
        let a = noise.sample3(Vector3::new(-3.0, 7.999_99, 0.125));
        let b = noise.sample3(Vector3::new(-3.0, 8.000_01, 0.125));
        assert!((a - b).abs() < 1e-3, "3D continuity broken");
    }

    #[test]
    fn fbm_normalized() {
        let noise = HyperbolicNoise::new(42);
        let cfg = FbmConfig {
            octaves: 6,
            ..FbmConfig::default()
        };
        let value = noise.fbm3(Vector3::new(12.5, -3.75, 0.0), cfg);
        assert!(value >= -1.0 && value <= 1.0);
    }
}
