//! Block lighting system for Hyperbolicraft.
//!
//! This module implements a fluid-like light propagation system where light spreads
//! outward from emitting blocks and decays over distance. Light is stored per-voxel
//! as RGB values with 4 bits per channel (16 levels per channel, 4096 possible colors).
//!
//! # Light Behavior by Block Type
//!
//! - **Air/Cutout blocks**: Light passes through with decay of 1 level per block
//! - **Translucent blocks**: Light passes through with decay of 3 levels per block
//! - **Solid blocks**: Light is completely blocked
//!
//! # Light Propagation
//!
//! Light propagates like a fluid at a constant tick rate, spreading one block per tick
//! in all six cardinal directions. This allows for efficient incremental updates rather
//! than recomputing all lighting at once.

use serde::{Deserialize, Serialize};

/// A light value with 4 bits per RGB channel.
///
/// Format: bits [11:8] = R, bits [7:4] = G, bits [3:0] = B
/// Each channel ranges from 0 (no light) to 15 (maximum intensity).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct LightValue(pub u16);

impl LightValue {
    /// No light (all channels at 0).
    pub const ZERO: Self = Self(0);

    /// Maximum light (all channels at 15).
    pub const MAX: Self = Self(0x0FFF);

    /// Maximum value for a single channel.
    pub const MAX_CHANNEL: u8 = 15;

    /// Creates a new light value from RGB components.
    ///
    /// Each component is clamped to [0, 15].
    #[inline]
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        let r = if r > 15 { 15 } else { r };
        let g = if g > 15 { 15 } else { g };
        let b = if b > 15 { 15 } else { b };
        Self(((r as u16) << 8) | ((g as u16) << 4) | (b as u16))
    }

    /// Returns the red component (0-15).
    #[inline]
    pub const fn r(self) -> u8 {
        ((self.0 >> 8) & 0x0F) as u8
    }

    /// Returns the green component (0-15).
    #[inline]
    pub const fn g(self) -> u8 {
        ((self.0 >> 4) & 0x0F) as u8
    }

    /// Returns the blue component (0-15).
    #[inline]
    pub const fn b(self) -> u8 {
        (self.0 & 0x0F) as u8
    }

    /// Returns the maximum channel value (brightness).
    #[inline]
    pub const fn brightness(self) -> u8 {
        let r = self.r();
        let g = self.g();
        let b = self.b();
        if r >= g && r >= b {
            r
        } else if g >= b {
            g
        } else {
            b
        }
    }

    /// Returns true if this light value is zero (no light).
    #[inline]
    pub const fn is_zero(self) -> bool {
        self.0 == 0
    }

    /// Decays the light by the specified amount in each channel.
    /// Returns ZERO if all channels would go below zero.
    #[inline]
    pub const fn decay(self, amount: u8) -> Self {
        let r = self.r().saturating_sub(amount);
        let g = self.g().saturating_sub(amount);
        let b = self.b().saturating_sub(amount);
        Self::new(r, g, b)
    }

    /// Returns the component-wise maximum of two light values.
    #[inline]
    pub const fn max(self, other: Self) -> Self {
        let r = if self.r() > other.r() {
            self.r()
        } else {
            other.r()
        };
        let g = if self.g() > other.g() {
            self.g()
        } else {
            other.g()
        };
        let b = if self.b() > other.b() {
            self.b()
        } else {
            other.b()
        };
        Self::new(r, g, b)
    }

    /// Returns true if self is brighter than other in any channel.
    #[inline]
    pub const fn any_brighter_than(self, other: Self) -> bool {
        self.r() > other.r() || self.g() > other.g() || self.b() > other.b()
    }

    /// Converts to normalized float RGB values (0.0 to 1.0).
    #[inline]
    pub fn to_rgb_f32(self) -> [f32; 3] {
        [
            self.r() as f32 / 15.0,
            self.g() as f32 / 15.0,
            self.b() as f32 / 15.0,
        ]
    }

    /// Returns the raw u16 value (for GPU upload).
    #[inline]
    pub const fn to_u16(self) -> u16 {
        self.0
    }
}

impl From<u16> for LightValue {
    #[inline]
    fn from(value: u16) -> Self {
        Self(value & 0x0FFF)
    }
}

impl From<LightValue> for u16 {
    #[inline]
    fn from(value: LightValue) -> Self {
        value.0
    }
}

/// How a block interacts with light propagation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[repr(u8)]
pub enum LightBehavior {
    /// Light passes through with minimal decay (1 level).
    /// Used for air and cutout blocks (flowers, saplings, etc).
    #[default]
    Transparent = 0,

    /// Light passes through with increased decay (3 levels).
    /// Used for water, ice, stained glass, etc.
    Translucent = 1,

    /// Light is completely blocked.
    /// Used for stone, dirt, wood, and most solid blocks.
    Opaque = 2,
}

impl LightBehavior {
    /// Returns the light decay amount when passing through this block type.
    /// Returns None for opaque blocks (light cannot pass through).
    #[inline]
    pub const fn decay_amount(self) -> Option<u8> {
        match self {
            LightBehavior::Transparent => Some(1),
            LightBehavior::Translucent => Some(3),
            LightBehavior::Opaque => None,
        }
    }

    /// Returns true if light can pass through this block type.
    #[inline]
    pub const fn allows_light(self) -> bool {
        !matches!(self, LightBehavior::Opaque)
    }
}

/// Light-related properties for a block type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct BlockLightInfo {
    /// Light emitted by this block type (zero for non-emitting blocks).
    pub emission: LightValue,

    /// How this block affects light propagation.
    pub behavior: LightBehavior,
}

impl BlockLightInfo {
    /// Creates a new block light info with no emission and transparent behavior.
    pub const fn transparent() -> Self {
        Self {
            emission: LightValue::ZERO,
            behavior: LightBehavior::Transparent,
        }
    }

    /// Creates a new block light info with no emission and translucent behavior.
    pub const fn translucent() -> Self {
        Self {
            emission: LightValue::ZERO,
            behavior: LightBehavior::Translucent,
        }
    }

    /// Creates a new block light info with no emission and opaque behavior.
    pub const fn opaque() -> Self {
        Self {
            emission: LightValue::ZERO,
            behavior: LightBehavior::Opaque,
        }
    }

    /// Creates a light-emitting block with the specified emission color.
    pub const fn emitter(r: u8, g: u8, b: u8) -> Self {
        Self {
            emission: LightValue::new(r, g, b),
            behavior: LightBehavior::Opaque, // Most light sources are solid
        }
    }

    /// Creates a light-emitting block with transparent behavior (like a torch).
    pub const fn emitter_transparent(r: u8, g: u8, b: u8) -> Self {
        Self {
            emission: LightValue::new(r, g, b),
            behavior: LightBehavior::Transparent,
        }
    }

    /// Returns true if this block emits light.
    #[inline]
    pub const fn emits_light(self) -> bool {
        !self.emission.is_zero()
    }
}

/// Stores light values for a chunk, similar to VoxelData.
///
/// Like VoxelData, this includes a 1-voxel margin on each side to handle
/// cross-chunk light propagation during rendering and surface extraction.
#[derive(Clone)]
pub enum LightData {
    /// All voxels, including margins, have the same light value.
    Uniform(LightValue),

    /// Dense storage of light values for each voxel.
    /// Size is (dimension + 2)^3 to include margins.
    Dense(Box<[LightValue]>),
}

impl Default for LightData {
    fn default() -> Self {
        LightData::Uniform(LightValue::ZERO)
    }
}

impl LightData {
    /// Creates a new LightData with all voxels set to zero light.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new dense LightData initialized to zero light.
    pub fn new_dense(dimension: u8) -> Self {
        let size = (usize::from(dimension) + 2).pow(3);
        LightData::Dense(vec![LightValue::ZERO; size].into_boxed_slice())
    }

    /// Gets the light value at the given index.
    #[inline]
    pub fn get(&self, index: usize) -> LightValue {
        match self {
            LightData::Uniform(value) => *value,
            LightData::Dense(data) => data[index],
        }
    }

    /// Gets a mutable reference to the dense data, converting from Uniform if necessary.
    pub fn data_mut(&mut self, dimension: u8) -> &mut [LightValue] {
        match self {
            LightData::Dense(data) => data,
            LightData::Uniform(value) => {
                let size = (usize::from(dimension) + 2).pow(3);
                *self = LightData::Dense(vec![*value; size].into_boxed_slice());
                match self {
                    LightData::Dense(data) => data,
                    _ => unreachable!(),
                }
            }
        }
    }

    /// Returns true if this is uniform (all same value).
    #[inline]
    pub fn is_uniform(&self) -> bool {
        matches!(self, LightData::Uniform(_))
    }

    /// Returns true if all light values are zero.
    pub fn is_dark(&self) -> bool {
        match self {
            LightData::Uniform(v) => v.is_zero(),
            LightData::Dense(data) => data.iter().all(|v| v.is_zero()),
        }
    }

    /// Serializes the light data (excluding margins) to bytes.
    /// Format: 2 bytes per voxel (u16 little-endian), dimension^3 voxels.
    pub fn serialize(&self, dimension: u8) -> Vec<u8> {
        let dim = dimension as usize;
        let mut bytes = Vec::with_capacity(dim * dim * dim * 2);

        for z in 0..dimension {
            for y in 0..dimension {
                for x in 0..dimension {
                    let index = coords_to_index(x, y, z, dimension);
                    let value = self.get(index);
                    bytes.extend_from_slice(&value.0.to_le_bytes());
                }
            }
        }

        bytes
    }

    /// Deserializes light data from bytes, creating a Dense LightData with void margins.
    /// Returns None if the data size doesn't match the expected dimension.
    pub fn deserialize(bytes: &[u8], dimension: u8) -> Option<Self> {
        let dim = dimension as usize;
        let expected_size = dim * dim * dim * 2;

        if bytes.len() != expected_size {
            return None;
        }

        let size = (dim + 2).pow(3);
        let mut data = vec![LightValue::ZERO; size];

        let mut byte_iter = bytes.chunks_exact(2);
        for z in 0..dimension {
            for y in 0..dimension {
                for x in 0..dimension {
                    let chunk = byte_iter.next()?;
                    let value = u16::from_le_bytes([chunk[0], chunk[1]]);
                    let index = coords_to_index(x, y, z, dimension);
                    data[index] = LightValue::from(value);
                }
            }
        }

        Some(LightData::Dense(data.into_boxed_slice()))
    }
}

/// Converts voxel coordinates to an index in the data array (including margins).
#[inline]
pub fn coords_to_index(x: u8, y: u8, z: u8, dimension: u8) -> usize {
    let chunk_size_with_margin = dimension as usize + 2;
    (x as usize + 1)
        + (y as usize + 1) * chunk_size_with_margin
        + (z as usize + 1) * chunk_size_with_margin.pow(2)
}

/// Converts an index back to voxel coordinates (excluding margin coordinates).
/// Returns None if the index corresponds to a margin voxel.
#[inline]
pub fn index_to_coords(index: usize, dimension: u8) -> Option<(u8, u8, u8)> {
    let chunk_size_with_margin = dimension as usize + 2;
    let z = index / chunk_size_with_margin.pow(2);
    let remainder = index % chunk_size_with_margin.pow(2);
    let y = remainder / chunk_size_with_margin;
    let x = remainder % chunk_size_with_margin;

    // Check if within non-margin bounds
    if x == 0 || x > dimension as usize || y == 0 || y > dimension as usize || z == 0 || z > dimension as usize {
        return None;
    }

    Some(((x - 1) as u8, (y - 1) as u8, (z - 1) as u8))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn light_value_components() {
        let light = LightValue::new(15, 8, 3);
        assert_eq!(light.r(), 15);
        assert_eq!(light.g(), 8);
        assert_eq!(light.b(), 3);
    }

    #[test]
    fn light_value_clamping() {
        let light = LightValue::new(20, 100, 255);
        assert_eq!(light.r(), 15);
        assert_eq!(light.g(), 15);
        assert_eq!(light.b(), 15);
    }

    #[test]
    fn light_value_decay() {
        let light = LightValue::new(10, 5, 2);
        let decayed = light.decay(3);
        assert_eq!(decayed.r(), 7);
        assert_eq!(decayed.g(), 2);
        assert_eq!(decayed.b(), 0);
    }

    #[test]
    fn light_value_max() {
        let a = LightValue::new(10, 5, 8);
        let b = LightValue::new(7, 12, 6);
        let max = a.max(b);
        assert_eq!(max.r(), 10);
        assert_eq!(max.g(), 12);
        assert_eq!(max.b(), 8);
    }

    #[test]
    fn light_data_serialization() {
        let dimension = 4u8;
        let mut data = LightData::new_dense(dimension);

        // Set some light values
        data.data_mut(dimension)[coords_to_index(1, 2, 3, dimension)] = LightValue::new(10, 5, 3);

        let serialized = data.serialize(dimension);
        let deserialized = LightData::deserialize(&serialized, dimension).unwrap();

        assert_eq!(
            deserialized.get(coords_to_index(1, 2, 3, dimension)),
            LightValue::new(10, 5, 3)
        );
    }

    #[test]
    fn coords_roundtrip() {
        let dimension = 12u8;
        for z in 0..dimension {
            for y in 0..dimension {
                for x in 0..dimension {
                    let index = coords_to_index(x, y, z, dimension);
                    let (rx, ry, rz) = index_to_coords(index, dimension).unwrap();
                    assert_eq!((x, y, z), (rx, ry, rz));
                }
            }
        }
    }
}
