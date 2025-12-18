//! Texture transparency classification for proper rendering order.
//!
//! This module analyzes the terrain atlas at load time to classify each tile
//! into one of three transparency classes:
//! - **Opaque**: All pixels are fully opaque (alpha = 255)
//! - **Cutout**: Has fully transparent pixels (alpha = 0) but no semi-transparency
//! - **Translucent**: Has semi-transparent pixels (0 < alpha < 255)
//!
//! The rendering order for depth ties is: Opaque < Cutout < Translucent

use std::fs::File;
use std::path::Path;

use anyhow::{Context, anyhow};
use tracing::trace;

/// Transparency classification for a block texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum TransparencyClass {
    /// All pixels are fully opaque (alpha = 255).
    /// Rendered first with depth write enabled.
    #[default]
    Opaque = 0,
    /// Has fully transparent pixels (alpha = 0) but no semi-transparency.
    /// Rendered second with alpha testing (discard if alpha < threshold).
    Cutout = 1,
    /// Has semi-transparent pixels (0 < alpha < 255).
    /// Rendered last with alpha blending, depth test but no depth write.
    Translucent = 2,
}

impl TransparencyClass {
    /// Returns the render priority (lower = rendered first).
    #[inline]
    pub fn priority(self) -> u8 {
        self as u8
    }

    /// Returns true if this class requires alpha testing or blending.
    #[inline]
    pub fn has_transparency(self) -> bool {
        !matches!(self, TransparencyClass::Opaque)
    }
}

/// Stores the transparency classification for all texture tiles in the atlas.
///
/// The terrain atlas is assumed to be a 16x16 grid of 16x16 pixel tiles (256x256 total),
/// giving 256 possible texture indices.
#[derive(Clone)]
pub struct TransparencyMap {
    /// Transparency class for each texture index (0-255).
    classes: [TransparencyClass; 256],
}

impl Default for TransparencyMap {
    fn default() -> Self {
        Self {
            classes: [TransparencyClass::Opaque; 256],
        }
    }
}

impl TransparencyMap {
    /// Creates a new transparency map with all textures classified as opaque.
    pub fn new() -> Self {
        Self::default()
    }

    /// Analyzes a terrain atlas PNG file and classifies each tile.
    ///
    /// The atlas is expected to be a 256x256 pixel image with a 16x16 grid of tiles,
    /// where each tile is 16x16 pixels.
    pub fn from_atlas(atlas_path: &Path) -> anyhow::Result<Self> {
        trace!(path=%atlas_path.display(), "analyzing terrain atlas for transparency");

        let file = File::open(atlas_path)
            .with_context(|| format!("opening {}", atlas_path.display()))?;
        let decoder = png::Decoder::new(file);
        let mut reader = decoder
            .read_info()
            .with_context(|| format!("decoding {}", atlas_path.display()))?;

        let info = reader.info();
        let (width, height) = (info.width, info.height);
        let color_type = info.color_type;

        // We need at minimum a 256x256 atlas for 16x16 tiles
        if width < 256 || height < 256 {
            return Err(anyhow!(
                "terrain atlas too small: {}x{}, expected at least 256x256",
                width,
                height
            ));
        }

        let bytes_per_pixel = match color_type {
            png::ColorType::Rgba => 4,
            png::ColorType::GrayscaleAlpha => 2,
            png::ColorType::Rgb => 3,
            png::ColorType::Grayscale => 1,
            png::ColorType::Indexed => {
                // For indexed images, we'd need to check the palette
                // For simplicity, treat as opaque unless palette has alpha
                return Ok(Self::new());
            }
        };

        let has_alpha = matches!(
            color_type,
            png::ColorType::Rgba | png::ColorType::GrayscaleAlpha
        );

        // Read the image data
        let mut raw_data = vec![0u8; reader.output_buffer_size()];
        reader
            .next_frame(&mut raw_data)
            .with_context(|| format!("reading {}", atlas_path.display()))?;

        let mut classes = [TransparencyClass::Opaque; 256];

        // If the image doesn't have an alpha channel, all tiles are opaque
        if !has_alpha {
            trace!("atlas has no alpha channel, all tiles are opaque");
            return Ok(Self { classes });
        }

        let tile_size = 16u32;
        let tiles_per_row = (width / tile_size).min(16) as usize;
        let tiles_per_col = (height / tile_size).min(16) as usize;

        // Analyze each tile
        for tile_y in 0..tiles_per_col {
            for tile_x in 0..tiles_per_row {
                let texture_index = tile_y * 16 + tile_x;
                if texture_index >= 256 {
                    break;
                }

                let class = Self::classify_tile(
                    &raw_data,
                    width as usize,
                    bytes_per_pixel,
                    tile_x * tile_size as usize,
                    tile_y * tile_size as usize,
                    tile_size as usize,
                );

                classes[texture_index] = class;

                // Log non-opaque tiles for debugging
                if class != TransparencyClass::Opaque {
                    trace!(
                        texture_index,
                        ?class,
                        tile_x,
                        tile_y,
                        "non-opaque tile found"
                    );
                }
            }
        }

        let opaque_count = classes.iter().filter(|&&c| c == TransparencyClass::Opaque).count();
        let cutout_count = classes.iter().filter(|&&c| c == TransparencyClass::Cutout).count();
        let translucent_count = classes.iter().filter(|&&c| c == TransparencyClass::Translucent).count();

        trace!(
            opaque = opaque_count,
            cutout = cutout_count,
            translucent = translucent_count,
            "classified terrain atlas tiles"
        );

        Ok(Self { classes })
    }

    /// Classifies a single tile based on its alpha values.
    fn classify_tile(
        data: &[u8],
        image_width: usize,
        bytes_per_pixel: usize,
        tile_x: usize,
        tile_y: usize,
        tile_size: usize,
    ) -> TransparencyClass {
        let mut has_transparent = false; // alpha = 0
        let mut has_semitransparent = false; // 0 < alpha < 255

        for py in 0..tile_size {
            for px in 0..tile_size {
                let x = tile_x + px;
                let y = tile_y + py;
                let pixel_offset = (y * image_width + x) * bytes_per_pixel;

                // Alpha is the last component
                let alpha = data[pixel_offset + bytes_per_pixel - 1];

                match alpha {
                    0 => has_transparent = true,
                    255 => {} // fully opaque, no change
                    _ => has_semitransparent = true,
                }

                // Early exit if we've found both types
                if has_semitransparent {
                    return TransparencyClass::Translucent;
                }
            }
        }

        if has_semitransparent {
            TransparencyClass::Translucent
        } else if has_transparent {
            TransparencyClass::Cutout
        } else {
            TransparencyClass::Opaque
        }
    }

    /// Gets the transparency class for a texture index.
    #[inline]
    pub fn get(&self, texture_index: u16) -> TransparencyClass {
        self.classes.get(texture_index as usize).copied().unwrap_or(TransparencyClass::Opaque)
    }

    /// Gets the raw classes array for uploading to GPU.
    #[inline]
    pub fn classes(&self) -> &[TransparencyClass; 256] {
        &self.classes
    }

    /// Converts the classes to a u8 array for GPU upload.
    pub fn as_u8_array(&self) -> [u8; 256] {
        let mut result = [0u8; 256];
        for (i, &class) in self.classes.iter().enumerate() {
            result[i] = class as u8;
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transparency_class_priority() {
        assert!(TransparencyClass::Opaque.priority() < TransparencyClass::Cutout.priority());
        assert!(TransparencyClass::Cutout.priority() < TransparencyClass::Translucent.priority());
    }

    #[test]
    fn test_default_map_is_opaque() {
        let map = TransparencyMap::new();
        for i in 0..256 {
            assert_eq!(map.get(i as u16), TransparencyClass::Opaque);
        }
    }

    #[test]
    fn test_classify_tile_all_opaque() {
        // 4x4 tile, RGBA, all opaque (alpha = 255)
        let data: Vec<u8> = (0..4*4).flat_map(|_| [255, 128, 64, 255]).collect();
        let class = TransparencyMap::classify_tile(&data, 4, 4, 0, 0, 4);
        assert_eq!(class, TransparencyClass::Opaque);
    }

    #[test]
    fn test_classify_tile_with_cutout() {
        // 4x4 tile, RGBA, one fully transparent pixel
        let mut data: Vec<u8> = (0..4*4).flat_map(|_| [255, 128, 64, 255]).collect();
        data[3] = 0; // First pixel fully transparent
        let class = TransparencyMap::classify_tile(&data, 4, 4, 0, 0, 4);
        assert_eq!(class, TransparencyClass::Cutout);
    }

    #[test]
    fn test_classify_tile_with_translucent() {
        // 4x4 tile, RGBA, one semi-transparent pixel
        let mut data: Vec<u8> = (0..4*4).flat_map(|_| [255, 128, 64, 255]).collect();
        data[3] = 128; // First pixel semi-transparent
        let class = TransparencyMap::classify_tile(&data, 4, 4, 0, 0, 4);
        assert_eq!(class, TransparencyClass::Translucent);
    }
}
