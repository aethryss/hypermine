use std::fs::File;
use std::path::PathBuf;

use anyhow::{Context, anyhow};
use ash::vk;
use lahar::DedicatedImage;
use tracing::trace;

use crate::loader::{LoadCtx, LoadFuture, Loadable};

pub struct Png {
    pub path: PathBuf,
}

impl Loadable for Png {
    type Output = DedicatedImage;

    fn load(self, handle: &LoadCtx) -> LoadFuture<'_, Self::Output> {
        Box::pin(async move {
            let full_path = handle
                .cfg
                .find_asset(&self.path)
                .ok_or_else(|| anyhow!("{} not found", self.path.display()))?;

            trace!(path=%full_path.display(), "loading PNG");
            let file = File::open(&full_path)
                .with_context(|| format!("opening {}", full_path.display()))?;
            let decoder = png::Decoder::new(file);
            let mut reader = decoder
                .read_info()
                .with_context(|| format!("decoding {}", full_path.display()))?;

            // Copy dimensions and color info
            let (width, height, bytes_per_pixel) = {
                let info = reader.info();
                let bpp = match info.color_type {
                    png::ColorType::Grayscale => 1,
                    png::ColorType::Rgb => 3,
                    png::ColorType::Indexed => 1,
                    png::ColorType::GrayscaleAlpha => 2,
                    png::ColorType::Rgba => 4,
                };
                (info.width, info.height, bpp)
            };

            // Allocate buffer for raw PNG data
            let raw_data_size = width as usize * height as usize * bytes_per_pixel;

            // Read PNG data into temporary buffer
            let mut raw_data = vec![0u8; raw_data_size];
            reader
                .next_frame(&mut raw_data)
                .with_context(|| format!("decoding {}", full_path.display()))?;

            // Convert to RGBA if needed
            let mut mem_data = vec![0u8; width as usize * height as usize * 4];
            match bytes_per_pixel {
                4 => mem_data.copy_from_slice(&raw_data), // Already RGBA
                3 => {
                    // Convert RGB to RGBA (add alpha channel)
                    for (i, chunk) in raw_data.chunks_exact(3).enumerate() {
                        mem_data[i * 4] = chunk[0];
                        mem_data[i * 4 + 1] = chunk[1];
                        mem_data[i * 4 + 2] = chunk[2];
                        mem_data[i * 4 + 3] = 255;
                    }
                }
                _ => anyhow::bail!(
                    "Unsupported PNG color type with {} bytes per pixel",
                    bytes_per_pixel
                ),
            }

            let mut mem = handle
                .staging
                .alloc(mem_data.len())
                .await
                .ok_or_else(|| anyhow!("{}: image too large", full_path.display()))?;

            // Copy the converted data into the staging buffer
            mem.copy_from_slice(&mem_data);

            unsafe {
                let image = DedicatedImage::new(
                    &handle.gfx.device,
                    &handle.gfx.memory_properties,
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(vk::Format::R8G8B8A8_SRGB)
                        .extent(vk::Extent3D {
                            width,
                            height,
                            depth: 1,
                        })
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST),
                );

                let range = vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                };
                let src = handle.staging.buffer();
                let buffer_offset = mem.offset();
                let dst = image.handle;

                handle
                    .transfer
                    .run(move |xf, cmd| {
                        xf.device.cmd_pipeline_barrier(
                            cmd,
                            vk::PipelineStageFlags::TOP_OF_PIPE,
                            vk::PipelineStageFlags::TRANSFER,
                            vk::DependencyFlags::default(),
                            &[],
                            &[],
                            &[vk::ImageMemoryBarrier::default()
                                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                                .old_layout(vk::ImageLayout::UNDEFINED)
                                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                                .image(dst)
                                .subresource_range(range)],
                        );
                        xf.device.cmd_copy_buffer_to_image(
                            cmd,
                            src,
                            dst,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            &[vk::BufferImageCopy {
                                buffer_offset,
                                image_subresource: vk::ImageSubresourceLayers {
                                    aspect_mask: vk::ImageAspectFlags::COLOR,
                                    mip_level: 0,
                                    base_array_layer: 0,
                                    layer_count: 1,
                                },
                                image_extent: vk::Extent3D {
                                    width,
                                    height,
                                    depth: 1,
                                },
                                ..Default::default()
                            }],
                        );
                        xf.stages |= vk::PipelineStageFlags::FRAGMENT_SHADER;
                        xf.image_barriers.push(
                            vk::ImageMemoryBarrier::default()
                                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                                .src_queue_family_index(xf.queue_family)
                                .dst_queue_family_index(xf.dst_queue_family)
                                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                                .image(dst)
                                .subresource_range(range),
                        );
                    })
                    .await?;

                trace!(
                    width = width,
                    height = height,
                    path = %full_path.display(),
                    "loaded PNG"
                );
                Ok(image)
            }
        })
    }
}
