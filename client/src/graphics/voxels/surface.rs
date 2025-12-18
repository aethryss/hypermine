use ash::{Device, vk};
use lahar::{DedicatedImage, DedicatedMapping};
use vk_shader_macros::include_glsl;

use super::surface_extraction::{DrawBuffer, TRANSPARENCY_CLASS_COUNT};
use crate::{Asset, Loader, graphics::Base};
use common::defer;

const VERT: &[u32] = include_glsl!("shaders/voxels.vert");
const FRAG: &[u32] = include_glsl!("shaders/voxels.frag");

/// Index of the Opaque transparency class
pub const OPAQUE: usize = 0;
/// Index of the Cutout transparency class
pub const CUTOUT: usize = 1;
/// Index of the Translucent transparency class
pub const TRANSLUCENT: usize = 2;

pub struct Surface {
    static_ds_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    /// One pipeline per transparency class: [Opaque, Cutout, Translucent]
    pipelines: [vk::Pipeline; TRANSPARENCY_CLASS_COUNT],
    descriptor_pool: vk::DescriptorPool,
    /// One descriptor set per transparency class
    descriptor_sets: [vk::DescriptorSet; TRANSPARENCY_CLASS_COUNT],
    colors: Asset<DedicatedImage>,
    colors_view: vk::ImageView,
    linear_sampler: vk::Sampler,
}

impl Surface {
    pub fn new(gfx: &Base, loader: &mut Loader, buffer: &DrawBuffer) -> Self {
        let device = &*gfx.device;
        unsafe {
            // Construct the shader modules
            let vert = device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(VERT), None)
                .unwrap();
            // Note that these only need to live until the pipeline itself is constructed
            let v_guard = defer(|| device.destroy_shader_module(vert, None));

            let frag = device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(FRAG), None)
                .unwrap();
            let f_guard = defer(|| device.destroy_shader_module(frag, None));

            let static_ds_layout = device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&[
                        vk::DescriptorSetLayoutBinding {
                            binding: 0,
                            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::VERTEX,
                            ..Default::default()
                        },
                        vk::DescriptorSetLayoutBinding {
                            binding: 1,
                            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::FRAGMENT,
                            ..Default::default()
                        },
                    ]),
                    None,
                )
                .unwrap();

            let descriptor_pool = device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(TRANSPARENCY_CLASS_COUNT as u32)
                        .pool_sizes(&[
                            vk::DescriptorPoolSize {
                                ty: vk::DescriptorType::STORAGE_BUFFER,
                                descriptor_count: TRANSPARENCY_CLASS_COUNT as u32,
                            },
                            vk::DescriptorPoolSize {
                                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                                descriptor_count: TRANSPARENCY_CLASS_COUNT as u32,
                            },
                        ]),
                    None,
                )
                .unwrap();
            let layouts = [static_ds_layout; TRANSPARENCY_CLASS_COUNT];
            let descriptor_sets_vec = device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(descriptor_pool)
                        .set_layouts(&layouts),
                )
                .unwrap();
            let descriptor_sets: [vk::DescriptorSet; TRANSPARENCY_CLASS_COUNT] =
                descriptor_sets_vec.try_into().unwrap();

            // Bind face buffers to each descriptor set
            for (i, &ds) in descriptor_sets.iter().enumerate() {
                device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default()
                        .dst_set(ds)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&[vk::DescriptorBufferInfo {
                            buffer: buffer.face_buffer(i),
                            offset: 0,
                            range: vk::WHOLE_SIZE,
                        }])],
                    &[],
                );
            }

            // Define the outward-facing interface of the shaders, incl. uniforms, samplers, etc.
            let pipeline_layout = device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&[gfx.common_layout, static_ds_layout])
                        .push_constant_ranges(&[vk::PushConstantRange {
                            stage_flags: vk::ShaderStageFlags::VERTEX,
                            offset: 0,
                            size: 4,
                        }]),
                    None,
                )
                .unwrap();

            let entry_point = cstr!("main").as_ptr();

            // Specialization constants for fragment shader:
            // constant_id 0: enable_alpha_test (bool, 4 bytes in Vulkan)
            // constant_id 1: alpha_cutoff (f32)
            // Note: VK_BOOL32 is 4 bytes

            // Specialization map entries
            let spec_entries = [
                vk::SpecializationMapEntry {
                    constant_id: 0,
                    offset: 0,
                    size: 4, // VK_BOOL32 is 4 bytes
                },
                vk::SpecializationMapEntry {
                    constant_id: 1,
                    offset: 4,
                    size: 4, // f32
                },
            ];

            // Specialization data for each transparency class
            // [enable_alpha_test: u32, alpha_cutoff: f32]
            // Opaque: enable alpha test with 0.5 threshold (safety for any alpha textures)
            let spec_data_opaque: [u8; 8] = {
                let mut data = [0u8; 8];
                data[0..4].copy_from_slice(&1u32.to_ne_bytes()); // enable_alpha_test = true
                data[4..8].copy_from_slice(&0.5f32.to_ne_bytes()); // alpha_cutoff = 0.5
                data
            };

            // Cutout: enable alpha test with 0.5 threshold
            let spec_data_cutout: [u8; 8] = {
                let mut data = [0u8; 8];
                data[0..4].copy_from_slice(&1u32.to_ne_bytes()); // enable_alpha_test = true
                data[4..8].copy_from_slice(&0.5f32.to_ne_bytes()); // alpha_cutoff = 0.5
                data
            };

            // Translucent: disable alpha test (render all semi-transparent pixels)
            let spec_data_translucent: [u8; 8] = {
                let mut data = [0u8; 8];
                data[0..4].copy_from_slice(&0u32.to_ne_bytes()); // enable_alpha_test = false
                data[4..8].copy_from_slice(&0.0f32.to_ne_bytes()); // alpha_cutoff unused
                data
            };

            let spec_info_opaque = vk::SpecializationInfo::default()
                .map_entries(&spec_entries)
                .data(&spec_data_opaque);

            let spec_info_cutout = vk::SpecializationInfo::default()
                .map_entries(&spec_entries)
                .data(&spec_data_cutout);

            let spec_info_translucent = vk::SpecializationInfo::default()
                .map_entries(&spec_entries)
                .data(&spec_data_translucent);

            // Shader stages for each pipeline type
            let shader_stages_opaque = [
                vk::PipelineShaderStageCreateInfo {
                    stage: vk::ShaderStageFlags::VERTEX,
                    module: vert,
                    p_name: entry_point,
                    ..Default::default()
                },
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(frag)
                    .name(std::ffi::CStr::from_ptr(entry_point))
                    .specialization_info(&spec_info_opaque),
            ];

            let shader_stages_cutout = [
                vk::PipelineShaderStageCreateInfo {
                    stage: vk::ShaderStageFlags::VERTEX,
                    module: vert,
                    p_name: entry_point,
                    ..Default::default()
                },
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(frag)
                    .name(std::ffi::CStr::from_ptr(entry_point))
                    .specialization_info(&spec_info_cutout),
            ];

            let shader_stages_translucent = [
                vk::PipelineShaderStageCreateInfo {
                    stage: vk::ShaderStageFlags::VERTEX,
                    module: vert,
                    p_name: entry_point,
                    ..Default::default()
                },
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(frag)
                    .name(std::ffi::CStr::from_ptr(entry_point))
                    .specialization_info(&spec_info_translucent),
            ];

            let vertex_binding = [vk::VertexInputBindingDescription {
                binding: 0,
                stride: TRANSFORM_SIZE as u32,
                input_rate: vk::VertexInputRate::INSTANCE,
            }];

            let vertex_attributes = [
                vk::VertexInputAttributeDescription {
                    location: 0,
                    binding: 0,
                    format: vk::Format::R32G32B32A32_SFLOAT,
                    offset: 0,
                },
                vk::VertexInputAttributeDescription {
                    location: 1,
                    binding: 0,
                    format: vk::Format::R32G32B32A32_SFLOAT,
                    offset: 16,
                },
                vk::VertexInputAttributeDescription {
                    location: 2,
                    binding: 0,
                    format: vk::Format::R32G32B32A32_SFLOAT,
                    offset: 32,
                },
                vk::VertexInputAttributeDescription {
                    location: 3,
                    binding: 0,
                    format: vk::Format::R32G32B32A32_SFLOAT,
                    offset: 48,
                },
            ];

            let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
                .vertex_binding_descriptions(&vertex_binding)
                .vertex_attribute_descriptions(&vertex_attributes);

            let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::default()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

            let viewport_state = vk::PipelineViewportStateCreateInfo::default()
                .scissor_count(1)
                .viewport_count(1);

            let rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
                .cull_mode(vk::CullModeFlags::BACK)
                .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                .polygon_mode(vk::PolygonMode::FILL)
                .line_width(1.0);

            let multisample_state = vk::PipelineMultisampleStateCreateInfo::default()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);

            let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let dynamic_state =
                vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

            // Depth/stencil states for each transparency class
            // Opaque and Cutout: depth test + write
            let depth_stencil_opaque = vk::PipelineDepthStencilStateCreateInfo::default()
                .depth_test_enable(true)
                .depth_write_enable(true)
                .depth_compare_op(vk::CompareOp::GREATER);

            // Translucent: depth test AND write
            // We need depth write so the fog pass knows there's geometry here
            // Translucent-translucent occlusion isn't ideal, but it's better than sky showing through
            let depth_stencil_translucent = vk::PipelineDepthStencilStateCreateInfo::default()
                .depth_test_enable(true)
                .depth_write_enable(true)
                .depth_compare_op(vk::CompareOp::GREATER);

            // Color blend states for each transparency class
            // Opaque: no blending, just overwrite (including alpha=1.0 for fog pass)
            let color_blend_opaque = [vk::PipelineColorBlendAttachmentState {
                blend_enable: vk::FALSE,
                color_write_mask: vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
                ..Default::default()
            }];
            let color_blend_state_opaque =
                vk::PipelineColorBlendStateCreateInfo::default().attachments(&color_blend_opaque);

            // Cutout: no blending (alpha test is done in fragment shader via discard)
            // Writes alpha=1.0 for fog pass
            let color_blend_cutout = [vk::PipelineColorBlendAttachmentState {
                blend_enable: vk::FALSE,
                color_write_mask: vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
                ..Default::default()
            }];
            let color_blend_state_cutout =
                vk::PipelineColorBlendStateCreateInfo::default().attachments(&color_blend_cutout);

            // Translucent: alpha blending for RGB colors, write alpha=1.0 for fog pass
            // Color blending: final_rgb = src_rgb * src_alpha + dst_rgb * (1 - src_alpha)
            // Alpha: just overwrite with 1.0 (no blending) so fog knows there's geometry
            let color_blend_translucent = [vk::PipelineColorBlendAttachmentState {
                blend_enable: vk::TRUE,
                src_color_blend_factor: vk::BlendFactor::SRC_ALPHA,
                dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                color_blend_op: vk::BlendOp::ADD,
                // Alpha: output = 1*1 + 0*dst = 1.0 (fragment shader outputs alpha=1.0 for translucent)
                src_alpha_blend_factor: vk::BlendFactor::ONE,
                dst_alpha_blend_factor: vk::BlendFactor::ZERO,
                alpha_blend_op: vk::BlendOp::ADD,
                // Write all channels including alpha
                color_write_mask: vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
                ..Default::default()
            }];
            let color_blend_state_translucent = vk::PipelineColorBlendStateCreateInfo::default()
                .attachments(&color_blend_translucent);

            // Create all 3 pipelines
            let pipeline_create_infos = [
                // Opaque pipeline
                vk::GraphicsPipelineCreateInfo::default()
                    .stages(&shader_stages_opaque)
                    .vertex_input_state(&vertex_input_state)
                    .input_assembly_state(&input_assembly_state)
                    .viewport_state(&viewport_state)
                    .rasterization_state(&rasterization_state)
                    .multisample_state(&multisample_state)
                    .depth_stencil_state(&depth_stencil_opaque)
                    .color_blend_state(&color_blend_state_opaque)
                    .dynamic_state(&dynamic_state)
                    .layout(pipeline_layout)
                    .render_pass(gfx.render_pass)
                    .subpass(0),
                // Cutout pipeline
                vk::GraphicsPipelineCreateInfo::default()
                    .stages(&shader_stages_cutout)
                    .vertex_input_state(&vertex_input_state)
                    .input_assembly_state(&input_assembly_state)
                    .viewport_state(&viewport_state)
                    .rasterization_state(&rasterization_state)
                    .multisample_state(&multisample_state)
                    .depth_stencil_state(&depth_stencil_opaque)
                    .color_blend_state(&color_blend_state_cutout)
                    .dynamic_state(&dynamic_state)
                    .layout(pipeline_layout)
                    .render_pass(gfx.render_pass)
                    .subpass(0),
                // Translucent pipeline
                vk::GraphicsPipelineCreateInfo::default()
                    .stages(&shader_stages_translucent)
                    .vertex_input_state(&vertex_input_state)
                    .input_assembly_state(&input_assembly_state)
                    .viewport_state(&viewport_state)
                    .rasterization_state(&rasterization_state)
                    .multisample_state(&multisample_state)
                    .depth_stencil_state(&depth_stencil_translucent)
                    .color_blend_state(&color_blend_state_translucent)
                    .dynamic_state(&dynamic_state)
                    .layout(pipeline_layout)
                    .render_pass(gfx.render_pass)
                    .subpass(0),
            ];

            let pipelines_vec = device
                .create_graphics_pipelines(gfx.pipeline_cache, &pipeline_create_infos, None)
                .unwrap();
            let pipelines: [vk::Pipeline; TRANSPARENCY_CLASS_COUNT] =
                pipelines_vec.try_into().unwrap();

            gfx.set_name(pipelines[OPAQUE], cstr!("voxels_opaque"));
            gfx.set_name(pipelines[CUTOUT], cstr!("voxels_cutout"));
            gfx.set_name(pipelines[TRANSLUCENT], cstr!("voxels_translucent"));

            // Clean up the shaders explicitly, so the defer guards don't hold onto references we're
            // moving into `Self` to be returned
            v_guard.invoke();
            f_guard.invoke();

            let colors = loader.load(
                "terrain atlas",
                crate::graphics::Png {
                    path: "materials/terrain.png".into(),
                },
            );

            Self {
                static_ds_layout,
                pipeline_layout,
                pipelines,
                descriptor_pool,
                descriptor_sets,
                colors,
                colors_view: vk::ImageView::null(),
                linear_sampler: gfx.linear_sampler,
            }
        }
    }

    pub unsafe fn bind(
        &mut self,
        device: &Device,
        loader: &Loader,
        dimension: u32,
        _common_ds: vk::DescriptorSet,
        frame: &Frame,
        cmd: vk::CommandBuffer,
    ) -> bool {
        unsafe {
            if self.colors_view == vk::ImageView::null() {
                if let Some(colors) = loader.get(self.colors) {
                    self.colors_view = device
                        .create_image_view(
                            &vk::ImageViewCreateInfo::default()
                                .image(colors.handle)
                                .view_type(vk::ImageViewType::TYPE_2D)
                                .format(vk::Format::R8G8B8A8_SRGB)
                                .subresource_range(vk::ImageSubresourceRange {
                                    aspect_mask: vk::ImageAspectFlags::COLOR,
                                    base_mip_level: 0,
                                    level_count: 1,
                                    base_array_layer: 0,
                                    layer_count: 1,
                                }),
                            None,
                        )
                        .unwrap();
                    // Update all descriptor sets with the texture
                    for &ds in &self.descriptor_sets {
                        device.update_descriptor_sets(
                            &[vk::WriteDescriptorSet::default()
                                .dst_set(ds)
                                .dst_binding(1)
                                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                                .image_info(&[vk::DescriptorImageInfo {
                                    sampler: self.linear_sampler,
                                    image_view: self.colors_view,
                                    image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                                }])],
                            &[],
                        );
                    }
                } else {
                    return false;
                }
            }

            // Bind vertex buffers and push constants (shared by all transparency classes)
            device.cmd_bind_vertex_buffers(cmd, 0, &[frame.transforms.buffer()], &[0]);

            device.cmd_push_constants(
                cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                &dimension.to_ne_bytes(),
            );

            true
        }
    }

    /// Bind the pipeline and descriptor set for a specific transparency class
    pub unsafe fn bind_transparency_class(
        &self,
        device: &Device,
        common_ds: vk::DescriptorSet,
        cmd: vk::CommandBuffer,
        class_idx: usize,
    ) {
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipelines[class_idx]);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[common_ds, self.descriptor_sets[class_idx]],
                &[],
            );
        }
    }

    pub unsafe fn draw(
        &self,
        device: &Device,
        cmd: vk::CommandBuffer,
        buffer: &DrawBuffer,
        chunk: u32,
        class_idx: usize,
    ) {
        unsafe {
            device.cmd_draw_indirect(
                cmd,
                buffer.indirect_buffer(class_idx),
                buffer.indirect_offset(chunk),
                1,
                16,
            );
        }
    }

    pub unsafe fn destroy(&mut self, device: &Device) {
        unsafe {
            for pipeline in &self.pipelines {
                device.destroy_pipeline(*pipeline, None);
            }
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.static_ds_layout, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            if self.colors_view != vk::ImageView::null() {
                device.destroy_image_view(self.colors_view, None);
            }
        }
    }
}

pub struct Frame {
    transforms: DedicatedMapping<[na::Matrix4<f32>]>,
}

impl Frame {
    pub fn new(gfx: &Base, count: u32) -> Self {
        unsafe {
            let transforms = DedicatedMapping::zeroed_array(
                &gfx.device,
                &gfx.memory_properties,
                vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                count as usize * TRANSFORM_SIZE as usize,
            );
            gfx.set_name(transforms.buffer(), cstr!("voxel transforms"));
            Self { transforms }
        }
    }

    pub fn transforms_mut(&mut self) -> &mut [na::Matrix4<f32>] {
        &mut self.transforms
    }
}

impl Frame {
    pub unsafe fn destroy(&mut self, device: &Device) {
        unsafe {
            self.transforms.destroy(device);
        }
    }
}

// 4x4 f32 matrix
pub const TRANSFORM_SIZE: vk::DeviceSize = 64;
