use ash::{Device, vk};
use lahar::DedicatedImage;
use vk_shader_macros::include_glsl;

use crate::graphics::Base;
use common::defer;

use super::surface::{OPAQUE, TRANSLUCENT, TRANSFORM_SIZE};
use super::surface_extraction::DrawBuffer;

const VERT: &[u32] = include_glsl!("shaders/skylight.vert");

pub struct SkylightShadow {
    pub extent: vk::Extent2D,

    render_pass: vk::RenderPass,
    framebuffer: vk::Framebuffer,

    depth: DedicatedImage,
    depth_view: vk::ImageView,

    sampler: vk::Sampler,

    static_ds_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_opaque: vk::DescriptorSet,
    descriptor_set_translucent: vk::DescriptorSet,

    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl SkylightShadow {
    pub fn new(gfx: &Base, buffer: &DrawBuffer, extent: vk::Extent2D) -> Self {
        let device = &*gfx.device;
        unsafe {
            let depth = DedicatedImage::new(
                device,
                &gfx.memory_properties,
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(vk::Format::D32_SFLOAT)
                    .extent(vk::Extent3D {
                        width: extent.width,
                        height: extent.height,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED),
            );
            gfx.set_name(depth.handle, cstr!("skylight_shadow_depth"));

            let depth_view = device
                .create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(depth.handle)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(vk::Format::D32_SFLOAT)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::DEPTH,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        }),
                    None,
                )
                .unwrap();
            gfx.set_name(depth_view, cstr!("skylight_shadow_depth_view"));

            let sampler = device
                .create_sampler(
                    &vk::SamplerCreateInfo::default()
                        .min_filter(vk::Filter::NEAREST)
                        .mag_filter(vk::Filter::NEAREST)
                        .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                    None,
                )
                .unwrap();
            gfx.set_name(sampler, cstr!("skylight_shadow_sampler"));

            // Depth-only render pass.
            let render_pass = device
                .create_render_pass(
                    &vk::RenderPassCreateInfo::default()
                        .attachments(&[vk::AttachmentDescription {
                            format: vk::Format::D32_SFLOAT,
                            samples: vk::SampleCountFlags::TYPE_1,
                            load_op: vk::AttachmentLoadOp::CLEAR,
                            store_op: vk::AttachmentStoreOp::STORE,
                            initial_layout: vk::ImageLayout::UNDEFINED,
                            final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                            ..Default::default()
                        }])
                        .subpasses(&[vk::SubpassDescription::default()
                            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                            .depth_stencil_attachment(&vk::AttachmentReference {
                                attachment: 0,
                                layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                            })])
                        .dependencies(&[vk::SubpassDependency {
                            src_subpass: vk::SUBPASS_EXTERNAL,
                            dst_subpass: 0,
                            src_stage_mask: vk::PipelineStageFlags::FRAGMENT_SHADER,
                            dst_stage_mask: vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                                | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                            src_access_mask: vk::AccessFlags::SHADER_READ,
                            dst_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                            dependency_flags: vk::DependencyFlags::BY_REGION,
                            ..Default::default()
                        }]),
                    None,
                )
                .unwrap();
            gfx.set_name(render_pass, cstr!("skylight_shadow_rp"));

            let framebuffer = device
                .create_framebuffer(
                    &vk::FramebufferCreateInfo::default()
                        .render_pass(render_pass)
                        .attachments(&[depth_view])
                        .width(extent.width)
                        .height(extent.height)
                        .layers(1),
                    None,
                )
                .unwrap();
            gfx.set_name(framebuffer, cstr!("skylight_shadow_fb"));

            // Descriptor set (set=1) for the opaque surfaces buffer.
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
                    ]),
                    None,
                )
                .unwrap();

            let descriptor_pool = device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(2)
                        .pool_sizes(&[vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::STORAGE_BUFFER,
                            descriptor_count: 2,
                        }]),
                    None,
                )
                .unwrap();

            let descriptor_sets = device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(descriptor_pool)
                        .set_layouts(&[static_ds_layout, static_ds_layout]),
                )
                .unwrap();

            let descriptor_set_opaque = descriptor_sets[0];
            let descriptor_set_translucent = descriptor_sets[1];

            device.update_descriptor_sets(
                &[
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set_opaque)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&[vk::DescriptorBufferInfo {
                            buffer: buffer.face_buffer(OPAQUE),
                            offset: 0,
                            range: vk::WHOLE_SIZE,
                        }]),
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set_translucent)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&[vk::DescriptorBufferInfo {
                            buffer: buffer.face_buffer(TRANSLUCENT),
                            offset: 0,
                            range: vk::WHOLE_SIZE,
                        }]),
                ],
                &[],
            );

            // Pipeline
            let vert = device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(VERT), None)
                .unwrap();
            let v_guard = defer(|| device.destroy_shader_module(vert, None));

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
            let shader_stages = [vk::PipelineShaderStageCreateInfo {
                stage: vk::ShaderStageFlags::VERTEX,
                module: vert,
                p_name: entry_point,
                ..Default::default()
            }];

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
                .cull_mode(vk::CullModeFlags::NONE)
                .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                .polygon_mode(vk::PolygonMode::FILL)
                .line_width(1.0);

            let multisample_state = vk::PipelineMultisampleStateCreateInfo::default()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);

            let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default()
                .depth_test_enable(true)
                .depth_write_enable(true)
                .depth_compare_op(vk::CompareOp::LESS);

            let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let dynamic_state =
                vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

            // No color attachments.
            let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default();

            let pipeline = device
                .create_graphics_pipelines(
                    gfx.pipeline_cache,
                    &[vk::GraphicsPipelineCreateInfo::default()
                        .stages(&shader_stages)
                        .vertex_input_state(&vertex_input_state)
                        .input_assembly_state(&input_assembly_state)
                        .viewport_state(&viewport_state)
                        .rasterization_state(&rasterization_state)
                        .multisample_state(&multisample_state)
                        .depth_stencil_state(&depth_stencil_state)
                        .color_blend_state(&color_blend_state)
                        .dynamic_state(&dynamic_state)
                        .layout(pipeline_layout)
                        .render_pass(render_pass)
                        .subpass(0)],
                    None,
                )
                .unwrap()[0];
            gfx.set_name(pipeline, cstr!("skylight_shadow_pipeline"));

            v_guard.invoke();

            Self {
                extent,
                render_pass,
                framebuffer,
                depth,
                depth_view,
                sampler,
                static_ds_layout,
                descriptor_pool,
                descriptor_set_opaque,
                descriptor_set_translucent,
                pipeline_layout,
                pipeline,
            }
        }
    }

    pub fn depth_view(&self) -> vk::ImageView {
        self.depth_view
    }

    pub fn sampler(&self) -> vk::Sampler {
        self.sampler
    }

    pub unsafe fn draw(
        &self,
        device: &Device,
        common_ds: vk::DescriptorSet,
        dimension: u32,
        cmd: vk::CommandBuffer,
        frame: &super::surface::Frame,
        buffer: &DrawBuffer,
        drawn_chunks: &[common::lru_slab::SlotId],
    ) {
        unsafe {
            // Transition for rendering.
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                vk::DependencyFlags::BY_REGION,
                &[],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::SHADER_READ)
                    .dst_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
                    .image(self.depth.handle)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::DEPTH,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })],
            );

            device.cmd_begin_render_pass(
                cmd,
                &vk::RenderPassBeginInfo::default()
                    .render_pass(self.render_pass)
                    .framebuffer(self.framebuffer)
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: self.extent,
                    })
                    .clear_values(&[vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue {
                            depth: 1.0,
                            stencil: 0,
                        },
                    }]),
                vk::SubpassContents::INLINE,
            );

            let viewports = [vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: self.extent.width as f32,
                height: self.extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            }];
            let scissors = [vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.extent,
            }];
            device.cmd_set_viewport(cmd, 0, &viewports);
            device.cmd_set_scissor(cmd, 0, &scissors);

            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            device.cmd_bind_vertex_buffers(cmd, 0, &[frame.transforms_buffer()], &[0]);
            device.cmd_push_constants(
                cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                &dimension.to_ne_bytes(),
            );

            // Occlusion policy:
            // - Opaque blocks occlude
            // - Translucent blocks (e.g. water) occlude
            // - Cutout blocks do NOT occlude (skip)
            for &(class_idx, ds) in &[
                (OPAQUE, self.descriptor_set_opaque),
                (TRANSLUCENT, self.descriptor_set_translucent),
            ] {
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline_layout,
                    0,
                    &[common_ds, ds],
                    &[],
                );
                for chunk in drawn_chunks {
                    device.cmd_draw_indirect(
                        cmd,
                        buffer.indirect_buffer(class_idx),
                        buffer.indirect_offset(chunk.0),
                        1,
                        16,
                    );
                }
            }

            device.cmd_end_render_pass(cmd);

            // Transition for sampling.
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::BY_REGION,
                &[],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .image(self.depth.handle)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::DEPTH,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })],
            );
        }
    }

    pub unsafe fn destroy(&mut self, device: &Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.static_ds_layout, None);
            device.destroy_framebuffer(self.framebuffer, None);
            device.destroy_render_pass(self.render_pass, None);
            device.destroy_image_view(self.depth_view, None);
            device.destroy_sampler(self.sampler, None);
            self.depth.destroy(device);
        }
    }
}
