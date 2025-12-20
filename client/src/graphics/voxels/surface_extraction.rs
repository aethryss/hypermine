use std::ffi::c_char;
use std::mem;

use ash::{Device, vk};
use lahar::{DedicatedBuffer, DedicatedMapping};
use vk_shader_macros::include_glsl;

use crate::graphics::{Base, TransparencyMap, VkDrawIndirectCommand, as_bytes};
use common::{defer, world::BlockRegistry};

const EXTRACT: &[u32] = include_glsl!("shaders/surface-extraction/extract.comp", target: vulkan1_1);

/// Number of transparency classes (opaque, cutout, translucent)
pub const TRANSPARENCY_CLASS_COUNT: usize = 3;

/// GPU-accelerated surface extraction from voxel chunks
pub struct SurfaceExtraction {
    params_layout: vk::DescriptorSetLayout,
    ds_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    extract: vk::Pipeline,
}

impl SurfaceExtraction {
    pub fn new(gfx: &Base) -> Self {
        let device = &*gfx.device;
        unsafe {
            let params_layout = device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&[
                        vk::DescriptorSetLayoutBinding {
                            binding: 0,
                            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            ..Default::default()
                        },
                        vk::DescriptorSetLayoutBinding {
                            binding: 1,
                            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            ..Default::default()
                        },
                        vk::DescriptorSetLayoutBinding {
                            binding: 2,
                            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            ..Default::default()
                        },
                    ]),
                    None,
                )
                .unwrap();
            let ds_layout = device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&[
                        vk::DescriptorSetLayoutBinding {
                            binding: 0,
                            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            ..Default::default()
                        },
                        vk::DescriptorSetLayoutBinding {
                            binding: 1,
                            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            ..Default::default()
                        },
                        vk::DescriptorSetLayoutBinding {
                            binding: 2,
                            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            ..Default::default()
                        },
                        vk::DescriptorSetLayoutBinding {
                            binding: 3,
                            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            ..Default::default()
                        },
                        vk::DescriptorSetLayoutBinding {
                            binding: 4,
                            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            ..Default::default()
                        },
                        vk::DescriptorSetLayoutBinding {
                            binding: 5,
                            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            ..Default::default()
                        },
                        vk::DescriptorSetLayoutBinding {
                            binding: 6,
                            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            ..Default::default()
                        },
                        vk::DescriptorSetLayoutBinding {
                            binding: 7,
                            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            ..Default::default()
                        },
                        // Light data buffer (same format as voxels)
                        vk::DescriptorSetLayoutBinding {
                            binding: 8,
                            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            ..Default::default()
                        },
                    ]),
                    None,
                )
                .unwrap();
            let pipeline_layout = device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&[params_layout, ds_layout])
                        .push_constant_ranges(&[vk::PushConstantRange {
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            offset: 0,
                            size: 4,
                        }]),
                    None,
                )
                .unwrap();

            let extract = device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(EXTRACT), None)
                .unwrap();
            let extract_guard = defer(|| device.destroy_shader_module(extract, None));

            let specialization_map_entries = [
                vk::SpecializationMapEntry {
                    constant_id: 0,
                    offset: 0,
                    size: 4,
                },
                vk::SpecializationMapEntry {
                    constant_id: 1,
                    offset: 4,
                    size: 4,
                },
                vk::SpecializationMapEntry {
                    constant_id: 2,
                    offset: 8,
                    size: 4,
                },
            ];
            let specialization = vk::SpecializationInfo::default()
                .map_entries(&specialization_map_entries)
                .data(as_bytes(&WORKGROUP_SIZE));

            let p_name = c"main".as_ptr() as *const c_char;
            let mut pipelines = device
                .create_compute_pipelines(
                    gfx.pipeline_cache,
                    &[vk::ComputePipelineCreateInfo {
                        stage: vk::PipelineShaderStageCreateInfo {
                            stage: vk::ShaderStageFlags::COMPUTE,
                            module: extract,
                            p_name,
                            p_specialization_info: &specialization,
                            ..Default::default()
                        },
                        layout: pipeline_layout,
                        ..Default::default()
                    }],
                    None,
                )
                .unwrap()
                .into_iter();

            // Free shader modules now that the actual pipelines are built
            extract_guard.invoke();

            let extract = pipelines.next().unwrap();
            gfx.set_name(extract, cstr!("extract"));

            Self {
                params_layout,
                ds_layout,
                pipeline_layout,
                extract,
            }
        }
    }

    pub unsafe fn destroy(&mut self, device: &Device) {
        unsafe {
            device.destroy_descriptor_set_layout(self.params_layout, None);
            device.destroy_descriptor_set_layout(self.ds_layout, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_pipeline(self.extract, None);
        }
    }
}

/// Scratch space for actually performing the extraction
pub struct ScratchBuffer {
    dimension: u32,
    params: DedicatedBuffer,
    texture_indices: DedicatedMapping<[u32]>, // Maps BlockID to texture_index
    transparency_classes: DedicatedMapping<[u32]>, // Maps texture_index to transparency class
    /// Size of a single entry in the voxel buffer
    voxel_buffer_unit: vk::DeviceSize,
    /// Size of a single entry in the state buffer (3 face counts)
    state_buffer_unit: vk::DeviceSize,
    voxels_staging: DedicatedMapping<[u16]>,
    voxels: DedicatedBuffer,
    /// Light data staging buffer (same layout as voxels)
    light_staging: DedicatedMapping<[u16]>,
    /// Light data GPU buffer
    light: DedicatedBuffer,
    state: DedicatedBuffer,
    descriptor_pool: vk::DescriptorPool,
    params_ds: vk::DescriptorSet,
    descriptor_sets: Vec<vk::DescriptorSet>,
    free_slots: Vec<u32>,
    concurrency: u32,
}

impl ScratchBuffer {
    pub fn new(
        gfx: &Base,
        ctx: &SurfaceExtraction,
        concurrency: u32,
        dimension: u32,
        transparency_map: &TransparencyMap,
    ) -> Self {
        let device = &*gfx.device;
        // Padded by 2 on each dimension so each voxel of interest has a full neighborhood
        let voxel_buffer_unit = round_up(
            mem::size_of::<u16>() as vk::DeviceSize * (dimension as vk::DeviceSize + 2).pow(3),
            // Pad at least to multiples of 4 so the shaders can safely read in 32 bit units
            gfx.limits.min_storage_buffer_offset_alignment.max(4),
        );
        let voxels_size = concurrency as vk::DeviceSize * voxel_buffer_unit;

        // 3 face counts, one per transparency class
        let state_buffer_unit = round_up(
            4 * TRANSPARENCY_CLASS_COUNT as vk::DeviceSize,
            gfx.limits.min_storage_buffer_offset_alignment,
        );
        unsafe {
            let params = DedicatedBuffer::new(
                device,
                &gfx.memory_properties,
                &vk::BufferCreateInfo::default()
                    .size(mem::size_of::<Params>() as vk::DeviceSize)
                    .usage(
                        vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            );
            gfx.set_name(params.handle, cstr!("surface extraction params"));

            // Create texture_indices lookup buffer (BlockID to texture_index mapping)
            let mut texture_indices_data = [0u32; 256];
            for (block_id, block) in BlockRegistry::all_blocks().iter().enumerate() {
                if block.id as usize != block_id {
                    eprintln!(
                        "ERROR: BLOCKS entry at index {} has id {}, but expected {}",
                        block_id, block.id, block_id
                    );
                }
                texture_indices_data[block_id] = block.texture_index as u32;
            }

            let mut texture_indices = DedicatedMapping::<[u32]>::zeroed_array(
                device,
                &gfx.memory_properties,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                256,
            );
            // Copy data into the mapped memory
            for (i, &val) in texture_indices_data.iter().enumerate() {
                texture_indices[i] = val;
            }
            gfx.set_name(texture_indices.buffer(), cstr!("texture indices"));

            // Create transparency_classes lookup buffer (block_id to TransparencyClass mapping)
            // We need to map block_id -> texture_index -> transparency_class
            let mut transparency_classes = DedicatedMapping::<[u32]>::zeroed_array(
                device,
                &gfx.memory_properties,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                256,
            );
            for (block_id, block) in BlockRegistry::all_blocks().iter().enumerate() {
                let texture_index = block.texture_index as usize;
                let class = transparency_map.classes().get(texture_index).copied()
                    .unwrap_or(crate::graphics::TransparencyClass::Opaque);
                transparency_classes[block_id] = class as u32;

                // Log non-opaque blocks for debugging
                if class != crate::graphics::TransparencyClass::Opaque {
                    tracing::info!(
                        block_id,
                        block_name = block.name,
                        texture_index,
                        ?class,
                        "non-opaque block"
                    );
                }
            }
            gfx.set_name(
                transparency_classes.buffer(),
                cstr!("transparency classes"),
            );

            let voxels_staging = DedicatedMapping::zeroed_array(
                device,
                &gfx.memory_properties,
                vk::BufferUsageFlags::TRANSFER_SRC,
                (voxels_size / mem::size_of::<u16>() as vk::DeviceSize) as usize,
            );
            gfx.set_name(voxels_staging.buffer(), cstr!("voxels staging"));

            let voxels = DedicatedBuffer::new(
                device,
                &gfx.memory_properties,
                &vk::BufferCreateInfo::default()
                    .size(voxels_size)
                    .usage(
                        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            );
            gfx.set_name(voxels.handle, cstr!("voxels"));

            // Light data buffers (same size and layout as voxels)
            let light_staging = DedicatedMapping::zeroed_array(
                device,
                &gfx.memory_properties,
                vk::BufferUsageFlags::TRANSFER_SRC,
                (voxels_size / mem::size_of::<u16>() as vk::DeviceSize) as usize,
            );
            gfx.set_name(light_staging.buffer(), cstr!("light staging"));

            let light = DedicatedBuffer::new(
                device,
                &gfx.memory_properties,
                &vk::BufferCreateInfo::default()
                    .size(voxels_size)
                    .usage(
                        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            );
            gfx.set_name(light.handle, cstr!("light"));

            let state = DedicatedBuffer::new(
                device,
                &gfx.memory_properties,
                &vk::BufferCreateInfo::default()
                    .size(state_buffer_unit * vk::DeviceSize::from(concurrency))
                    .usage(
                        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            );
            gfx.set_name(state.handle, cstr!("surface extraction state"));

            let descriptor_pool = device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(concurrency + 1)
                        .pool_sizes(&[
                            vk::DescriptorPoolSize {
                                ty: vk::DescriptorType::UNIFORM_BUFFER,
                                // params + texture_indices + transparency_classes
                                descriptor_count: 3,
                            },
                            vk::DescriptorPoolSize {
                                ty: vk::DescriptorType::STORAGE_BUFFER,
                                // Per task: voxels + light + state + 3 indirect + 3 faces = 9
                                descriptor_count: 9 * concurrency,
                            },
                        ]),
                    None,
                )
                .unwrap();
            let mut layouts = Vec::with_capacity(concurrency as usize + 1);
            layouts.resize(concurrency as usize, ctx.ds_layout);
            layouts.push(ctx.params_layout);
            let mut descriptor_sets = device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(descriptor_pool)
                        .set_layouts(&layouts),
                )
                .unwrap();

            let params_ds = descriptor_sets.pop().unwrap();
            device.update_descriptor_sets(
                &[
                    vk::WriteDescriptorSet::default()
                        .dst_set(params_ds)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .buffer_info(&[vk::DescriptorBufferInfo {
                            buffer: params.handle,
                            offset: 0,
                            range: vk::WHOLE_SIZE,
                        }]),
                    vk::WriteDescriptorSet::default()
                        .dst_set(params_ds)
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .buffer_info(&[vk::DescriptorBufferInfo {
                            buffer: texture_indices.buffer(),
                            offset: 0,
                            range: vk::WHOLE_SIZE,
                        }]),
                    vk::WriteDescriptorSet::default()
                        .dst_set(params_ds)
                        .dst_binding(2)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .buffer_info(&[vk::DescriptorBufferInfo {
                            buffer: transparency_classes.buffer(),
                            offset: 0,
                            range: vk::WHOLE_SIZE,
                        }]),
                ],
                &[],
            );

            Self {
                dimension,
                params,
                texture_indices,
                transparency_classes,
                voxel_buffer_unit,
                state_buffer_unit,
                voxels_staging,
                voxels,
                light_staging,
                light,
                state,
                descriptor_pool,
                params_ds,
                descriptor_sets,
                free_slots: (0..concurrency).collect(),
                concurrency,
            }
        }
    }

    pub fn alloc(&mut self) -> Option<u32> {
        self.free_slots.pop()
    }

    pub fn free(&mut self, index: u32) {
        debug_assert!(
            !self.free_slots.contains(&index),
            "double-free of surface extraction scratch slot"
        );
        self.free_slots.push(index);
    }

    /// Includes a one-voxel margin around the entire volume
    pub fn storage(&mut self, index: u32) -> &mut [u16] {
        let start = index as usize * (self.voxel_buffer_unit as usize / mem::size_of::<u16>());
        let length = (self.dimension + 2).pow(3) as usize;
        &mut self.voxels_staging[start..start + length]
    }

    /// Light data storage, same layout as voxel storage
    /// Each u16 contains a 12-bit RGB light value (4 bits per channel)
    pub fn light_storage(&mut self, index: u32) -> &mut [u16] {
        let start = index as usize * (self.voxel_buffer_unit as usize / mem::size_of::<u16>());
        let length = (self.dimension + 2).pow(3) as usize;
        &mut self.light_staging[start..start + length]
    }

    pub unsafe fn extract(
        &mut self,
        device: &Device,
        ctx: &SurfaceExtraction,
        indirect_buffers: [vk::Buffer; TRANSPARENCY_CLASS_COUNT],
        face_buffers: [vk::Buffer; TRANSPARENCY_CLASS_COUNT],
        cmd: vk::CommandBuffer,
        tasks: &[ExtractTask],
    ) {
        unsafe {
            // Prevent overlap with the last batch of work
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                Default::default(),
                &[vk::MemoryBarrier {
                    src_access_mask: vk::AccessFlags::SHADER_READ,
                    dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                    ..Default::default()
                }],
                &[],
                &[],
            );
            // HACKITY HACK: Queue submit synchronization validation thinks we're
            // racing with the preceding chunk draws. Our logic to allocate unique
            // ranges should be preventing this, so this may be a false positive.
            // However, if that's true, why does the validation error only trigger a
            // handful of times at startup? Perhaps we're freeing and reusing
            // storage before the previous draw completes, and validation is somehow
            // smart enough to notice?
            for face_buffer in face_buffers {
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::VERTEX_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    Default::default(),
                    &[],
                    &[vk::BufferMemoryBarrier {
                        buffer: face_buffer,
                        src_access_mask: vk::AccessFlags::SHADER_READ,
                        dst_access_mask: vk::AccessFlags::SHADER_WRITE,
                        offset: 0,
                        size: vk::WHOLE_SIZE,
                        ..Default::default()
                    }],
                    &[],
                );
            }

            // Prepare shared state
            device.cmd_update_buffer(
                cmd,
                self.params.handle,
                0,
                as_bytes(&Params {
                    dimension: self.dimension,
                }),
            );
            device.cmd_fill_buffer(cmd, self.state.handle, 0, vk::WHOLE_SIZE, 0);

            let voxel_count = (self.dimension + 2).pow(3) as usize;
            let voxels_range =
                voxel_count as vk::DeviceSize * mem::size_of::<u16>() as vk::DeviceSize;
            let max_faces = 3 * (self.dimension.pow(3) + self.dimension.pow(2));
            let dispatch = dispatch_sizes(self.dimension);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                ctx.pipeline_layout,
                0,
                &[self.params_ds],
                &[],
            );

            // Prepare each task
            for task in tasks {
                assert!(
                    task.index < self.concurrency,
                    "index {} out of bounds for concurrency {}",
                    task.index,
                    self.concurrency
                );
                let index = task.index as usize;

                let voxels_offset = self.voxel_buffer_unit * index as vk::DeviceSize;

                device.update_descriptor_sets(
                    &[
                        vk::WriteDescriptorSet::default()
                            .dst_set(self.descriptor_sets[index])
                            .dst_binding(0)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(&[vk::DescriptorBufferInfo {
                                buffer: self.voxels.handle,
                                offset: voxels_offset,
                                range: voxels_range,
                            }]),
                        vk::WriteDescriptorSet::default()
                            .dst_set(self.descriptor_sets[index])
                            .dst_binding(1)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(&[vk::DescriptorBufferInfo {
                                buffer: self.state.handle,
                                offset: self.state_buffer_unit * vk::DeviceSize::from(task.index),
                                range: 4 * TRANSPARENCY_CLASS_COUNT as vk::DeviceSize,
                            }]),
                        // Indirect buffers for each transparency class
                        vk::WriteDescriptorSet::default()
                            .dst_set(self.descriptor_sets[index])
                            .dst_binding(2)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(&[vk::DescriptorBufferInfo {
                                buffer: indirect_buffers[0],
                                offset: task.indirect_offsets[0],
                                range: INDIRECT_SIZE,
                            }]),
                        vk::WriteDescriptorSet::default()
                            .dst_set(self.descriptor_sets[index])
                            .dst_binding(3)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(&[vk::DescriptorBufferInfo {
                                buffer: indirect_buffers[1],
                                offset: task.indirect_offsets[1],
                                range: INDIRECT_SIZE,
                            }]),
                        vk::WriteDescriptorSet::default()
                            .dst_set(self.descriptor_sets[index])
                            .dst_binding(4)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(&[vk::DescriptorBufferInfo {
                                buffer: indirect_buffers[2],
                                offset: task.indirect_offsets[2],
                                range: INDIRECT_SIZE,
                            }]),
                        // Face buffers for each transparency class
                        vk::WriteDescriptorSet::default()
                            .dst_set(self.descriptor_sets[index])
                            .dst_binding(5)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(&[vk::DescriptorBufferInfo {
                                buffer: face_buffers[0],
                                offset: task.face_offsets[0],
                                range: max_faces as vk::DeviceSize * FACE_SIZE,
                            }]),
                        vk::WriteDescriptorSet::default()
                            .dst_set(self.descriptor_sets[index])
                            .dst_binding(6)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(&[vk::DescriptorBufferInfo {
                                buffer: face_buffers[1],
                                offset: task.face_offsets[1],
                                range: max_faces as vk::DeviceSize * FACE_SIZE,
                            }]),
                        vk::WriteDescriptorSet::default()
                            .dst_set(self.descriptor_sets[index])
                            .dst_binding(7)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(&[vk::DescriptorBufferInfo {
                                buffer: face_buffers[2],
                                offset: task.face_offsets[2],
                                range: max_faces as vk::DeviceSize * FACE_SIZE,
                            }]),
                        // Light data buffer (binding 8)
                        vk::WriteDescriptorSet::default()
                            .dst_set(self.descriptor_sets[index])
                            .dst_binding(8)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(&[vk::DescriptorBufferInfo {
                                buffer: self.light.handle,
                                offset: voxels_offset, // Same offset as voxels
                                range: voxels_range,
                            }]),
                    ],
                    &[],
                );

                device.cmd_copy_buffer(
                    cmd,
                    self.voxels_staging.buffer(),
                    self.voxels.handle,
                    &[vk::BufferCopy {
                        src_offset: voxels_offset,
                        dst_offset: voxels_offset,
                        size: voxels_range,
                    }],
                );
                // Copy light data to GPU
                device.cmd_copy_buffer(
                    cmd,
                    self.light_staging.buffer(),
                    self.light.handle,
                    &[vk::BufferCopy {
                        src_offset: voxels_offset,
                        dst_offset: voxels_offset,
                        size: voxels_range,
                    }],
                );
                // Initialize indirect commands for all 3 transparency classes
                for class_idx in 0..TRANSPARENCY_CLASS_COUNT {
                    device.cmd_update_buffer(
                        cmd,
                        indirect_buffers[class_idx],
                        task.indirect_offsets[class_idx],
                        as_bytes(&VkDrawIndirectCommand {
                            vertex_count: 0,
                            instance_count: 1,
                            first_vertex: (task.face_offsets[class_idx] / FACE_SIZE) as u32 * 6,
                            first_instance: task.draw_id,
                        }),
                    );
                }
            }

            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                Default::default(),
                &[vk::MemoryBarrier {
                    src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                    dst_access_mask: vk::AccessFlags::SHADER_READ
                        | vk::AccessFlags::SHADER_WRITE
                        | vk::AccessFlags::UNIFORM_READ,
                    ..Default::default()
                }],
                &[],
                &[],
            );

            // Write faces to memory
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, ctx.extract);
            for task in tasks {
                device.cmd_push_constants(
                    cmd,
                    ctx.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &u32::from(task.reverse_winding).to_ne_bytes(),
                );
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    ctx.pipeline_layout,
                    1,
                    &[self.descriptor_sets[task.index as usize]],
                    &[],
                );
                device.cmd_dispatch(cmd, dispatch.x, dispatch.y, dispatch.z);
            }

            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::VERTEX_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
                Default::default(),
                &[vk::MemoryBarrier {
                    src_access_mask: vk::AccessFlags::SHADER_WRITE,
                    dst_access_mask: vk::AccessFlags::SHADER_READ
                        | vk::AccessFlags::INDIRECT_COMMAND_READ,
                    ..Default::default()
                }],
                &[],
                &[],
            );
        }
    }

    pub unsafe fn destroy(&mut self, device: &Device) {
        unsafe {
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.params.destroy(device);
            self.texture_indices.destroy(device);
            self.transparency_classes.destroy(device);
            self.voxels_staging.destroy(device);
            self.voxels.destroy(device);
            self.light_staging.destroy(device);
            self.light.destroy(device);
            self.state.destroy(device);
        }
    }
}

/// Specifies a single chunk's worth of surface extraction work
#[derive(Debug, Copy, Clone)]
pub struct ExtractTask {
    /// Offset into indirect buffers for each transparency class
    pub indirect_offsets: [vk::DeviceSize; TRANSPARENCY_CLASS_COUNT],
    /// Offset into face buffers for each transparency class
    pub face_offsets: [vk::DeviceSize; TRANSPARENCY_CLASS_COUNT],
    pub index: u32,
    pub draw_id: u32,
    pub reverse_winding: bool,
}

fn dispatch_sizes(dimension: u32) -> na::Vector3<u32> {
    fn divide_rounding_up(x: u32, y: u32) -> u32 {
        debug_assert!(x > 0 && y > 0);
        (x - 1) / y + 1
    }

    // We add 1 to each dimension because we only look at negative-facing faces of the target voxel
    na::Vector3::new(
        // Extending the X axis accounts for 3 possible faces per voxel
        divide_rounding_up((dimension + 1) * 3, WORKGROUP_SIZE[0]),
        divide_rounding_up(dimension + 1, WORKGROUP_SIZE[1]),
        divide_rounding_up(dimension + 1, WORKGROUP_SIZE[2]),
    )
}

#[repr(C)]
#[derive(Copy, Clone)]
struct Params {
    dimension: u32,
}

/// Manages storage for ready-to-render voxels
pub struct DrawBuffer {
    /// One indirect buffer per transparency class
    indirect: [DedicatedBuffer; TRANSPARENCY_CLASS_COUNT],
    /// One face buffer per transparency class
    faces: [DedicatedBuffer; TRANSPARENCY_CLASS_COUNT],
    dimension: u32,
    face_buffer_unit: vk::DeviceSize,
    count: u32,
}

impl DrawBuffer {
    /// Allocate a buffer suitable for rendering at most `count` chunks having `dimension` voxels
    /// along each edge
    pub fn new(gfx: &Base, count: u32, dimension: u32) -> Self {
        let device = &*gfx.device;

        let max_faces = 3 * (dimension.pow(3) + dimension.pow(2));
        let face_buffer_unit = round_up(
            max_faces as vk::DeviceSize * FACE_SIZE,
            gfx.limits.min_storage_buffer_offset_alignment,
        );
        let face_buffer_size = count as vk::DeviceSize * face_buffer_unit;

        unsafe {
            let class_names = ["opaque", "cutout", "translucent"];
            let indirect = std::array::from_fn(|i| {
                let buf = DedicatedBuffer::new(
                    device,
                    &gfx.memory_properties,
                    &vk::BufferCreateInfo::default()
                        .size(count as vk::DeviceSize * INDIRECT_SIZE)
                        .usage(
                            vk::BufferUsageFlags::STORAGE_BUFFER
                                | vk::BufferUsageFlags::INDIRECT_BUFFER
                                | vk::BufferUsageFlags::TRANSFER_DST,
                        )
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                );
                let name = format!("indirect_{}\0", class_names[i]);
                gfx.set_name(buf.handle, std::ffi::CStr::from_bytes_with_nul(name.as_bytes()).unwrap());
                buf
            });

            let faces = std::array::from_fn(|i| {
                let buf = DedicatedBuffer::new(
                    device,
                    &gfx.memory_properties,
                    &vk::BufferCreateInfo::default()
                        .size(face_buffer_size)
                        .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                );
                let name = format!("faces_{}\0", class_names[i]);
                gfx.set_name(buf.handle, std::ffi::CStr::from_bytes_with_nul(name.as_bytes()).unwrap());
                buf
            });

            Self {
                indirect,
                faces,
                dimension,
                face_buffer_unit,
                count,
            }
        }
    }

    /// Buffers containing face data, one per transparency class
    pub fn face_buffers(&self) -> [vk::Buffer; TRANSPARENCY_CLASS_COUNT] {
        std::array::from_fn(|i| self.faces[i].handle)
    }

    /// Buffers containing face counts for use with cmd_draw_indirect, one per transparency class
    pub fn indirect_buffers(&self) -> [vk::Buffer; TRANSPARENCY_CLASS_COUNT] {
        std::array::from_fn(|i| self.indirect[i].handle)
    }

    /// Buffer containing face data for a specific transparency class
    pub fn face_buffer(&self, class_idx: usize) -> vk::Buffer {
        self.faces[class_idx].handle
    }

    /// Buffer containing face counts for use with cmd_draw_indirect for a specific transparency class
    pub fn indirect_buffer(&self, class_idx: usize) -> vk::Buffer {
        self.indirect[class_idx].handle
    }

    /// The offset into the face buffer at which a chunk's face data can be found
    pub fn face_offset(&self, chunk: u32) -> vk::DeviceSize {
        assert!(chunk < self.count);
        vk::DeviceSize::from(chunk) * self.face_buffer_unit
    }

    /// The offsets into all face buffers at which a chunk's face data can be found
    pub fn face_offsets(&self, chunk: u32) -> [vk::DeviceSize; TRANSPARENCY_CLASS_COUNT] {
        let offset = self.face_offset(chunk);
        [offset; TRANSPARENCY_CLASS_COUNT]
    }

    /// The offset into the indirect buffer at which a chunk's face data can be found
    pub fn indirect_offset(&self, chunk: u32) -> vk::DeviceSize {
        assert!(chunk < self.count);
        vk::DeviceSize::from(chunk) * INDIRECT_SIZE
    }

    /// The offsets into all indirect buffers at which a chunk's face data can be found
    pub fn indirect_offsets(&self, chunk: u32) -> [vk::DeviceSize; TRANSPARENCY_CLASS_COUNT] {
        let offset = self.indirect_offset(chunk);
        [offset; TRANSPARENCY_CLASS_COUNT]
    }

    /// Number of voxels along a chunk edge
    pub fn dimension(&self) -> u32 {
        self.dimension
    }

    pub unsafe fn destroy(&mut self, device: &Device) {
        unsafe {
            for buf in &mut self.indirect {
                buf.destroy(device);
            }
            for buf in &mut self.faces {
                buf.destroy(device);
            }
        }
    }
}

// Size of the VkDrawIndirectCommand struct
const INDIRECT_SIZE: vk::DeviceSize = 16;

// Size of a Surface struct: 3 uint32 fields (pos_axis, occlusion_mat, light) = 12 bytes
const FACE_SIZE: vk::DeviceSize = 12;

const WORKGROUP_SIZE: [u32; 3] = [4, 4, 4];

fn round_up(value: vk::DeviceSize, alignment: vk::DeviceSize) -> vk::DeviceSize {
    value.div_ceil(alignment) * alignment
}
