use crate::error::{Result, VeraError};

#[cfg(feature = "gpu")]
use {
    bytemuck::{Pod, Zeroable},
    std::collections::HashMap,
    wgpu::util::DeviceExt,
};

/// Rendering backend selection
#[derive(Debug, Clone, Copy)]
pub enum RenderingBackend {
    Cpu,
    #[cfg(feature = "gpu")]
    Gpu,
}

/// VeRA image renderer
pub struct Renderer {
    backend: RenderingBackend,
    #[cfg(feature = "gpu")]
    gpu_state: Option<GpuState>,
}

#[cfg(feature = "gpu")]
struct GpuState {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter: wgpu::Adapter,
    vector_pipeline: wgpu::RenderPipeline,
    raster_pipeline: wgpu::RenderPipeline,
    composite_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    texture_cache: std::collections::HashMap<String, wgpu::Texture>,
    buffer_pool: BufferPool,
}

#[cfg(feature = "gpu")]
struct BufferPool {
    vertex_buffers: Vec<wgpu::Buffer>,
    uniform_buffers: Vec<wgpu::Buffer>,
    storage_buffers: Vec<wgpu::Buffer>,
    texture_buffers: Vec<wgpu::Buffer>,
}

#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
    tex_coords: [f32; 2],
    color: [f32; 4],
}

#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    resolution: [f32; 2],
    zoom_level: f32,
    time: f32,
}

#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct VectorUniforms {
    transform: [[f32; 4]; 4],
    color: [f32; 4],
    stroke_width: f32,
    opacity: f32,
    blend_mode: u32,
    _padding: u32,
}

impl Renderer {
    /// Create a new renderer with CPU backend
    pub fn new_cpu() -> Self {
        Self {
            backend: RenderingBackend::Cpu,
            #[cfg(feature = "gpu")]
            gpu_state: None,
        }
    }

    /// Create a new renderer with GPU backend
    #[cfg(feature = "gpu")]
    pub async fn new_gpu() -> Result<Self> {
        Self::new_gpu_impl().await
    }

    #[cfg(feature = "gpu")]
    async fn new_gpu_impl() -> Result<Self> {
        // Create wgpu instance
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::default(),
        });

        // Request adapter
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or(VeraError::GpuError("Failed to request adapter".to_string()))?;

        // Check adapter limits and features
        let adapter_info = adapter.get_info();
        log::info!(
            "Using GPU: {} ({:?})",
            adapter_info.name,
            adapter_info.backend
        );

        // Request device and queue
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("VeRA GPU Device"),
                    required_features: wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY
                        | wgpu::Features::TEXTURE_BINDING_ARRAY
                        | wgpu::Features::BUFFER_BINDING_ARRAY
                        | wgpu::Features::PARTIALLY_BOUND_BINDING_ARRAY,
                    required_limits: wgpu::Limits::downlevel_webgl2_defaults()
                        .using_resolution(adapter.limits()),
                    trace: None,
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| VeraError::GpuError(format!("Failed to request device: {e}")))?;

        // Create pipelines and GPU state
        let gpu_state = Self::create_gpu_state(device, queue, adapter).await?;

        Ok(Self {
            backend: RenderingBackend::Gpu,
            gpu_state: Some(gpu_state),
        })
    }

    #[cfg(feature = "gpu")]
    async fn create_gpu_state(
        device: wgpu::Device,
        queue: wgpu::Queue,
        adapter: wgpu::Adapter,
    ) -> Result<GpuState> {
        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("VeRA Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // Create render pipelines
        let vector_pipeline = Self::create_vector_pipeline(&device, &bind_group_layout)?;
        let raster_pipeline = Self::create_raster_pipeline(&device, &bind_group_layout)?;

        // Create compute pipeline for compositing
        let composite_pipeline = Self::create_composite_pipeline(&device)?;

        // Create buffer pool
        let buffer_pool = BufferPool {
            vertex_buffers: Vec::new(),
            uniform_buffers: Vec::new(),
            storage_buffers: Vec::new(),
            texture_buffers: Vec::new(),
        };

        Ok(GpuState {
            device,
            queue,
            adapter,
            vector_pipeline,
            raster_pipeline,
            composite_pipeline,
            bind_group_layout,
            texture_cache: HashMap::new(),
            buffer_pool,
        })
    }

    /// Render a region of the image
    pub fn render_region(
        &self,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        zoom_level: u8,
        decoder: &mut crate::decoder::Decoder<impl std::io::Read + std::io::Seek>,
    ) -> Result<image::RgbaImage> {
        match self.backend {
            RenderingBackend::Cpu => self.render_cpu(x, y, width, height, zoom_level, decoder),
            #[cfg(feature = "gpu")]
            RenderingBackend::Gpu => self.render_gpu(x, y, width, height, zoom_level, decoder),
        }
    }

    /// CPU rendering implementation
    fn render_cpu(
        &self,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        zoom_level: u8,
        decoder: &mut crate::decoder::Decoder<impl std::io::Read + std::io::Seek>,
    ) -> Result<image::RgbaImage> {
        // 1. First, render raster tiles as the base layer
        let mut result_image = decoder.decode_region(x, y, width, height, zoom_level)?;

        // 2. Get vector data and render vector layers on top
        let vector_bytes = decoder.vector_data()?;
        if !vector_bytes.is_empty() {
            let metadata = decoder.metadata()?;
            let vector_data = crate::vector::VectorData::from_bytes(
                &vector_bytes,
                &metadata.compression.vector_compression,
            )?;

            // Render vector layers on top of raster tiles
            self.render_vector_layers(
                &mut result_image,
                &vector_data,
                x,
                y,
                width,
                height,
                zoom_level,
            )?;
        }

        Ok(result_image)
    }

    /// Render vector layers onto an existing image
    fn render_vector_layers(
        &self,
        target: &mut image::RgbaImage,
        vector_data: &crate::vector::VectorData,
        region_x: u32,
        region_y: u32,
        region_width: u32,
        region_height: u32,
        zoom_level: u8,
    ) -> Result<()> {
        use crate::vector::BoundingBox;

        let scale = 1.0 / (1 << zoom_level) as f32;

        // Define the region bounding box
        let region_bounds = BoundingBox {
            x: region_x as f32,
            y: region_y as f32,
            width: region_width as f32,
            height: region_height as f32,
        };

        // Render each layer
        for layer in &vector_data.layers {
            // Skip layers that don't intersect with our region
            if !layer.bounds.intersects(&region_bounds) {
                continue;
            }

            // Create a temporary image for this layer
            let mut layer_image = image::RgbaImage::new(region_width, region_height);

            // Render each path in the layer
            for path in &layer.paths {
                if path.bounds.intersects(&region_bounds) {
                    self.render_vector_path(&mut layer_image, path, region_x, region_y, scale)?;
                }
            }

            // Composite the layer onto the target using the layer's blend mode and opacity
            self.composite_layer(target, &layer_image, layer)?;
        }

        Ok(())
    }

    /// Render a single vector path
    fn render_vector_path(
        &self,
        target: &mut image::RgbaImage,
        path: &crate::vector::VectorPath,
        offset_x: u32,
        offset_y: u32,
        scale: f32,
    ) -> Result<()> {
        // Simple rasterization - this is a basic implementation
        // In a real implementation, you'd use a proper vector graphics library

        if let Some(fill_color) = path.fill {
            // Convert fill color to pixel
            let pixel = image::Rgba([
                (fill_color[0] * 255.0) as u8,
                (fill_color[1] * 255.0) as u8,
                (fill_color[2] * 255.0) as u8,
                (fill_color[3] * 255.0) as u8,
            ]);

            // Simple path rendering - just fill the bounding box for now
            // This is a placeholder; real vector rendering would trace the actual path
            let scaled_bounds = crate::vector::BoundingBox {
                x: (path.bounds.x - offset_x as f32) * scale,
                y: (path.bounds.y - offset_y as f32) * scale,
                width: path.bounds.width * scale,
                height: path.bounds.height * scale,
            };

            let start_x = scaled_bounds.x.max(0.0) as u32;
            let start_y = scaled_bounds.y.max(0.0) as u32;
            let end_x = (scaled_bounds.x + scaled_bounds.width).min(target.width() as f32) as u32;
            let end_y = (scaled_bounds.y + scaled_bounds.height).min(target.height() as f32) as u32;

            for y in start_y..end_y {
                for x in start_x..end_x {
                    if x < target.width() && y < target.height() {
                        target.put_pixel(x, y, pixel);
                    }
                }
            }
        }

        Ok(())
    }

    /// Composite a layer onto the target image
    fn composite_layer(
        &self,
        target: &mut image::RgbaImage,
        layer_image: &image::RgbaImage,
        layer: &crate::vector::VectorLayer,
    ) -> Result<()> {
        // Simple alpha blending for now
        for y in 0..target.height() {
            for x in 0..target.width() {
                let layer_pixel = layer_image.get_pixel(x, y);
                let target_pixel = target.get_pixel(x, y);

                if layer_pixel[3] > 0 {
                    // Apply layer opacity
                    let alpha = (layer_pixel[3] as f32 / 255.0) * layer.opacity;

                    let new_pixel = image::Rgba([
                        ((layer_pixel[0] as f32 * alpha) + (target_pixel[0] as f32 * (1.0 - alpha)))
                            as u8,
                        ((layer_pixel[1] as f32 * alpha) + (target_pixel[1] as f32 * (1.0 - alpha)))
                            as u8,
                        ((layer_pixel[2] as f32 * alpha) + (target_pixel[2] as f32 * (1.0 - alpha)))
                            as u8,
                        target_pixel[3].max(layer_pixel[3]),
                    ]);

                    target.put_pixel(x, y, new_pixel);
                }
            }
        }

        Ok(())
    }

    /// GPU rendering implementation
    #[cfg(feature = "gpu")]
    fn render_gpu(
        &self,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        zoom_level: u8,
        decoder: &mut crate::decoder::Decoder<impl std::io::Read + std::io::Seek>,
    ) -> Result<image::RgbaImage> {
        let gpu_state = self
            .gpu_state
            .as_ref()
            .ok_or_else(|| VeraError::GpuError("GPU state not initialized".to_string()))?;

        // Create render target texture
        let target_texture = gpu_state.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Render Target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let target_view = target_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create command encoder
        let mut encoder =
            gpu_state
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("VeRA Render Encoder"),
                });

        // Render raster tiles first
        self.render_raster_tiles_gpu(
            &mut encoder,
            &target_view,
            width,
            height,
            x,
            y,
            zoom_level,
            decoder,
            gpu_state,
        )?;

        // Render vector layers on top
        self.render_vector_layers_gpu(
            &mut encoder,
            &target_view,
            width,
            height,
            x,
            y,
            zoom_level,
            decoder,
            gpu_state,
        )?;

        // Submit commands
        gpu_state.queue.submit(std::iter::once(encoder.finish()));

        // Read back the result
        self.read_texture_to_image(&target_texture, width, height, gpu_state)
    }

    #[cfg(feature = "gpu")]
    fn render_raster_tiles_gpu(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        width: u32,
        height: u32,
        region_x: u32,
        region_y: u32,
        zoom_level: u8,
        decoder: &mut crate::decoder::Decoder<impl std::io::Read + std::io::Seek>,
        gpu_state: &GpuState,
    ) -> Result<()> {
        // Get intersecting tiles
        let tiles =
            decoder.get_intersecting_tiles(region_x, region_y, width, height, zoom_level)?;

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Raster Tiles Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        render_pass.set_pipeline(&gpu_state.raster_pipeline);

        // Render each tile
        for (tile_x, tile_y) in tiles {
            self.render_single_tile_gpu(
                &mut render_pass,
                tile_x,
                tile_y,
                zoom_level,
                region_x,
                region_y,
                width,
                height,
                decoder,
                gpu_state,
            )?;
        }

        Ok(())
    }

    #[cfg(feature = "gpu")]
    fn render_single_tile_gpu(
        &self,
        render_pass: &mut wgpu::RenderPass,
        tile_x: u32,
        tile_y: u32,
        zoom_level: u8,
        region_x: u32,
        region_y: u32,
        width: u32,
        height: u32,
        decoder: &mut crate::decoder::Decoder<impl std::io::Read + std::io::Seek>,
        gpu_state: &GpuState,
    ) -> Result<()> {
        // Load tile texture to GPU
        let tile_image = decoder.decode_tile(zoom_level, tile_x, tile_y)?;
        let tile_texture = self.upload_image_to_texture(&tile_image, gpu_state)?;

        // Create vertices for tile quad
        let tile_size = decoder.tile_size()?;
        let scale = 1 << zoom_level;
        let tile_pixel_x = tile_x * tile_size * scale;
        let tile_pixel_y = tile_y * tile_size * scale;

        // Calculate normalized coordinates
        let norm_x = (tile_pixel_x as f32 - region_x as f32) / width as f32 * 2.0 - 1.0;
        let norm_y = (tile_pixel_y as f32 - region_y as f32) / height as f32 * 2.0 - 1.0;
        let norm_w = (tile_size * scale) as f32 / width as f32 * 2.0;
        let norm_h = (tile_size * scale) as f32 / height as f32 * 2.0;

        let vertices = [
            Vertex {
                position: [norm_x, norm_y, 0.0],
                tex_coords: [0.0, 0.0],
                color: [1.0, 1.0, 1.0, 1.0],
            },
            Vertex {
                position: [norm_x + norm_w, norm_y, 0.0],
                tex_coords: [1.0, 0.0],
                color: [1.0, 1.0, 1.0, 1.0],
            },
            Vertex {
                position: [norm_x, norm_y + norm_h, 0.0],
                tex_coords: [0.0, 1.0],
                color: [1.0, 1.0, 1.0, 1.0],
            },
            Vertex {
                position: [norm_x + norm_w, norm_y + norm_h, 0.0],
                tex_coords: [1.0, 1.0],
                color: [1.0, 1.0, 1.0, 1.0],
            },
        ];

        let vertex_buffer =
            gpu_state
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Tile Vertex Buffer"),
                    contents: bytemuck::cast_slice(&vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });

        let indices: &[u16] = &[0, 1, 2, 2, 1, 3];
        let index_buffer = gpu_state
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tile Index Buffer"),
                contents: bytemuck::cast_slice(indices),
                usage: wgpu::BufferUsages::INDEX,
            });

        // Create bind group for this tile
        let tile_bind_group = self.create_tile_bind_group(&tile_texture, gpu_state)?;

        // Render the tile
        render_pass.set_bind_group(0, &tile_bind_group, &[]);
        render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        render_pass.draw_indexed(0..indices.len() as u32, 0, 0..1);

        Ok(())
    }

    #[cfg(feature = "gpu")]
    fn render_vector_layers_gpu(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        width: u32,
        height: u32,
        region_x: u32,
        region_y: u32,
        zoom_level: u8,
        decoder: &mut crate::decoder::Decoder<impl std::io::Read + std::io::Seek>,
        gpu_state: &GpuState,
    ) -> Result<()> {
        // Get vector data
        let vector_bytes = decoder.vector_data()?;
        if vector_bytes.is_empty() {
            return Ok(());
        }

        let metadata = decoder.metadata()?;
        let vector_data = crate::vector::VectorData::from_bytes(
            &vector_bytes,
            &metadata.compression.vector_compression,
        )?;

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Vector Layers Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        render_pass.set_pipeline(&gpu_state.vector_pipeline);

        // Render vector layers using tessellation
        for layer in &vector_data.layers {
            self.render_vector_layer_gpu(
                &mut render_pass,
                layer,
                region_x,
                region_y,
                width,
                height,
                zoom_level,
                gpu_state,
            )?;
        }

        Ok(())
    }

    #[cfg(feature = "gpu")]
    fn render_vector_layer_gpu(
        &self,
        render_pass: &mut wgpu::RenderPass,
        layer: &crate::vector::VectorLayer,
        region_x: u32,
        region_y: u32,
        width: u32,
        height: u32,
        zoom_level: u8,
        gpu_state: &GpuState,
    ) -> Result<()> {
        use crate::vector::BoundingBox;

        let scale = 1.0 / (1 << zoom_level) as f32;
        let region_bounds = BoundingBox {
            x: region_x as f32,
            y: region_y as f32,
            width: width as f32,
            height: height as f32,
        };

        if !layer.bounds.intersects(&region_bounds) {
            return Ok(());
        }

        // Tessellate vector paths to GPU vertices
        let vertices =
            self.tessellate_vector_paths(&layer.paths, region_x, region_y, width, height, scale)?;

        if vertices.is_empty() {
            return Ok(());
        }

        let vertex_buffer =
            gpu_state
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Vector Vertex Buffer"),
                    contents: bytemuck::cast_slice(&vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });

        // Create uniforms for vector rendering
        let vector_uniforms = VectorUniforms {
            transform: self.create_transform_matrix(region_x, region_y, width, height, scale),
            color: layer
                .paths
                .first()
                .and_then(|p| p.fill)
                .unwrap_or([1.0, 1.0, 1.0, 1.0]),
            stroke_width: layer.paths.first().map(|p| p.stroke_width).unwrap_or(1.0),
            opacity: layer.opacity,
            blend_mode: 0, // Normal blend mode
            _padding: 0,
        };

        let uniform_buffer =
            gpu_state
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Vector Uniforms"),
                    contents: bytemuck::cast_slice(&[vector_uniforms]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let bind_group = gpu_state
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Vector Bind Group"),
                layout: &gpu_state.bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                }],
            });

        render_pass.set_bind_group(0, &bind_group, &[]);
        render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        render_pass.draw(0..vertices.len() as u32, 0..1);

        Ok(())
    }

    /// Get current rendering backend
    pub fn backend(&self) -> RenderingBackend {
        self.backend
    }

    // GPU helper methods
    #[cfg(feature = "gpu")]
    fn create_vector_pipeline(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Result<wgpu::RenderPipeline> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Vector Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_shaders.rs").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Vector Pipeline Layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Vector Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x3,
                        },
                        wgpu::VertexAttribute {
                            offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                        wgpu::VertexAttribute {
                            offset: std::mem::size_of::<[f32; 5]>() as wgpu::BufferAddress,
                            shader_location: 2,
                            format: wgpu::VertexFormat::Float32x4,
                        },
                    ],
                }],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        Ok(pipeline)
    }

    #[cfg(feature = "gpu")]
    fn create_raster_pipeline(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Result<wgpu::RenderPipeline> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Raster Shader"),
            source: wgpu::ShaderSource::Wgsl(crate::gpu_shaders::RASTER_VERTEX_SHADER.into()),
        });

        let fragment_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Raster Fragment Shader"),
            source: wgpu::ShaderSource::Wgsl(crate::gpu_shaders::RASTER_FRAGMENT_SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Raster Pipeline Layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Raster Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x3,
                        },
                        wgpu::VertexAttribute {
                            offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                        wgpu::VertexAttribute {
                            offset: std::mem::size_of::<[f32; 5]>() as wgpu::BufferAddress,
                            shader_location: 2,
                            format: wgpu::VertexFormat::Float32x4,
                        },
                    ],
                }],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &fragment_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        Ok(pipeline)
    }

    #[cfg(feature = "gpu")]
    fn create_composite_pipeline(device: &wgpu::Device) -> Result<wgpu::ComputePipeline> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Composite Shader"),
            source: wgpu::ShaderSource::Wgsl(crate::gpu_shaders::COMPOSITE_COMPUTE_SHADER.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Composite Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Composite Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Composite Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(pipeline)
    }

    #[cfg(feature = "gpu")]
    fn upload_image_to_texture(
        &self,
        image: &image::RgbaImage,
        gpu_state: &GpuState,
    ) -> Result<wgpu::Texture> {
        let (width, height) = image.dimensions();

        let texture = gpu_state.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Image Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        gpu_state.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            image.as_raw(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        Ok(texture)
    }

    #[cfg(feature = "gpu")]
    fn create_tile_bind_group(
        &self,
        texture: &wgpu::Texture,
        gpu_state: &GpuState,
    ) -> Result<wgpu::BindGroup> {
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = gpu_state.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Tile Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let uniforms = Uniforms {
            view_proj: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            resolution: [1024.0, 1024.0],
            zoom_level: 1.0,
            time: 0.0,
        };

        let uniform_buffer =
            gpu_state
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Tile Uniforms"),
                    contents: bytemuck::cast_slice(&[uniforms]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let bind_group = gpu_state
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Tile Bind Group"),
                layout: &gpu_state.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
            });

        Ok(bind_group)
    }

    #[cfg(feature = "gpu")]
    fn read_texture_to_image(
        &self,
        texture: &wgpu::Texture,
        width: u32,
        height: u32,
        gpu_state: &GpuState,
    ) -> Result<image::RgbaImage> {
        let output_buffer = gpu_state.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: (width * height * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder =
            gpu_state
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Texture Copy Encoder"),
                });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &output_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * width),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        gpu_state.queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = output_buffer.slice(..);
        let (sender, receiver) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        gpu_state.device.poll(wgpu::MaintainBase::Wait);
        futures::executor::block_on(receiver)
            .unwrap()
            .map_err(|e| VeraError::GpuError(format!("Failed to map buffer: {:?}", e)))?;

        let data = buffer_slice.get_mapped_range();
        let image = image::RgbaImage::from_raw(width, height, data.to_vec())
            .ok_or_else(|| VeraError::GpuError("Failed to create image from buffer".to_string()))?;

        drop(data);
        output_buffer.unmap();

        Ok(image)
    }

    #[cfg(feature = "gpu")]
    fn tessellate_vector_paths(
        &self,
        paths: &[crate::vector::VectorPath],
        region_x: u32,
        region_y: u32,
        width: u32,
        height: u32,
        _scale: f32,
    ) -> Result<Vec<Vertex>> {
        let mut vertices = Vec::new();

        for path in paths {
            // Simple tessellation - in production would use Lyon or similar
            if let Some(fill_color) = path.fill {
                // Create triangles for filled regions
                let bounds = &path.bounds;

                // Simple quad for now
                let x1 = (bounds.x - region_x as f32) / width as f32 * 2.0 - 1.0;
                let y1 = (bounds.y - region_y as f32) / height as f32 * 2.0 - 1.0;
                let x2 = ((bounds.x + bounds.width) - region_x as f32) / width as f32 * 2.0 - 1.0;
                let y2 = ((bounds.y + bounds.height) - region_y as f32) / height as f32 * 2.0 - 1.0;

                vertices.extend_from_slice(&[
                    Vertex {
                        position: [x1, y1, 0.0],
                        tex_coords: [0.0, 0.0],
                        color: fill_color,
                    },
                    Vertex {
                        position: [x2, y1, 0.0],
                        tex_coords: [1.0, 0.0],
                        color: fill_color,
                    },
                    Vertex {
                        position: [x1, y2, 0.0],
                        tex_coords: [0.0, 1.0],
                        color: fill_color,
                    },
                    Vertex {
                        position: [x2, y1, 0.0],
                        tex_coords: [1.0, 0.0],
                        color: fill_color,
                    },
                    Vertex {
                        position: [x2, y2, 0.0],
                        tex_coords: [1.0, 1.0],
                        color: fill_color,
                    },
                    Vertex {
                        position: [x1, y2, 0.0],
                        tex_coords: [0.0, 1.0],
                        color: fill_color,
                    },
                ]);
            }
        }

        Ok(vertices)
    }

    #[cfg(feature = "gpu")]
    fn create_transform_matrix(
        &self,
        region_x: u32,
        region_y: u32,
        width: u32,
        height: u32,
        scale: f32,
    ) -> [[f32; 4]; 4] {
        // Create a simple transform matrix
        let scale_x = 2.0 / width as f32;
        let scale_y = 2.0 / height as f32;
        let offset_x = -1.0 - region_x as f32 * scale_x;
        let offset_y = -1.0 - region_y as f32 * scale_y;

        [
            [scale_x * scale, 0.0, 0.0, 0.0],
            [0.0, scale_y * scale, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [offset_x, offset_y, 0.0, 1.0],
        ]
    }
}
