use crate::error::{Result, VeraError};

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
    _device: wgpu::Device,
    _queue: wgpu::Queue,
    // Additional GPU state will be added here
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
        // TODO: Implement GPU rendering with correct wgpu API
        // For now, fall back to CPU rendering
        Err(VeraError::GpuError(
            "GPU rendering not implemented yet with current wgpu version".to_string(),
        ))
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
        let _gpu_state = self
            .gpu_state
            .as_ref()
            .ok_or_else(|| VeraError::GpuError("GPU state not initialized".to_string()))?;

        // For now, fall back to CPU rendering
        // Future enhancement: implement actual GPU rendering
        self.render_cpu(x, y, width, height, zoom_level, decoder)
    }

    /// Get current rendering backend
    pub fn backend(&self) -> RenderingBackend {
        self.backend
    }
}
