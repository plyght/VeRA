use std::io::Write;

use image::GenericImageView;

use crate::error::{Result, VeraError};
use crate::metadata::{Metadata, SegmentationMethod};

/// VeRA file encoder
pub struct Encoder<W> {
    writer: W,
    metadata: Metadata,
}

impl<W: Write> Encoder<W> {
    /// Create a new encoder
    pub fn new(writer: W, width: u32, height: u32) -> Self {
        let metadata = Metadata::new(width, height);
        Self { writer, metadata }
    }

    /// Set segmentation method
    pub fn with_segmentation(mut self, method: SegmentationMethod) -> Self {
        self.metadata.segmentation = method;
        self
    }

    /// Set tile size
    pub fn with_tile_size(mut self, size: u32) -> Result<Self> {
        if size == 0 || (size & (size - 1)) != 0 {
            return Err(VeraError::EncodingError(
                "Tile size must be a power of two".to_string()
            ));
        }
        self.metadata.tile_size = size;
        Ok(self)
    }

    /// Set maximum zoom level
    pub fn with_max_zoom_level(mut self, level: u8) -> Result<Self> {
        if level > crate::MAX_ZOOM_LEVELS {
            return Err(VeraError::EncodingError(
                format!("Zoom level exceeds maximum of {}", crate::MAX_ZOOM_LEVELS)
            ));
        }
        self.metadata.max_zoom_level = level;
        Ok(self)
    }

    /// Encode an image to VeRA format
    pub fn encode(mut self, image: &image::DynamicImage) -> Result<()> {
        self.validate_input(image)?;
        
        // TODO: Implement actual encoding pipeline:
        // 1. Segment image into vector and raster regions
        // 2. Generate vector paths for vector regions
        // 3. Create tile pyramid for raster regions
        // 4. Compress and write all data to container
        
        // Placeholder: write basic header and metadata
        let header = crate::container::Header::new();
        header.write_to(&mut self.writer)?;
        
        let metadata_bytes = self.metadata.to_cbor()?;
        self.writer.write_all(&metadata_bytes)?;
        
        Ok(())
    }

    /// Validate input image
    fn validate_input(&self, image: &image::DynamicImage) -> Result<()> {
        let (width, height) = image.dimensions();
        
        if width != self.metadata.width || height != self.metadata.height {
            return Err(VeraError::EncodingError(
                "Image dimensions don't match encoder configuration".to_string()
            ));
        }
        
        if width > crate::MAX_DIMENSION || height > crate::MAX_DIMENSION {
            return Err(VeraError::EncodingError(
                format!("Image dimensions exceed maximum of {}", crate::MAX_DIMENSION)
            ));
        }
        
        Ok(())
    }
}