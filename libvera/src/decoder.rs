use std::io::{Read, Seek};

use image::GenericImageView;

use crate::error::{Result, VeraError};
use crate::format::VeraFormat;
use crate::metadata::Metadata;

/// `VeRA` file decoder
pub struct Decoder<R> {
    format: VeraFormat<R>,
}

impl<R: Read + Seek> Decoder<R> {
    /// Create a new decoder from a reader
    ///
    /// # Errors
    /// Returns an error if the format cannot be loaded
    pub fn new(reader: R) -> Result<Self> {
        let format = VeraFormat::open(reader)?;
        Ok(Self { format })
    }

    /// Get file metadata
    ///
    /// # Errors
    /// Returns an error if metadata is not available
    pub fn metadata(&self) -> Result<&Metadata> {
        self.format.metadata()
    }

    /// Get image dimensions
    ///
    /// # Errors
    /// Returns an error if dimensions cannot be determined
    pub fn dimensions(&self) -> Result<(u32, u32)> {
        self.format.dimensions()
    }

    /// Decode a specific tile
    ///
    /// # Errors
    /// Returns an error if the tile cannot be found or decoded
    pub fn decode_tile(&mut self, level: u8, x: u32, y: u32) -> Result<image::RgbaImage> {
        if !self.format.has_tile(level, x, y) {
            return Err(VeraError::TileNotFound { level, x, y });
        }

        // Split the borrow to avoid mutable/immutable borrow conflict
        let entry =
            {
                let tile_index =
                    self.format.container.tile_index.as_ref().ok_or_else(|| {
                        VeraError::CorruptedData("No tile index loaded".to_string())
                    })?;

                tile_index
                    .get_tile(level, x, y)
                    .ok_or_else(|| VeraError::TileNotFound { level, x, y })?
                    .clone()
            };

        let compressed_data = self.format.container.read_tile_data(&entry)?;

        let metadata = self.metadata()?;
        let compression = &metadata.compression.raster_compression;

        let image = {
            use crate::tiles::TileCompressor;
            TileCompressor::decompress(&compressed_data, compression)?
        };

        Ok(image)
    }

    /// Decode a region of the image at a specific zoom level
    ///
    /// # Errors
    /// Returns an error if tiles cannot be decoded or region is invalid
    pub fn decode_region(
        &mut self,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        level: u8,
    ) -> Result<image::RgbaImage> {
        let tiles = self
            .format
            .get_intersecting_tiles(x, y, width, height, level)?;

        if tiles.is_empty() {
            return Ok(image::RgbaImage::new(width, height));
        }

        let mut result_image = image::RgbaImage::new(width, height);
        let tile_size = self.format.tile_size()?;
        let scale = 1 << level;

        for (tile_x, tile_y) in tiles {
            let tile_image = self.decode_tile(level, tile_x, tile_y)?;

            // Calculate where this tile should be placed in the result image
            let tile_pixel_x = tile_x * tile_size * scale;
            let tile_pixel_y = tile_y * tile_size * scale;

            // Calculate the overlap between the tile and the requested region
            let overlay_x = x.saturating_sub(tile_pixel_x);
            let overlay_y = y.saturating_sub(tile_pixel_y);

            let dest_x = tile_pixel_x.saturating_sub(x);
            let dest_y = tile_pixel_y.saturating_sub(y);

            // Only overlay if the tile intersects with our region
            if dest_x < width && dest_y < height {
                // Create a view of the tile that fits in our region
                let crop_width = (tile_size * scale - overlay_x).min(width - dest_x);
                let crop_height = (tile_size * scale - overlay_y).min(height - dest_y);

                if crop_width > 0 && crop_height > 0 {
                    let cropped_tile = tile_image
                        .view(
                            overlay_x / scale,
                            overlay_y / scale,
                            crop_width / scale,
                            crop_height / scale,
                        )
                        .to_image();

                    // Scale the cropped tile if necessary
                    let final_tile = if scale > 1 {
                        image::DynamicImage::ImageRgba8(cropped_tile)
                            .resize_exact(
                                crop_width,
                                crop_height,
                                image::imageops::FilterType::Nearest,
                            )
                            .to_rgba8()
                    } else {
                        cropped_tile
                    };

                    image::imageops::overlay(
                        &mut result_image,
                        &final_tile,
                        i64::from(dest_x),
                        i64::from(dest_y),
                    );
                }
            }
        }

        Ok(result_image)
    }

    /// Get vector data
    ///
    /// # Errors
    /// Returns an error if vector data cannot be read or decompressed
    pub fn vector_data(&mut self) -> Result<Vec<u8>> {
        if self.format.container.header.vector_size == 0 {
            return Ok(Vec::new());
        }

        let compressed_data = self.format.container.read_vector_data()?;

        let metadata = self.metadata()?;
        let compression = &metadata.compression.vector_compression;

        let vector_data = {
            use crate::vector::VectorData;
            VectorData::from_bytes(&compressed_data, compression)?
        };

        // Return the decompressed CBOR data
        cbor4ii::serde::to_vec(vec![], &vector_data)
            .map_err(|e| VeraError::CborError(e.to_string()))
    }

    /// Check if the file is valid
    ///
    /// # Errors
    /// Returns an error if the file is corrupted or invalid
    pub fn validate(&self) -> Result<()> {
        self.format.validate()
    }
}
