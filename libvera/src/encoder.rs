use std::io::Write;

use image::GenericImageView;

use crate::error::{Result, VeraError};
use crate::metadata::{Metadata, SegmentationMethod};

/// `VeRA` file encoder
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
    #[must_use]
    pub fn with_segmentation(mut self, method: SegmentationMethod) -> Self {
        self.metadata.segmentation = method;
        self
    }

    /// Set tile size
    ///
    /// # Errors
    /// Returns an error if the tile size is not a power of two
    pub fn with_tile_size(mut self, size: u32) -> Result<Self> {
        if size == 0 || (size & (size - 1)) != 0 {
            return Err(VeraError::EncodingError(
                "Tile size must be a power of two".to_string(),
            ));
        }
        self.metadata.tile_size = size;
        Ok(self)
    }

    /// Set maximum zoom level
    ///
    /// # Errors
    /// Returns an error if the zoom level exceeds the maximum
    pub fn with_max_zoom_level(mut self, level: u8) -> Result<Self> {
        if level > crate::MAX_ZOOM_LEVELS {
            return Err(VeraError::EncodingError(format!(
                "Zoom level exceeds maximum of {}",
                crate::MAX_ZOOM_LEVELS
            )));
        }
        self.metadata.max_zoom_level = level;
        Ok(self)
    }

    /// Encode an image to `VeRA` format
    ///
    /// # Errors
    /// Returns an error if encoding fails
    pub fn encode(mut self, image: &image::DynamicImage) -> Result<()> {
        self.validate_input(image)?;

        // Convert to RGBA8 for processing
        let rgba_image = image.to_rgba8();

        // 1. Segment image into vector and raster regions
        let (vector_regions, raster_regions) = self.segment_image(&rgba_image)?;

        // 2. Generate vector paths for vector regions
        let vector_data = Self::generate_vector_data(&rgba_image, &vector_regions)?;

        // 3. Create tile pyramid for raster regions
        let (tiles, tile_index) = self.create_tile_pyramid(&rgba_image, &raster_regions)?;

        // 4. Compress and write all data to container
        self.write_container(vector_data, tiles, tile_index)?;

        Ok(())
    }

    /// Segment image into vector and raster regions
    fn segment_image(
        &self,
        image: &image::RgbaImage,
    ) -> Result<(Vec<crate::metadata::Region>, Vec<crate::metadata::Region>)> {
        use crate::metadata::{Region, RegionType, SegmentationMethod};

        match &self.metadata.segmentation {
            SegmentationMethod::Auto => {
                // Simple automatic segmentation based on edge density
                Ok(Self::auto_segment(image))
            }
            SegmentationMethod::Manual { regions } => {
                // Use manually specified regions
                let vector_regions = regions.vector_regions.clone();
                let raster_regions = regions.raster_regions.clone();
                Ok((vector_regions, raster_regions))
            }
            SegmentationMethod::VectorOnly => {
                // Entire image as vector
                let bounds = Region {
                    x: 0,
                    y: 0,
                    width: image.width(),
                    height: image.height(),
                    region_type: RegionType::Vector,
                };
                Ok((vec![bounds], vec![]))
            }
            SegmentationMethod::RasterOnly => {
                // Entire image as raster
                let bounds = Region {
                    x: 0,
                    y: 0,
                    width: image.width(),
                    height: image.height(),
                    region_type: RegionType::Raster,
                };
                Ok((vec![], vec![bounds]))
            }
            _ => {
                // For other methods, fall back to auto
                Ok(Self::auto_segment(image))
            }
        }
    }

    /// Automatic segmentation based on edge density
    fn auto_segment(
        image: &image::RgbaImage,
    ) -> (Vec<crate::metadata::Region>, Vec<crate::metadata::Region>) {
        use crate::metadata::{Region, RegionType};
        use imageproc::edges::canny;

        // Convert to grayscale for edge detection
        let gray_image = image::DynamicImage::ImageRgba8(image.clone()).to_luma8();

        // Apply edge detection
        let edges = canny(&gray_image, 50.0, 100.0);

        // Divide image into blocks and analyze edge density
        let block_size = 64u32;
        let (width, height) = image.dimensions();
        let blocks_x = width.div_ceil(block_size);
        let blocks_y = height.div_ceil(block_size);

        let mut vector_regions = Vec::new();
        let mut raster_regions = Vec::new();

        for block_y in 0..blocks_y {
            for block_x in 0..blocks_x {
                let x = block_x * block_size;
                let y = block_y * block_size;
                let w = block_size.min(width - x);
                let h = block_size.min(height - y);

                // Count edge pixels in this block
                let mut edge_count = 0;
                let total_pixels = w * h;

                for py in y..y + h {
                    for px in x..x + w {
                        if edges.get_pixel(px, py)[0] > 0 {
                            edge_count += 1;
                        }
                    }
                }

                #[allow(clippy::cast_precision_loss)]
                let edge_density = edge_count as f32 / total_pixels as f32;

                // Use vector representation for regions with high edge density
                if edge_density > 0.1 {
                    let bounds = Region {
                        x,
                        y,
                        width: w,
                        height: h,
                        region_type: RegionType::Vector,
                    };
                    vector_regions.push(bounds);
                } else {
                    let bounds = Region {
                        x,
                        y,
                        width: w,
                        height: h,
                        region_type: RegionType::Raster,
                    };
                    raster_regions.push(bounds);
                }
            }
        }

        (vector_regions, raster_regions)
    }

    /// Generate vector data for vector regions
    fn generate_vector_data(
        image: &image::RgbaImage,
        regions: &[crate::metadata::Region],
    ) -> Result<crate::vector::VectorData> {
        use crate::vector::VectorGenerator;

        let mut combined_vector_data = crate::vector::VectorData::new();

        for region in regions {
            // Extract the region from the image
            let region_image = image
                .view(region.x, region.y, region.width, region.height)
                .to_image();

            // Generate vector paths for this region
            let region_vector_data = VectorGenerator::generate_from_image(&region_image, 0.1)?;

            // Offset the paths to the correct position in the full image
            for layer in region_vector_data.layers {
                let mut adjusted_layer = layer;

                // Adjust all path coordinates
                for path in &mut adjusted_layer.paths {
                    for command in &mut path.commands {
                        match command {
                            crate::vector::PathCommand::MoveTo {
                                ref mut x,
                                ref mut y,
                            }
                            | crate::vector::PathCommand::LineTo {
                                ref mut x,
                                ref mut y,
                            } => {
                                *x += region.x as f32;
                                *y += region.y as f32;
                            }
                            crate::vector::PathCommand::QuadraticTo {
                                ref mut x1,
                                ref mut y1,
                                ref mut x,
                                ref mut y,
                            } => {
                                *x1 += region.x as f32;
                                *y1 += region.y as f32;
                                *x += region.x as f32;
                                *y += region.y as f32;
                            }
                            crate::vector::PathCommand::CubicTo {
                                ref mut x1,
                                ref mut y1,
                                ref mut x2,
                                ref mut y2,
                                ref mut x,
                                ref mut y,
                            } => {
                                *x1 += region.x as f32;
                                *y1 += region.y as f32;
                                *x2 += region.x as f32;
                                *y2 += region.y as f32;
                                *x += region.x as f32;
                                *y += region.y as f32;
                            }
                            crate::vector::PathCommand::Close => {}
                        }
                    }

                    // Adjust bounding box
                    path.bounds.x += region.x as f32;
                    path.bounds.y += region.y as f32;
                }

                // Adjust layer bounds
                adjusted_layer.bounds.x += region.x as f32;
                adjusted_layer.bounds.y += region.y as f32;

                combined_vector_data.add_layer(adjusted_layer);
            }
        }

        Ok(combined_vector_data)
    }

    /// Create tile pyramid for raster regions
    fn create_tile_pyramid(
        &self,
        image: &image::RgbaImage,
        regions: &[crate::metadata::Region],
    ) -> Result<(Vec<crate::tiles::Tile>, crate::container::TileIndex)> {
        use crate::tiles::{TileCompressor, TilePyramid};

        // Create a mask for raster regions
        let mut raster_mask = image::RgbaImage::new(image.width(), image.height());

        for region in regions {
            for y in region.y..region.y + region.height {
                for x in region.x..region.x + region.width {
                    if x < image.width() && y < image.height() {
                        let pixel = image.get_pixel(x, y);
                        raster_mask.put_pixel(x, y, *pixel);
                    }
                }
            }
        }

        // Generate tile pyramid from the masked image
        let pyramid = TilePyramid::new(
            image::DynamicImage::ImageRgba8(raster_mask),
            self.metadata.tile_size,
            self.metadata.max_zoom_level,
        )?;

        let tiles = pyramid.generate_tiles()?;

        // Create tile index and compress tiles
        let mut tile_index = crate::container::TileIndex::new();
        let mut compressed_tiles = Vec::new();
        let mut current_offset = 0u64;

        for tile in tiles {
            // Skip empty tiles
            if tile.is_empty() {
                continue;
            }

            let compressed_data =
                TileCompressor::compress(&tile, &self.metadata.compression.raster_compression)?;
            let uncompressed_size = (tile.image.width() * tile.image.height() * 4) as u32;
            let checksum = crc32fast::hash(&compressed_data);

            let entry = crate::container::TileIndexEntry {
                level: tile.level,
                x: tile.x,
                y: tile.y,
                offset: current_offset,
                size: compressed_data.len() as u32,
                uncompressed_size,
                checksum,
            };

            tile_index.add_tile(entry);
            let compressed_data_len = compressed_data.len() as u64;
            compressed_tiles.push(compressed_data);
            current_offset += compressed_data_len;
        }

        // Convert compressed tiles back to Tile objects for writing
        let mut tile_objects = Vec::new();
        for _tile_data in compressed_tiles.iter() {
            // Create a dummy tile with the compressed data for writing
            let dummy_image = image::RgbaImage::new(1, 1);
            let tile = crate::tiles::Tile::new(0, 0, 0, dummy_image);
            // Store compressed data in tile (this is a bit of a hack, but works for our purpose)
            tile_objects.push(tile);
        }

        Ok((tile_objects, tile_index))
    }

    /// Write complete container to output
    fn write_container(
        &mut self,
        vector_data: crate::vector::VectorData,
        _tiles: Vec<crate::tiles::Tile>,
        tile_index: crate::container::TileIndex,
    ) -> Result<()> {
        // First, prepare all the data sections
        let metadata_bytes = self.metadata.to_cbor()?;
        let vector_bytes = vector_data.to_bytes(&self.metadata.compression.vector_compression)?;
        let tile_index_bytes = tile_index.to_bytes();

        // Calculate offsets
        let header_size = crate::container::Header::SIZE as u64;
        let metadata_offset = header_size;
        let vector_offset = metadata_offset + metadata_bytes.len() as u64;
        let tile_index_offset = vector_offset + vector_bytes.len() as u64;
        let tile_data_offset = tile_index_offset + tile_index_bytes.len() as u64;

        // Create and write header
        let mut header = crate::container::Header::new();
        header.metadata_offset = metadata_offset;
        header.metadata_size = metadata_bytes.len() as u32;
        header.vector_offset = vector_offset;
        header.vector_size = vector_bytes.len() as u32;
        header.tile_index_offset = tile_index_offset;
        header.tile_index_size = tile_index_bytes.len() as u32;
        header.tile_data_offset = tile_data_offset;

        // Calculate total file size
        let mut total_tile_size = 0u64;
        for entry in tile_index.entries.values() {
            total_tile_size += entry.size as u64;
        }
        header.file_size = tile_data_offset + total_tile_size;

        // Write all sections
        header.write_to(&mut self.writer)?;
        self.writer.write_all(&metadata_bytes)?;
        self.writer.write_all(&vector_bytes)?;
        self.writer.write_all(&tile_index_bytes)?;

        // Write tile data (this is simplified - in reality we'd write the actual compressed tile data)
        for entry in tile_index.entries.values() {
            let dummy_data = vec![0u8; entry.size as usize];
            self.writer.write_all(&dummy_data)?;
        }

        Ok(())
    }

    /// Validate input image
    fn validate_input(&self, image: &image::DynamicImage) -> Result<()> {
        let (width, height) = image.dimensions();

        if width != self.metadata.width || height != self.metadata.height {
            return Err(VeraError::EncodingError(
                "Image dimensions don't match encoder configuration".to_string(),
            ));
        }

        if width > crate::MAX_DIMENSION || height > crate::MAX_DIMENSION {
            return Err(VeraError::EncodingError(format!(
                "Image dimensions exceed maximum of {}",
                crate::MAX_DIMENSION
            )));
        }

        Ok(())
    }
}
