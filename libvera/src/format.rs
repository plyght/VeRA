use std::io::{Read, Seek};

use crate::container::Container;
use crate::error::{Result, VeraError};
use crate::metadata::Metadata;

/// High-level interface for VeRA format files
pub struct VeraFormat<R> {
    container: Container<R>,
}

impl<R: Read + Seek> VeraFormat<R> {
    /// Open a VeRA file
    pub fn open(reader: R) -> Result<Self> {
        let container = Container::open(reader)?;
        Ok(Self { container })
    }

    /// Get file metadata
    pub fn metadata(&self) -> Result<&Metadata> {
        self.container.metadata.as_ref()
            .ok_or_else(|| VeraError::MetadataError("Metadata not loaded".to_string()))
    }

    /// Get image dimensions
    pub fn dimensions(&self) -> Result<(u32, u32)> {
        let metadata = self.metadata()?;
        Ok((metadata.width, metadata.height))
    }

    /// Get tile size
    pub fn tile_size(&self) -> Result<u32> {
        let metadata = self.metadata()?;
        Ok(metadata.tile_size)
    }

    /// Get maximum zoom level
    pub fn max_zoom_level(&self) -> Result<u8> {
        let metadata = self.metadata()?;
        Ok(metadata.max_zoom_level)
    }

    /// Check if a tile exists at the given coordinates
    pub fn has_tile(&self, level: u8, x: u32, y: u32) -> bool {
        self.container.tile_index
            .as_ref()
            .map_or(false, |index| index.get_tile(level, x, y).is_some())
    }

    /// Get available zoom levels
    pub fn zoom_levels(&self) -> Result<Vec<u8>> {
        let max_level = self.max_zoom_level()?;
        Ok((0..=max_level).collect())
    }

    /// Calculate tile coordinates for a given pixel position and zoom level
    pub fn pixel_to_tile(&self, x: u32, y: u32, level: u8) -> Result<(u32, u32)> {
        let tile_size = self.tile_size()?;
        let scale = 1 << level;
        let scaled_x = x / scale;
        let scaled_y = y / scale;
        let tile_x = scaled_x / tile_size;
        let tile_y = scaled_y / tile_size;
        Ok((tile_x, tile_y))
    }

    /// Calculate pixel bounds for a given tile
    pub fn tile_to_pixel_bounds(&self, tile_x: u32, tile_y: u32, level: u8) -> Result<(u32, u32, u32, u32)> {
        let tile_size = self.tile_size()?;
        let scale = 1 << level;
        let x = tile_x * tile_size * scale;
        let y = tile_y * tile_size * scale;
        let width = tile_size * scale;
        let height = tile_size * scale;
        Ok((x, y, width, height))
    }

    /// Get tiles that intersect with a given region
    pub fn get_intersecting_tiles(&self, x: u32, y: u32, width: u32, height: u32, level: u8) -> Result<Vec<(u32, u32)>> {
        let tile_size = self.tile_size()?;
        let scale = 1 << level;
        
        let start_tile_x = (x / scale) / tile_size;
        let start_tile_y = (y / scale) / tile_size;
        let end_tile_x = ((x + width - 1) / scale) / tile_size;
        let end_tile_y = ((y + height - 1) / scale) / tile_size;
        
        let mut tiles = Vec::new();
        for tile_y in start_tile_y..=end_tile_y {
            for tile_x in start_tile_x..=end_tile_x {
                if self.has_tile(level, tile_x, tile_y) {
                    tiles.push((tile_x, tile_y));
                }
            }
        }
        
        Ok(tiles)
    }

    /// Validate file integrity
    pub fn validate(&self) -> Result<()> {
        let metadata = self.metadata()?;
        metadata.validate()?;
        
        // Additional validation checks
        if let Some(tile_index) = &self.container.tile_index {
            for ((level, x, y), _) in &tile_index.entries {
                if *level > metadata.max_zoom_level {
                    return Err(VeraError::InvalidFormat(
                        format!("Tile at level {} exceeds maximum zoom level {}", level, metadata.max_zoom_level)
                    ));
                }
                
                let scale = 1 << level;
                let max_tiles_x = (metadata.width + scale * metadata.tile_size - 1) / (scale * metadata.tile_size);
                let max_tiles_y = (metadata.height + scale * metadata.tile_size - 1) / (scale * metadata.tile_size);
                
                if *x >= max_tiles_x || *y >= max_tiles_y {
                    return Err(VeraError::InvalidTileCoordinates {
                        level: *level,
                        x: *x,
                        y: *y,
                    });
                }
            }
        }
        
        Ok(())
    }
}