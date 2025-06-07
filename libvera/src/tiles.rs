use image::{DynamicImage, RgbaImage, GenericImageView};

use crate::error::{Result, VeraError};
use crate::metadata::RasterCompression;

/// Tile data structure
#[derive(Debug, Clone)]
pub struct Tile {
    /// Zoom level
    pub level: u8,
    /// X coordinate
    pub x: u32,
    /// Y coordinate  
    pub y: u32,
    /// Tile image data
    pub image: RgbaImage,
}

impl Tile {
    /// Create a new tile
    pub fn new(level: u8, x: u32, y: u32, image: RgbaImage) -> Self {
        Self { level, x, y, image }
    }

    /// Get tile dimensions
    pub fn dimensions(&self) -> (u32, u32) {
        self.image.dimensions()
    }

    /// Check if tile is empty (all transparent pixels)
    pub fn is_empty(&self) -> bool {
        self.image.pixels().all(|p| p[3] == 0)
    }
}

/// Tile pyramid generator
pub struct TilePyramid {
    /// Base image
    base_image: DynamicImage,
    /// Tile size
    tile_size: u32,
    /// Maximum zoom level
    max_level: u8,
}

impl TilePyramid {
    /// Create a new tile pyramid generator
    pub fn new(image: DynamicImage, tile_size: u32, max_level: u8) -> Result<Self> {
        if tile_size == 0 || (tile_size & (tile_size - 1)) != 0 {
            return Err(VeraError::ImageProcessingError(
                "Tile size must be a power of two".to_string()
            ));
        }

        Ok(Self {
            base_image: image,
            tile_size,
            max_level,
        })
    }

    /// Generate all tiles for the pyramid
    pub fn generate_tiles(&self) -> Result<Vec<Tile>> {
        let mut tiles = Vec::new();

        for level in 0..=self.max_level {
            let level_tiles = self.generate_level_tiles(level)?;
            tiles.extend(level_tiles);
        }

        Ok(tiles)
    }

    /// Generate tiles for a specific level
    pub fn generate_level_tiles(&self, level: u8) -> Result<Vec<Tile>> {
        let scale = 1 << level;
        let (base_width, base_height) = self.base_image.dimensions();
        
        let level_width = base_width / scale;
        let level_height = base_height / scale;
        
        if level_width == 0 || level_height == 0 {
            return Ok(Vec::new());
        }

        let scaled_image = self.base_image.resize(
            level_width,
            level_height,
            image::imageops::FilterType::Lanczos3,
        );

        let tiles_x = (level_width + self.tile_size - 1) / self.tile_size;
        let tiles_y = (level_height + self.tile_size - 1) / self.tile_size;

        let mut tiles = Vec::new();

        for tile_y in 0..tiles_y {
            for tile_x in 0..tiles_x {
                let x = tile_x * self.tile_size;
                let y = tile_y * self.tile_size;
                
                let width = (self.tile_size).min(level_width - x);
                let height = (self.tile_size).min(level_height - y);

                let tile_image = scaled_image.crop_imm(x, y, width, height).to_rgba8();
                
                // Pad tile to full tile size if necessary
                let padded_tile = if width < self.tile_size || height < self.tile_size {
                    let mut padded = RgbaImage::new(self.tile_size, self.tile_size);
                    image::imageops::overlay(&mut padded, &tile_image, 0, 0);
                    padded
                } else {
                    tile_image
                };

                let tile = Tile::new(level, tile_x, tile_y, padded_tile);
                tiles.push(tile);
            }
        }

        Ok(tiles)
    }

    /// Get tile at specific coordinates
    pub fn get_tile(&self, level: u8, x: u32, y: u32) -> Result<Option<Tile>> {
        if level > self.max_level {
            return Ok(None);
        }

        let scale = 1 << level;
        let (base_width, base_height) = self.base_image.dimensions();
        
        let level_width = base_width / scale;
        let level_height = base_height / scale;
        
        let tiles_x = (level_width + self.tile_size - 1) / self.tile_size;
        let tiles_y = (level_height + self.tile_size - 1) / self.tile_size;

        if x >= tiles_x || y >= tiles_y {
            return Ok(None);
        }

        let tiles = self.generate_level_tiles(level)?;
        Ok(tiles.into_iter().find(|t| t.x == x && t.y == y))
    }
}

/// Tile compression utilities
pub struct TileCompressor;

impl TileCompressor {
    /// Compress a tile using the specified compression method
    pub fn compress(tile: &Tile, compression: &RasterCompression) -> Result<Vec<u8>> {
        match compression {
            RasterCompression::Png => {
                Self::compress_png(tile)
            }
            RasterCompression::Jpeg { quality } => {
                Self::compress_jpeg(tile, *quality)
            }
            RasterCompression::WebP { quality, lossless } => {
                Self::compress_webp(tile, *quality, *lossless)
            }
            RasterCompression::Avif { quality } => {
                Self::compress_avif(tile, *quality)
            }
        }
    }

    /// Decompress tile data
    pub fn decompress(data: &[u8], compression: &RasterCompression) -> Result<RgbaImage> {
        match compression {
            RasterCompression::Png => {
                Self::decompress_png(data)
            }
            RasterCompression::Jpeg { .. } => {
                Self::decompress_jpeg(data)
            }
            RasterCompression::WebP { .. } => {
                Self::decompress_webp(data)
            }
            RasterCompression::Avif { .. } => {
                Self::decompress_avif(data)
            }
        }
    }

    fn compress_png(tile: &Tile) -> Result<Vec<u8>> {
        let mut buffer = Vec::new();
        tile.image.write_to(&mut std::io::Cursor::new(&mut buffer), image::ImageFormat::Png)
            .map_err(|e| VeraError::EncodingError(format!("PNG compression failed: {}", e)))?;
        Ok(buffer)
    }

    fn compress_jpeg(tile: &Tile, quality: u8) -> Result<Vec<u8>> {
        // Convert RGBA to RGB for JPEG
        let rgb_image = image::DynamicImage::ImageRgba8(tile.image.clone()).to_rgb8();
        
        let mut buffer = Vec::new();
        let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buffer, quality);
        encoder.encode_image(&rgb_image)
            .map_err(|e| VeraError::EncodingError(format!("JPEG compression failed: {}", e)))?;
        Ok(buffer)
    }

    fn compress_webp(_tile: &Tile, _quality: u8, _lossless: bool) -> Result<Vec<u8>> {
        // TODO: Implement WebP compression
        Err(VeraError::EncodingError("WebP compression not implemented yet".to_string()))
    }

    fn compress_avif(_tile: &Tile, _quality: u8) -> Result<Vec<u8>> {
        // TODO: Implement AVIF compression  
        Err(VeraError::EncodingError("AVIF compression not implemented yet".to_string()))
    }

    fn decompress_png(data: &[u8]) -> Result<RgbaImage> {
        let image = image::load_from_memory_with_format(data, image::ImageFormat::Png)
            .map_err(|e| VeraError::DecodingError(format!("PNG decompression failed: {}", e)))?;
        Ok(image.to_rgba8())
    }

    fn decompress_jpeg(data: &[u8]) -> Result<RgbaImage> {
        let image = image::load_from_memory_with_format(data, image::ImageFormat::Jpeg)
            .map_err(|e| VeraError::DecodingError(format!("JPEG decompression failed: {}", e)))?;
        Ok(image.to_rgba8())
    }

    fn decompress_webp(_data: &[u8]) -> Result<RgbaImage> {
        // TODO: Implement WebP decompression
        Err(VeraError::DecodingError("WebP decompression not implemented yet".to_string()))
    }

    fn decompress_avif(_data: &[u8]) -> Result<RgbaImage> {
        // TODO: Implement AVIF decompression
        Err(VeraError::DecodingError("AVIF decompression not implemented yet".to_string()))
    }
}