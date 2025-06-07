use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::{Result, VeraError};

/// Color space specification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ColorSpace {
    Srgb,
    DisplayP3,
    Rec2020,
    ProPhotoRgb,
    Custom { 
        name: String,
        icc_profile: Option<Vec<u8>>,
    },
}

impl Default for ColorSpace {
    fn default() -> Self {
        Self::Srgb
    }
}

/// Compression settings for different data types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionSettings {
    pub vector_compression: VectorCompression,
    pub raster_compression: RasterCompression,
}

impl Default for CompressionSettings {
    fn default() -> Self {
        Self {
            vector_compression: VectorCompression::Flate2 { level: 6 },
            raster_compression: RasterCompression::Avif { quality: 80 },
        }
    }
}

/// Vector data compression methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VectorCompression {
    None,
    Flate2 { level: u32 },
    Lz4,
}

/// Raster tile compression methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RasterCompression {
    Avif { quality: u8 },
    WebP { quality: u8, lossless: bool },
    Jpeg { quality: u8 },
    Png,
}

/// Segmentation method used for vector/raster separation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SegmentationMethod {
    EdgeDetection { threshold: f32 },
    MachineLearning { model: String },
    Manual { regions: Vec<Region> },
    Hybrid { 
        edge_threshold: f32,
        ml_model: Option<String>,
    },
}

impl Default for SegmentationMethod {
    fn default() -> Self {
        Self::EdgeDetection { threshold: 0.1 }
    }
}

/// Geometric region specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Region {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub region_type: RegionType,
}

/// Type of region content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegionType {
    Vector,
    Raster,
}

/// VeRA file metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metadata {
    /// VeRA format version
    pub version: u32,
    
    /// Original image dimensions
    pub width: u32,
    pub height: u32,
    
    /// Color space information
    pub color_space: ColorSpace,
    
    /// Tile size in pixels
    pub tile_size: u32,
    
    /// Maximum zoom level
    pub max_zoom_level: u8,
    
    /// Number of vector regions
    pub vector_regions: u32,
    
    /// Number of raster tiles
    pub raster_tiles: u32,
    
    /// Compression settings
    pub compression: CompressionSettings,
    
    /// Segmentation method used
    pub segmentation: SegmentationMethod,
    
    /// Creation timestamp (Unix timestamp)
    pub created_at: u64,
    
    /// Source image hash (for integrity checking)
    pub source_hash: String,
    
    /// Custom metadata fields
    pub custom: HashMap<String, String>,
    
    /// Encoder information
    pub encoder: EncoderInfo,
}

/// Information about the encoder used
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncoderInfo {
    pub name: String,
    pub version: String,
    pub settings: HashMap<String, String>,
}

impl Default for EncoderInfo {
    fn default() -> Self {
        Self {
            name: "vera-enc".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            settings: HashMap::new(),
        }
    }
}

impl Metadata {
    /// Create new metadata with required fields
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            version: crate::VERA_VERSION,
            width,
            height,
            color_space: ColorSpace::default(),
            tile_size: crate::DEFAULT_TILE_SIZE,
            max_zoom_level: 10,
            vector_regions: 0,
            raster_tiles: 0,
            compression: CompressionSettings::default(),
            segmentation: SegmentationMethod::default(),
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            source_hash: String::new(),
            custom: HashMap::new(),
            encoder: EncoderInfo::default(),
        }
    }

    /// Serialize metadata to CBOR bytes
    pub fn to_cbor(&self) -> Result<Vec<u8>> {
        cbor4ii::serde::to_vec(vec![], self)
            .map_err(|e| VeraError::CborError(e.to_string()))
    }

    /// Deserialize metadata from CBOR bytes
    pub fn from_cbor(data: &[u8]) -> Result<Self> {
        cbor4ii::serde::from_slice(data)
            .map_err(|e| VeraError::CborDeError(e.to_string()))
    }

    /// Validate metadata consistency
    pub fn validate(&self) -> Result<()> {
        if self.version > crate::VERA_VERSION {
            return Err(VeraError::UnsupportedVersion { 
                version: self.version 
            });
        }

        if self.width == 0 || self.height == 0 {
            return Err(VeraError::InvalidFormat(
                "Image dimensions must be greater than zero".to_string()
            ));
        }

        if self.width > crate::MAX_DIMENSION || self.height > crate::MAX_DIMENSION {
            return Err(VeraError::InvalidFormat(
                format!("Image dimensions exceed maximum of {}", crate::MAX_DIMENSION)
            ));
        }

        if self.tile_size == 0 || (self.tile_size & (self.tile_size - 1)) != 0 {
            return Err(VeraError::InvalidFormat(
                "Tile size must be a power of two and greater than zero".to_string()
            ));
        }

        if self.max_zoom_level > crate::MAX_ZOOM_LEVELS {
            return Err(VeraError::InvalidFormat(
                format!("Zoom level exceeds maximum of {}", crate::MAX_ZOOM_LEVELS)
            ));
        }

        Ok(())
    }

    /// Calculate expected number of tiles for given dimensions and zoom levels
    pub fn calculate_tile_count(&self) -> u32 {
        let mut total = 0;
        for level in 0..=self.max_zoom_level {
            let scale = 1 << level;
            let level_width = (self.width + scale - 1) / scale;
            let level_height = (self.height + scale - 1) / scale;
            let tiles_x = (level_width + self.tile_size - 1) / self.tile_size;
            let tiles_y = (level_height + self.tile_size - 1) / self.tile_size;
            total += tiles_x * tiles_y;
        }
        total
    }
}