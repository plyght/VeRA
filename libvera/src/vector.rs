use serde::{Deserialize, Serialize};

use crate::error::{Result, VeraError};
use crate::metadata::VectorCompression;

/// Vector path data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorPath {
    /// Path commands (move, line, curve, etc.)
    pub commands: Vec<PathCommand>,
    /// Fill color (RGBA)
    pub fill: Option<[f32; 4]>,
    /// Stroke color (RGBA)
    pub stroke: Option<[f32; 4]>,
    /// Stroke width
    pub stroke_width: f32,
    /// Bounding box
    pub bounds: BoundingBox,
}

/// Path command types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PathCommand {
    MoveTo { x: f32, y: f32 },
    LineTo { x: f32, y: f32 },
    QuadraticTo { x1: f32, y1: f32, x: f32, y: f32 },
    CubicTo { x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32 },
    Close,
}

/// Bounding box
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl BoundingBox {
    /// Check if this bounding box intersects with another
    pub fn intersects(&self, other: &BoundingBox) -> bool {
        self.x < other.x + other.width
            && self.x + self.width > other.x
            && self.y < other.y + other.height
            && self.y + self.height > other.y
    }

    /// Check if this bounding box contains a point
    pub fn contains_point(&self, x: f32, y: f32) -> bool {
        x >= self.x && x <= self.x + self.width && y >= self.y && y <= self.y + self.height
    }
}

/// Vector layer containing multiple paths
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorLayer {
    /// Layer name
    pub name: String,
    /// Layer opacity (0.0 - 1.0)
    pub opacity: f32,
    /// Blend mode
    pub blend_mode: BlendMode,
    /// Vector paths in this layer
    pub paths: Vec<VectorPath>,
    /// Layer bounding box
    pub bounds: BoundingBox,
}

/// Blend modes for vector layers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BlendMode {
    Normal,
    Multiply,
    Screen,
    Overlay,
    SoftLight,
    HardLight,
    ColorDodge,
    ColorBurn,
    Darken,
    Lighten,
    Difference,
    Exclusion,
}

impl Default for BlendMode {
    fn default() -> Self {
        Self::Normal
    }
}

/// Complete vector data for a VeRA file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorData {
    /// Vector layers
    pub layers: Vec<VectorLayer>,
    /// Total bounding box
    pub bounds: BoundingBox,
}

impl VectorData {
    /// Create new empty vector data
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            bounds: BoundingBox {
                x: 0.0,
                y: 0.0,
                width: 0.0,
                height: 0.0,
            },
        }
    }

    /// Add a vector layer
    pub fn add_layer(&mut self, layer: VectorLayer) {
        // Update bounds to include new layer
        if self.layers.is_empty() {
            self.bounds = layer.bounds.clone();
        } else {
            let new_x = self.bounds.x.min(layer.bounds.x);
            let new_y = self.bounds.y.min(layer.bounds.y);
            let new_width = (self.bounds.x + self.bounds.width)
                .max(layer.bounds.x + layer.bounds.width) - new_x;
            let new_height = (self.bounds.y + self.bounds.height)
                .max(layer.bounds.y + layer.bounds.height) - new_y;
            
            self.bounds = BoundingBox {
                x: new_x,
                y: new_y,
                width: new_width,
                height: new_height,
            };
        }
        
        self.layers.push(layer);
    }

    /// Get layers that intersect with a given region
    pub fn get_intersecting_layers(&self, region: &BoundingBox) -> Vec<&VectorLayer> {
        self.layers
            .iter()
            .filter(|layer| layer.bounds.intersects(region))
            .collect()
    }

    /// Serialize to bytes with compression
    pub fn to_bytes(&self, compression: &VectorCompression) -> Result<Vec<u8>> {
        let serialized = cbor4ii::serde::to_vec(vec![], self)
            .map_err(|e| VeraError::CborError(e.to_string()))?;

        match compression {
            VectorCompression::None => Ok(serialized),
            VectorCompression::Flate2 { level } => {
                use flate2::{write::DeflateEncoder, Compression};
                use std::io::Write;

                let mut encoder = DeflateEncoder::new(Vec::new(), Compression::new(*level));
                encoder.write_all(&serialized)
                    .map_err(|e| VeraError::EncodingError(format!("Compression failed: {}", e)))?;
                encoder.finish()
                    .map_err(|e| VeraError::EncodingError(format!("Compression failed: {}", e)))
            }
            VectorCompression::Lz4 => {
                // TODO: Implement LZ4 compression
                Err(VeraError::EncodingError("LZ4 compression not implemented yet".to_string()))
            }
        }
    }

    /// Deserialize from bytes with decompression
    pub fn from_bytes(data: &[u8], compression: &VectorCompression) -> Result<Self> {
        let decompressed = match compression {
            VectorCompression::None => data.to_vec(),
            VectorCompression::Flate2 { .. } => {
                use flate2::read::DeflateDecoder;
                use std::io::Read;

                let mut decoder = DeflateDecoder::new(data);
                let mut decompressed = Vec::new();
                decoder.read_to_end(&mut decompressed)
                    .map_err(|e| VeraError::DecodingError(format!("Decompression failed: {}", e)))?;
                decompressed
            }
            VectorCompression::Lz4 => {
                // TODO: Implement LZ4 decompression
                return Err(VeraError::DecodingError("LZ4 decompression not implemented yet".to_string()));
            }
        };

        cbor4ii::serde::from_slice(&decompressed)
            .map_err(|e| VeraError::CborDeError(e.to_string()))
    }
}

impl Default for VectorData {
    fn default() -> Self {
        Self::new()
    }
}

/// Vector path generator from image regions
pub struct VectorGenerator;

impl VectorGenerator {
    /// Generate vector paths from an image region using edge detection
    pub fn generate_from_image(
        image: &image::RgbaImage,
        _threshold: f32,
    ) -> Result<VectorData> {
        // TODO: Implement actual vectorization
        // 1. Apply edge detection
        // 2. Trace contours
        // 3. Simplify paths
        // 4. Convert to vector commands
        
        // Placeholder implementation
        let mut vector_data = VectorData::new();
        
        let layer = VectorLayer {
            name: "Generated".to_string(),
            opacity: 1.0,
            blend_mode: BlendMode::Normal,
            paths: Vec::new(),
            bounds: BoundingBox {
                x: 0.0,
                y: 0.0,
                width: image.width() as f32,
                height: image.height() as f32,
            },
        };
        
        vector_data.add_layer(layer);
        Ok(vector_data)
    }

    /// Generate vector paths using Lyon library
    pub fn generate_with_lyon(
        image: &image::RgbaImage,
        _tolerance: f32,
    ) -> Result<VectorData> {
        // TODO: Implement Lyon-based vectorization
        // This would use lyon::path and lyon::algorithms for path generation
        
        Self::generate_from_image(image, 0.1)
    }
}