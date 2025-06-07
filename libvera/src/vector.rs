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
    MoveTo {
        x: f32,
        y: f32,
    },
    LineTo {
        x: f32,
        y: f32,
    },
    QuadraticTo {
        x1: f32,
        y1: f32,
        x: f32,
        y: f32,
    },
    CubicTo {
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
        x: f32,
        y: f32,
    },
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
                .max(layer.bounds.x + layer.bounds.width)
                - new_x;
            let new_height = (self.bounds.y + self.bounds.height)
                .max(layer.bounds.y + layer.bounds.height)
                - new_y;

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
                encoder
                    .write_all(&serialized)
                    .map_err(|e| VeraError::EncodingError(format!("Compression failed: {}", e)))?;
                encoder
                    .finish()
                    .map_err(|e| VeraError::EncodingError(format!("Compression failed: {}", e)))
            }
            VectorCompression::Lz4 => match lz4::block::compress(&serialized, None, false) {
                Ok(compressed) => Ok(compressed),
                Err(e) => Err(VeraError::EncodingError(format!(
                    "LZ4 compression failed: {}",
                    e
                ))),
            },
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
                decoder.read_to_end(&mut decompressed).map_err(|e| {
                    VeraError::DecodingError(format!("Decompression failed: {}", e))
                })?;
                decompressed
            }
            VectorCompression::Lz4 => match lz4::block::decompress(data, None) {
                Ok(decompressed) => decompressed,
                Err(e) => {
                    return Err(VeraError::DecodingError(format!(
                        "LZ4 decompression failed: {}",
                        e
                    )))
                }
            },
        };

        cbor4ii::serde::from_slice(&decompressed).map_err(|e| VeraError::CborDeError(e.to_string()))
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
    pub fn generate_from_image(image: &image::RgbaImage, threshold: f32) -> Result<VectorData> {
        use imageproc::edges::canny;

        let mut vector_data = VectorData::new();

        // Convert to grayscale for edge detection
        let gray_image = image::DynamicImage::ImageRgba8(image.clone()).to_luma8();

        // Apply Canny edge detection
        let low_threshold = (threshold * 255.0) as u8;
        let high_threshold = ((threshold * 2.0).min(1.0) * 255.0) as u8;
        let edges = canny(&gray_image, low_threshold as f32, high_threshold as f32);

        // Find contours and create paths
        let paths = Self::trace_contours(&edges, image)?;

        let layer = VectorLayer {
            name: "Generated".to_string(),
            opacity: 1.0,
            blend_mode: BlendMode::Normal,
            paths,
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

    /// Trace contours from edge-detected image
    fn trace_contours(
        edges: &image::GrayImage,
        original: &image::RgbaImage,
    ) -> Result<Vec<VectorPath>> {
        let mut paths = Vec::new();
        let (width, height) = edges.dimensions();

        // Simple contour tracing - find connected edge pixels
        let mut visited = vec![vec![false; height as usize]; width as usize];

        for y in 0..height {
            for x in 0..width {
                if edges.get_pixel(x, y)[0] > 0 && !visited[x as usize][y as usize] {
                    let path = Self::trace_single_contour(edges, x, y, &mut visited, original)?;
                    if path.commands.len() > 2 {
                        paths.push(path);
                    }
                }
            }
        }

        Ok(paths)
    }

    /// Trace a single contour starting from a point
    fn trace_single_contour(
        edges: &image::GrayImage,
        start_x: u32,
        start_y: u32,
        visited: &mut [Vec<bool>],
        original: &image::RgbaImage,
    ) -> Result<VectorPath> {
        let mut commands = Vec::new();
        let mut current_x = start_x;
        let mut current_y = start_y;
        let (width, height) = edges.dimensions();

        // Start the path
        commands.push(PathCommand::MoveTo {
            x: current_x as f32,
            y: current_y as f32,
        });

        let mut min_x = current_x as f32;
        let mut min_y = current_y as f32;
        let mut max_x = current_x as f32;
        let mut max_y = current_y as f32;

        visited[current_x as usize][current_y as usize] = true;

        // Follow the contour
        loop {
            let mut found_next = false;

            // Check 8-connected neighbors
            for dx in -1..=1i32 {
                for dy in -1..=1i32 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }

                    let next_x = current_x as i32 + dx;
                    let next_y = current_y as i32 + dy;

                    if next_x >= 0 && next_x < width as i32 && next_y >= 0 && next_y < height as i32
                    {
                        let nx = next_x as u32;
                        let ny = next_y as u32;

                        if edges.get_pixel(nx, ny)[0] > 0 && !visited[nx as usize][ny as usize] {
                            visited[nx as usize][ny as usize] = true;
                            current_x = nx;
                            current_y = ny;

                            commands.push(PathCommand::LineTo {
                                x: current_x as f32,
                                y: current_y as f32,
                            });

                            min_x = min_x.min(current_x as f32);
                            min_y = min_y.min(current_y as f32);
                            max_x = max_x.max(current_x as f32);
                            max_y = max_y.max(current_y as f32);

                            found_next = true;
                            break;
                        }
                    }
                }
                if found_next {
                    break;
                }
            }

            if !found_next {
                break;
            }
        }

        // Close the path if it's long enough
        if commands.len() > 3 {
            commands.push(PathCommand::Close);
        }

        // Sample color from the original image
        let sample_x = ((min_x + max_x) / 2.0) as u32;
        let sample_y = ((min_y + max_y) / 2.0) as u32;
        let color_pixel = original.get_pixel(
            sample_x.min(original.width() - 1),
            sample_y.min(original.height() - 1),
        );

        let fill_color = [
            color_pixel[0] as f32 / 255.0,
            color_pixel[1] as f32 / 255.0,
            color_pixel[2] as f32 / 255.0,
            color_pixel[3] as f32 / 255.0,
        ];

        Ok(VectorPath {
            commands,
            fill: Some(fill_color),
            stroke: None,
            stroke_width: 0.0,
            bounds: BoundingBox {
                x: min_x,
                y: min_y,
                width: max_x - min_x,
                height: max_y - min_y,
            },
        })
    }

    /// Generate vector paths using Lyon library
    pub fn generate_with_lyon(image: &image::RgbaImage, tolerance: f32) -> Result<VectorData> {
        // For now, use the basic edge detection approach
        // Future enhancement: implement proper Lyon-based tessellation
        Self::generate_from_image(image, tolerance)
    }
}
