#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]

//! VeRA format encoder library
//!
//! This library provides encoding functionality for the VeRA image format.

use image::GenericImageView;

pub use vera::{Encoder, Result, VeraError};

/// Re-export encoding related types from vera
pub mod encoding {
    pub use vera::metadata::{
        CompressionSettings, RasterCompression, SegmentationMethod, VectorCompression,
    };
    pub use vera::tiles::{TileCompressor, TilePyramid};
    pub use vera::vector::VectorGenerator;
}

/// Convenience function to encode an image to VeRA format
pub fn encode_image_to_vera<W: std::io::Write>(
    writer: W,
    image: &image::DynamicImage,
    tile_size: u32,
    max_zoom: u8,
) -> Result<()> {
    let (width, height) = image.dimensions();
    let encoder = Encoder::new(writer, width, height)
        .with_tile_size(tile_size)?
        .with_max_zoom_level(max_zoom)?;

    encoder.encode(image)
}
