#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::module_name_repetitions)]

//! VeRA - A hybrid vector-raster image format for infinite zoom photography
//!
//! This library provides the core functionality for decoding and rendering VeRA format images.
//! VeRA combines vector graphics for geometric regions with high-quality raster tiles for
//! photographic content, enabling infinite zoom without visible quality loss.

pub mod container;
pub mod decoder;
pub mod encoder;
pub mod error;
pub mod format;
pub mod metadata;
pub mod renderer;
pub mod tiles;
pub mod vector;

#[cfg(feature = "wasm")]
pub mod wasm;

#[cfg(feature = "ffi")]
pub mod ffi;

pub use decoder::Decoder;
pub use encoder::Encoder;
pub use error::{Result, VeraError};
pub use format::VeraFormat;
pub use metadata::Metadata;
pub use renderer::Renderer;

/// VeRA format version
pub const VERA_VERSION: u32 = 1;

/// Magic bytes for VeRA files
pub const VERA_MAGIC: &[u8; 4] = b"VERA";

/// Maximum supported image dimensions
pub const MAX_DIMENSION: u32 = 65536;

/// Default tile size in pixels
pub const DEFAULT_TILE_SIZE: u32 = 512;

/// Maximum zoom levels supported
pub const MAX_ZOOM_LEVELS: u8 = 20;
