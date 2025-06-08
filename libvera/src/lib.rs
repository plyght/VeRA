#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(
    clippy::module_name_repetitions,
    clippy::missing_panics_doc,
    clippy::unnecessary_wraps,
    clippy::missing_errors_doc,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::doc_markdown,
    clippy::unnecessary_cast,
    clippy::explicit_iter_loop,
    clippy::needless_pass_by_value,
    clippy::cast_lossless,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::missing_const_for_thread_local,
    clippy::option_if_let_else,
    clippy::uninlined_format_args,
    clippy::trailing_empty_array,
    clippy::not_unsafe_ptr_arg_deref,
    clippy::ptr_as_ptr,
    clippy::unnecessary_map_or,
    clippy::for_kv_map,
    clippy::manual_div_ceil,
    clippy::derive_partial_eq_without_eq,
    clippy::redundant_closure_for_method_calls,
    clippy::large_enum_variant,
    clippy::no_effect_underscore_binding,
    clippy::use_self,
    clippy::cast_possible_wrap,
    clippy::manual_midpoint,
    clippy::too_many_arguments,
    clippy::unused_self,
    clippy::suboptimal_flops
)]

//! `VeRA` - A hybrid vector-raster image format for infinite zoom photography
//!
//! This library provides the core functionality for decoding and rendering `VeRA` format images.
//! `VeRA` combines vector graphics for geometric regions with high-quality raster tiles for
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

/// `VeRA` format version
pub const VERA_VERSION: u32 = 1;

/// Magic bytes for `VeRA` files
pub const VERA_MAGIC: &[u8; 4] = b"VERA";

/// Maximum supported image dimensions
pub const MAX_DIMENSION: u32 = 65536;

/// Default tile size in pixels
pub const DEFAULT_TILE_SIZE: u32 = 512;

/// Maximum zoom levels supported
pub const MAX_ZOOM_LEVELS: u8 = 20;
