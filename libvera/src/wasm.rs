//! WebAssembly bindings for libvera

use js_sys::{Array, Uint8Array};
use wasm_bindgen::prelude::*;

use crate::{Decoder, Result, VeraError};

/// JavaScript-friendly error type
#[wasm_bindgen]
pub struct VeraWasmError {
    message: String,
}

#[wasm_bindgen]
impl VeraWasmError {
    #[wasm_bindgen(getter)]
    pub fn message(&self) -> String {
        self.message.clone()
    }
}

impl From<VeraError> for VeraWasmError {
    fn from(error: VeraError) -> Self {
        Self {
            message: error.to_string(),
        }
    }
}

/// WASM wrapper for VeRA decoder
#[wasm_bindgen]
pub struct VeraWasmDecoder {
    decoder: Decoder<std::io::Cursor<Vec<u8>>>,
}

#[wasm_bindgen]
impl VeraWasmDecoder {
    /// Create a new decoder from byte array
    #[wasm_bindgen(constructor)]
    pub fn new(data: &[u8]) -> Result<VeraWasmDecoder, VeraWasmError> {
        let cursor = std::io::Cursor::new(data.to_vec());
        let decoder = Decoder::new(cursor)?;
        Ok(Self { decoder })
    }

    /// Get image dimensions
    #[wasm_bindgen]
    pub fn dimensions(&self) -> Result<Array, VeraWasmError> {
        let (width, height) = self.decoder.dimensions()?;
        let array = Array::new();
        array.push(&JsValue::from(width));
        array.push(&JsValue::from(height));
        Ok(array)
    }

    /// Get tile size
    #[wasm_bindgen]
    pub fn tile_size(&self) -> Result<u32, VeraWasmError> {
        let metadata = self.decoder.metadata()?;
        Ok(metadata.tile_size)
    }

    /// Get maximum zoom level
    #[wasm_bindgen]
    pub fn max_zoom_level(&self) -> Result<u8, VeraWasmError> {
        let metadata = self.decoder.metadata()?;
        Ok(metadata.max_zoom_level)
    }

    /// Decode a tile and return as RGBA bytes
    #[wasm_bindgen]
    pub fn decode_tile(&mut self, level: u8, x: u32, y: u32) -> Result<Uint8Array, VeraWasmError> {
        let image = self.decoder.decode_tile(level, x, y)?;
        let pixels = image.as_raw();
        let array = Uint8Array::new_with_length(pixels.len() as u32);
        array.copy_from(pixels);
        Ok(array)
    }

    /// Decode a region and return as RGBA bytes
    #[wasm_bindgen]
    pub fn decode_region(
        &mut self,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        level: u8,
    ) -> Result<Uint8Array, VeraWasmError> {
        let image = self.decoder.decode_region(x, y, width, height, level)?;
        let pixels = image.as_raw();
        let array = Uint8Array::new_with_length(pixels.len() as u32);
        array.copy_from(pixels);
        Ok(array)
    }

    /// Get metadata as JSON string
    #[wasm_bindgen]
    pub fn metadata_json(&self) -> Result<String, VeraWasmError> {
        let metadata = self.decoder.metadata()?;
        serde_json::to_string(metadata).map_err(|e| VeraWasmError {
            message: format!("Failed to serialize metadata: {}", e),
        })
    }
}

/// Utility functions for WASM
#[wasm_bindgen]
pub struct VeraWasmUtils;

#[wasm_bindgen]
impl VeraWasmUtils {
    /// Check if data is a valid VeRA file
    #[wasm_bindgen]
    pub fn is_vera_file(data: &[u8]) -> bool {
        if data.len() < 4 {
            return false;
        }
        &data[0..4] == crate::VERA_MAGIC
    }

    /// Get VeRA format version from file
    #[wasm_bindgen]
    pub fn get_version(data: &[u8]) -> Option<u32> {
        if data.len() < 8 {
            return None;
        }
        if &data[0..4] != crate::VERA_MAGIC {
            return None;
        }
        Some(u32::from_le_bytes([data[4], data[5], data[6], data[7]]))
    }

    /// Calculate tile coordinates for a pixel position
    #[wasm_bindgen]
    pub fn pixel_to_tile(x: u32, y: u32, level: u8, tile_size: u32) -> Array {
        let scale = 1 << level;
        let scaled_x = x / scale;
        let scaled_y = y / scale;
        let tile_x = scaled_x / tile_size;
        let tile_y = scaled_y / tile_size;

        let array = Array::new();
        array.push(&JsValue::from(tile_x));
        array.push(&JsValue::from(tile_y));
        array
    }

    /// Calculate pixel bounds for a tile
    #[wasm_bindgen]
    pub fn tile_to_pixel_bounds(tile_x: u32, tile_y: u32, level: u8, tile_size: u32) -> Array {
        let scale = 1 << level;
        let x = tile_x * tile_size * scale;
        let y = tile_y * tile_size * scale;
        let width = tile_size * scale;
        let height = tile_size * scale;

        let array = Array::new();
        array.push(&JsValue::from(x));
        array.push(&JsValue::from(y));
        array.push(&JsValue::from(width));
        array.push(&JsValue::from(height));
        array
    }
}

/// Initialize WASM module
#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}
