//! C FFI bindings for libvera

use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_uint};
use std::ptr;

use crate::{Decoder, Result, VeraError};

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = RefCell::new(None);
}

fn set_last_error(error: &VeraError) {
    let error_message = match CString::new(format!("{}", error)) {
        Ok(msg) => msg,
        Err(_) => CString::new("Failed to format error message").unwrap(),
    };

    LAST_ERROR.with(|e| {
        *e.borrow_mut() = Some(error_message);
    });
}

/// Error codes for FFI
#[repr(C)]
pub enum VeraErrorCode {
    Success = 0,
    InvalidFormat = 1,
    UnsupportedVersion = 2,
    CorruptedData = 3,
    InvalidTileCoordinates = 4,
    TileNotFound = 5,
    VectorParsingError = 6,
    RenderingError = 7,
    GpuError = 8,
    ImageProcessingError = 9,
    EncodingError = 10,
    DecodingError = 11,
    MetadataError = 12,
    IoError = 13,
    UnknownError = 99,
}

impl From<&VeraError> for VeraErrorCode {
    fn from(error: &VeraError) -> Self {
        match error {
            VeraError::InvalidFormat(_) => Self::InvalidFormat,
            VeraError::UnsupportedVersion { .. } => Self::UnsupportedVersion,
            VeraError::CorruptedData(_) => Self::CorruptedData,
            VeraError::InvalidTileCoordinates { .. } => Self::InvalidTileCoordinates,
            VeraError::TileNotFound { .. } => Self::TileNotFound,
            VeraError::VectorParsingError(_) => Self::VectorParsingError,
            VeraError::RenderingError(_) => Self::RenderingError,
            VeraError::GpuError(_) => Self::GpuError,
            VeraError::ImageProcessingError(_) => Self::ImageProcessingError,
            VeraError::EncodingError(_) => Self::EncodingError,
            VeraError::DecodingError(_) => Self::DecodingError,
            VeraError::MetadataError(_) => Self::MetadataError,
            VeraError::IoError(_) => Self::IoError,
            _ => Self::UnknownError,
        }
    }
}

/// Opaque handle to VeRA decoder
pub struct VeraDecoder {
    _private: [u8; 0],
}

/// Create a new VeRA decoder from file path
#[no_mangle]
pub extern "C" fn vera_decoder_open(path: *const c_char) -> *mut VeraDecoder {
    if path.is_null() {
        return ptr::null_mut();
    }

    let path_str = unsafe {
        match CStr::from_ptr(path).to_str() {
            Ok(s) => s,
            Err(_) => return ptr::null_mut(),
        }
    };

    let file = match std::fs::File::open(path_str) {
        Ok(f) => f,
        Err(e) => {
            let vera_error = VeraError::IoError(e.to_string());
            set_last_error(&vera_error);
            return ptr::null_mut();
        }
    };

    let decoder = match Decoder::new(file) {
        Ok(d) => d,
        Err(e) => {
            set_last_error(&e);
            return ptr::null_mut();
        }
    };

    Box::into_raw(Box::new(decoder)) as *mut VeraDecoder
}

/// Get image dimensions
#[no_mangle]
pub extern "C" fn vera_decoder_dimensions(
    decoder: *mut VeraDecoder,
    width: *mut c_uint,
    height: *mut c_uint,
) -> VeraErrorCode {
    if decoder.is_null() || width.is_null() || height.is_null() {
        return VeraErrorCode::UnknownError;
    }

    let decoder = unsafe { &mut *(decoder as *mut Decoder<std::fs::File>) };

    match decoder.dimensions() {
        Ok((w, h)) => {
            unsafe {
                *width = w;
                *height = h;
            }
            VeraErrorCode::Success
        }
        Err(ref e) => {
            set_last_error(e);
            e.into()
        }
    }
}

/// Decode a tile
#[no_mangle]
pub extern "C" fn vera_decoder_decode_tile(
    decoder: *mut VeraDecoder,
    level: u8,
    x: c_uint,
    y: c_uint,
    output: *mut u8,
    output_size: c_uint,
) -> VeraErrorCode {
    if decoder.is_null() || output.is_null() {
        return VeraErrorCode::UnknownError;
    }

    let decoder = unsafe { &mut *(decoder as *mut Decoder<std::fs::File>) };

    match decoder.decode_tile(level, x, y) {
        Ok(image) => {
            let pixels = image.as_raw();
            let required_size = pixels.len();

            if output_size < required_size as c_uint {
                return VeraErrorCode::UnknownError;
            }

            unsafe {
                ptr::copy_nonoverlapping(pixels.as_ptr(), output, required_size);
            }

            VeraErrorCode::Success
        }
        Err(ref e) => {
            set_last_error(e);
            e.into()
        }
    }
}

/// Free a VeRA decoder
#[no_mangle]
pub extern "C" fn vera_decoder_free(decoder: *mut VeraDecoder) {
    if !decoder.is_null() {
        unsafe {
            let _ = Box::from_raw(decoder as *mut Decoder<std::fs::File>);
        }
    }
}

/// Get last error message
#[no_mangle]
pub extern "C" fn vera_get_error_message() -> *const c_char {
    LAST_ERROR.with(|e| match e.borrow().as_ref() {
        Some(error_string) => error_string.as_ptr(),
        None => b"No error information available\0".as_ptr() as *const c_char,
    })
}
