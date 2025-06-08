use thiserror::Error;

/// Result type for VeRA operations
pub type Result<T> = std::result::Result<T, VeraError>;

/// Errors that can occur when working with VeRA files
#[derive(Error, Debug)]
pub enum VeraError {
    #[error("Invalid VeRA file format: {0}")]
    InvalidFormat(String),

    #[error("Unsupported VeRA version: {version}")]
    UnsupportedVersion { version: u32 },

    #[error("Corrupted file data: {0}")]
    CorruptedData(String),

    #[error("Invalid tile coordinates: level={level}, x={x}, y={y}")]
    InvalidTileCoordinates { level: u8, x: u32, y: u32 },

    #[error("Tile not found: level={level}, x={x}, y={y}")]
    TileNotFound { level: u8, x: u32, y: u32 },

    #[error("Vector data parsing failed: {0}")]
    VectorParsingError(String),

    #[error("Rendering failed: {0}")]
    RenderingError(String),

    #[error("GPU initialization failed: {0}")]
    GpuError(String),

    #[error("Image processing error: {0}")]
    ImageProcessingError(String),

    #[error("Encoding error: {0}")]
    EncodingError(String),

    #[error("Decoding error: {0}")]
    DecodingError(String),

    #[error("Metadata error: {0}")]
    MetadataError(String),

    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("CBOR serialization error: {0}")]
    CborError(String),

    #[error("CBOR deserialization error: {0}")]
    CborDeError(String),

    #[error("Image format error: {0}")]
    ImageError(#[from] image::ImageError),

    #[error("ML processing error: {0}")]
    MlError(String),

    #[error("Memory allocation error: {0}")]
    MemoryError(String),

    #[error("Performance measurement error: {0}")]
    PerformanceError(String),

    #[error("Plugin system error: {0}")]
    PluginError(String),

    #[error("Streaming operation error: {0}")]
    StreamingError(String),
}

impl VeraError {
    /// Returns true if this error is recoverable
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Self::TileNotFound { .. }
                | Self::InvalidTileCoordinates { .. }
                | Self::RenderingError(_)
                | Self::GpuError(_)
        )
    }

    /// Returns true if this error indicates a security issue
    pub fn is_security_issue(&self) -> bool {
        matches!(
            self,
            Self::CorruptedData(_) | Self::InvalidFormat(_) | Self::UnsupportedVersion { .. }
        )
    }
}
