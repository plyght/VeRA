use vera::*;

#[test]
fn test_metadata_creation() {
    let metadata = Metadata::new(1920, 1080);
    assert_eq!(metadata.width, 1920);
    assert_eq!(metadata.height, 1080);
    assert_eq!(metadata.version, VERA_VERSION);
}

#[test]
fn test_metadata_validation() {
    let metadata = Metadata::new(1920, 1080);
    assert!(metadata.validate().is_ok());
}

#[test]
fn test_metadata_cbor_serialization() {
    let metadata = Metadata::new(1920, 1080);
    let serialized = metadata.to_cbor().expect("Failed to serialize");
    let deserialized = Metadata::from_cbor(&serialized).expect("Failed to deserialize");

    assert_eq!(metadata.width, deserialized.width);
    assert_eq!(metadata.height, deserialized.height);
    assert_eq!(metadata.version, deserialized.version);
}

#[test]
fn test_header_creation() {
    let header = container::Header::new();
    assert_eq!(header.magic, *VERA_MAGIC);
    assert_eq!(header.version, VERA_VERSION);
}

#[test]
fn test_header_validation() {
    let header = container::Header::new();
    assert!(header.validate().is_ok());
}

#[test]
fn test_vector_data_creation() {
    let vector_data = vector::VectorData::new();
    assert!(vector_data.layers.is_empty());
}

#[test]
fn test_tile_index_creation() {
    let index = container::TileIndex::new();
    assert!(index.entries.is_empty());
}

#[test]
fn test_constants() {
    assert_eq!(VERA_MAGIC, b"VERA");
    assert_eq!(VERA_VERSION, 1);
    assert_eq!(DEFAULT_TILE_SIZE, 512);
    assert_eq!(MAX_ZOOM_LEVELS, 20);
    assert_eq!(MAX_DIMENSION, 65536);
}
