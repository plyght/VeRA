use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom, Write};

use crate::error::{Result, VeraError};
use crate::metadata::Metadata;

/// VeRA file container layout
///
/// File structure:
/// - Header (32 bytes)
/// - Metadata section (variable size)
/// - Vector data section (variable size)
/// - Tile index (variable size)
/// - Tile data sections (variable size)
/// - Footer (32 bytes)

/// VeRA file header
#[derive(Debug, Clone)]
pub struct Header {
    /// Magic bytes: "VERA"
    pub magic: [u8; 4],
    /// Format version
    pub version: u32,
    /// Header size in bytes
    pub header_size: u32,
    /// Metadata section offset
    pub metadata_offset: u64,
    /// Metadata section size
    pub metadata_size: u32,
    /// Vector data section offset
    pub vector_offset: u64,
    /// Vector data section size
    pub vector_size: u32,
    /// Tile index offset
    pub tile_index_offset: u64,
    /// Tile index size
    pub tile_index_size: u32,
    /// Tile data offset
    pub tile_data_offset: u64,
    /// Total file size
    pub file_size: u64,
}

impl Header {
    pub const SIZE: usize = 64;

    /// Create a new header
    pub fn new() -> Self {
        Self {
            magic: *crate::VERA_MAGIC,
            version: crate::VERA_VERSION,
            header_size: Self::SIZE as u32,
            metadata_offset: Self::SIZE as u64,
            metadata_size: 0,
            vector_offset: 0,
            vector_size: 0,
            tile_index_offset: 0,
            tile_index_size: 0,
            tile_data_offset: 0,
            file_size: 0,
        }
    }

    /// Read header from a reader
    pub fn read_from<R: Read>(reader: &mut R) -> Result<Self> {
        let mut buffer = [0u8; Self::SIZE];
        reader.read_exact(&mut buffer)?;

        let magic = [buffer[0], buffer[1], buffer[2], buffer[3]];
        if magic != *crate::VERA_MAGIC {
            return Err(VeraError::InvalidFormat(format!(
                "Invalid magic bytes: {:?}",
                magic
            )));
        }

        let version = u32::from_le_bytes([buffer[4], buffer[5], buffer[6], buffer[7]]);
        let header_size = u32::from_le_bytes([buffer[8], buffer[9], buffer[10], buffer[11]]);
        let metadata_offset = u64::from_le_bytes([
            buffer[12], buffer[13], buffer[14], buffer[15], buffer[16], buffer[17], buffer[18],
            buffer[19],
        ]);
        let metadata_size = u32::from_le_bytes([buffer[20], buffer[21], buffer[22], buffer[23]]);
        let vector_offset = u64::from_le_bytes([
            buffer[24], buffer[25], buffer[26], buffer[27], buffer[28], buffer[29], buffer[30],
            buffer[31],
        ]);
        let vector_size = u32::from_le_bytes([buffer[32], buffer[33], buffer[34], buffer[35]]);
        let tile_index_offset = u64::from_le_bytes([
            buffer[36], buffer[37], buffer[38], buffer[39], buffer[40], buffer[41], buffer[42],
            buffer[43],
        ]);
        let tile_index_size = u32::from_le_bytes([buffer[44], buffer[45], buffer[46], buffer[47]]);
        let tile_data_offset = u64::from_le_bytes([
            buffer[48], buffer[49], buffer[50], buffer[51], buffer[52], buffer[53], buffer[54],
            buffer[55],
        ]);
        let file_size = u64::from_le_bytes([
            buffer[56], buffer[57], buffer[58], buffer[59], buffer[60], buffer[61], buffer[62],
            buffer[63],
        ]);

        Ok(Self {
            magic,
            version,
            header_size,
            metadata_offset,
            metadata_size,
            vector_offset,
            vector_size,
            tile_index_offset,
            tile_index_size,
            tile_data_offset,
            file_size,
        })
    }

    /// Write header to a writer
    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<()> {
        let mut buffer = [0u8; Self::SIZE];

        buffer[0..4].copy_from_slice(&self.magic);
        buffer[4..8].copy_from_slice(&self.version.to_le_bytes());
        buffer[8..12].copy_from_slice(&self.header_size.to_le_bytes());
        buffer[12..20].copy_from_slice(&self.metadata_offset.to_le_bytes());
        buffer[20..24].copy_from_slice(&self.metadata_size.to_le_bytes());
        buffer[24..32].copy_from_slice(&self.vector_offset.to_le_bytes());
        buffer[32..36].copy_from_slice(&self.vector_size.to_le_bytes());
        buffer[36..44].copy_from_slice(&self.tile_index_offset.to_le_bytes());
        buffer[44..48].copy_from_slice(&self.tile_index_size.to_le_bytes());
        buffer[48..56].copy_from_slice(&self.tile_data_offset.to_le_bytes());
        buffer[56..64].copy_from_slice(&self.file_size.to_le_bytes());

        writer.write_all(&buffer)?;
        Ok(())
    }

    /// Validate header values
    pub fn validate(&self) -> Result<()> {
        if self.magic != *crate::VERA_MAGIC {
            return Err(VeraError::InvalidFormat("Invalid magic bytes".to_string()));
        }

        if self.version > crate::VERA_VERSION {
            return Err(VeraError::UnsupportedVersion {
                version: self.version,
            });
        }

        if self.header_size != Self::SIZE as u32 {
            return Err(VeraError::InvalidFormat("Invalid header size".to_string()));
        }

        if self.metadata_offset < self.header_size as u64 {
            return Err(VeraError::InvalidFormat(
                "Metadata offset is before end of header".to_string(),
            ));
        }

        Ok(())
    }
}

impl Default for Header {
    fn default() -> Self {
        Self::new()
    }
}

/// Tile index entry
#[derive(Debug, Clone)]
pub struct TileIndexEntry {
    /// Zoom level
    pub level: u8,
    /// Tile X coordinate
    pub x: u32,
    /// Tile Y coordinate
    pub y: u32,
    /// Offset in tile data section
    pub offset: u64,
    /// Compressed tile size
    pub size: u32,
    /// Uncompressed tile size
    pub uncompressed_size: u32,
    /// CRC32 checksum
    pub checksum: u32,
}

impl TileIndexEntry {
    /// Serialize to bytes
    pub fn to_bytes(&self) -> [u8; 29] {
        let mut buffer = [0u8; 29];
        buffer[0] = self.level;
        buffer[1..5].copy_from_slice(&self.x.to_le_bytes());
        buffer[5..9].copy_from_slice(&self.y.to_le_bytes());
        buffer[9..17].copy_from_slice(&self.offset.to_le_bytes());
        buffer[17..21].copy_from_slice(&self.size.to_le_bytes());
        buffer[21..25].copy_from_slice(&self.uncompressed_size.to_le_bytes());
        buffer[25..29].copy_from_slice(&self.checksum.to_le_bytes());
        buffer
    }

    /// Deserialize from bytes
    pub fn from_bytes(buffer: &[u8; 29]) -> Self {
        Self {
            level: buffer[0],
            x: u32::from_le_bytes([buffer[1], buffer[2], buffer[3], buffer[4]]),
            y: u32::from_le_bytes([buffer[5], buffer[6], buffer[7], buffer[8]]),
            offset: u64::from_le_bytes([
                buffer[9], buffer[10], buffer[11], buffer[12], buffer[13], buffer[14], buffer[15],
                buffer[16],
            ]),
            size: u32::from_le_bytes([buffer[17], buffer[18], buffer[19], buffer[20]]),
            uncompressed_size: u32::from_le_bytes([buffer[21], buffer[22], buffer[23], buffer[24]]),
            checksum: u32::from_le_bytes([buffer[25], buffer[26], buffer[27], buffer[28]]),
        }
    }
}

/// Tile index for efficient tile lookup
#[derive(Debug, Clone)]
pub struct TileIndex {
    /// Map from (level, x, y) to tile entry
    pub entries: HashMap<(u8, u32, u32), TileIndexEntry>,
}

impl TileIndex {
    /// Create a new empty tile index
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Add a tile entry
    pub fn add_tile(&mut self, entry: TileIndexEntry) {
        self.entries.insert((entry.level, entry.x, entry.y), entry);
    }

    /// Get a tile entry
    pub fn get_tile(&self, level: u8, x: u32, y: u32) -> Option<&TileIndexEntry> {
        self.entries.get(&(level, x, y))
    }

    /// Serialize index to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buffer = Vec::new();

        let count = self.entries.len() as u32;
        buffer.extend_from_slice(&count.to_le_bytes());

        for entry in self.entries.values() {
            buffer.extend_from_slice(&entry.to_bytes());
        }

        buffer
    }

    /// Deserialize index from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < 4 {
            return Err(VeraError::CorruptedData("Tile index too small".to_string()));
        }

        let count = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let expected_size = 4 + count * 29;

        if data.len() != expected_size {
            return Err(VeraError::CorruptedData(
                "Tile index size mismatch".to_string(),
            ));
        }

        let mut index = Self::new();

        for i in 0..count {
            let offset = 4 + i * 29;
            let entry_data: [u8; 29] = data[offset..offset + 29]
                .try_into()
                .map_err(|_| VeraError::CorruptedData("Invalid tile entry".to_string()))?;

            let entry = TileIndexEntry::from_bytes(&entry_data);
            index.add_tile(entry);
        }

        Ok(index)
    }
}

impl Default for TileIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// VeRA container for reading and writing VeRA files
pub struct Container<T> {
    /// Underlying reader or writer
    pub inner: T,
    /// File header
    pub header: Header,
    /// File metadata
    pub metadata: Option<Metadata>,
    /// Tile index
    pub tile_index: Option<TileIndex>,
}

impl<R: Read + Seek> Container<R> {
    /// Open a VeRA file for reading
    pub fn open(mut reader: R) -> Result<Self> {
        let header = Header::read_from(&mut reader)?;
        header.validate()?;

        let mut container = Self {
            inner: reader,
            header,
            metadata: None,
            tile_index: None,
        };

        container.load_metadata()?;
        container.load_tile_index()?;

        Ok(container)
    }

    /// Load metadata from the file
    fn load_metadata(&mut self) -> Result<()> {
        self.inner
            .seek(SeekFrom::Start(self.header.metadata_offset))?;

        let mut buffer = vec![0u8; self.header.metadata_size as usize];
        self.inner.read_exact(&mut buffer)?;

        let metadata = Metadata::from_cbor(&buffer)?;
        metadata.validate()?;

        self.metadata = Some(metadata);
        Ok(())
    }

    /// Load tile index from the file
    fn load_tile_index(&mut self) -> Result<()> {
        self.inner
            .seek(SeekFrom::Start(self.header.tile_index_offset))?;

        let mut buffer = vec![0u8; self.header.tile_index_size as usize];
        self.inner.read_exact(&mut buffer)?;

        let tile_index = TileIndex::from_bytes(&buffer)?;
        self.tile_index = Some(tile_index);
        Ok(())
    }

    /// Read vector data
    pub fn read_vector_data(&mut self) -> Result<Vec<u8>> {
        self.inner
            .seek(SeekFrom::Start(self.header.vector_offset))?;

        let mut buffer = vec![0u8; self.header.vector_size as usize];
        self.inner.read_exact(&mut buffer)?;

        Ok(buffer)
    }

    /// Read tile data
    pub fn read_tile_data(&mut self, entry: &TileIndexEntry) -> Result<Vec<u8>> {
        let offset = self.header.tile_data_offset + entry.offset;
        self.inner.seek(SeekFrom::Start(offset))?;

        let mut buffer = vec![0u8; entry.size as usize];
        self.inner.read_exact(&mut buffer)?;

        // Verify checksum
        let computed_checksum = crc32fast::hash(&buffer);
        if computed_checksum != entry.checksum {
            return Err(VeraError::CorruptedData(
                "Tile data checksum mismatch".to_string(),
            ));
        }

        Ok(buffer)
    }
}
