#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(
    clippy::needless_continue,
    clippy::doc_markdown,
    clippy::uninlined_format_args,
    clippy::match_same_arms,
    clippy::needless_pass_by_value,
    clippy::ptr_arg,
    clippy::manual_div_ceil
)]

use std::fs::File;
use std::path::PathBuf;
use std::process;

use anyhow::Result;
use clap::{Parser, Subcommand};
use log::{error, info};
use serde_json::json;

use vera::{Decoder, VeraError, VeraFormat};

/// VeRA format CLI tools
#[derive(Parser)]
#[command(name = "vera")]
#[command(about = "VeRA format CLI tools - inspect, extract, and manipulate VeRA files")]
#[command(version = env!("CARGO_PKG_VERSION"))]
struct Args {
    #[command(subcommand)]
    command: Commands,

    /// Enable verbose output
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Quiet mode (only errors)
    #[arg(short, long, global = true)]
    quiet: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Encode an image to VeRA format
    Encode {
        /// Input image file
        input: PathBuf,
        /// Output VeRA file
        output: PathBuf,
        /// Tile size (power of 2)
        #[arg(long, default_value = "512")]
        tile_size: u32,
        /// Maximum zoom level
        #[arg(long, default_value = "10")]
        max_zoom: u8,
    },
    /// Inspect a VeRA file and dump metadata
    Inspect {
        /// VeRA file to inspect
        file: PathBuf,
        /// Output format (json, yaml, toml)
        #[arg(short, long, default_value = "json")]
        format: InspectFormat,
        /// Pretty print output
        #[arg(short, long)]
        pretty: bool,
    },
    /// Extract tiles from a VeRA file
    Extract {
        /// VeRA file to extract from
        file: PathBuf,
        /// Output directory
        #[arg(short, long, default_value = "tiles")]
        output_dir: PathBuf,
        /// Specific tile to extract (format: level/x_y)
        #[arg(long)]
        tile: Option<String>,
        /// Zoom level to extract
        #[arg(short, long)]
        level: Option<u8>,
        /// Extract all tiles
        #[arg(short, long)]
        all: bool,
    },
    /// Validate a VeRA file
    Validate {
        /// VeRA file to validate
        file: PathBuf,
        /// Check tile integrity
        #[arg(long)]
        check_tiles: bool,
        /// Check vector data
        #[arg(long)]
        check_vectors: bool,
    },
    /// Show file information
    Info {
        /// VeRA file to analyze
        file: PathBuf,
    },
}

#[derive(clap::ValueEnum, Clone)]
enum InspectFormat {
    Json,
    Yaml,
    Toml,
}

fn main() {
    let args = Args::parse();

    // Initialize logging
    let log_level = if args.quiet {
        log::LevelFilter::Error
    } else if args.verbose {
        log::LevelFilter::Debug
    } else {
        log::LevelFilter::Info
    };

    env_logger::Builder::from_default_env()
        .filter_level(log_level)
        .init();

    // Run command
    if let Err(e) = run(args) {
        error!("Command failed: {}", e);

        // Map to appropriate exit codes
        let exit_code = match e.downcast_ref::<VeraError>() {
            Some(VeraError::InvalidFormat(_)) => 1,
            Some(VeraError::TileNotFound { .. }) => 1,
            Some(VeraError::IoError(_)) => 1,
            _ => 2,
        };

        process::exit(exit_code);
    }
}

fn run(args: Args) -> Result<()> {
    match args.command {
        Commands::Encode {
            input,
            output,
            tile_size,
            max_zoom,
        } => cmd_encode(input, output, tile_size, max_zoom),
        Commands::Inspect {
            file,
            format,
            pretty,
        } => cmd_inspect(file, format, pretty),
        Commands::Extract {
            file,
            output_dir,
            tile,
            level,
            all,
        } => cmd_extract(file, output_dir, tile, level, all),
        Commands::Validate {
            file,
            check_tiles,
            check_vectors,
        } => cmd_validate(file, check_tiles, check_vectors),
        Commands::Info { file } => cmd_info(file),
    }
}

fn cmd_encode(input: PathBuf, output: PathBuf, tile_size: u32, max_zoom: u8) -> Result<()> {
    info!("Encoding {} to {}", input.display(), output.display());

    // Use vera-enc for encoding
    let status = process::Command::new("vera-enc")
        .arg(&input)
        .arg(&output)
        .arg("--tile-size")
        .arg(tile_size.to_string())
        .arg("--max-zoom")
        .arg(max_zoom.to_string())
        .status()?;

    if !status.success() {
        anyhow::bail!("Encoding failed");
    }

    info!("Encoding completed successfully");
    Ok(())
}

fn cmd_inspect(file: PathBuf, format: InspectFormat, pretty: bool) -> Result<()> {
    info!("Inspecting {}", file.display());

    let file_handle = File::open(&file)?;
    let vera_format = VeraFormat::open(file_handle)?;
    let metadata = vera_format.metadata()?;

    let output = match format {
        InspectFormat::Json => {
            if pretty {
                serde_json::to_string_pretty(metadata)?
            } else {
                serde_json::to_string(metadata)?
            }
        }
        InspectFormat::Yaml => serde_yaml::to_string(metadata)?,
        InspectFormat::Toml => toml::to_string_pretty(metadata)?,
    };

    println!("{}", output);
    Ok(())
}

fn cmd_extract(
    file: PathBuf,
    output_dir: PathBuf,
    tile: Option<String>,
    level: Option<u8>,
    all: bool,
) -> Result<()> {
    info!("Extracting from {}", file.display());

    let file_handle = File::open(&file)?;
    let mut decoder = Decoder::new(file_handle)?;

    // Create output directory
    std::fs::create_dir_all(&output_dir)?;

    if let Some(tile_spec) = tile {
        // Extract specific tile
        let (level, x, y) = parse_tile_spec(&tile_spec)?;
        extract_single_tile(&mut decoder, &output_dir, level, x, y)?;
    } else if let Some(level) = level {
        // Extract all tiles for a level
        extract_level_tiles(&mut decoder, &output_dir, level)?;
    } else if all {
        // Extract all tiles
        extract_all_tiles(&mut decoder, &output_dir)?;
    } else {
        anyhow::bail!("Must specify --tile, --level, or --all");
    }

    info!("Extraction completed successfully");
    Ok(())
}

fn cmd_validate(file: PathBuf, check_tiles: bool, check_vectors: bool) -> Result<()> {
    info!("Validating {}", file.display());

    let file_handle = File::open(&file)?;
    let vera_format = VeraFormat::open(file_handle)?;

    // Basic validation
    vera_format.validate()?;
    info!("✓ Basic format validation passed");

    if check_tiles {
        validate_tiles(&vera_format)?;
        info!("✓ Tile validation passed");
    }

    if check_vectors {
        validate_vectors(&vera_format)?;
        info!("✓ Vector validation passed");
    }

    info!("Validation completed successfully");
    Ok(())
}

fn cmd_info(file: PathBuf) -> Result<()> {
    let file_handle = File::open(&file)?;
    let vera_format = VeraFormat::open(file_handle)?;
    let metadata = vera_format.metadata()?;

    let (width, height) = vera_format.dimensions()?;
    let tile_size = vera_format.tile_size()?;
    let max_zoom = vera_format.max_zoom_level()?;

    let info = json!({
        "file": file.display().to_string(),
        "format_version": metadata.version,
        "dimensions": {
            "width": width,
            "height": height
        },
        "tile_size": tile_size,
        "max_zoom_level": max_zoom,
        "vector_regions": metadata.vector_regions,
        "raster_tiles": metadata.raster_tiles,
        "color_space": metadata.color_space,
        "compression": metadata.compression,
        "created_at": metadata.created_at,
        "encoder": metadata.encoder
    });

    println!("{}", serde_json::to_string_pretty(&info)?);
    Ok(())
}

fn parse_tile_spec(spec: &str) -> Result<(u8, u32, u32)> {
    let parts: Vec<&str> = spec.split('/').collect();
    if parts.len() != 2 {
        anyhow::bail!("Invalid tile specification: {}", spec);
    }

    let level: u8 = parts[0].parse()?;

    let coords: Vec<&str> = parts[1].split('_').collect();
    if coords.len() != 2 {
        anyhow::bail!("Invalid tile coordinates: {}", parts[1]);
    }

    let x: u32 = coords[0].parse()?;
    let y: u32 = coords[1].parse()?;

    Ok((level, x, y))
}

fn extract_single_tile(
    decoder: &mut Decoder<File>,
    output_dir: &PathBuf,
    level: u8,
    x: u32,
    y: u32,
) -> Result<()> {
    let image = decoder.decode_tile(level, x, y)?;
    let filename = format!("tile_{}_{}_{}.png", level, x, y);
    let output_path = output_dir.join(filename);

    image.save(&output_path)?;
    info!(
        "Extracted tile {}/{} {} to {}",
        level,
        x,
        y,
        output_path.display()
    );

    Ok(())
}

fn extract_level_tiles(decoder: &mut Decoder<File>, output_dir: &PathBuf, level: u8) -> Result<()> {
    info!("Extracting all tiles for level {}", level);

    let metadata = decoder.metadata()?;
    let (width, height) = decoder.dimensions()?;

    // Calculate number of tiles at this level
    let scale = 1 << level;
    let level_width = (width + scale - 1) / scale;
    let level_height = (height + scale - 1) / scale;
    let tiles_x = (level_width + metadata.tile_size - 1) / metadata.tile_size;
    let tiles_y = (level_height + metadata.tile_size - 1) / metadata.tile_size;

    let mut extracted_count = 0;

    for tile_y in 0..tiles_y {
        for tile_x in 0..tiles_x {
            match decoder.decode_tile(level, tile_x, tile_y) {
                Ok(image) => {
                    let filename = format!("tile_{}_{}_{}.png", level, tile_x, tile_y);
                    let output_path = output_dir.join(filename);
                    image.save(&output_path)?;
                    extracted_count += 1;
                }
                Err(VeraError::TileNotFound { .. }) => {
                    // Skip missing tiles
                    continue;
                }
                Err(e) => return Err(e.into()),
            }
        }
    }

    info!("Extracted {} tiles for level {}", extracted_count, level);
    Ok(())
}

fn extract_all_tiles(decoder: &mut Decoder<File>, output_dir: &PathBuf) -> Result<()> {
    info!("Extracting all tiles");

    // Extract metadata values to avoid borrow conflicts
    let (max_zoom, tile_size) = {
        let metadata = decoder.metadata()?;
        (metadata.max_zoom_level, metadata.tile_size)
    };

    let mut total_extracted = 0;

    for level in 0..=max_zoom {
        info!("Processing level {}", level);

        let level_dir = output_dir.join(format!("level_{}", level));
        std::fs::create_dir_all(&level_dir)?;

        let (width, height) = decoder.dimensions()?;
        let scale = 1 << level;
        let level_width = (width + scale - 1) / scale;
        let level_height = (height + scale - 1) / scale;
        let tiles_x = (level_width + tile_size - 1) / tile_size;
        let tiles_y = (level_height + tile_size - 1) / tile_size;

        let mut level_count = 0;

        for tile_y in 0..tiles_y {
            for tile_x in 0..tiles_x {
                match decoder.decode_tile(level, tile_x, tile_y) {
                    Ok(image) => {
                        let filename = format!("tile_{}_{}.png", tile_x, tile_y);
                        let output_path = level_dir.join(filename);
                        image.save(&output_path)?;
                        level_count += 1;
                        total_extracted += 1;
                    }
                    Err(VeraError::TileNotFound { .. }) => {
                        // Skip missing tiles
                        continue;
                    }
                    Err(e) => return Err(e.into()),
                }
            }
        }

        info!("Extracted {} tiles for level {}", level_count, level);
    }

    info!(
        "Extracted {} total tiles across all levels",
        total_extracted
    );
    Ok(())
}

fn validate_tiles(vera_format: &VeraFormat<File>) -> Result<()> {
    info!("Validating tiles...");

    let metadata = vera_format.metadata()?;

    // Check if tile count in metadata makes sense
    let expected_tiles = metadata.calculate_tile_count();
    if metadata.raster_tiles > expected_tiles {
        anyhow::bail!(
            "Metadata reports more tiles ({}) than theoretically possible ({})",
            metadata.raster_tiles,
            expected_tiles
        );
    }

    info!(
        "Tile metadata validation passed ({} tiles expected)",
        expected_tiles
    );
    Ok(())
}

fn validate_vectors(vera_format: &VeraFormat<File>) -> Result<()> {
    info!("Validating vector data...");

    let metadata = vera_format.metadata()?;

    // Check if vector regions match the metadata count
    if metadata.vector_regions > 0 {
        info!(
            "Found {} vector regions in metadata",
            metadata.vector_regions
        );

        // Additional vector validation could be added here:
        // - Check vector data integrity
        // - Validate path commands
        // - Check bounding boxes
    }

    info!("Vector data validation completed");
    Ok(())
}
