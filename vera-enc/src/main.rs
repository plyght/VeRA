#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(
    clippy::doc_markdown,
    clippy::uninlined_format_args,
    clippy::match_same_arms,
    clippy::needless_pass_by_value,
    clippy::cast_precision_loss
)]

use std::path::PathBuf;
use std::process;

use anyhow::Result;
use clap::{Parser, ValueEnum};
use image::GenericImageView;
use indicatif::{ProgressBar, ProgressStyle};
use log::{error, info, warn};

use vera::{Encoder, VeraError};

/// VeRA format encoder
#[derive(Parser)]
#[command(name = "vera-enc")]
#[command(about = "VeRA format encoder - converts images to VeRA format")]
#[command(version = env!("CARGO_PKG_VERSION"))]
struct Args {
    /// Input image file
    #[arg(value_name = "INPUT")]
    input: PathBuf,

    /// Output VeRA file
    #[arg(value_name = "OUTPUT")]
    output: PathBuf,

    /// Tile size (must be power of 2)
    #[arg(short, long, default_value = "512")]
    tile_size: u32,

    /// Maximum zoom level
    #[arg(short, long, default_value = "10")]
    max_zoom: u8,

    /// Segmentation method
    #[arg(short, long, default_value = "edge-detection")]
    segmentation: SegmentationMode,

    /// Edge detection threshold (0.0-1.0)
    #[arg(long, default_value = "0.1")]
    edge_threshold: f32,

    /// Vector compression quality (0-100)
    #[arg(long, default_value = "80")]
    vector_quality: u8,

    /// Raster compression quality (0-100)
    #[arg(long, default_value = "80")]
    raster_quality: u8,

    /// Raster compression format
    #[arg(long, default_value = "avif")]
    raster_format: RasterFormat,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Quiet mode (only errors)
    #[arg(short, long)]
    quiet: bool,

    /// Number of threads for processing
    #[arg(short = 'j', long, default_value = "0")]
    threads: usize,
}

#[derive(ValueEnum, Clone)]
enum SegmentationMode {
    EdgeDetection,
    Manual,
    Hybrid,
}

#[derive(ValueEnum, Clone)]
enum RasterFormat {
    Avif,
    WebP,
    Jpeg,
    Png,
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

    // Run encoder
    if let Err(e) = run(args) {
        error!("Encoding failed: {}", e);

        // Map to appropriate exit codes
        let exit_code = match e.downcast_ref::<VeraError>() {
            Some(VeraError::InvalidFormat(_)) => 1,
            Some(VeraError::ImageError(_)) => 1,
            Some(VeraError::IoError(_)) => 1,
            _ => 2,
        };

        process::exit(exit_code);
    }
}

fn run(args: Args) -> Result<()> {
    info!("VeRA Encoder v{}", env!("CARGO_PKG_VERSION"));
    info!("Input: {}", args.input.display());
    info!("Output: {}", args.output.display());

    // Validate arguments
    validate_args(&args)?;

    // Set up progress reporting
    let progress = if args.quiet {
        None
    } else {
        Some(create_progress_bar())
    };

    // Load input image
    if let Some(ref pb) = progress {
        pb.set_message("Loading input image...");
    }

    let image = image::open(&args.input)?;
    let (width, height) = image.dimensions();

    info!("Loaded image: {}x{} pixels", width, height);

    if let Some(ref pb) = progress {
        pb.set_message("Initializing encoder...");
        pb.inc(10);
    }

    // Create encoder
    let output_file = std::fs::File::create(&args.output)?;
    let mut encoder = Encoder::new(output_file, width, height)
        .with_tile_size(args.tile_size)?
        .with_max_zoom_level(args.max_zoom)?;

    // Configure segmentation
    let segmentation = match args.segmentation {
        SegmentationMode::EdgeDetection => vera::metadata::SegmentationMethod::EdgeDetection {
            threshold: args.edge_threshold,
        },
        SegmentationMode::Manual => vera::metadata::SegmentationMethod::Manual {
            regions: vera::metadata::ManualRegions {
                vector_regions: Vec::new(),
                raster_regions: Vec::new(),
            },
        },
        SegmentationMode::Hybrid => vera::metadata::SegmentationMethod::Hybrid {
            edge_threshold: args.edge_threshold,
            ml_model: None,
        },
    };

    encoder = encoder.with_segmentation(segmentation);

    if let Some(ref pb) = progress {
        pb.set_message("Encoding image...");
        pb.inc(20);
    }

    // Encode image
    encoder.encode(&image)?;

    if let Some(ref pb) = progress {
        pb.finish_with_message("Encoding complete!");
    }

    let output_size = std::fs::metadata(&args.output)?.len();
    let input_size = std::fs::metadata(&args.input)?.len();
    let compression_ratio = input_size as f64 / output_size as f64;

    info!(
        "Output file: {} ({} bytes)",
        args.output.display(),
        output_size
    );
    info!("Compression ratio: {:.2}x", compression_ratio);
    info!("Encoding completed successfully");

    Ok(())
}

fn validate_args(args: &Args) -> Result<()> {
    // Check input file exists
    if !args.input.exists() {
        anyhow::bail!("Input file does not exist: {}", args.input.display());
    }

    // Check output directory exists
    if let Some(parent) = args.output.parent() {
        if !parent.exists() {
            anyhow::bail!("Output directory does not exist: {}", parent.display());
        }
    }

    // Validate tile size
    if args.tile_size == 0 || (args.tile_size & (args.tile_size - 1)) != 0 {
        anyhow::bail!("Tile size must be a power of 2, got {}", args.tile_size);
    }

    if args.tile_size < 64 || args.tile_size > 2048 {
        anyhow::bail!(
            "Tile size must be between 64 and 2048, got {}",
            args.tile_size
        );
    }

    // Validate zoom level
    if args.max_zoom > vera::MAX_ZOOM_LEVELS {
        anyhow::bail!(
            "Maximum zoom level is {}, got {}",
            vera::MAX_ZOOM_LEVELS,
            args.max_zoom
        );
    }

    // Validate edge threshold
    if !(0.0..=1.0).contains(&args.edge_threshold) {
        anyhow::bail!(
            "Edge threshold must be between 0.0 and 1.0, got {}",
            args.edge_threshold
        );
    }

    // Validate quality settings
    if args.vector_quality > 100 {
        anyhow::bail!(
            "Vector quality must be between 0 and 100, got {}",
            args.vector_quality
        );
    }

    if args.raster_quality > 100 {
        anyhow::bail!(
            "Raster quality must be between 0 and 100, got {}",
            args.raster_quality
        );
    }

    // Warn about conflicting flags
    if args.quiet && args.verbose {
        warn!("Both --quiet and --verbose specified, using --quiet");
    }

    Ok(())
}

fn create_progress_bar() -> ProgressBar {
    let pb = ProgressBar::new(100);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
            .expect("Failed to create progress bar template")
            .progress_chars("##-"),
    );
    pb
}
