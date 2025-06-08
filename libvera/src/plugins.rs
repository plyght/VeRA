//! VeRA format extensions and plugin system
//!
//! This module provides a flexible plugin architecture for extending VeRA capabilities:
//! - Custom compression algorithms
//! - Format extensions and new data types
//! - Processing pipelines and filters
//! - Custom renderers and output formats
//! - Third-party integrations

#[cfg(feature = "plugins")]
use {
    inventory,
    libloading::{Library, Symbol},
    std::collections::HashMap,
    std::ffi::{CStr, CString, OsStr},
    std::path::{Path, PathBuf},
    std::sync::{Arc, Mutex, RwLock},
};

use crate::error::{Result, VeraError};
use crate::metadata::{Metadata, RasterCompression, VectorCompression};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::fmt::Debug;

/// Plugin manager for VeRA format extensions
pub struct PluginManager {
    #[cfg(feature = "plugins")]
    loaded_plugins: HashMap<String, Arc<dyn Plugin>>,

    #[cfg(feature = "plugins")]
    dynamic_libraries: HashMap<String, Library>,

    registry: PluginRegistry,

    #[cfg(feature = "plugins")]
    plugin_search_paths: Vec<PathBuf>,
}

/// Registry of available plugins and their metadata
#[derive(Debug, Clone)]
pub struct PluginRegistry {
    compression_plugins: HashMap<String, CompressionPluginInfo>,
    renderer_plugins: HashMap<String, RendererPluginInfo>,
    filter_plugins: HashMap<String, FilterPluginInfo>,
    format_plugins: HashMap<String, FormatPluginInfo>,
}

/// Base trait for all VeRA plugins
pub trait Plugin: Send + Sync + Debug {
    /// Plugin identifier
    fn id(&self) -> &str;

    /// Plugin version
    fn version(&self) -> &str;

    /// Plugin description
    fn description(&self) -> &str;

    /// Plugin capabilities
    fn capabilities(&self) -> PluginCapabilities;

    /// Initialize the plugin
    fn initialize(&mut self, config: &PluginConfig) -> Result<()>;

    /// Cleanup the plugin
    fn cleanup(&mut self) -> Result<()>;

    /// Get plugin as any for downcasting
    fn as_any(&self) -> &dyn Any;
}

/// Plugin capability flags
#[derive(Debug, Clone)]
pub struct PluginCapabilities {
    pub compression: bool,
    pub rendering: bool,
    pub filtering: bool,
    pub format_extension: bool,
    pub metadata_processing: bool,
    pub streaming: bool,
    pub gpu_acceleration: bool,
}

/// Plugin configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginConfig {
    pub enabled: bool,
    pub priority: i32,
    pub settings: HashMap<String, PluginSetting>,
}

/// Plugin setting value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PluginSetting {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Array(Vec<PluginSetting>),
    Object(HashMap<String, PluginSetting>),
}

/// Compression plugin trait
pub trait CompressionPlugin: Plugin {
    /// Compress data with plugin-specific algorithm
    fn compress(&self, data: &[u8], quality: Option<u8>) -> Result<Vec<u8>>;

    /// Decompress data with plugin-specific algorithm
    fn decompress(&self, data: &[u8]) -> Result<Vec<u8>>;

    /// Get compression type identifier
    fn compression_type(&self) -> &str;

    /// Get supported quality range
    fn quality_range(&self) -> Option<(u8, u8)>;

    /// Get estimated compression ratio
    fn estimate_ratio(&self, data: &[u8]) -> f32;
}

/// Renderer plugin trait
pub trait RendererPlugin: Plugin {
    /// Render tiles with custom renderer
    fn render_tile(&self, tile_data: &[u8], width: u32, height: u32) -> Result<image::RgbaImage>;

    /// Render vector data with custom renderer
    fn render_vector(
        &self,
        vector_data: &[u8],
        width: u32,
        height: u32,
    ) -> Result<image::RgbaImage>;

    /// Get supported output formats
    fn output_formats(&self) -> Vec<String>;

    /// Check if GPU acceleration is available
    fn supports_gpu(&self) -> bool;
}

/// Filter plugin trait for image processing
pub trait FilterPlugin: Plugin {
    /// Apply filter to image data
    fn apply_filter(
        &self,
        image: &image::RgbaImage,
        params: &FilterParams,
    ) -> Result<image::RgbaImage>;

    /// Get filter parameters schema
    fn parameter_schema(&self) -> FilterParameterSchema;

    /// Preview filter effect (for UI)
    fn preview(&self, image: &image::RgbaImage, params: &FilterParams) -> Result<image::RgbaImage>;
}

/// Format extension plugin trait
pub trait FormatExtensionPlugin: Plugin {
    /// Parse custom format data
    fn parse_data(&self, data: &[u8]) -> Result<FormatExtensionData>;

    /// Serialize custom format data
    fn serialize_data(&self, data: &FormatExtensionData) -> Result<Vec<u8>>;

    /// Get extension identifier
    fn extension_id(&self) -> &str;

    /// Get extension version
    fn extension_version(&self) -> u32;
}

/// Filter parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterParams {
    pub params: HashMap<String, PluginSetting>,
}

/// Filter parameter schema definition
#[derive(Debug, Clone)]
pub struct FilterParameterSchema {
    pub parameters: Vec<ParameterDefinition>,
}

/// Parameter definition for filters
#[derive(Debug, Clone)]
pub struct ParameterDefinition {
    pub name: String,
    pub param_type: ParameterType,
    pub default_value: PluginSetting,
    pub description: String,
    pub constraints: Option<ParameterConstraints>,
}

/// Parameter type
#[derive(Debug, Clone)]
pub enum ParameterType {
    String,
    Integer,
    Float,
    Boolean,
    Choice(Vec<String>),
    Range(f64, f64),
}

/// Parameter constraints
#[derive(Debug, Clone)]
pub enum ParameterConstraints {
    Range(f64, f64),
    MinMax(f64, f64),
    StringLength(usize, usize),
    Regex(String),
}

/// Custom format extension data
#[derive(Debug, Clone)]
pub struct FormatExtensionData {
    pub extension_id: String,
    pub version: u32,
    pub data: Vec<u8>,
    pub metadata: HashMap<String, String>,
}

/// Plugin information structures
#[derive(Debug, Clone)]
pub struct CompressionPluginInfo {
    pub id: String,
    pub name: String,
    pub version: String,
    pub compression_type: String,
    pub quality_range: Option<(u8, u8)>,
    pub typical_ratio: f32,
}

#[derive(Debug, Clone)]
pub struct RendererPluginInfo {
    pub id: String,
    pub name: String,
    pub version: String,
    pub output_formats: Vec<String>,
    pub gpu_support: bool,
}

#[derive(Debug, Clone)]
pub struct FilterPluginInfo {
    pub id: String,
    pub name: String,
    pub version: String,
    pub description: String,
    pub category: String,
}

#[derive(Debug, Clone)]
pub struct FormatPluginInfo {
    pub id: String,
    pub name: String,
    pub version: String,
    pub extension_id: String,
    pub supported_versions: Vec<u32>,
}

/// Plugin discovery and loading
impl PluginManager {
    /// Create a new plugin manager
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "plugins")]
            loaded_plugins: HashMap::new(),

            #[cfg(feature = "plugins")]
            dynamic_libraries: HashMap::new(),

            registry: PluginRegistry::new(),

            #[cfg(feature = "plugins")]
            plugin_search_paths: vec![
                PathBuf::from("./plugins"),
                PathBuf::from("/usr/local/lib/vera/plugins"),
                PathBuf::from("/usr/lib/vera/plugins"),
            ],
        }
    }

    /// Add search path for plugins
    #[cfg(feature = "plugins")]
    pub fn add_search_path<P: AsRef<Path>>(&mut self, path: P) {
        self.plugin_search_paths.push(path.as_ref().to_path_buf());
    }

    /// Discover and load all available plugins
    #[cfg(feature = "plugins")]
    pub fn discover_plugins(&mut self) -> Result<()> {
        // Discover static plugins registered with inventory
        self.discover_static_plugins();

        // Discover dynamic plugins in search paths
        for search_path in &self.plugin_search_paths.clone() {
            if search_path.exists() && search_path.is_dir() {
                self.discover_dynamic_plugins(search_path)?;
            }
        }

        log::info!("Discovered {} plugins", self.loaded_plugins.len());
        Ok(())
    }

    #[cfg(feature = "plugins")]
    fn discover_static_plugins(&mut self) {
        // Use inventory to discover statically compiled plugins
        for plugin_factory in inventory::iter::<StaticPluginFactory> {
            match plugin_factory.create() {
                Ok(plugin) => {
                    let id = plugin.id().to_string();
                    self.loaded_plugins.insert(id.clone(), plugin);
                    log::info!("Loaded static plugin: {}", id);
                }
                Err(e) => {
                    log::warn!("Failed to load static plugin: {}", e);
                }
            }
        }
    }

    #[cfg(feature = "plugins")]
    fn discover_dynamic_plugins(&mut self, search_path: &Path) -> Result<()> {
        let entries = std::fs::read_dir(search_path).map_err(|e| {
            VeraError::PluginError(format!("Failed to read plugin directory: {}", e))
        })?;

        for entry in entries {
            let entry = entry.map_err(|e| {
                VeraError::PluginError(format!("Failed to read directory entry: {}", e))
            })?;
            let path = entry.path();

            if path.extension() == Some(OsStr::new("so"))
                || path.extension() == Some(OsStr::new("dll"))
                || path.extension() == Some(OsStr::new("dylib"))
            {
                self.load_dynamic_plugin(&path)?;
            }
        }

        Ok(())
    }

    #[cfg(feature = "plugins")]
    fn load_dynamic_plugin(&mut self, plugin_path: &Path) -> Result<()> {
        unsafe {
            let library = Library::new(plugin_path).map_err(|e| {
                VeraError::PluginError(format!("Failed to load plugin library: {}", e))
            })?;

            // Look for plugin creation function
            let create_plugin: Symbol<extern "C" fn() -> *mut dyn Plugin> =
                library.get(b"create_plugin").map_err(|e| {
                    VeraError::PluginError(format!("Plugin missing create_plugin function: {}", e))
                })?;

            let plugin_ptr = create_plugin();
            if plugin_ptr.is_null() {
                return Err(VeraError::PluginError(
                    "Plugin creation returned null".to_string(),
                ));
            }

            let plugin = Box::from_raw(plugin_ptr);
            let plugin_arc: Arc<dyn Plugin> = Arc::from(plugin);

            let id = plugin_arc.id().to_string();
            self.loaded_plugins.insert(id.clone(), plugin_arc);

            let library_name = plugin_path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            self.dynamic_libraries.insert(library_name, library);

            log::info!("Loaded dynamic plugin: {} from {:?}", id, plugin_path);
        }

        Ok(())
    }

    /// Get plugin by ID
    pub fn get_plugin(&self, id: &str) -> Option<Arc<dyn Plugin>> {
        #[cfg(feature = "plugins")]
        {
            self.loaded_plugins.get(id).cloned()
        }

        #[cfg(not(feature = "plugins"))]
        None
    }

    /// List all loaded plugins
    pub fn list_plugins(&self) -> Vec<String> {
        #[cfg(feature = "plugins")]
        {
            self.loaded_plugins.keys().cloned().collect()
        }

        #[cfg(not(feature = "plugins"))]
        Vec::new()
    }

    /// Get compression plugin by type
    pub fn get_compression_plugin(
        &self,
        compression_type: &str,
    ) -> Option<Arc<dyn CompressionPlugin>> {
        #[cfg(feature = "plugins")]
        {
            for plugin in self.loaded_plugins.values() {
                if plugin.capabilities().compression {
                    if let Some(compression_plugin) =
                        plugin.as_any().downcast_ref::<dyn CompressionPlugin>()
                    {
                        if compression_plugin.compression_type() == compression_type {
                            // This is a bit tricky with trait objects - in real implementation
                            // we'd need a different approach
                            return None; // Placeholder
                        }
                    }
                }
            }
        }
        None
    }

    /// Register custom compression algorithm
    pub fn register_compression(&mut self, plugin: Arc<dyn CompressionPlugin>) -> Result<()> {
        let compression_type = plugin.compression_type().to_string();

        // Register with metadata system
        self.registry.compression_plugins.insert(
            plugin.id().to_string(),
            CompressionPluginInfo {
                id: plugin.id().to_string(),
                name: format!("{} Compression", plugin.id()),
                version: plugin.version().to_string(),
                compression_type: compression_type.clone(),
                quality_range: plugin.quality_range(),
                typical_ratio: 0.5, // Would calculate from benchmarks
            },
        );

        #[cfg(feature = "plugins")]
        {
            self.loaded_plugins.insert(plugin.id().to_string(), plugin);
        }

        log::info!("Registered compression plugin: {}", compression_type);
        Ok(())
    }

    /// Apply filter plugin to image
    pub fn apply_filter(
        &self,
        filter_id: &str,
        image: &image::RgbaImage,
        params: &FilterParams,
    ) -> Result<image::RgbaImage> {
        #[cfg(feature = "plugins")]
        {
            if let Some(plugin) = self.get_plugin(filter_id) {
                if plugin.capabilities().filtering {
                    if let Some(filter_plugin) = plugin.as_any().downcast_ref::<dyn FilterPlugin>()
                    {
                        return filter_plugin.apply_filter(image, params);
                    }
                }
            }
        }

        Err(VeraError::PluginError(format!(
            "Filter plugin not found: {}",
            filter_id
        )))
    }

    /// Get plugin registry
    pub fn registry(&self) -> &PluginRegistry {
        &self.registry
    }
}

impl PluginRegistry {
    fn new() -> Self {
        Self {
            compression_plugins: HashMap::new(),
            renderer_plugins: HashMap::new(),
            filter_plugins: HashMap::new(),
            format_plugins: HashMap::new(),
        }
    }

    /// Get available compression plugins
    pub fn compression_plugins(&self) -> &HashMap<String, CompressionPluginInfo> {
        &self.compression_plugins
    }

    /// Get available renderer plugins
    pub fn renderer_plugins(&self) -> &HashMap<String, RendererPluginInfo> {
        &self.renderer_plugins
    }

    /// Get available filter plugins
    pub fn filter_plugins(&self) -> &HashMap<String, FilterPluginInfo> {
        &self.filter_plugins
    }

    /// Get available format extension plugins
    pub fn format_plugins(&self) -> &HashMap<String, FormatPluginInfo> {
        &self.format_plugins
    }
}

/// Static plugin factory for compile-time registration
pub trait StaticPluginFactory: Send + Sync {
    fn create(&self) -> Result<Arc<dyn Plugin>>;
}

// Register static plugin factories with inventory
inventory::collect!(StaticPluginFactory);

/// Example compression plugin implementation
#[derive(Debug)]
pub struct ExampleCompressionPlugin {
    id: String,
    version: String,
    initialized: bool,
}

impl ExampleCompressionPlugin {
    pub fn new() -> Self {
        Self {
            id: "example_compression".to_string(),
            version: "1.0.0".to_string(),
            initialized: false,
        }
    }
}

impl Plugin for ExampleCompressionPlugin {
    fn id(&self) -> &str {
        &self.id
    }

    fn version(&self) -> &str {
        &self.version
    }

    fn description(&self) -> &str {
        "Example compression plugin for demonstration"
    }

    fn capabilities(&self) -> PluginCapabilities {
        PluginCapabilities {
            compression: true,
            rendering: false,
            filtering: false,
            format_extension: false,
            metadata_processing: false,
            streaming: false,
            gpu_acceleration: false,
        }
    }

    fn initialize(&mut self, _config: &PluginConfig) -> Result<()> {
        self.initialized = true;
        log::info!("Initialized example compression plugin");
        Ok(())
    }

    fn cleanup(&mut self) -> Result<()> {
        self.initialized = false;
        log::info!("Cleaned up example compression plugin");
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl CompressionPlugin for ExampleCompressionPlugin {
    fn compress(&self, data: &[u8], _quality: Option<u8>) -> Result<Vec<u8>> {
        // Example compression (just return original data)
        Ok(data.to_vec())
    }

    fn decompress(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Example decompression (just return original data)
        Ok(data.to_vec())
    }

    fn compression_type(&self) -> &str {
        "example"
    }

    fn quality_range(&self) -> Option<(u8, u8)> {
        Some((1, 100))
    }

    fn estimate_ratio(&self, _data: &[u8]) -> f32 {
        1.0 // No compression in this example
    }
}

/// Example filter plugin
#[derive(Debug)]
pub struct ExampleFilterPlugin {
    id: String,
    version: String,
}

impl ExampleFilterPlugin {
    pub fn new() -> Self {
        Self {
            id: "example_filter".to_string(),
            version: "1.0.0".to_string(),
        }
    }
}

impl Plugin for ExampleFilterPlugin {
    fn id(&self) -> &str {
        &self.id
    }

    fn version(&self) -> &str {
        &self.version
    }

    fn description(&self) -> &str {
        "Example filter plugin for demonstration"
    }

    fn capabilities(&self) -> PluginCapabilities {
        PluginCapabilities {
            compression: false,
            rendering: false,
            filtering: true,
            format_extension: false,
            metadata_processing: false,
            streaming: false,
            gpu_acceleration: false,
        }
    }

    fn initialize(&mut self, _config: &PluginConfig) -> Result<()> {
        log::info!("Initialized example filter plugin");
        Ok(())
    }

    fn cleanup(&mut self) -> Result<()> {
        log::info!("Cleaned up example filter plugin");
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl FilterPlugin for ExampleFilterPlugin {
    fn apply_filter(
        &self,
        image: &image::RgbaImage,
        params: &FilterParams,
    ) -> Result<image::RgbaImage> {
        // Example filter: adjust brightness
        let brightness = params
            .params
            .get("brightness")
            .and_then(|v| match v {
                PluginSetting::Float(f) => Some(*f as f32),
                PluginSetting::Integer(i) => Some(*i as f32),
                _ => None,
            })
            .unwrap_or(0.0);

        let mut result = image.clone();
        for pixel in result.pixels_mut() {
            pixel[0] = (pixel[0] as f32 + brightness).clamp(0.0, 255.0) as u8;
            pixel[1] = (pixel[1] as f32 + brightness).clamp(0.0, 255.0) as u8;
            pixel[2] = (pixel[2] as f32 + brightness).clamp(0.0, 255.0) as u8;
        }

        Ok(result)
    }

    fn parameter_schema(&self) -> FilterParameterSchema {
        FilterParameterSchema {
            parameters: vec![ParameterDefinition {
                name: "brightness".to_string(),
                param_type: ParameterType::Range(-100.0, 100.0),
                default_value: PluginSetting::Float(0.0),
                description: "Brightness adjustment".to_string(),
                constraints: Some(ParameterConstraints::Range(-100.0, 100.0)),
            }],
        }
    }

    fn preview(&self, image: &image::RgbaImage, params: &FilterParams) -> Result<image::RgbaImage> {
        // For preview, apply filter to a smaller version
        let preview_size = 256;
        let (width, height) = image.dimensions();

        if width <= preview_size && height <= preview_size {
            self.apply_filter(image, params)
        } else {
            let aspect_ratio = width as f32 / height as f32;
            let (preview_width, preview_height) = if aspect_ratio > 1.0 {
                (preview_size, (preview_size as f32 / aspect_ratio) as u32)
            } else {
                ((preview_size as f32 * aspect_ratio) as u32, preview_size)
            };

            let small_image = image::DynamicImage::ImageRgba8(image.clone())
                .resize_exact(
                    preview_width,
                    preview_height,
                    image::imageops::FilterType::Lanczos3,
                )
                .to_rgba8();

            self.apply_filter(&small_image, params)
        }
    }
}

/// Factory for example plugins
pub struct ExamplePluginFactory;

impl StaticPluginFactory for ExamplePluginFactory {
    fn create(&self) -> Result<Arc<dyn Plugin>> {
        Ok(Arc::new(ExampleCompressionPlugin::new()))
    }
}

inventory::submit!(ExamplePluginFactory);

/// Plugin configuration management
pub struct PluginConfigManager {
    configs: HashMap<String, PluginConfig>,
    config_file_path: PathBuf,
}

impl PluginConfigManager {
    pub fn new<P: AsRef<Path>>(config_file: P) -> Self {
        Self {
            configs: HashMap::new(),
            config_file_path: config_file.as_ref().to_path_buf(),
        }
    }

    /// Load plugin configurations from file
    pub fn load_configs(&mut self) -> Result<()> {
        if self.config_file_path.exists() {
            let content = std::fs::read_to_string(&self.config_file_path).map_err(|e| {
                VeraError::PluginError(format!("Failed to read config file: {}", e))
            })?;

            self.configs = serde_json::from_str(&content).map_err(|e| {
                VeraError::PluginError(format!("Failed to parse config file: {}", e))
            })?;
        }

        Ok(())
    }

    /// Save plugin configurations to file
    pub fn save_configs(&self) -> Result<()> {
        let content = serde_json::to_string_pretty(&self.configs)
            .map_err(|e| VeraError::PluginError(format!("Failed to serialize configs: {}", e)))?;

        std::fs::write(&self.config_file_path, content)
            .map_err(|e| VeraError::PluginError(format!("Failed to write config file: {}", e)))?;

        Ok(())
    }

    /// Get configuration for a plugin
    pub fn get_config(&self, plugin_id: &str) -> Option<&PluginConfig> {
        self.configs.get(plugin_id)
    }

    /// Set configuration for a plugin
    pub fn set_config(&mut self, plugin_id: String, config: PluginConfig) {
        self.configs.insert(plugin_id, config);
    }
}

/// Extension metadata for custom format data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionMetadata {
    pub extension_id: String,
    pub version: u32,
    pub data_size: u64,
    pub checksum: Option<String>,
    pub compression: Option<String>,
    pub custom_fields: HashMap<String, String>,
}

/// Format extension integration with VeRA metadata
impl Metadata {
    /// Add format extension data
    pub fn add_extension(&mut self, extension_id: String, data: FormatExtensionData) -> Result<()> {
        let metadata_key = format!("extension.{}", extension_id);
        let serialized = serde_json::to_string(&ExtensionMetadata {
            extension_id: data.extension_id,
            version: data.version,
            data_size: data.data.len() as u64,
            checksum: None, // Would calculate actual checksum
            compression: None,
            custom_fields: data.metadata,
        })
        .map_err(|e| {
            VeraError::InvalidFormat(format!("Failed to serialize extension metadata: {}", e))
        })?;

        self.custom.insert(metadata_key, serialized);
        Ok(())
    }

    /// Get format extension data
    pub fn get_extension(&self, extension_id: &str) -> Option<ExtensionMetadata> {
        let metadata_key = format!("extension.{}", extension_id);
        self.custom
            .get(&metadata_key)
            .and_then(|json| serde_json::from_str(json).ok())
    }

    /// List all format extensions
    pub fn list_extensions(&self) -> Vec<String> {
        self.custom
            .keys()
            .filter(|key| key.starts_with("extension."))
            .map(|key| key.strip_prefix("extension.").unwrap().to_string())
            .collect()
    }
}
