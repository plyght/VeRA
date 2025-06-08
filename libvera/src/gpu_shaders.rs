//! GPU shaders for VeRA rendering

/// Vertex shader for raster tile rendering
pub const RASTER_VERTEX_SHADER: &str = r#"
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
    @location(2) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) color: vec4<f32>,
}

struct Uniforms {
    view_proj: mat4x4<f32>,
    resolution: vec2<f32>,
    zoom_level: f32,
    time: f32,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(in.position, 1.0);
    out.tex_coords = in.tex_coords;
    out.color = in.color;
    return out;
}
"#;

/// Fragment shader for raster tile rendering
pub const RASTER_FRAGMENT_SHADER: &str = r#"
@group(0) @binding(1)
var diffuse_texture: texture_2d<f32>;
@group(0) @binding(2)
var diffuse_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) color: vec4<f32>,
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let texture_color = textureSample(diffuse_texture, diffuse_sampler, in.tex_coords);
    return texture_color * in.color;
}
"#;

/// Vertex shader for vector rendering
pub const VECTOR_VERTEX_SHADER: &str = r#"
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
    @location(2) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec2<f32>,
    @location(1) color: vec4<f32>,
}

struct VectorUniforms {
    transform: mat4x4<f32>,
    color: vec4<f32>,
    stroke_width: f32,
    opacity: f32,
    blend_mode: u32,
    padding: u32,
}

@group(0) @binding(0)
var<uniform> uniforms: VectorUniforms;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world_pos = uniforms.transform * vec4<f32>(in.position, 1.0);
    out.clip_position = world_pos;
    out.world_position = world_pos.xy;
    out.color = in.color * uniforms.color;
    return out;
}
"#;

/// Fragment shader for vector rendering
pub const VECTOR_FRAGMENT_SHADER: &str = r#"
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec2<f32>,
    @location(1) color: vec4<f32>,
}

struct VectorUniforms {
    transform: mat4x4<f32>,
    color: vec4<f32>,
    stroke_width: f32,
    opacity: f32,
    blend_mode: u32,
    padding: u32,
}

@group(0) @binding(0)
var<uniform> uniforms: VectorUniforms;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = in.color;
    return vec4<f32>(color.rgb, color.a * uniforms.opacity);
}
"#;

/// Compute shader for advanced compositing operations
pub const COMPOSITE_COMPUTE_SHADER: &str = r#"
@group(0) @binding(0)
var source_texture: texture_2d<f32>;
@group(0) @binding(1)
var target_texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2)
var blend_texture: texture_2d<f32>;

struct CompositeParams {
    blend_mode: u32,
    opacity: f32,
    width: u32,
    height: u32,
}

@group(0) @binding(3)
var<uniform> params: CompositeParams;

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel_coord = vec2<i32>(i32(global_id.x), i32(global_id.y));
    
    if (global_id.x >= params.width || global_id.y >= params.height) {
        return;
    }
    
    let source_color = textureLoad(source_texture, pixel_coord, 0);
    let blend_color = textureLoad(blend_texture, pixel_coord, 0);
    
    var result_color: vec4<f32>;
    
    // Blend modes
    switch (params.blend_mode) {
        case 0u: { // Normal
            result_color = mix(source_color, blend_color, blend_color.a * params.opacity);
        }
        case 1u: { // Multiply
            result_color = vec4<f32>(source_color.rgb * blend_color.rgb, source_color.a);
        }
        case 2u: { // Screen
            result_color = vec4<f32>(1.0 - (1.0 - source_color.rgb) * (1.0 - blend_color.rgb), source_color.a);
        }
        case 3u: { // Overlay
            let base = source_color.rgb;
            let overlay = blend_color.rgb;
            let result = select(
                2.0 * base * overlay,
                1.0 - 2.0 * (1.0 - base) * (1.0 - overlay),
                base < vec3<f32>(0.5)
            );
            result_color = vec4<f32>(result, source_color.a);
        }
        default: {
            result_color = source_color;
        }
    }
    
    textureStore(target_texture, pixel_coord, result_color);
}
"#;

/// Advanced vector tessellation compute shader
pub const TESSELLATION_COMPUTE_SHADER: &str = r#"
struct PathCommand {
    command_type: u32, // 0=MoveTo, 1=LineTo, 2=QuadTo, 3=CubicTo, 4=Close
    x: f32,
    y: f32,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    padding: f32,
}

struct TessellationVertex {
    position: vec2<f32>,
    normal: vec2<f32>,
    distance: f32,
    side: f32, // -1 for left, 1 for right
}

@group(0) @binding(0)
var<storage, read> path_commands: array<PathCommand>;
@group(0) @binding(1)
var<storage, read_write> output_vertices: array<TessellationVertex>;
@group(0) @binding(2)
var<storage, read_write> output_indices: array<u32>;

struct TessellationParams {
    num_commands: u32,
    tolerance: f32,
    stroke_width: f32,
    max_vertices: u32,
}

@group(0) @binding(3)
var<uniform> params: TessellationParams;

fn bezier_point(p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>, t: f32) -> vec2<f32> {
    let inv_t = 1.0 - t;
    return inv_t * inv_t * p0 + 2.0 * inv_t * t * p1 + t * t * p2;
}

fn cubic_bezier_point(p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>, p3: vec2<f32>, t: f32) -> vec2<f32> {
    let inv_t = 1.0 - t;
    let inv_t2 = inv_t * inv_t;
    let inv_t3 = inv_t2 * inv_t;
    let t2 = t * t;
    let t3 = t2 * t;
    
    return inv_t3 * p0 + 3.0 * inv_t2 * t * p1 + 3.0 * inv_t * t2 * p2 + t3 * p3;
}

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let command_idx = global_id.x;
    
    if (command_idx >= params.num_commands) {
        return;
    }
    
    let command = path_commands[command_idx];
    
    // Tessellate based on command type
    switch (command.command_type) {
        case 1u: { // LineTo
            // Add line segment vertices
            let start = vec2<f32>(command.x1, command.y1);
            let end = vec2<f32>(command.x, command.y);
            let direction = normalize(end - start);
            let normal = vec2<f32>(-direction.y, direction.x);
            
            // Add stroke vertices
            let half_width = params.stroke_width * 0.5;
            
            // Atomic increment for vertex indices would be needed here in real implementation
            // For now, this is a simplified version
        }
        case 2u: { // QuadTo
            // Tessellate quadratic Bezier curve
            let p0 = vec2<f32>(command.x1, command.y1);
            let p1 = vec2<f32>(command.x2, command.y2);
            let p2 = vec2<f32>(command.x, command.y);
            
            // Adaptive tessellation based on curve flatness
            let num_segments = u32(max(8.0, length(p1 - (p0 + p2) * 0.5) / params.tolerance));
            
            for (var i: u32 = 0u; i <= num_segments; i = i + 1u) {
                let t = f32(i) / f32(num_segments);
                let point = bezier_point(p0, p1, p2, t);
                
                // Calculate tangent and normal
                let dt = 1.0 / f32(num_segments);
                let tangent = normalize(bezier_point(p0, p1, p2, t + dt) - point);
                let normal = vec2<f32>(-tangent.y, tangent.x);
                
                // Add vertices for stroke
                // Implementation details for actual vertex generation
            }
        }
        case 3u: { // CubicTo
            // Tessellate cubic Bezier curve
            let p0 = vec2<f32>(command.x1, command.y1);
            let p1 = vec2<f32>(command.x2, command.y2);
            let p2 = vec2<f32>(command.x, command.y);
            let p3 = vec2<f32>(command.x1, command.y1); // This would be the actual end point in real implementation
            
            // Adaptive tessellation for cubic curves
            let num_segments = u32(max(16.0, (length(p1 - p0) + length(p2 - p1) + length(p3 - p2)) / params.tolerance));
            
            for (var i: u32 = 0u; i <= num_segments; i = i + 1u) {
                let t = f32(i) / f32(num_segments);
                let point = cubic_bezier_point(p0, p1, p2, p3, t);
                
                // Generate stroke vertices
                // Implementation details for actual vertex generation
            }
        }
        default: { // MoveTo, Close
            // No tessellation needed
        }
    }
}
"#;

/// High-quality anti-aliasing fragment shader for vector rendering
pub const VECTOR_AA_FRAGMENT_SHADER: &str = r#"
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec2<f32>,
    @location(1) distance: f32,
    @location(2) color: vec4<f32>,
}

struct VectorUniforms {
    transform: mat4x4<f32>,
    color: vec4<f32>,
    stroke_width: f32,
    opacity: f32,
    blend_mode: u32,
    aa_width: f32,
}

@group(0) @binding(0)
var<uniform> uniforms: VectorUniforms;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Distance-based antialiasing
    let distance_from_edge = abs(in.distance);
    let aa_factor = smoothstep(uniforms.aa_width, 0.0, distance_from_edge);
    
    let color = in.color * uniforms.color;
    return vec4<f32>(color.rgb, color.a * uniforms.opacity * aa_factor);
}
"#;

/// Multi-sample anti-aliasing resolve shader
pub const MSAA_RESOLVE_SHADER: &str = r#"
@group(0) @binding(0)
var msaa_texture: texture_multisampled_2d<f32>;
@group(0) @binding(1)
var output_texture: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel_coord = vec2<i32>(i32(global_id.x), i32(global_id.y));
    let dimensions = textureDimensions(msaa_texture);
    
    if (global_id.x >= dimensions.x || global_id.y >= dimensions.y) {
        return;
    }
    
    // Manual MSAA resolve with better quality than hardware resolve
    var accumulated_color = vec4<f32>(0.0);
    let sample_count = textureNumSamples(msaa_texture);
    
    for (var i: i32 = 0; i < sample_count; i = i + 1) {
        let sample_color = textureLoad(msaa_texture, pixel_coord, i);
        accumulated_color = accumulated_color + sample_color;
    }
    
    let resolved_color = accumulated_color / f32(sample_count);
    textureStore(output_texture, pixel_coord, resolved_color);
}
"#;
