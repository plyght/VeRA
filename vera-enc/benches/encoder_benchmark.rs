use criterion::{black_box, criterion_group, criterion_main, Criterion};
use libvera::*;

fn bench_encoder_creation(c: &mut Criterion) {
    c.bench_function("encoder_creation", |b| {
        b.iter(|| {
            let buffer = std::io::Cursor::new(Vec::new());
            black_box(Encoder::new(buffer, black_box(1920), black_box(1080)))
        })
    });
}

fn bench_tile_pyramid_creation(c: &mut Criterion) {
    let image = image::DynamicImage::new_rgba8(512, 512);
    
    c.bench_function("tile_pyramid_creation", |b| {
        b.iter(|| {
            black_box(tiles::TilePyramid::new(
                black_box(image.clone()),
                black_box(256),
                black_box(5)
            ))
        })
    });
}

criterion_group!(
    benches,
    bench_encoder_creation,
    bench_tile_pyramid_creation
);
criterion_main!(benches);