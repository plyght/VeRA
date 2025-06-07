use criterion::{black_box, criterion_group, criterion_main, Criterion};
use libvera::*;

fn bench_metadata_creation(c: &mut Criterion) {
    c.bench_function("metadata_creation", |b| {
        b.iter(|| black_box(Metadata::new(black_box(1920), black_box(1080))))
    });
}

fn bench_metadata_serialization(c: &mut Criterion) {
    let metadata = Metadata::new(1920, 1080);

    c.bench_function("metadata_cbor_serialize", |b| {
        b.iter(|| black_box(metadata.to_cbor().unwrap()))
    });
}

fn bench_metadata_deserialization(c: &mut Criterion) {
    let metadata = Metadata::new(1920, 1080);
    let serialized = metadata.to_cbor().unwrap();

    c.bench_function("metadata_cbor_deserialize", |b| {
        b.iter(|| black_box(Metadata::from_cbor(black_box(&serialized)).unwrap()))
    });
}

fn bench_header_creation(c: &mut Criterion) {
    c.bench_function("header_creation", |b| {
        b.iter(|| black_box(container::Header::new()))
    });
}

criterion_group!(
    benches,
    bench_metadata_creation,
    bench_metadata_serialization,
    bench_metadata_deserialization,
    bench_header_creation
);
criterion_main!(benches);
