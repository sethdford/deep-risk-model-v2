use criterion::{black_box, criterion_group, criterion_main, Criterion};
use deep_risk_model::transformer::{TransformerConfig, TransformerComponent};
use deep_risk_model::transformer::attention::MultiHeadAttention;
use ndarray::Array2;

fn attention_benchmark(c: &mut Criterion) {
    let d_model = 512;
    let n_heads = 8;
    
    // Create the attention module directly with d_model and n_heads
    let attention = MultiHeadAttention::new(d_model, n_heads).unwrap();
    let batch_size = 32;
    
    // Create a 2D input array with shape [batch_size, d_model]
    let input = Array2::zeros((batch_size, d_model));
    
    c.bench_function("multi_head_attention", |b| {
        b.iter(|| {
            attention.forward(black_box(&input)).unwrap()
        })
    });
}

criterion_group!(benches, attention_benchmark);
criterion_main!(benches); 