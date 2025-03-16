use criterion::{black_box, criterion_group, criterion_main, Criterion};
use deep_risk_model::{
    transformer::{TransformerConfig, Transformer, TransformerComponent},
    TransformerRiskModel,
    types::{RiskModel, MarketData},
};
use ndarray::{Array2};

fn transformer_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("transformer");
    group.sample_size(10); // Reduce sample size for faster benchmarking
    
    // Test different model sizes
    for &d_model in &[32, 64] {  // Reduced test sizes for initial benchmarking
        group.bench_function(format!("forward_pass_{}", d_model), |b| {
            let config = TransformerConfig {
                d_model,
                n_heads: 4,
                d_ff: d_model * 4,
                dropout: 0.1,
                max_seq_len: 50,  // Reduced sequence length
                n_layers: 2,      // Reduced layers
                num_static_features: 5,
                num_temporal_features: 10,
                hidden_size: 32,
            };
            
            let transformer = Transformer::new(config).unwrap();
            // Create a 2D input array with shape [batch_size, d_model]
            let batch_size = 10;
            let input = Array2::ones((batch_size, d_model));
            
            b.iter(|| {
                black_box(transformer.forward(black_box(&input)).unwrap());
            });
        });
    }
    
    group.finish();
}

fn risk_model_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("risk_model");
    group.sample_size(10); // Reduce sample size for faster benchmarking
    
    // Test with different d_model values
    for &d_model in &[64] {  // Use only 64 for now since that's what we've fixed in the tests
        group.bench_function(format!("covariance_estimation_{}", d_model), |b| {
            let n_heads = 8;
            let d_ff = 256;
            let n_layers = 2;  // Reduced layers
            
            // Create the TransformerRiskModel with the new constructor signature
            let model = TransformerRiskModel::new(
                d_model,
                n_heads,
                d_ff,
                n_layers
            ).unwrap();
            
            // Create sample data with appropriate dimensions
            let n_samples = 100;
            let n_assets = d_model;  // Make sure n_assets matches d_model
            let features = Array2::ones((n_samples, n_assets));
            let returns = Array2::ones((n_samples, n_assets));
            let market_data = MarketData::new(returns, features);
            
            let rt = tokio::runtime::Runtime::new().unwrap();
            b.iter(|| rt.block_on(async {
                black_box(model.estimate_covariance(&market_data).await.unwrap())
            }));
        });
    }
    
    group.finish();
}

criterion_group!(benches, transformer_benchmark, risk_model_benchmark);
criterion_main!(benches); 