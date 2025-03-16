use deep_risk_model::prelude::{
    TransformerRiskModel, RiskModel, MarketData, 
    MemoryConfig, SparseTensor, ChunkedProcessor, GradientCheckpointer, MemoryPool
};
use ndarray::{Array2, Array};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use std::time::Instant;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Deep Risk Model - Memory Optimization Example");
    println!("=============================================\n");
    
    // Create a transformer risk model
    let d_model = 64;
    let n_heads = 8;
    let d_ff = 256;
    let n_layers = 3;
    
    println!("Creating transformer risk model with:");
    println!("  d_model = {}", d_model);
    println!("  n_heads = {}", n_heads);
    println!("  d_ff = {}", d_ff);
    println!("  n_layers = {}", n_layers);
    
    let model = TransformerRiskModel::new(d_model, n_heads, d_ff, n_layers)?;
    
    // Generate synthetic market data (large dataset)
    let n_samples = 10000;  // 10,000 samples
    let n_assets = 100;     // 100 assets
    
    println!("\nGenerating synthetic market data:");
    println!("  n_samples = {}", n_samples);
    println!("  n_assets = {}", n_assets);
    
    let features = Array::random((n_samples, n_assets), Uniform::new(-1.0, 1.0));
    let returns = Array::random((n_samples, n_assets), Uniform::new(-0.05, 0.05));
    let market_data = MarketData::new(returns, features);
    
    // 1. Sparse Tensor Example
    println!("\n1. Sparse Tensor Example");
    println!("------------------------");
    
    // Create a weight matrix with sparsity
    let mut dense_weights = Array2::zeros((d_model, d_ff));
    
    // Fill only 20% of the weights (80% sparsity)
    let sparsity = 0.8;
    let num_nonzero = ((d_model * d_ff) as f32 * (1.0 - sparsity)) as usize;
    
    for _ in 0..num_nonzero {
        let i = rand::random::<usize>() % d_model;
        let j = rand::random::<usize>() % d_ff;
        dense_weights[[i, j]] = rand::random::<f32>() * 2.0 - 1.0;
    }
    
    // Convert to sparse representation
    let threshold = 1e-6;
    let sparse_weights = SparseTensor::from_dense(&dense_weights.view(), threshold);
    
    // Compare memory usage
    let dense_memory = d_model * d_ff * std::mem::size_of::<f32>();
    let sparse_memory = sparse_weights.memory_usage();
    
    println!("  Dense matrix shape: ({}, {})", d_model, d_ff);
    println!("  Sparsity: {:.1}%", sparsity * 100.0);
    println!("  Dense memory usage: {} bytes", dense_memory);
    println!("  Sparse memory usage: {} bytes", sparse_memory);
    println!("  Memory reduction: {:.1}x", dense_memory as f32 / sparse_memory as f32);
    
    // Verify correctness
    let reconstructed = sparse_weights.to_dense();
    let mut sum_squared_error = 0.0;
    let mut max_diff: f32 = 0.0;
    
    for i in 0..dense_weights.shape()[0] {
        for j in 0..dense_weights.shape()[1] {
            let diff = (dense_weights[[i, j]] - reconstructed[[i, j]]).abs();
            sum_squared_error += diff * diff;
            max_diff = max_diff.max(diff);
        }
    }
    println!("  Maximum reconstruction error: {:.2e}", max_diff);
    
    // 2. Chunked Processing Example
    println!("\n2. Chunked Processing Example");
    println!("----------------------------");
    
    // Configure chunked processing
    let memory_config = MemoryConfig {
        use_chunked_processing: true,
        chunk_size: 1000,
        ..Default::default()
    };
    
    let mut chunked_processor = ChunkedProcessor::new(memory_config.clone(), n_samples);
    
    // Process in chunks and measure time
    println!("  Processing {} samples in chunks of {}", n_samples, memory_config.chunk_size);
    let start = Instant::now();
    
    let chunk_results = chunked_processor.process_in_chunks(&market_data.features().view(), |chunk| {
        // Simple processing: compute mean of each feature
        let means = chunk.mean_axis(ndarray::Axis(0))
            .ok_or_else(|| deep_risk_model::error::ModelError::ComputationError(
                "Failed to compute mean".into()
            ))?;
        Ok(means)
    })?;
    
    let chunked_duration = start.elapsed();
    println!("  Chunked processing time: {:?}", chunked_duration);
    println!("  Number of chunks processed: {}", chunk_results.len());
    
    // 3. Memory Pool Example
    println!("\n3. Memory Pool Example");
    println!("--------------------");
    
    // Create a memory pool with 100MB limit
    let max_memory = 100 * 1024 * 1024; // 100MB
    let mut memory_pool = MemoryPool::new(max_memory);
    
    println!("  Created memory pool with {} MB limit", max_memory / (1024 * 1024));
    
    // Allocate and release tensors
    let shapes = [
        (1000, 64),
        (2000, 32),
        (500, 128),
        (1000, 64), // Same as first, should be reused
    ];
    
    for (i, &(rows, cols)) in shapes.iter().enumerate() {
        println!("  Allocating tensor {}: ({}, {})", i+1, rows, cols);
        let tensor = memory_pool.allocate(&[rows, cols])?;
        println!("    Memory usage: {:.2} MB", memory_pool.get_memory_usage() as f32 / (1024.0 * 1024.0));
        
        // Do some work with the tensor
        // ...
        
        // Release the tensor back to the pool
        memory_pool.release(tensor);
        println!("    Released tensor back to pool");
    }
    
    // 4. Gradient Checkpointing Example
    println!("\n4. Gradient Checkpointing Example");
    println!("-------------------------------");
    
    // Configure gradient checkpointing
    let checkpoint_config = MemoryConfig {
        use_checkpointing: true,
        checkpoint_segments: 4,
        ..Default::default()
    };
    
    let checkpointer = GradientCheckpointer::new(checkpoint_config.clone());
    
    // Create a smaller dataset for this example
    let small_samples = 1000;
    let small_features = Array::random((small_samples, n_assets), Uniform::new(-1.0, 1.0));
    
    println!("  Processing sequence with {} samples", small_samples);
    println!("  Using {} checkpoint segments", checkpoint_config.checkpoint_segments);
    
    // Note: This will return an error because we haven't implemented combine_segment_results
    // for the specific return type. In a real implementation, you would implement this for
    // your specific data types.
    let checkpoint_result = checkpointer.process_sequence(&small_features.view(), |segment| {
        // Process segment
        println!("    Processing segment with shape: {:?}", segment.shape());
        
        // Return segment mean as a placeholder
        let segment_mean = segment.mean_axis(ndarray::Axis(0))
            .ok_or_else(|| deep_risk_model::error::ModelError::ComputationError(
                "Failed to compute mean".into()
            ))?;
        
        Ok(segment_mean)
    });
    
    match checkpoint_result {
        Ok(_) => println!("  Checkpoint processing succeeded"),
        Err(e) => println!("  Expected error: {}", e),
    }
    
    println!("\nMemory Optimization Example Completed");
    
    Ok(())
} 