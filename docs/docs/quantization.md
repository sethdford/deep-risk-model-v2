---
layout: default
title: Quantization
nav_order: 5
---

# Quantization

Quantization is a technique to reduce the precision of model weights and activations, resulting in smaller model sizes and faster inference times. The Deep Risk Model library provides comprehensive quantization support to optimize models for deployment.

## Overview

The `quantization` module offers several features for model compression:

- **Per-tensor quantization**: Apply the same scale factor to an entire tensor
- **Per-channel quantization**: Apply different scale factors to each channel
- **Multiple precision options**: Int8, Int16, Float16, and Float32
- **Quantization-aware training**: Train models with simulated quantization
- **Post-training quantization**: Quantize pre-trained models

## Quantization Configuration

The `QuantizationConfig` struct allows you to configure various quantization options:

```rust
use deep_risk_model::prelude::*;

let config = QuantizationConfig {
    precision: Precision::Int8,
    per_channel: true,
    symmetric: true,
    calibration_method: CalibrationMethod::MinMax,
};
```

### Precision Options

The library supports the following precision types:

- `Precision::Int8`: 8-bit integer quantization (smallest size, lowest precision)
- `Precision::Int16`: 16-bit integer quantization
- `Precision::Float16`: 16-bit floating point (half precision)
- `Precision::Float32`: 32-bit floating point (full precision, no quantization)

## Quantizing a Tensor

You can quantize individual tensors using the `QuantizedTensor` struct:

```rust
use deep_risk_model::prelude::*;

// Create a tensor to quantize
let tensor = Tensor::new(&[1.2, -0.5, 3.7, -2.1, 0.8, -1.5], &[6]);

// Create a quantization configuration
let config = QuantizationConfig {
    precision: Precision::Int8,
    per_channel: false,
    symmetric: true,
    calibration_method: CalibrationMethod::MinMax,
};

// Quantize the tensor
let quantized_tensor = QuantizedTensor::quantize(&tensor, &config);

// Get the quantized data
let quantized_data = quantized_tensor.data();

// Dequantize back to floating point
let dequantized = quantized_tensor.dequantize();

// Calculate quantization error
let mse = tensor.mean_squared_error(&dequantized);
println!("Quantization MSE: {}", mse);

// Calculate memory savings
let original_size = tensor.memory_usage();
let quantized_size = quantized_tensor.memory_usage();
println!("Memory reduction: {}%", 100.0 * (1.0 - quantized_size as f64 / original_size as f64));
```

## Per-Channel Quantization

Per-channel quantization applies different scale factors to each channel, which can improve accuracy for weights with varying distributions across channels:

```rust
use deep_risk_model::prelude::*;

// Create a 2D tensor with shape [3, 4] (3 channels, 4 values per channel)
let tensor = Tensor::new(
    &[
        1.0, 2.0, 3.0, 4.0,    // Channel 1
        0.1, 0.2, 0.3, 0.4,    // Channel 2
        10.0, 20.0, 30.0, 40.0 // Channel 3
    ],
    &[3, 4]
);

// Configure per-channel quantization
let config = QuantizationConfig {
    precision: Precision::Int8,
    per_channel: true,  // Enable per-channel quantization
    symmetric: true,
    calibration_method: CalibrationMethod::MinMax,
};

// Quantize the tensor
let quantized_tensor = QuantizedTensor::quantize(&tensor, &config);

// Dequantize back to floating point
let dequantized = quantized_tensor.dequantize();

// Calculate quantization error
let mse = tensor.mean_squared_error(&dequantized);
println!("Per-channel quantization MSE: {}", mse);
```

## Quantizing a Model

To quantize an entire model, use the `Quantizer` struct:

```rust
use deep_risk_model::prelude::*;

// Create a transformer risk model
let model = TransformerRiskModel::new(
    ModelConfig {
        input_dim: 100,
        hidden_dim: 512,
        num_layers: 6,
        num_heads: 8,
        dropout: 0.1,
    },
    MemoryConfig::default(),
);

// Create a quantizer with configuration
let quantizer = Quantizer::new(QuantizationConfig {
    precision: Precision::Int8,
    per_channel: true,
    symmetric: true,
    calibration_method: CalibrationMethod::MinMax,
});

// Quantize the model
let quantized_model = quantizer.quantize_model(&model);

// Use the quantized model for inference
let input_data = MarketData::generate_synthetic(100, 100);
let risk_factors = quantized_model.generate_risk_factors(&input_data)?;

// Compare memory usage
let original_size = model.memory_usage();
let quantized_size = quantized_model.memory_usage();
println!("Model size reduction: {}%", 100.0 * (1.0 - quantized_size as f64 / original_size as f64));
```

## Calibration Methods

The library supports different calibration methods to determine the optimal scale factors:

- `CalibrationMethod::MinMax`: Uses the minimum and maximum values
- `CalibrationMethod::Percentile`: Uses percentile values to exclude outliers
- `CalibrationMethod::Entropy`: Minimizes information loss during quantization
- `CalibrationMethod::MSE`: Minimizes mean squared error

Example with percentile calibration:

```rust
use deep_risk_model::prelude::*;

let config = QuantizationConfig {
    precision: Precision::Int8,
    per_channel: true,
    symmetric: true,
    calibration_method: CalibrationMethod::Percentile(0.01), // Exclude top and bottom 1%
};

let quantized_tensor = QuantizedTensor::quantize(&tensor, &config);
```

## Quantization-Aware Training

For best results, you can train models with simulated quantization:

```rust
use deep_risk_model::prelude::*;

// Create a model with quantization-aware training
let mut model = TransformerRiskModel::new_with_quantization(
    ModelConfig {
        input_dim: 100,
        hidden_dim: 512,
        num_layers: 6,
        num_heads: 8,
        dropout: 0.1,
    },
    QuantizationConfig {
        precision: Precision::Int8,
        per_channel: true,
        symmetric: true,
        calibration_method: CalibrationMethod::MinMax,
    },
);

// Train the model with quantization awareness
model.train(&training_data, &validation_data, TrainingConfig {
    epochs: 10,
    batch_size: 32,
    learning_rate: 0.001,
    quantization_aware: true,
});
```

## Best Practices

Here are some recommendations for effective quantization:

1. **Start with higher precision**: Begin with Float16 or Int16 and gradually reduce to Int8 if accuracy is acceptable
2. **Use per-channel quantization** for weights, especially in convolutional and linear layers
3. **Apply symmetric quantization** for weights and asymmetric for activations
4. **Consider quantization-aware training** for critical models to minimize accuracy loss
5. **Benchmark performance** to ensure quantization provides the expected speedup
6. **Monitor accuracy metrics** to ensure quantization doesn't significantly impact model performance

## Performance Considerations

Quantization involves trade-offs between model size, inference speed, and accuracy:

- Int8 quantization typically reduces model size by 75% compared to Float32
- Inference speed can improve by 2-4x with optimized Int8 operations
- Accuracy loss varies by model and task, but is typically 1-2% for Int8 quantization

## Example

Here's a complete example demonstrating quantization techniques:

```rust
use deep_risk_model::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a transformer risk model
    let model = TransformerRiskModel::new(
        ModelConfig {
            input_dim: 100,
            hidden_dim: 512,
            num_layers: 6,
            num_heads: 8,
            dropout: 0.1,
        },
        MemoryConfig::default(),
    );

    // Generate synthetic market data
    let market_data = MarketData::generate_synthetic(1000, 100);
    
    // Run inference with the full-precision model
    let fp32_result = model.generate_risk_factors(&market_data)?;
    
    // Create quantization configurations for different precisions
    let int8_config = QuantizationConfig {
        precision: Precision::Int8,
        per_channel: true,
        symmetric: true,
        calibration_method: CalibrationMethod::MinMax,
    };
    
    let int16_config = QuantizationConfig {
        precision: Precision::Int16,
        per_channel: true,
        symmetric: true,
        calibration_method: CalibrationMethod::MinMax,
    };
    
    let float16_config = QuantizationConfig {
        precision: Precision::Float16,
        per_channel: false,
        symmetric: false,
        calibration_method: CalibrationMethod::MinMax,
    };
    
    // Quantize the model with different precisions
    let quantizer = Quantizer::new(int8_config);
    let int8_model = quantizer.quantize_model(&model);
    
    let quantizer = Quantizer::new(int16_config);
    let int16_model = quantizer.quantize_model(&model);
    
    let quantizer = Quantizer::new(float16_config);
    let float16_model = quantizer.quantize_model(&model);
    
    // Run inference with quantized models
    let int8_result = int8_model.generate_risk_factors(&market_data)?;
    let int16_result = int16_model.generate_risk_factors(&market_data)?;
    let float16_result = float16_model.generate_risk_factors(&market_data)?;
    
    // Compare results
    println!("Int8 MSE: {}", fp32_result.mean_squared_error(&int8_result));
    println!("Int16 MSE: {}", fp32_result.mean_squared_error(&int16_result));
    println!("Float16 MSE: {}", fp32_result.mean_squared_error(&float16_result));
    
    // Compare memory usage
    let fp32_size = model.memory_usage();
    let int8_size = int8_model.memory_usage();
    let int16_size = int16_model.memory_usage();
    let float16_size = float16_model.memory_usage();
    
    println!("FP32 model size: {} MB", fp32_size / (1024 * 1024));
    println!("Int8 model size: {} MB ({}% reduction)", 
             int8_size / (1024 * 1024), 
             100.0 * (1.0 - int8_size as f64 / fp32_size as f64));
    println!("Int16 model size: {} MB ({}% reduction)", 
             int16_size / (1024 * 1024), 
             100.0 * (1.0 - int16_size as f64 / fp32_size as f64));
    println!("Float16 model size: {} MB ({}% reduction)", 
             float16_size / (1024 * 1024), 
             100.0 * (1.0 - float16_size as f64 / fp32_size as f64));
    
    Ok(())
}
```

For more detailed examples, see the [quantization example]({{ site.url }}/examples/quantization) in the examples section. 