# Deep Risk Model: Theoretical Foundations

This document provides a detailed explanation of the theoretical foundations behind the Deep Risk Model, based on the research paper by [Lin et al. (2021)](https://arxiv.org/abs/2107.05201).

## 1. Problem Formulation

### 1.1 Traditional Risk Factor Models

Traditional factor models decompose asset returns \( r_t \) into systematic and idiosyncratic components:

\[ r_t = Bf_t + \epsilon_t \]

where:
- \( r_t \) is the vector of asset returns at time t
- \( B \) is the factor loading matrix
- \( f_t \) is the vector of factor returns
- \( \epsilon_t \) is the idiosyncratic return vector

### 1.2 Deep Learning Extension

Our model extends this by learning the factor structure through a deep neural network:

\[ f_t = \text{GAT}(\text{GRU}(X_t)) \]

where:
- \( X_t \) is the market data tensor
- \( \text{GRU}(\cdot) \) captures temporal dependencies
- \( \text{GAT}(\cdot) \) models cross-sectional relationships

## 2. Architecture Components

### 2.1 Gated Recurrent Unit (GRU)

The GRU processes temporal sequences with the following equations:

\[ z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z) \]
\[ r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r) \]
\[ \tilde{h}_t = \tanh(W_h x_t + U_h(r_t \odot h_{t-1}) + b_h) \]
\[ h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t \]

where:
- \( z_t \) is the update gate
- \( r_t \) is the reset gate
- \( h_t \) is the hidden state
- \( \odot \) denotes element-wise multiplication

### 2.2 Graph Attention Network (GAT)

The GAT layer computes attention scores between assets:

\[ \alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}_i} \exp(e_{ik})} \]
\[ e_{ij} = a(Wh_i, Wh_j) \]

where:
- \( \alpha_{ij} \) is the attention coefficient
- \( h_i \) is the feature vector of asset i
- \( W \) is a learnable weight matrix
- \( a(\cdot) \) is a shared attention mechanism

## 3. Loss Function

The model is trained with a multi-component loss function:

\[ \mathcal{L} = \mathcal{L}_{\text{factor}} + \lambda_1 \mathcal{L}_{\text{ortho}} + \lambda_2 \mathcal{L}_{\text{stable}} \]

where:
- \( \mathcal{L}_{\text{factor}} \) measures factor explanatory power
- \( \mathcal{L}_{\text{ortho}} \) ensures factor orthogonality
- \( \mathcal{L}_{\text{stable}} \) promotes factor stability

### 3.1 Factor Loss

\[ \mathcal{L}_{\text{factor}} = \|r_t - Bf_t\|_2^2 \]

### 3.2 Orthogonality Loss

\[ \mathcal{L}_{\text{ortho}} = \|F^TF - I\|_F^2 \]

where \( F \) is the matrix of factor returns.

### 3.3 Stability Loss

\[ \mathcal{L}_{\text{stable}} = \|f_t - f_{t-1}\|_2^2 \]

## 4. Covariance Estimation

The covariance matrix is estimated as:

\[ \Sigma = B\Sigma_fB^T + D \]

where:
- \( \Sigma_f \) is the factor covariance matrix
- \( D \) is a diagonal matrix of idiosyncratic variances

## 5. Implementation Details

### 5.1 Hyperparameters

Our implementation uses the following default hyperparameters:
- Input size: 64 (market features)
- Hidden size: 64 (GRU state dimension)
- Number of attention heads: 4
- Head dimension: 16
- Number of GRU layers: 2
- Output size: 3 (risk factors)

### 5.2 Training Process

The model is trained using:
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 32
- Training epochs: 100
- Early stopping patience: 10

## 6. Performance Metrics

The model's performance is evaluated using:

1. Explained Variance (R²):
   \[ R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2} \]

2. Portfolio Risk Reduction:
   \[ \text{Risk Reduction} = \frac{\sigma_{\text{baseline}} - \sigma_{\text{model}}}{\sigma_{\text{baseline}}} \]

3. Factor Stability:
   \[ \text{Stability} = \text{corr}(f_t, f_{t-1}) \]

## Temporal Fusion Transformer (TFT)

## Architecture Overview
The Temporal Fusion Transformer is a state-of-the-art architecture for interpretable temporal forecasting, particularly suited for risk modeling. Our implementation includes several key components:

### Variable Selection Network (VSN)
- Processes both static and temporal features independently
- Uses GRU-based feature processing for context-aware selection
- Applies softmax-based importance weighting for feature selection
- Outputs selected features with importance scores

### Static-Temporal Feature Processing
- Static features are processed once and enriched across all time steps
- Temporal features undergo sequential processing via GRU layers
- Static enrichment uses attention mechanism to enhance temporal features
- Repeated static context ensures consistent feature influence across time

### Memory-Efficient Implementation
- Gradient checkpointing support for handling long sequences
- Configurable number of segments for memory-performance tradeoff
- Optional checkpointing for variable selection and attention layers
- Efficient state management during forward pass

### Attention Mechanisms
- Multi-head attention for temporal self-attention
- Static enrichment attention for feature interaction
- Scaled dot-product attention with efficient memory usage
- Supports variable sequence lengths through segmentation

### Gating Layer
- Controls information flow between static and temporal paths
- Sigmoid-based gating mechanism for feature importance
- Learnable parameters for adaptive feature selection
- Ensures relevant feature contribution at each time step

## Implementation Details

### Static Enrichment
```rust
// Static context is repeated for each time step
let mut repeated_static = Array3::zeros((batch_size, seq_len, hidden_size));
for i in 0..batch_size {
    for j in 0..seq_len {
        repeated_static.slice_mut(s![i, j, ..])
            .assign(&static_context.slice(s![i, 0, ..]));
    }
}
```

### Gradient Checkpointing
```rust
pub struct CheckpointConfig {
    pub enabled: bool,
    pub num_segments: usize,
    pub checkpoint_vsn: bool,
    pub checkpoint_attention: bool,
}
```

The gradient checkpointing mechanism:
1. Divides sequence into configurable number of segments
2. Processes each segment independently
3. Manages memory by clearing intermediate states
4. Concatenates segment outputs for final prediction

### Memory Optimization Strategies
1. **Segment Processing**
   - Divide long sequences into manageable segments
   - Process each segment with independent memory space
   - Concatenate results while maintaining temporal coherence

2. **Selective Checkpointing**
   - Configure which components use checkpointing
   - Balance memory usage vs. computation time
   - Optional checkpointing for VSN and attention layers

3. **State Management**
   - Clear intermediate states between segments
   - Efficient memory reuse for sequential processing
   - Maintain essential context across segments

## Mathematical Details

### Variable Selection
The variable selection process combines GRU-based processing with importance weighting:
\[
h_t = \text{GRU}(x_t, h_{t-1})
\]
\[
\alpha_t = \text{softmax}(W h_t + b)
\]
\[
y_t = \alpha_t \odot x_t
\]

### Static Enrichment
Static enrichment uses multi-head attention to enhance temporal features:
\[
Q = W_Q T, K = W_K S, V = W_V S
\]
\[
\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V
\]
where T represents temporal features and S represents static features.

### Gating Mechanism
The gating layer controls information flow:
\[
g_t = \sigma(W_g h_t + b_g)
\]
\[
y_t = g_t \odot h_t
\]

## Performance Considerations

### Memory Usage
- Memory complexity: O(B * S * H) where:
  - B: Batch size
  - S: Sequence length
  - H: Hidden dimension
- Reduced to O(B * (S/N) * H) with N segments

### Computational Efficiency
- Linear time complexity in sequence length
- Parallel processing within segments
- Efficient attention computation through segmentation

### Numerical Stability
- Proper initialization of weights
- Gradient norm clipping during training
- Stable softmax and attention computations

## Testing Coverage
- Unit tests for each component
- Integration tests for full forward pass
- Memory usage validation
- Numerical stability checks
- Shape compatibility verification

## References
1. Lim et al. (2021) - Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting
2. Vaswani et al. (2017) - Attention Is All You Need
3. Chen et al. (2016) - Training Deep Nets with Sublinear Memory Cost 

# Advanced Factor Generation and Analysis

## Factor Quality Metrics

The Deep Risk Model implements several key metrics to evaluate and optimize risk factor quality:

1. **Information Coefficient (IC)**
   - Measures the correlation between factor values and future returns
   - Higher IC indicates better predictive power
   - Formula: \[ IC = \frac{1}{N} \sum_{i=1}^N corr(f_i, r_i) \]
   where \(f_i\) is the factor value and \(r_i\) is the return for asset i

2. **Variance Inflation Factor (VIF)**
   - Detects multicollinearity between factors
   - VIF > 5 indicates potential redundancy
   - Formula: \[ VIF_j = \frac{1}{1 - R^2_j} \]
   where \(R^2_j\) is from regressing factor j on all other factors

3. **T-Statistic**
   - Measures statistical significance of factors
   - Higher absolute values indicate stronger significance
   - Formula: \[ t = \frac{\bar{f}}{\sigma_f / \sqrt{n}} \]
   where \(\bar{f}\) is factor mean, \(\sigma_f\) is standard deviation

4. **Explained Variance Ratio**
   - Quantifies factor contribution to total variance
   - Higher values indicate more important factors
   - Formula: \[ EVR = \frac{Var(f)}{Var(r)} \]
   where \(Var(f)\) is factor variance and \(Var(r)\) is return variance

## Factor Orthogonalization

The model employs Gram-Schmidt orthogonalization to ensure factors are uncorrelated:

1. **Process**
   ```rust
   // For each factor i
   for i in 1..n_factors {
       // Subtract projections onto previous factors
       for j in 0..i {
           proj = <f_i, f_j> / <f_j, f_j>
           f_i = f_i - proj * f_j
       }
       // Normalize
       f_i = f_i / ||f_i||
   }
   ```

2. **Benefits**
   - Eliminates multicollinearity
   - Improves factor interpretability
   - Enhances numerical stability

## Adaptive Factor Selection

The model dynamically selects optimal factors based on quality metrics:

1. **Selection Criteria**
   - Explained variance ratio > threshold (default: 0.1)
   - VIF < maximum (default: 5.0)
   - |t-statistic| > significance level (default: 1.96)

2. **Algorithm**
   ```rust
   let selected_factors = factors.filter(|f| {
       f.explained_variance >= min_threshold &&
       f.vif <= max_vif &&
       f.t_statistic.abs() > significance_level
   });
   ```

3. **Advantages**
   - Removes redundant factors
   - Focuses on significant predictors
   - Adapts to changing market conditions

## Implementation Details

1. **Factor Generation Pipeline**
   ```rust
   // 1. Generate initial factors using transformer
   let initial_factors = transformer.forward(market_data);
   
   // 2. Orthogonalize factors
   let orthogonal_factors = gram_schmidt(initial_factors);
   
   // 3. Calculate quality metrics
   let metrics = calculate_metrics(orthogonal_factors);
   
   // 4. Select optimal factors
   let final_factors = select_factors(orthogonal_factors, metrics);
   ```

2. **Memory Optimization**
   - In-place orthogonalization
   - Efficient matrix operations
   - Batch processing for large datasets

3. **Performance Considerations**
   - Parallel metric calculation
   - Optimized linear algebra routines
   - Caching of intermediate results

## Validation and Testing

1. **Unit Tests**
   - Orthogonality verification
   - Metric calculation accuracy
   - Selection criteria validation

2. **Integration Tests**
   - End-to-end factor generation
   - Performance benchmarks
   - Memory usage monitoring

3. **Quality Assurance**
   ```rust
   #[test]
   fn test_factor_quality() {
       let metrics = model.get_factor_metrics(data);
       assert!(metrics.information_coefficient.abs() <= 1.0);
       assert!(metrics.vif >= 1.0);
       assert!(metrics.explained_variance >= 0.0);
   }
   ```

## Future Enhancements

1. **Planned Improvements**
   - GPU acceleration for large matrices
   - Online factor updating
   - Adaptive thresholding

2. **Research Directions**
   - Alternative orthogonalization methods
   - Dynamic factor number selection
   - Regime-dependent metrics 