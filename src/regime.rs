use ndarray::{Array1, Array2, s};
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use crate::error::ModelError;
use crate::types::MarketData;
use std::fmt;
use std::hash::Hash;

/// Market regime types that can be detected
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum RegimeType {
    /// Low volatility, stable returns
    LowVolatility,
    /// Normal market conditions
    Normal,
    /// High volatility, unstable returns
    HighVolatility,
    /// Crisis conditions
    Crisis,
}

impl fmt::Display for RegimeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegimeType::LowVolatility => write!(f, "Low Volatility"),
            RegimeType::Normal => write!(f, "Normal"),
            RegimeType::HighVolatility => write!(f, "High Volatility"),
            RegimeType::Crisis => write!(f, "Crisis"),
        }
    }
}

/// Configuration for the HMM-based regime detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeConfig {
    /// Number of regimes to detect
    pub n_regimes: usize,
    /// Number of iterations for training
    pub max_iter: usize,
    /// Convergence threshold
    pub tol: f32,
    /// Random seed for initialization
    pub random_seed: Option<u64>,
    /// Minimum probability for regime assignment
    pub min_prob: f32,
}

impl Default for RegimeConfig {
    fn default() -> Self {
        Self {
            n_regimes: 4,
            max_iter: 100,
            tol: 1e-6,
            random_seed: None,
            min_prob: 0.6,
        }
    }
}

/// Hidden Markov Model for market regime detection
#[derive(Debug, Clone)]
pub struct MarketRegimeHMM {
    /// Number of regimes
    n_regimes: usize,
    /// Initial state probabilities
    initial_probs: Array1<f32>,
    /// Transition matrix
    transition_matrix: Array2<f32>,
    /// Emission means for each regime
    emission_means: Array1<f32>,
    /// Emission variances for each regime
    emission_vars: Array1<f32>,
    /// Configuration
    config: RegimeConfig,
    /// Current regime
    current_regime: Option<RegimeType>,
    /// Regime history
    regime_history: Vec<RegimeType>,
    /// Probability history
    probability_history: Vec<Array1<f32>>,
    /// Trained flag
    trained: bool,
}

impl Default for MarketRegimeHMM {
    fn default() -> Self {
        Self::new()
    }
}

impl MarketRegimeHMM {
    /// Create a new MarketRegimeHMM with default configuration
    pub fn new() -> Self {
        Self::with_config(RegimeConfig::default())
    }

    /// Create a new MarketRegimeHMM with custom configuration
    pub fn with_config(config: RegimeConfig) -> Self {
        let n_regimes = config.n_regimes;
        
        // Initialize with uniform probabilities
        let initial_probs = Array1::ones(n_regimes) / (n_regimes as f32);
        
        // Initialize transition matrix with equal probabilities
        let transition_matrix = Array2::ones((n_regimes, n_regimes)) / (n_regimes as f32);
        
        Self {
            n_regimes,
            initial_probs,
            transition_matrix,
            emission_means: Array1::zeros(n_regimes),
            emission_vars: Array1::ones(n_regimes),
            config,
            current_regime: None,
            regime_history: Vec::new(),
            probability_history: Vec::new(),
            trained: false,
        }
    }
    
    /// Create a new MarketRegimeHMM with existing state
    pub fn with_state(
        &self, 
        current_regime: Option<RegimeType>, 
        regime_history: Vec<RegimeType>, 
        probability_history: Vec<Array1<f32>>
    ) -> Self {
        Self {
            n_regimes: self.n_regimes,
            initial_probs: self.initial_probs.clone(),
            transition_matrix: self.transition_matrix.clone(),
            emission_means: self.emission_means.clone(),
            emission_vars: self.emission_vars.clone(),
            config: self.config.clone(),
            current_regime,
            regime_history,
            probability_history,
            trained: self.trained,
        }
    }
    
    /// Train the HMM using the Baum-Welch algorithm
    pub fn train(&mut self, data: &Array1<f32>) -> Result<(), ModelError> {
        if data.len() < 10 {
            return Err(ModelError::InvalidInput(
                "Training data must have at least 10 observations".to_string()
            ));
        }
        
        // Initialize emission parameters based on data statistics
        self.initialize_emission_params(data)?;
        
        let mut prev_log_likelihood = f32::NEG_INFINITY;
        
        for iter in 0..self.config.max_iter {
            // Forward-backward algorithm
            let (alpha, beta, log_likelihood) = self.forward_backward(data)?;
            
            // Check for convergence
            if (log_likelihood - prev_log_likelihood).abs() < self.config.tol && iter > 0 {
                break;
            }
            prev_log_likelihood = log_likelihood;
            
            // Compute state and transition probabilities
            let (gamma, xi) = self.compute_probabilities(&alpha, &beta, data)?;
            
            // Update parameters
            self.update_parameters(&gamma, &xi, data)?;
        }
        
        self.trained = true;
        Ok(())
    }
    
    /// Initialize emission parameters based on data statistics
    fn initialize_emission_params(&mut self, data: &Array1<f32>) -> Result<(), ModelError> {
        let mean = data.mean().unwrap_or(0.0);
        let var = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
        
        let _rng = match self.config.random_seed {
            Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
            None => rand::rngs::StdRng::from_entropy(),
        };
        
        // Set means around the data mean with some variance
        for i in 0..self.n_regimes {
            let regime_factor = match i {
                0 => -2.0, // Low volatility
                1 => -0.5, // Normal (negative bias)
                2 => 0.5,  // Normal (positive bias)
                3 => 2.0,  // High volatility/crisis
                _ => 0.0,  // Default
            };
            
            self.emission_means[i] = mean + regime_factor * (var.sqrt() / 2.0);
            
            // Set variances based on regime type
            let var_factor = match i {
                0 => 0.5,  // Low volatility
                1 => 1.0,  // Normal
                2 => 2.0,  // High volatility
                3 => 4.0,  // Crisis
                _ => 1.0,  // Default
            };
            
            self.emission_vars[i] = var * var_factor;
        }
        
        Ok(())
    }
    
    /// Forward-backward algorithm for HMM
    fn forward_backward(&self, data: &Array1<f32>) -> Result<(Array2<f32>, Array2<f32>, f32), ModelError> {
        let t = data.len();
        
        // Forward pass (alpha)
        let mut alpha = Array2::zeros((t, self.n_regimes));
        
        // Initialize first step with initial probabilities and first emission
        for i in 0..self.n_regimes {
            alpha[[0, i]] = self.initial_probs[i] * self.emission_prob(data[0], i);
        }
        
        // Forward recursion
        for t in 1..t {
            for j in 0..self.n_regimes {
                let mut sum = 0.0;
                for i in 0..self.n_regimes {
                    sum += alpha[[t-1, i]] * self.transition_matrix[[i, j]];
                }
                alpha[[t, j]] = sum * self.emission_prob(data[t], j);
            }
        }
        
        // Backward pass (beta)
        let mut beta = Array2::zeros((t, self.n_regimes));
        
        // Initialize last step
        for i in 0..self.n_regimes {
            beta[[t-1, i]] = 1.0;
        }
        
        // Backward recursion
        for t in (0..t-1).rev() {
            for i in 0..self.n_regimes {
                let mut sum = 0.0;
                for j in 0..self.n_regimes {
                    sum += self.transition_matrix[[i, j]] * self.emission_prob(data[t+1], j) * beta[[t+1, j]];
                }
                beta[[t, i]] = sum;
            }
        }
        
        // Compute log-likelihood
        let log_likelihood = alpha.row(t-1).sum().ln();
        
        Ok((alpha, beta, log_likelihood))
    }
    
    /// Compute state and transition probabilities
    fn compute_probabilities(
        &self, 
        alpha: &Array2<f32>, 
        beta: &Array2<f32>, 
        data: &Array1<f32>
    ) -> Result<(Array2<f32>, Array3<f32>), ModelError> {
        let t = data.len();
        
        // Compute gamma (state probabilities)
        let mut gamma = Array2::zeros((t, self.n_regimes));
        for t in 0..t {
            let normalizer = alpha.row(t).dot(&beta.row(t));
            for i in 0..self.n_regimes {
                gamma[[t, i]] = alpha[[t, i]] * beta[[t, i]] / normalizer;
            }
        }
        
        // Compute xi (transition probabilities)
        let mut xi = Array3::zeros((t-1, self.n_regimes, self.n_regimes));
        for t in 0..t-1 {
            let mut normalizer = 0.0;
            for i in 0..self.n_regimes {
                for j in 0..self.n_regimes {
                    xi[[t, i, j]] = alpha[[t, i]] 
                        * self.transition_matrix[[i, j]] 
                        * self.emission_prob(data[t+1], j) 
                        * beta[[t+1, j]];
                    normalizer += xi[[t, i, j]];
                }
            }
            
            if normalizer > 0.0 {
                for i in 0..self.n_regimes {
                    for j in 0..self.n_regimes {
                        xi[[t, i, j]] /= normalizer;
                    }
                }
            }
        }
        
        Ok((gamma, xi))
    }
    
    /// Update HMM parameters based on computed probabilities
    fn update_parameters(
        &mut self, 
        gamma: &Array2<f32>, 
        xi: &Array3<f32>, 
        data: &Array1<f32>
    ) -> Result<(), ModelError> {
        let t = data.len();
        
        // Update initial probabilities
        for i in 0..self.n_regimes {
            self.initial_probs[i] = gamma[[0, i]];
        }
        
        // Update transition matrix
        for i in 0..self.n_regimes {
            let mut row_sum = 0.0;
            for j in 0..self.n_regimes {
                let mut num = 0.0;
                let mut den = 0.0;
                
                for t in 0..t-1 {
                    num += xi[[t, i, j]];
                    den += gamma[[t, i]];
                }
                
                if den > 0.0 {
                    self.transition_matrix[[i, j]] = num / den;
                }
                row_sum += self.transition_matrix[[i, j]];
            }
            
            // Normalize row
            if row_sum > 0.0 {
                for j in 0..self.n_regimes {
                    self.transition_matrix[[i, j]] /= row_sum;
                }
            }
        }
        
        // Update emission parameters
        for j in 0..self.n_regimes {
            let mut sum_gamma = 0.0;
            let mut sum_gamma_x = 0.0;
            let mut sum_gamma_x2 = 0.0;
            
            for t in 0..t {
                sum_gamma += gamma[[t, j]];
                sum_gamma_x += gamma[[t, j]] * data[t];
                sum_gamma_x2 += gamma[[t, j]] * data[t].powi(2);
            }
            
            if sum_gamma > 0.0 {
                // Update mean
                self.emission_means[j] = sum_gamma_x / sum_gamma;
                
                // Update variance (with minimum value to ensure stability)
                self.emission_vars[j] = (sum_gamma_x2 / sum_gamma) - self.emission_means[j].powi(2);
                self.emission_vars[j] = self.emission_vars[j].max(1e-6);
            }
        }
        
        Ok(())
    }
    
    /// Calculate emission probability for a given observation and state
    fn emission_prob(&self, x: f32, state: usize) -> f32 {
        let mean = self.emission_means[state];
        let var = self.emission_vars[state];
        
        // Gaussian emission probability
        let exponent = -0.5 * (x - mean).powi(2) / var;
        (2.0 * std::f32::consts::PI * var).sqrt().recip() * exponent.exp()
    }
    
    /// Predict the regime for a given observation
    pub fn predict(&self, x: f32) -> Result<(RegimeType, Option<RegimeType>, Vec<RegimeType>, Vec<Array1<f32>>), ModelError> {
        if !self.trained {
            return Err(ModelError::InvalidInput("Model has not been trained yet".to_string()));
        }
        
        // Calculate emission probabilities for each regime
        let mut probs = Array1::zeros(self.n_regimes);
        for i in 0..self.n_regimes {
            probs[i] = self.emission_prob(x, i);
        }
        
        // If we have a current regime, use transition probabilities
        if let Some(current) = self.current_regime {
            let current_idx = self.regime_to_index(current);
            for i in 0..self.n_regimes {
                probs[i] *= self.transition_matrix[[current_idx, i]];
            }
        }
        
        // Normalize probabilities
        let sum = probs.sum();
        if sum > 0.0 {
            probs /= sum;
        }
        
        // Find the most likely regime
        let mut max_prob = 0.0;
        let mut max_idx = 0;
        for i in 0..self.n_regimes {
            if probs[i] > max_prob {
                max_prob = probs[i];
                max_idx = i;
            }
        }
        
        // Only assign a regime if probability is above threshold
        if max_prob < self.config.min_prob {
            // If below threshold, maintain current regime if available
            if let Some(regime) = self.current_regime {
                return Ok((regime, Some(regime), self.regime_history.clone(), self.probability_history.clone()));
            }
        }
        
        let regime = self.index_to_regime(max_idx);
        
        // Return the regime and updated state
        let mut new_history = self.regime_history.clone();
        let mut new_probs = self.probability_history.clone();
        new_history.push(regime);
        new_probs.push(probs.clone());
        
        Ok((regime, Some(regime), new_history, new_probs))
    }
    
    /// Predict regimes for a sequence of observations
    pub fn predict_sequence(&self, data: &Array1<f32>) -> Result<Vec<RegimeType>, ModelError> {
        if !self.trained {
            return Err(ModelError::InvalidInput("Model has not been trained yet".to_string()));
        }
        
        let mut regimes = Vec::with_capacity(data.len());
        let mut current_regime = None;
        let mut regime_history = Vec::new();
        let mut probability_history = Vec::new();
        
        for &x in data.iter() {
            // Create a temporary HMM with the current state
            let temp_hmm = self.with_state(current_regime, regime_history.clone(), probability_history.clone());
            
            // Get prediction and update state
            let (regime, new_current, new_history, new_probs) = temp_hmm.predict(x)?;
            current_regime = new_current;
            regime_history = new_history;
            probability_history = new_probs;
            regimes.push(regime);
        }
        
        Ok(regimes)
    }
    
    /// Get the current regime
    pub fn current_regime(&self) -> Option<RegimeType> {
        self.current_regime
    }
    
    /// Get the regime history
    pub fn regime_history(&self) -> &[RegimeType] {
        &self.regime_history
    }
    
    /// Get the probability history
    pub fn probability_history(&self) -> &[Array1<f32>] {
        &self.probability_history
    }
    
    /// Get the transition matrix
    pub fn transition_matrix(&self) -> &Array2<f32> {
        &self.transition_matrix
    }
    
    /// Get the emission means
    pub fn emission_means(&self) -> &Array1<f32> {
        &self.emission_means
    }
    
    /// Get the emission variances
    pub fn emission_vars(&self) -> &Array1<f32> {
        &self.emission_vars
    }
    
    /// Convert regime to index
    fn regime_to_index(&self, regime: RegimeType) -> usize {
        match regime {
            RegimeType::LowVolatility => 0,
            RegimeType::Normal => 1,
            RegimeType::HighVolatility => 2,
            RegimeType::Crisis => 3,
        }
    }
    
    /// Convert index to regime
    fn index_to_regime(&self, index: usize) -> RegimeType {
        match index {
            0 => RegimeType::LowVolatility,
            1 => RegimeType::Normal,
            2 => RegimeType::HighVolatility,
            3 => RegimeType::Crisis,
            _ => RegimeType::Normal, // Default
        }
    }
}

/// Market regime detection module
#[derive(Debug)]
pub struct MarketRegimeDetector {
    hmm: MarketRegimeHMM,
    window_size: usize,
    trained: bool,
}

impl MarketRegimeDetector {
    /// Create a new market regime detector
    pub fn new(window_size: usize) -> Self {
        Self {
            hmm: MarketRegimeHMM::new(),
            window_size,
            trained: false,
        }
    }
    
    /// Create a new market regime detector with custom configuration
    pub fn with_config(window_size: usize, config: RegimeConfig) -> Self {
        Self {
            hmm: MarketRegimeHMM::with_config(config),
            window_size,
            trained: false,
        }
    }
    
    /// Train the regime detector on market data
    pub fn train(&mut self, data: &MarketData) -> Result<(), ModelError> {
        let returns = data.returns();
        
        if returns.shape()[0] < self.window_size {
            return Err(ModelError::InvalidInput(
                format!("Data length ({}) must be >= window size ({})", returns.shape()[0], self.window_size)
            ));
        }
        
        // Calculate volatility as the feature for regime detection
        let volatility = self.calculate_volatility(returns)?;
        
        // Train HMM on volatility
        self.hmm.train(&volatility)?;
        self.trained = true;
        
        Ok(())
    }
    
    /// Detect regime for new market data
    pub fn detect_regime(&self, data: &MarketData) -> Result<RegimeType, ModelError> {
        if !self.trained {
            return Err(ModelError::InvalidInput("Detector has not been trained yet".to_string()));
        }
        
        let returns = data.returns();
        
        if returns.shape()[0] < self.window_size {
            return Err(ModelError::InvalidInput(
                format!("Data length ({}) must be >= window size ({})", returns.shape()[0], self.window_size)
            ));
        }
        
        // Calculate volatility
        let volatility = self.calculate_volatility(returns)?;
        
        // Use the last volatility value for prediction
        let last_vol = volatility[volatility.len() - 1];
        
        // Clone the HMM to avoid mutating self
        let hmm_clone = self.hmm.clone();
        
        // Predict regime and return only the regime
        let (regime, _, _, _) = hmm_clone.predict(last_vol)?;
        Ok(regime)
    }
    
    /// Detect regimes for a sequence of market data
    pub fn detect_regimes(&self, data: &MarketData) -> Result<Vec<RegimeType>, ModelError> {
        if !self.trained {
            return Err(ModelError::InvalidInput("Detector has not been trained yet".to_string()));
        }
        
        let returns = data.returns();
        
        if returns.shape()[0] < self.window_size {
            return Err(ModelError::InvalidInput(
                format!("Data length ({}) must be >= window size ({})", returns.shape()[0], self.window_size)
            ));
        }
        
        // Calculate volatility
        let volatility = self.calculate_volatility(returns)?;
        
        // Clone the HMM to avoid mutating self
        let hmm_clone = self.hmm.clone();
        
        // Predict regimes
        hmm_clone.predict_sequence(&volatility)
    }
    
    /// Calculate volatility from returns
    fn calculate_volatility(&self, returns: &Array2<f32>) -> Result<Array1<f32>, ModelError> {
        let n_samples = returns.shape()[0];
        let n_assets = returns.shape()[1];
        
        if n_samples < self.window_size {
            return Err(ModelError::InvalidInput(
                format!("Number of samples ({}) must be >= window size ({})", n_samples, self.window_size)
            ));
        }
        
        // Calculate rolling volatility
        let mut volatility = Array1::zeros(n_samples - self.window_size + 1);
        
        for i in 0..volatility.len() {
            let window = returns.slice(s![i..i+self.window_size, ..]);
            
            // Calculate average volatility across assets
            let mut vol_sum = 0.0;
            for j in 0..n_assets {
                let asset_returns = window.column(j);
                let std_dev = asset_returns.std(0.0);
                vol_sum += std_dev;
            }
            
            volatility[i] = vol_sum / (n_assets as f32);
        }
        
        Ok(volatility)
    }
    
    /// Get the current regime
    pub fn current_regime(&self) -> Option<RegimeType> {
        self.hmm.current_regime()
    }
    
    /// Get the regime history
    pub fn regime_history(&self) -> &[RegimeType] {
        self.hmm.regime_history()
    }
    
    /// Get the transition matrix
    pub fn transition_matrix(&self) -> &Array2<f32> {
        self.hmm.transition_matrix()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Normal;
    
    #[test]
    fn test_regime_detector_initialization() {
        let detector = MarketRegimeDetector::new(20);
        assert_eq!(detector.window_size, 20);
        assert!(!detector.trained);
    }
    
    #[test]
    fn test_hmm_initialization() {
        let hmm = MarketRegimeHMM::new();
        assert_eq!(hmm.n_regimes, 4);
        assert!(!hmm.trained);
        
        // Check initial probabilities sum to 1
        assert!((hmm.initial_probs.sum() - 1.0).abs() < 1e-6);
        
        // Check transition matrix rows sum to 1
        for i in 0..hmm.n_regimes {
            let row_sum = hmm.transition_matrix.row(i).sum();
            assert!((row_sum - 1.0).abs() < 1e-6);
        }
    }
    
    #[test]
    fn test_hmm_training() -> Result<(), ModelError> {
        let mut hmm = MarketRegimeHMM::new();
        
        // Generate synthetic data with two regimes
        let low_vol_data = Array1::random(100, Normal::new(0.0, 0.5)?);
        let high_vol_data = Array1::random(100, Normal::new(0.0, 2.0)?);
        
        let mut combined_data = Array1::zeros(200);
        for i in 0..100 {
            combined_data[i] = low_vol_data[i];
            combined_data[i+100] = high_vol_data[i];
        }
        
        // Train HMM
        hmm.train(&combined_data)?;
        
        // Check that model is trained
        assert!(hmm.trained);
        
        // Check that emission parameters are reasonable
        let vars = hmm.emission_vars();
        assert!(vars.iter().all(|&v| v > 0.0));
        
        Ok(())
    }
    
    #[test]
    fn test_regime_detection() -> Result<(), ModelError> {
        // Create detector with a fixed random seed for deterministic results
        let mut config = RegimeConfig::default();
        config.random_seed = Some(42);
        config.min_prob = 0.1; // Lower the threshold to make regime detection more sensitive
        let mut detector = MarketRegimeDetector::with_config(10, config);
        
        // Generate synthetic returns data
        let n_samples = 100;
        let n_assets = 5;
        
        // First 50 samples: low volatility
        let low_vol_returns = Array2::random((50, n_assets), Normal::new(0.001, 0.01)?);
        
        // Next 50 samples: high volatility (with much higher volatility to ensure different regimes)
        let high_vol_returns = Array2::random((50, n_assets), Normal::new(-0.002, 0.2)?);
        
        // Combine data
        let mut returns = Array2::zeros((n_samples, n_assets));
        for i in 0..50 {
            for j in 0..n_assets {
                returns[[i, j]] = low_vol_returns[[i, j]];
                returns[[i+50, j]] = high_vol_returns[[i, j]];
            }
        }
        
        // Create market data
        let features = Array2::zeros((n_samples, 10)); // Dummy features
        let market_data = MarketData::new(returns, features);
        
        // Train detector
        detector.train(&market_data)?;
        
        // Detect regimes
        let regimes = detector.detect_regimes(&market_data)?;
        
        // Check that we have the expected number of regimes
        assert_eq!(regimes.len(), n_samples - detector.window_size + 1);
        
        // Check that we have different regimes
        let mut has_different_regimes = false;
        for i in 1..regimes.len() {
            if regimes[i] != regimes[0] {
                has_different_regimes = true;
                break;
            }
        }
        assert!(has_different_regimes, "Failed to detect different regimes");
        
        Ok(())
    }
}

// Additional types for ndarray operations
use ndarray::Array3; 