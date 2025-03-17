use deep_risk_model::prelude::{
    MarketData
};
use deep_risk_model::regime::RegimeType;
use deep_risk_model::regime_risk_model::RegimeAwareRiskModel;
use deep_risk_model::backtest::{Backtest, BacktestResults, HistoricalScenarioGenerator, StressScenarioGenerator};
use ndarray::Array2;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Market Regime Detection and Backtesting Example ===");
    
    // Check if BLAS is available
    if !cfg!(feature = "blas-enabled") {
        println!("This example requires the 'blas-enabled' feature to be enabled.");
        println!("Try running with: cargo run --example regime_example --features blas-enabled");
        return Ok(());
    }
    
    // Generate synthetic market data
    println!("Generating synthetic market data...");
    let (market_data, regime_changes) = generate_synthetic_data()?;
    
    // Create regime-aware risk model
    println!("Creating regime-aware risk model...");
    let d_model = 64;  // Updated to match TransformerRiskModel requirements
    let n_heads = 8;
    let d_ff = 256;
    let n_layers = 3;
    let window_size = 10;  // Reduced from 20 to be smaller than train_window
    
    // Create a custom config with smaller max_seq_len
    let mut config = deep_risk_model::transformer::TransformerConfig::new(
        64, // n_assets
        d_model,
        n_heads,
        d_ff,
        n_layers
    );
    config.max_seq_len = 10; // Use a smaller max_seq_len for the example
    
    let mut model = RegimeAwareRiskModel::with_transformer_config(
        d_model, n_heads, d_ff, n_layers, window_size, config
    )?;
    
    // Create backtest
    println!("Setting up backtest...");
    let train_window = 20;  // Reduced from 50 to be smaller than the data segments
    let test_window = 50;   // Reduced from 150 to be smaller than the data segments
    let rebalance_freq = 10;
    let risk_aversion = 1.0;
    
    let backtest = Backtest::new(train_window, test_window, rebalance_freq, risk_aversion);
    
    // Run backtest
    println!("Running backtest...");
    let results = backtest.run(&mut model, &market_data).await?;
    
    // Print results
    println!("\nBacktest Results:");
    results.print_summary();
    
    // Compare detected regimes with actual regime changes
    println!("\nRegime Detection Accuracy:");
    print_regime_detection_accuracy(&results, &regime_changes);
    
    // Run scenario analysis
    println!("\nRunning scenario analysis...");
    let scenario_results = run_scenario_analysis(&backtest, &mut model, &market_data).await?;
    
    // Print scenario results
    println!("\nScenario Analysis Results:");
    for (name, results) in &scenario_results {
        println!("\n=== Scenario: {} ===", name);
        println!("Sharpe Ratio: {:.2}", results.sharpe_ratio());
        println!("Volatility: {:.2}%", results.volatility() * 100.0);
        println!("Max Drawdown: {:.2}%", results.max_drawdown() * 100.0);
    }
    
    Ok(())
}

/// Generate synthetic market data with regime changes
fn generate_synthetic_data() -> Result<(MarketData, Vec<(usize, RegimeType)>), Box<dyn Error>> {
    let n_samples = 1000;  // Increased from 500 to ensure enough data
    let n_assets = 64;  // Updated to match d_model
    let n_features = 64;  // Updated to match d_model
    
    // Define regime changes
    let regime_changes = vec![
        (0, RegimeType::Normal),
        (200, RegimeType::LowVolatility),
        (400, RegimeType::Normal),
        (600, RegimeType::HighVolatility),
        (800, RegimeType::Crisis),
    ];
    
    // Initialize returns and features
    let mut returns = Array2::zeros((n_samples, n_assets));
    let mut features = Array2::zeros((n_samples, n_features));
    
    // Generate data for each regime
    for i in 0..regime_changes.len() {
        let (start_idx, regime) = regime_changes[i];
        let end_idx = if i < regime_changes.len() - 1 {
            regime_changes[i + 1].0
        } else {
            n_samples
        };
        
        // Set parameters based on regime
        let (mean_return, volatility) = match regime {
            RegimeType::LowVolatility => (0.001, 0.01),
            RegimeType::Normal => (0.0005, 0.015),
            RegimeType::HighVolatility => (0.0, 0.025),
            RegimeType::Crisis => (-0.002, 0.04),
        };
        
        // Generate returns for this regime
        let regime_returns = Array2::random((end_idx - start_idx, n_assets), Normal::new(mean_return, volatility)?);
        
        // Generate features for this regime
        let regime_features = Array2::random((end_idx - start_idx, n_features), Normal::new(0.0, 1.0)?);
        
        // Copy to main arrays
        for j in 0..(end_idx - start_idx) {
            for k in 0..n_assets {
                returns[[start_idx + j, k]] = regime_returns[[j, k]];
            }
            
            for k in 0..n_features {
                features[[start_idx + j, k]] = regime_features[[j, k]];
            }
        }
    }
    
    // Create market data
    let market_data = MarketData::new(returns, features);
    
    Ok((market_data, regime_changes))
}

/// Print regime detection accuracy
fn print_regime_detection_accuracy(results: &BacktestResults, actual_changes: &[(usize, RegimeType)]) {
    let detected_changes = results.regime_transitions();
    
    println!("Actual regime changes:");
    for (idx, regime) in actual_changes {
        println!("  Period {}: {}", idx, regime);
    }
    
    println!("\nDetected regime changes:");
    for (idx, regime) in detected_changes {
        println!("  Period {}: {}", idx, regime);
    }
    
    // Calculate simple accuracy metric
    let mut correct_detections = 0;
    let mut total_detections = 0;
    
    for (actual_idx, actual_regime) in actual_changes {
        // Find closest detection
        let mut closest_detection = None;
        let mut min_distance = usize::MAX;
        
        for (detected_idx, detected_regime) in detected_changes {
            let distance = if *detected_idx > *actual_idx {
                *detected_idx - *actual_idx
            } else {
                *actual_idx - *detected_idx
            };
            
            if distance < min_distance {
                min_distance = distance;
                closest_detection = Some((*detected_idx, *detected_regime));
            }
        }
        
        // Check if detection is close enough and correct
        if let Some((detected_idx, detected_regime)) = closest_detection {
            total_detections += 1;
            
            // Allow for some delay in detection (within 20 periods)
            if min_distance <= 20 && detected_regime == *actual_regime {
                correct_detections += 1;
                println!("  ✓ Correctly detected {} at period {} (actual: {})", 
                    detected_regime, detected_idx, actual_idx);
            } else {
                println!("  ✗ Incorrectly detected {} at period {} (actual: {} at period {})", 
                    detected_regime, detected_idx, actual_regime, actual_idx);
            }
        }
    }
    
    // Print accuracy
    if total_detections > 0 {
        let accuracy = (correct_detections as f32) / (total_detections as f32) * 100.0;
        println!("\nDetection accuracy: {:.1}% ({}/{} correct)", 
            accuracy, correct_detections, total_detections);
    } else {
        println!("\nNo regime detections to evaluate.");
    }
}

/// Run scenario analysis
async fn run_scenario_analysis(
    backtest: &Backtest,
    model: &mut RegimeAwareRiskModel,
    market_data: &MarketData
) -> Result<std::collections::HashMap<String, BacktestResults>, Box<dyn Error>> {
    // Create historical scenario generator
    let historical_generator = HistoricalScenarioGenerator::new(vec![
        ("Normal".to_string(), 0, 200),
        ("HighVol".to_string(), 300, 500),
    ]);
    
    // Create stress scenario generator
    let stress_generator = StressScenarioGenerator::new(
        vec![
            ("HighVol".to_string(), 2.0),
            ("LowVol".to_string(), 0.5),
        ],
        vec![
            ("HighCorr".to_string(), 0.8),
            ("LowCorr".to_string(), 0.2),
        ],
        vec![
            ("Crash".to_string(), -0.02),
            ("Rally".to_string(), 0.02),
        ],
    );
    
    // Run historical scenarios
    let mut all_results = std::collections::HashMap::new();
    
    // Run historical scenarios
    println!("Running historical scenarios...");
    let historical_results = backtest.run_scenario(model, market_data, &historical_generator).await?;
    
    // Add historical results
    for (name, results) in historical_results {
        all_results.insert(format!("Historical_{}", name), results);
    }
    
    // Run stress scenarios
    println!("Running stress scenarios...");
    let stress_results = backtest.run_scenario(model, market_data, &stress_generator).await?;
    
    // Add stress results
    for (name, results) in stress_results {
        all_results.insert(name, results);
    }
    
    Ok(all_results)
} 