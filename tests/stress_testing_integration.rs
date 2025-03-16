use deep_risk_model::prelude::{
    MarketData, RegimeAwareRiskModel, RegimeConfig, RegimeType, RegimeParameters,
    Backtest, EnhancedStressScenarioGenerator, StressTestExecutor, StressTestSettings,
    ReportDetail, StressScenario, HistoricalPeriod, ScenarioCombinationSettings,
    ScenarioGenerator, TransformerConfig
};
use ndarray::{Array2, Array1};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;
use std::collections::HashMap;
use std::time::Duration;

#[tokio::test]
async fn test_stress_testing_integration() -> Result<(), Box<dyn std::error::Error>> {
    // Create model
    let d_model = 5;
    let n_heads = 2;
    let d_ff = 16;
    let n_layers = 1;
    let window_size = 10;
    
    // Create a custom transformer config with smaller max_seq_len
    let transformer_config = TransformerConfig {
        d_model,
        n_heads,
        d_ff,
        n_layers,
        dropout: 0.1,
        max_seq_len: 50, // Set to match the backtest window size
        num_static_features: 0,
        num_temporal_features: 5,
        hidden_size: 16,
    };
    
    // Create the model with the custom transformer config
    let mut model = RegimeAwareRiskModel::with_transformer_config(
        d_model, n_heads, d_ff, n_layers, window_size, transformer_config
    )?;
    
    // Set up regime parameters
    model.set_regime_parameters(
        RegimeType::LowVolatility,
        RegimeParameters {
            volatility_scale: 0.8,
            correlation_scale: 0.9,
            risk_aversion: 1.0,
        }
    );
    
    model.set_regime_parameters(
        RegimeType::HighVolatility,
        RegimeParameters {
            volatility_scale: 1.5,
            correlation_scale: 1.2,
            risk_aversion: 1.0,
        }
    );
    
    // Generate synthetic data
    let n_samples = 200;
    let n_assets = 5;
    let n_features = 5;
    
    // First 100 samples: low volatility
    let low_vol_returns = Array2::random((100, n_assets), Normal::new(0.001, 0.01)?);
    let low_vol_features = Array2::random((100, n_features), Normal::new(0.0, 1.0)?);
    
    // Next 100 samples: high volatility
    let high_vol_returns = Array2::random((100, n_assets), Normal::new(-0.002, 0.05)?);
    let high_vol_features = Array2::random((100, n_features), Normal::new(0.0, 2.0)?);
    
    // Combine data
    let mut returns = Array2::zeros((n_samples, n_assets));
    let mut features = Array2::zeros((n_samples, n_features));
    
    for i in 0..100 {
        for j in 0..n_assets {
            returns[[i, j]] = low_vol_returns[[i, j]];
            returns[[i+100, j]] = high_vol_returns[[i, j]];
        }
        
        for j in 0..n_features {
            features[[i, j]] = low_vol_features[[i, j]];
            features[[i+100, j]] = high_vol_features[[i, j]];
        }
    }
    
    // Create market data
    let market_data = MarketData::new(returns, features);
    
    // Create backtest
    let backtest = Backtest::new(50, 150, 10, 1.0);
    
    // Create enhanced stress scenario generator
    let mut generator = EnhancedStressScenarioGenerator::new();
    
    // Add custom scenarios
    generator.add_scenario(StressScenario {
        name: "Custom Crash".to_string(),
        description: "Custom market crash scenario".to_string(),
        volatility_multiplier: 2.5,
        correlation_shift: 0.7,
        return_shock: -0.03,
        asset_shocks: HashMap::new(),
        sector_shocks: HashMap::new(),
        regime_transitions: HashMap::new(),
        severity: 8,
        probability: 0.05,
    });
    
    // Add historical period
    generator.add_historical_period(HistoricalPeriod {
        name: "Test Crisis".to_string(),
        description: "Test crisis period".to_string(),
        start_date: "2020-01-01".to_string(),
        end_date: "2020-03-31".to_string(),
        regime: RegimeType::Crisis,
        scaling_factor: 1.2,
    });
    
    // Set combination settings
    generator.set_combination_settings(ScenarioCombinationSettings {
        max_combinations: 2,
        use_weighted: true,
        min_probability: 0.01,
    });
    
    // Create stress test settings
    let settings = StressTestSettings {
        parallel_scenarios: 1,
        incremental: false,
        progress_interval: Duration::from_secs(1),
        report_detail: ReportDetail::Standard,
    };
    
    // Create stress test executor
    let executor = StressTestExecutor::new(backtest, settings);
    
    // Run stress tests
    let results = executor.run_stress_tests(&mut model, &market_data, &generator).await?;
    
    // Verify results
    assert!(!results.scenario_results.is_empty());
    assert!(!results.comparisons.is_empty());
    
    // Generate report
    let report = executor.generate_report(&results);
    assert!(!report.is_empty());
    
    // Print summary
    println!("=== Stress Test Integration Test ===");
    println!("Number of scenarios: {}", results.scenario_results.len());
    println!("Base scenario Sharpe ratio: {:.2}", results.base_results.sharpe_ratio());
    
    // Print worst scenario
    let mut worst_scenario = ("None", f32::MAX);
    for (name, comparison) in &results.comparisons {
        if comparison.sharpe_diff < worst_scenario.1 {
            worst_scenario = (name, comparison.sharpe_diff);
        }
    }
    
    println!("Worst scenario: {} (Sharpe diff: {:.2})", worst_scenario.0, worst_scenario.1);
    
    Ok(())
}

#[tokio::test]
async fn test_scenario_combination() -> Result<(), Box<dyn std::error::Error>> {
    // Create synthetic data
    let n_samples = 100;
    let n_assets = 5;
    let n_features = 5;
    
    let returns = Array2::random((n_samples, n_assets), Normal::new(0.001, 0.02)?);
    let features = Array2::random((n_samples, n_features), Normal::new(0.0, 1.0)?);
    
    let market_data = MarketData::new(returns, features);
    
    // Create enhanced stress scenario generator with multiple scenarios
    let mut generator = EnhancedStressScenarioGenerator::new();
    
    // Add scenarios with different characteristics
    generator.add_scenario(StressScenario {
        name: "VolShock".to_string(),
        description: "Volatility shock".to_string(),
        volatility_multiplier: 2.0,
        correlation_shift: 0.0,
        return_shock: 0.0,
        asset_shocks: HashMap::new(),
        sector_shocks: HashMap::new(),
        regime_transitions: HashMap::new(),
        severity: 6,
        probability: 0.1,
    });
    
    generator.add_scenario(StressScenario {
        name: "CorrShock".to_string(),
        description: "Correlation shock".to_string(),
        volatility_multiplier: 1.0,
        correlation_shift: 0.8,
        return_shock: 0.0,
        asset_shocks: HashMap::new(),
        sector_shocks: HashMap::new(),
        regime_transitions: HashMap::new(),
        severity: 5,
        probability: 0.1,
    });
    
    generator.add_scenario(StressScenario {
        name: "ReturnShock".to_string(),
        description: "Return shock".to_string(),
        volatility_multiplier: 1.0,
        correlation_shift: 0.0,
        return_shock: -0.02,
        asset_shocks: HashMap::new(),
        sector_shocks: HashMap::new(),
        regime_transitions: HashMap::new(),
        severity: 7,
        probability: 0.1,
    });
    
    // Enable combinations
    generator.set_combination_settings(ScenarioCombinationSettings {
        max_combinations: 2,
        use_weighted: true,
        min_probability: 0.001,
    });
    
    // Generate scenarios
    let scenarios = generator.generate_scenarios(&market_data)?;
    
    // Verify we have individual scenarios and combinations
    assert!(scenarios.len() > 3); // Should have at least the 3 base scenarios plus combinations
    
    // Check for combination scenarios
    let has_combinations = scenarios.iter().any(|(name, _)| name.contains('+'));
    assert!(has_combinations, "No combination scenarios were generated");
    
    Ok(())
} 