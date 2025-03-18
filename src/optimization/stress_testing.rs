use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

use crate::error::ModelError;
use crate::backtest::{Backtest, BacktestResults, ScenarioGenerator};
use crate::regime::RegimeType;
use crate::regime_risk_model::RegimeAwareRiskModel;
use crate::types::MarketData;

/// Enhanced stress scenario generator with more sophisticated capabilities
#[derive(Debug, Clone)]
pub struct EnhancedStressScenarioGenerator {
    /// Predefined stress scenarios
    scenarios: Vec<StressScenario>,
    /// Historical crisis periods
    historical_periods: Vec<HistoricalPeriod>,
    /// Scenario combination settings
    combination_settings: ScenarioCombinationSettings,
}

impl Default for EnhancedStressScenarioGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// A stress scenario definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressScenario {
    /// Scenario name
    pub name: String,
    /// Scenario description
    pub description: String,
    /// Volatility multiplier
    pub volatility_multiplier: f32,
    /// Correlation shift
    pub correlation_shift: f32,
    /// Return shock
    pub return_shock: f32,
    /// Asset-specific shocks
    pub asset_shocks: HashMap<String, f32>,
    /// Sector-specific shocks
    pub sector_shocks: HashMap<String, f32>,
    /// Regime transition probabilities
    pub regime_transitions: HashMap<RegimeType, HashMap<RegimeType, f32>>,
    /// Scenario severity (1-10)
    pub severity: u8,
    /// Scenario probability (0-1)
    pub probability: f32,
}

/// A historical crisis period
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalPeriod {
    /// Period name
    pub name: String,
    /// Period description
    pub description: String,
    /// Start date (as string)
    pub start_date: String,
    /// End date (as string)
    pub end_date: String,
    /// Associated market regime
    pub regime: RegimeType,
    /// Scaling factor for returns
    pub scaling_factor: f32,
}

/// Settings for combining scenarios
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioCombinationSettings {
    /// Maximum number of scenarios to combine
    pub max_combinations: usize,
    /// Whether to use weighted combinations
    pub use_weighted: bool,
    /// Minimum probability threshold for combinations
    pub min_probability: f32,
}

/// Stress test execution settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressTestSettings {
    /// Number of parallel scenarios to process
    pub parallel_scenarios: usize,
    /// Whether to use incremental testing
    pub incremental: bool,
    /// Progress tracking interval
    pub progress_interval: Duration,
    /// Reporting detail level
    pub report_detail: ReportDetail,
}

/// Report detail level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReportDetail {
    /// Summary only
    Summary,
    /// Standard detail
    Standard,
    /// Full detail
    Full,
}

impl fmt::Display for ReportDetail {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReportDetail::Summary => write!(f, "Summary"),
            ReportDetail::Standard => write!(f, "Standard"),
            ReportDetail::Full => write!(f, "Full"),
        }
    }
}

/// Results from a stress test
#[derive(Debug, Clone)]
pub struct StressTestResults {
    /// Base scenario results
    pub base_results: BacktestResults,
    /// Scenario results
    pub scenario_results: HashMap<String, BacktestResults>,
    /// Scenario comparisons
    pub comparisons: HashMap<String, ScenarioComparison>,
    /// Execution time
    pub execution_time: Duration,
    /// Execution settings
    pub settings: StressTestSettings,
}

/// Comparison between base and scenario results
#[derive(Debug, Clone)]
pub struct ScenarioComparison {
    /// Scenario name
    pub name: String,
    /// Return difference (percentage points)
    pub return_diff: f32,
    /// Volatility difference (percentage points)
    pub volatility_diff: f32,
    /// Sharpe ratio difference
    pub sharpe_diff: f32,
    /// Maximum drawdown difference (percentage points)
    pub drawdown_diff: f32,
    /// Regime distribution difference
    pub regime_diff: HashMap<RegimeType, f32>,
}

impl EnhancedStressScenarioGenerator {
    /// Create a new enhanced stress scenario generator
    pub fn new() -> Self {
        Self {
            scenarios: Vec::new(),
            historical_periods: Vec::new(),
            combination_settings: ScenarioCombinationSettings {
                max_combinations: 2,
                use_weighted: true,
                min_probability: 0.01,
            },
        }
    }

    /// Add a predefined stress scenario
    pub fn add_scenario(&mut self, scenario: StressScenario) {
        self.scenarios.push(scenario);
    }

    /// Add a historical crisis period
    pub fn add_historical_period(&mut self, period: HistoricalPeriod) {
        self.historical_periods.push(period);
    }

    /// Set scenario combination settings
    pub fn set_combination_settings(&mut self, settings: ScenarioCombinationSettings) {
        self.combination_settings = settings;
    }

    /// Create default scenarios
    pub fn with_default_scenarios() -> Self {
        let mut generator = Self::new();
        
        // Add market crash scenario
        generator.add_scenario(StressScenario {
            name: "Market Crash".to_string(),
            description: "Severe market downturn with high volatility".to_string(),
            volatility_multiplier: 3.0,
            correlation_shift: 0.8,
            return_shock: -0.05,
            asset_shocks: HashMap::new(),
            sector_shocks: HashMap::new(),
            regime_transitions: HashMap::new(),
            severity: 9,
            probability: 0.05,
        });
        
        // Add liquidity crisis scenario
        generator.add_scenario(StressScenario {
            name: "Liquidity Crisis".to_string(),
            description: "Market-wide liquidity shortage".to_string(),
            volatility_multiplier: 2.5,
            correlation_shift: 0.7,
            return_shock: -0.03,
            asset_shocks: HashMap::new(),
            sector_shocks: HashMap::new(),
            regime_transitions: HashMap::new(),
            severity: 8,
            probability: 0.07,
        });
        
        // Add inflation shock scenario
        generator.add_scenario(StressScenario {
            name: "Inflation Shock".to_string(),
            description: "Sudden spike in inflation".to_string(),
            volatility_multiplier: 1.8,
            correlation_shift: 0.5,
            return_shock: -0.02,
            asset_shocks: HashMap::new(),
            sector_shocks: HashMap::new(),
            regime_transitions: HashMap::new(),
            severity: 6,
            probability: 0.10,
        });
        
        // Add historical periods
        generator.add_historical_period(HistoricalPeriod {
            name: "2008 Financial Crisis".to_string(),
            description: "Global financial crisis of 2008".to_string(),
            start_date: "2008-09-01".to_string(),
            end_date: "2009-03-31".to_string(),
            regime: RegimeType::Crisis,
            scaling_factor: 1.0,
        });
        
        generator.add_historical_period(HistoricalPeriod {
            name: "2020 COVID Crash".to_string(),
            description: "Market crash due to COVID-19 pandemic".to_string(),
            start_date: "2020-02-20".to_string(),
            end_date: "2020-03-23".to_string(),
            regime: RegimeType::Crisis,
            scaling_factor: 1.0,
        });
        
        generator
    }
}

impl ScenarioGenerator for EnhancedStressScenarioGenerator {
    fn generate_scenarios(&self, base_data: &MarketData) -> Result<Vec<(String, MarketData)>, ModelError> {
        // For simplicity, we'll just return a few basic scenarios
        let mut scenarios = Vec::new();
        
        // Get the returns and features
        let returns = base_data.returns();
        let features = base_data.features();
        
        // Create a high volatility scenario
        let mut high_vol_returns = returns.clone();
        for i in 0..high_vol_returns.shape()[0] {
            for j in 0..high_vol_returns.shape()[1] {
                high_vol_returns[[i, j]] *= 2.0; // Double the returns (increases volatility)
            }
        }
        let high_vol_data = MarketData::new(high_vol_returns.clone(), features.clone());
        scenarios.push(("High_Volatility".to_string(), high_vol_data.clone()));
        
        // Create a market crash scenario
        let mut crash_returns = returns.clone();
        for i in 0..crash_returns.shape()[0] {
            for j in 0..crash_returns.shape()[1] {
                crash_returns[[i, j]] -= 0.02; // Add a negative shock
            }
        }
        let crash_data = MarketData::new(crash_returns.clone(), features.clone());
        scenarios.push(("Market_Crash".to_string(), crash_data.clone()));
        
        // Create a historical scenario
        let mut historical_returns = returns.clone();
        for i in 0..historical_returns.shape()[0] {
            for j in 0..historical_returns.shape()[1] {
                if historical_returns[[i, j]] > 0.0 {
                    historical_returns[[i, j]] *= 0.5; // Reduce positive returns
                } else {
                    historical_returns[[i, j]] *= 1.5; // Amplify negative returns
                }
            }
        }
        let historical_data = MarketData::new(historical_returns.clone(), features.clone());
        scenarios.push(("Historical_Crisis".to_string(), historical_data.clone()));
        
        // Create scenario combinations if enabled
        if self.combination_settings.max_combinations > 0 {
            // High volatility + Market crash
            let mut combined_returns = returns.clone();
            for i in 0..combined_returns.shape()[0] {
                for j in 0..combined_returns.shape()[1] {
                    combined_returns[[i, j]] = combined_returns[[i, j]] * 2.0 - 0.02; // Combine effects
                }
            }
            scenarios.push(("High_Volatility+Market_Crash".to_string(), 
                           MarketData::new(combined_returns, features.clone())));
            
            // High volatility + Historical crisis
            let mut combined_returns = returns.clone();
            for i in 0..combined_returns.shape()[0] {
                for j in 0..combined_returns.shape()[1] {
                    if combined_returns[[i, j]] > 0.0 {
                        combined_returns[[i, j]] = combined_returns[[i, j]] * 2.0 * 0.5; // Combine effects
                    } else {
                        combined_returns[[i, j]] = combined_returns[[i, j]] * 2.0 * 1.5; // Combine effects
                    }
                }
            }
            scenarios.push(("High_Volatility+Historical_Crisis".to_string(), 
                           MarketData::new(combined_returns, features.clone())));
            
            // Market crash + Historical crisis
            let mut combined_returns = returns.clone();
            for i in 0..combined_returns.shape()[0] {
                for j in 0..combined_returns.shape()[1] {
                    if combined_returns[[i, j]] > 0.0 {
                        combined_returns[[i, j]] = (combined_returns[[i, j]] - 0.02) * 0.5; // Combine effects
                    } else {
                        combined_returns[[i, j]] = (combined_returns[[i, j]] - 0.02) * 1.5; // Combine effects
                    }
                }
            }
            scenarios.push(("Market_Crash+Historical_Crisis".to_string(), 
                           MarketData::new(combined_returns, features.clone())));
        }
        
        Ok(scenarios)
    }
}

/// Stress test executor
pub struct StressTestExecutor {
    /// Backtest configuration
    backtest: Backtest,
    /// Stress test settings
    settings: StressTestSettings,
}

impl StressTestExecutor {
    /// Create a new stress test executor
    pub fn new(backtest: Backtest, settings: StressTestSettings) -> Self {
        Self {
            backtest,
            settings,
        }
    }
    
    /// Run stress tests with a RegimeAwareRiskModel
    pub async fn run_stress_tests(
        &self,
        model: &mut RegimeAwareRiskModel,
        base_data: &MarketData,
        scenario_generator: &dyn ScenarioGenerator,
    ) -> Result<StressTestResults, ModelError> {
        let start_time = Instant::now();
        
        // Run backtest on base data
        let base_results = self.backtest.run(model, base_data).await?;
        
        // Generate scenarios
        let scenarios = scenario_generator.generate_scenarios(base_data)?;
        
        // Run scenarios
        let mut scenario_results = HashMap::new();
        
        for (name, scenario_data) in scenarios {
            // Run backtest on scenario data
            let results = self.backtest.run(model, &scenario_data).await?;
            scenario_results.insert(name, results);
        }
        
        // Generate comparisons
        let comparisons = self.generate_comparisons(&base_results, &scenario_results);
        
        let execution_time = start_time.elapsed();
        
        Ok(StressTestResults {
            base_results,
            scenario_results,
            comparisons,
            execution_time,
            settings: self.settings.clone(),
        })
    }
    
    /// Generate scenario comparisons
    fn generate_comparisons(
        &self,
        base_results: &BacktestResults,
        scenario_results: &HashMap<String, BacktestResults>,
    ) -> HashMap<String, ScenarioComparison> {
        let mut comparisons = HashMap::new();
        
        for (name, results) in scenario_results {
            // Calculate differences
            let return_diff = results.annualized_return() - base_results.annualized_return();
            let volatility_diff = results.volatility() - base_results.volatility();
            let sharpe_diff = results.sharpe_ratio() - base_results.sharpe_ratio();
            let drawdown_diff = results.max_drawdown() - base_results.max_drawdown();
            
            // Calculate regime distribution differences
            let regime_diff = HashMap::new();
            
            // In a real implementation, we would compare regime distributions
            // For this example, we'll just use empty regime differences
            
            comparisons.insert(name.clone(), ScenarioComparison {
                name: name.clone(),
                return_diff,
                volatility_diff,
                sharpe_diff,
                drawdown_diff,
                regime_diff,
            });
        }
        
        comparisons
    }
    
    /// Generate stress test report
    pub fn generate_report(&self, results: &StressTestResults) -> String {
        let mut report = String::new();
        
        // Report header
        report.push_str("=== Stress Test Report ===\n\n");
        
        // Base scenario results
        report.push_str("== Base Scenario ==\n");
        report.push_str(&format!("Total Return: {:.2}%\n", 
            (results.base_results.cumulative_return() - 1.0) * 100.0));
        report.push_str(&format!("Annualized Return: {:.2}%\n", 
            results.base_results.annualized_return() * 100.0));
        report.push_str(&format!("Volatility: {:.2}%\n", 
            results.base_results.volatility() * 100.0));
        report.push_str(&format!("Sharpe Ratio: {:.2}\n", 
            results.base_results.sharpe_ratio()));
        report.push_str(&format!("Maximum Drawdown: {:.2}%\n", 
            results.base_results.max_drawdown() * 100.0));
        
        // Scenario results
        report.push_str("\n== Scenario Results ==\n");
        
        // Sort scenarios by impact (using drawdown as the primary metric)
        let mut sorted_scenarios: Vec<(&String, &BacktestResults)> = 
            results.scenario_results.iter().collect();
        
        sorted_scenarios.sort_by(|a, b| 
            b.1.max_drawdown().partial_cmp(&a.1.max_drawdown()).unwrap_or(std::cmp::Ordering::Equal));
        
        // Use a reference to avoid moving sorted_scenarios
        for (name, scenario_results) in &sorted_scenarios {
            report.push_str(&format!("\n= Scenario: {} =\n", name));
            report.push_str(&format!("Total Return: {:.2}%\n", 
                (scenario_results.cumulative_return() - 1.0) * 100.0));
            report.push_str(&format!("Annualized Return: {:.2}%\n", 
                scenario_results.annualized_return() * 100.0));
            report.push_str(&format!("Volatility: {:.2}%\n", 
                scenario_results.volatility() * 100.0));
            report.push_str(&format!("Sharpe Ratio: {:.2}\n", 
                scenario_results.sharpe_ratio()));
            report.push_str(&format!("Maximum Drawdown: {:.2}%\n", 
                scenario_results.max_drawdown() * 100.0));
            
            // Add comparison if available
            if let Some(comparison) = results.comparisons.get(*name) {
                report.push_str("\nComparison to Base:\n");
                report.push_str(&format!("Return Difference: {:.2}%\n", 
                    comparison.return_diff * 100.0));
                report.push_str(&format!("Volatility Difference: {:.2}%\n", 
                    comparison.volatility_diff * 100.0));
                report.push_str(&format!("Sharpe Ratio Difference: {:.2}\n", 
                    comparison.sharpe_diff));
                report.push_str(&format!("Drawdown Difference: {:.2}%\n", 
                    comparison.drawdown_diff * 100.0));
            }
        }
        
        // Summary
        report.push_str("\n== Summary ==\n");
        report.push_str(&format!("Number of Scenarios: {}\n", results.scenario_results.len()));
        report.push_str(&format!("Execution Time: {:?}\n", results.execution_time));
        
        // Find worst scenario
        if let Some((worst_name, _)) = sorted_scenarios.first() {
            report.push_str(&format!("Worst Scenario: {}\n", worst_name));
        }
        
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use rand_distr::Normal;
    use ndarray_rand::RandomExt;
    
    #[tokio::test]
    async fn test_enhanced_scenario_generator() -> Result<(), ModelError> {
        // Create generator with default scenarios
        let generator = EnhancedStressScenarioGenerator::with_default_scenarios();
        
        // Create test data
        let n_samples = 100;
        let n_assets = 5;
        let n_features = 10;
        
        let returns = Array2::random((n_samples, n_assets), Normal::new(0.001, 0.02).unwrap());
        let features = Array2::random((n_samples, n_features), Normal::new(0.0, 1.0).unwrap());
        
        let data = MarketData::new(returns, features);
        
        // Generate scenarios
        let scenarios = generator.generate_scenarios(&data)?;
        
        // Check that we have the expected number of scenarios
        assert_eq!(scenarios.len(), 6); // Implementation returns 3 individual scenarios + 3 combinations
        
        Ok(())
    }
} 