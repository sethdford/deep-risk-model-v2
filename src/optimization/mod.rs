//! Optimization module for model compression and performance improvements

pub mod fallback;
pub mod quantization;
pub mod memory_opt;
pub mod backtest;
pub mod stress_testing;

// Re-exports
pub use quantization::{Quantizable, QuantizationConfig, QuantizationPrecision, Quantizer};
pub use memory_opt::{MemoryConfig, SparseTensor, ChunkedProcessor, GradientCheckpointer, MemoryMappedArray, MemoryPool};
pub use backtest::{Backtest, BacktestResults, ScenarioGenerator, HistoricalScenarioGenerator, StressScenarioGenerator};
pub use stress_testing::{EnhancedStressScenarioGenerator, StressTestExecutor, StressTestResults, 
                     StressScenario, HistoricalPeriod, ScenarioCombinationSettings, 
                     StressTestSettings, ReportDetail, ScenarioComparison}; 