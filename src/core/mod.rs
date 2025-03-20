//! Core module containing fundamental types and utilities

pub mod error;
pub mod types;
pub mod utils;
pub mod linalg;
pub mod random;

// Re-exports for backward compatibility
pub use error::ModelError;
pub use types::{MarketData, RiskFactors, RiskModel, ModelConfig, MCPConfig}; 