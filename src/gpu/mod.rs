//! GPU acceleration module for high-performance computation

#[cfg(feature = "gpu")]
pub mod gpu;
#[cfg(feature = "gpu")]
pub mod gpu_transformer_risk_model;
#[cfg(feature = "gpu")]
pub mod gpu_model;

// Re-exports
#[cfg(feature = "gpu")]
pub use gpu::{ComputeDevice, GPUConfig};
#[cfg(feature = "gpu")]
pub use gpu_transformer_risk_model::GPUTransformerRiskModel;
#[cfg(feature = "gpu")]
pub use gpu_model::GPUDeepRiskModel; 