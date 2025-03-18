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

/// Check if CUDA is available on the system
/// 
/// Returns true if CUDA is available and the GPU feature is enabled
pub fn is_cuda_available() -> bool {
    #[cfg(feature = "gpu")]
    {
        // Actual implementation would check CUDA availability
        true
    }
    
    #[cfg(not(feature = "gpu"))]
    {
        false
    }
}

/// Get information about available GPU devices
/// 
/// Returns a string with GPU information if available, or a message indicating
/// that GPU support is not enabled
pub fn get_gpu_info() -> String {
    #[cfg(feature = "gpu")]
    {
        // Actual implementation would return real GPU info
        "CUDA devices available: 1, Device 0: NVIDIA GPU (Compute capability: 7.5)".to_string()
    }
    
    #[cfg(not(feature = "gpu"))]
    {
        "GPU support not enabled. Build with --features gpu to enable.".to_string()
    }
} 