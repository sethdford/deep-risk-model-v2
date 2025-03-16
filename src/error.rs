use anyhow;
use std::error::Error as StdError;
use std::fmt;

/// Custom error types for the deep risk modeling library.
/// 
/// This module defines the various error types that can occur during model
/// operations, including initialization, training, and inference. The errors
/// are categorized into several variants to help with error handling and
/// debugging:
/// 
/// - Dimension errors: When input shapes don't match expected dimensions
/// - Configuration errors: When model parameters are invalid
/// - Training errors: When model training fails
/// - Data errors: When input data is invalid or corrupted
/// - Inference errors: When model prediction fails
/// - I/O errors: When file operations fail
/// - Shape errors: When array operations fail due to incompatible shapes
/// - Parse errors: When data parsing fails
/// - Computation errors: When numerical operations fail
/// 
/// # Example
/// 
/// ```rust
/// use deep_risk_model::error::ModelError;
/// 
/// fn validate_input_shape(shape: &[usize]) -> Result<(), ModelError> {
///     if shape.len() != 2 {
///         return Err(ModelError::InvalidDimension(
///             format!("Expected 2D input, got {}D", shape.len())
///         ));
///     }
///     Ok(())
/// }
/// ```
#[derive(Debug)]
pub enum ModelError {
    /// Error when input dimensions do not match expected dimensions
    InvalidDimension(String),

    /// Error when model configuration is invalid
    InvalidConfig(String),

    /// Error when model training fails
    TrainingError(String),

    /// Error when input data is invalid
    DataError(String),

    /// Error when model inference fails
    InferenceError(String),

    /// Error when file I/O operations fail
    IO(std::io::Error),

    /// Error when array operations fail due to shape mismatch
    Shape(ndarray::ShapeError),

    /// Error when parsing fails
    Parse(std::num::ParseFloatError),

    /// Error when dimensions don't match
    DimensionMismatch(String),

    /// Error when numerical computation fails
    ComputationError(String),
    
    /// Error when numerical operations fail (e.g., singular matrix)
    NumericalError(String),

    /// Error when an operation is not supported
    UnsupportedOperation(String),

    /// Other unspecified errors
    Other(String),

    /// External errors from dependencies
    External(Box<dyn StdError + Send + Sync>),

    /// Error when initialization fails
    InitializationError(String),

    /// Error when input is invalid
    InvalidInput(String),
    
    /// Error when a feature is not implemented
    NotImplemented(String),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelError::InvalidDimension(msg) => write!(f, "Invalid dimension: {}", msg),
            ModelError::InvalidConfig(msg) => write!(f, "Invalid configuration: {}", msg),
            ModelError::TrainingError(msg) => write!(f, "Training error: {}", msg),
            ModelError::DataError(msg) => write!(f, "Data error: {}", msg),
            ModelError::InferenceError(msg) => write!(f, "Inference error: {}", msg),
            ModelError::IO(err) => write!(f, "IO error: {}", err),
            ModelError::Shape(err) => write!(f, "Shape error: {}", err),
            ModelError::Parse(err) => write!(f, "Parse error: {}", err),
            ModelError::DimensionMismatch(msg) => write!(f, "Dimension mismatch: {}", msg),
            ModelError::ComputationError(msg) => write!(f, "Computation error: {}", msg),
            ModelError::NumericalError(msg) => write!(f, "Numerical error: {}", msg),
            ModelError::UnsupportedOperation(msg) => write!(f, "Unsupported operation: {}", msg),
            ModelError::Other(msg) => write!(f, "Other error: {}", msg),
            ModelError::External(err) => write!(f, "External error: {}", err),
            ModelError::InitializationError(msg) => write!(f, "Initialization error: {}", msg),
            ModelError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            ModelError::NotImplemented(msg) => write!(f, "Not implemented: {}", msg),
        }
    }
}

impl StdError for ModelError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            ModelError::External(err) => Some(err.as_ref()),
            _ => None,
        }
    }
}

impl From<Box<dyn StdError + Send + Sync>> for ModelError {
    fn from(err: Box<dyn StdError + Send + Sync>) -> Self {
        ModelError::External(err)
    }
}

impl From<anyhow::Error> for ModelError {
    fn from(err: anyhow::Error) -> Self {
        ModelError::External(Box::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            err.to_string(),
        )))
    }
}

impl From<std::io::Error> for ModelError {
    fn from(err: std::io::Error) -> Self {
        ModelError::External(Box::new(err))
    }
}

impl From<ndarray::ShapeError> for ModelError {
    fn from(err: ndarray::ShapeError) -> Self {
        ModelError::External(Box::new(err))
    }
}

impl From<std::num::ParseFloatError> for ModelError {
    fn from(err: std::num::ParseFloatError) -> Self {
        ModelError::External(Box::new(err))
    }
}

#[cfg(not(feature = "no_blas"))]
impl From<ndarray_linalg::error::LinalgError> for ModelError {
    fn from(err: ndarray_linalg::error::LinalgError) -> Self {
        ModelError::ComputationError(format!("Linear algebra error: {}", err))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;

    #[test]
    fn test_error_display() {
        let err = ModelError::InvalidDimension("Expected (10, 5), got (10, 3)".to_string());
        assert_eq!(
            err.to_string(),
            "Invalid dimension: Expected (10, 5), got (10, 3)"
        );

        let err = ModelError::InvalidConfig("Learning rate must be positive".to_string());
        assert_eq!(
            err.to_string(),
            "Invalid configuration: Learning rate must be positive"
        );
    }

    #[test]
    fn test_error_conversion() {
        let io_err = io::Error::new(io::ErrorKind::NotFound, "File not found");
        let model_err: ModelError = io_err.into();
        match model_err {
            ModelError::External(_) => (),
            _ => panic!("Expected External variant"),
        }
    }
}