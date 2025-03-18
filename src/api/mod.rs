//! API module for exposing models through HTTP and Lambda functions

pub mod api;
pub mod lambda;

// Re-exports
pub use api::{AppState, run_server}; 