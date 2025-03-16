/// API Module for Deep Risk Model
///
/// This module provides a reusable API implementation for the Deep Risk Model.
/// It can be integrated into other applications or used as a library component.
/// 
/// Features:
/// - Configurable model parameters
/// - Training endpoint
/// - Risk factor generation
/// - Covariance estimation
/// - Health check
///
/// Example usage:
/// ```rust,no_run
/// use deep_risk_model::prelude::{ModelConfig, run_server};
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     // Create a model configuration
///     let config = ModelConfig::new(64, 8, 3, 0.1);
///     
///     run_server(config).await?;
///     Ok(())
/// }
/// ```
use crate::{
    prelude::{DeepRiskModel, MarketData, ModelConfig, ModelError, RiskModel},
};
use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::cors::{Any, CorsLayer};

#[derive(Clone)]
pub struct AppState {
    model: Arc<Mutex<DeepRiskModel>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainRequest {
    features: Vec<Vec<f32>>,
    returns: Vec<Vec<f32>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RiskFactorsResponse {
    factors: Vec<Vec<f32>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CovarianceResponse {
    covariance: Vec<Vec<f32>>,
}

pub async fn run_server(_config: ModelConfig) -> Result<(), ModelError> {
    // Create a model with appropriate dimensions
    // For this example, we'll use 100 assets and 10 factors
    let model = DeepRiskModel::new(100, 10)?;
    let state = AppState {
        model: Arc::new(Mutex::new(model)),
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/train", post(train))
        .route("/risk_factors", post(generate_risk_factors))
        .route("/covariance", post(estimate_covariance))
        .route("/health", get(health_check))
        .layer(cors)
        .with_state(state);

    let addr = "0.0.0.0:3000";
    println!("Server running on {}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await
        .map_err(|e| ModelError::IO(e))?;
    axum::serve(listener, app).await
        .map_err(|e| ModelError::IO(e))?;

    Ok(())
}

async fn train(
    State(state): State<AppState>,
    Json(request): Json<TrainRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let features = convert_to_array2(&request.features)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    let returns = convert_to_array2(&request.returns)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    
    let data = MarketData::new(features, returns);
    let mut model = state.model.lock().await;
    
    model
        .train(&data)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    Ok(StatusCode::OK)
}

async fn generate_risk_factors(
    State(state): State<AppState>,
    Json(request): Json<TrainRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let features = convert_to_array2(&request.features)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    let returns = convert_to_array2(&request.returns)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    
    let data = MarketData::new(features, returns);
    let model = state.model.lock().await;
    
    let risk_factors = model
        .generate_risk_factors(&data)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    Ok(Json(RiskFactorsResponse {
        factors: convert_from_array2(risk_factors.factors()),
    }))
}

async fn estimate_covariance(
    State(state): State<AppState>,
    Json(request): Json<TrainRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let features = convert_to_array2(&request.features)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    let returns = convert_to_array2(&request.returns)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    
    let data = MarketData::new(features, returns);
    let model = state.model.lock().await;
    
    let covariance = model
        .estimate_covariance(&data)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    Ok(Json(CovarianceResponse {
        covariance: convert_from_array2(&covariance),
    }))
}

async fn health_check() -> impl IntoResponse {
    StatusCode::OK
}

fn convert_to_array2(data: &[Vec<f32>]) -> Result<Array2<f32>, ModelError> {
    if data.is_empty() {
        return Err(ModelError::InvalidInput("Empty input data".to_string()));
    }
    
    let n_rows = data.len();
    let n_cols = data[0].len();
    
    let mut flat_data = Vec::with_capacity(n_rows * n_cols);
    for row in data {
        if row.len() != n_cols {
            return Err(ModelError::InvalidInput(
                "Inconsistent row lengths".to_string(),
            ));
        }
        flat_data.extend(row);
    }
    
    Ok(Array2::from_shape_vec((n_rows, n_cols), flat_data)
        .map_err(|e| ModelError::InvalidInput(e.to_string()))?)
}

fn convert_from_array2(array: &Array2<f32>) -> Vec<Vec<f32>> {
    let shape = array.shape();
    let mut result = Vec::with_capacity(shape[0]);
    
    for row in array.rows() {
        result.push(row.to_vec());
    }
    
    result
} 