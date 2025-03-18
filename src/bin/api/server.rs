/// API Server for Deep Risk Model
///
/// This binary provides a standalone HTTP API server for the Deep Risk Model.
/// It exposes endpoints for:
/// - Health check: GET /health
/// - Risk factor generation: POST /risk-factors
///
/// The server handles both BLAS-enabled and no-BLAS modes with appropriate
/// fallbacks and error handling.
use axum::{
    routing::{get, post},
    extract::State,
    Json,
    Router,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use deep_risk_model::prelude::{
    MarketData,
    RiskModel,
    DeepRiskModel,
};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::cors::{CorsLayer, Any};
use tracing::info;

#[derive(Debug, Deserialize)]
struct MarketDataRequest {
    features: Vec<Vec<f32>>,
    returns: Vec<Vec<f32>>,
}

#[derive(Debug, Serialize)]
struct RiskFactorsResponse {
    factors: Vec<Vec<f32>>,
    covariance: Vec<Vec<f32>>,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
    message: String,
}

type SharedState = Arc<Mutex<DeepRiskModel>>;

async fn health_check() -> &'static str {
    "OK"
}

#[cfg(not(feature = "no-blas"))]
async fn generate_risk_factors(
    State(model_state): State<SharedState>,
    Json(request): Json<MarketDataRequest>,
) -> Result<Json<RiskFactorsResponse>, String> {
    // Convert request data to ndarray
    let n_samples = request.features.len();
    if n_samples == 0 {
        return Err("Empty features array".to_string());
    }
    
    let feature_cols = request.features[0].len();
    let n_assets = request.returns[0].len();
    
    // Validate that feature dimensions are correct (should be 2 * n_assets)
    if feature_cols != 2 * n_assets {
        return Err(format!("Feature columns ({}) should be twice the number of assets ({})", feature_cols, n_assets));
    }
    
    let mut features = Array2::zeros((n_samples, feature_cols));
    let mut returns = Array2::zeros((n_samples, n_assets));
    
    for i in 0..n_samples {
        if request.features[i].len() != feature_cols {
            return Err(format!("Inconsistent feature dimensions at row {}", i));
        }
        if request.returns[i].len() != n_assets {
            return Err(format!("Inconsistent return dimensions at row {}", i));
        }
        
        for j in 0..feature_cols {
            features[[i, j]] = request.features[i][j];
        }
        
        for j in 0..n_assets {
            returns[[i, j]] = request.returns[i][j];
        }
    }
    
    // Create market data
    let market_data = MarketData::new(returns, features);
    
    // Check if we need to create a new model with the correct dimensions
    let mut model_guard = model_state.lock().await;
    if model_guard.n_assets() != n_assets {
        info!("Creating new model with {} assets (previous: {})", n_assets, model_guard.n_assets());
        // Create a new model with the correct number of assets
        // Use 10% of assets as factors, with a minimum of 2
        let n_factors = std::cmp::max(2, (n_assets as f32 * 0.1) as usize);
        *model_guard = DeepRiskModel::new(
            n_assets, 
            n_factors,
            50,  // max_seq_len
            64,  // d_model
            4,   // n_heads
            256, // d_ff
            3    // n_layers
        )
            .map_err(|e| e.to_string())?;
        
        // Set lower thresholds for factor selection in demo mode
        model_guard.set_factor_selection_thresholds(0.01, 10.0, 0.01)
            .map_err(|e| e.to_string())?;
    }
    
    // Generate risk factors
    let risk_factors = model_guard.generate_risk_factors(&market_data)
        .await
        .map_err(|e| e.to_string())?;
    
    // Convert to response format
    let response = RiskFactorsResponse {
        factors: risk_factors.factors()
            .outer_iter()
            .map(|row| row.to_vec())
            .collect(),
        covariance: risk_factors.covariance()
            .outer_iter()
            .map(|row| row.to_vec())
            .collect(),
    };
    
    Ok(Json(response))
}

#[cfg(feature = "no-blas")]
async fn generate_risk_factors(
    _state: State<SharedState>,
    _request: Json<MarketDataRequest>,
) -> impl IntoResponse {
    let error_response = ErrorResponse {
        error: "BLAS Support Required".to_string(),
        message: "This endpoint requires BLAS support for matrix operations. The server is running in no-blas mode.".to_string(),
    };
    
    (StatusCode::NOT_IMPLEMENTED, Json(error_response))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();
    
    // Initialize model with default parameters
    // The actual model will be created dynamically based on input data
    #[cfg(not(feature = "no-blas"))]
    let model = DeepRiskModel::new(
        5,   // n_assets
        2,   // n_factors
        50,  // max_seq_len
        64,  // d_model
        4,   // n_heads
        256, // d_ff
        3    // n_layers
    )?;
    
    #[cfg(feature = "no-blas")]
    let model = {
        // In no-blas mode, use a smaller configuration to avoid matrix inversion issues
        info!("Running in no-blas mode with limited functionality");
        DeepRiskModel::new(
            5,   // n_assets
            2,   // n_factors
            20,  // max_seq_len
            32,  // d_model
            2,   // n_heads
            64,  // d_ff
            2    // n_layers
        )?
    };
    
    let shared_state = Arc::new(Mutex::new(model));
    
    // Configure CORS
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);
    
    // Build router
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/risk-factors", post(generate_risk_factors))
        .layer(cors)
        .with_state(shared_state);
    
    // Get port from environment or use default
    let port = std::env::var("PORT").unwrap_or_else(|_| "3000".to_string());
    let addr = format!("0.0.0.0:{}", port);
    
    info!("Starting server on {}", addr);
    
    // Start server
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    
    Ok(())
} 