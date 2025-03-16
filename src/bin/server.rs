use axum::{
    routing::{get, post},
    extract::State,
    Json,
    Router,
};
use deep_risk_model::{
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

type SharedState = Arc<Mutex<DeepRiskModel>>;

async fn health_check() -> &'static str {
    "OK"
}

async fn generate_risk_factors(
    State(model): State<SharedState>,
    Json(request): Json<MarketDataRequest>,
) -> Result<Json<RiskFactorsResponse>, String> {
    // Convert request data to ndarray
    let n_samples = request.features.len();
    let n_assets = request.features[0].len();
    
    let mut features = Array2::zeros((n_samples, n_assets));
    let mut returns = Array2::zeros((n_samples, n_assets));
    
    for i in 0..n_samples {
        for j in 0..n_assets {
            features[[i, j]] = request.features[i][j];
            returns[[i, j]] = request.returns[i][j];
        }
    }
    
    // Create market data
    let market_data = MarketData::new(returns, features);
    
    // Generate risk factors
    let model = model.lock().await;
    let risk_factors = model.generate_risk_factors(&market_data)
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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();
    
    // Initialize model
    let model = DeepRiskModel::new(100, 10)?;
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