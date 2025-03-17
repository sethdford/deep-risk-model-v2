use deep_risk_model::prelude::*;
use ndarray::Array2;
use serde_json::{Value, from_str};
use std::fs::File;
use std::io::Read;
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Read the test event from file
    let mut file = File::open("test_event_direct.json")?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    
    // Parse the JSON
    let json: Value = from_str(&contents)?;
    
    // Extract the matrix from the JSON
    let matrix = &json["matrix"];
    let rows = matrix.as_array().unwrap().len();
    let cols = matrix[0].as_array().unwrap().len();
    
    // Convert to ndarray
    let mut data_array = Array2::<f32>::zeros((rows, cols));
    for i in 0..rows {
        for j in 0..cols {
            data_array[[i, j]] = matrix[i][j].as_f64().unwrap() as f32;
        }
    }
    
    println!("Input matrix shape: {:?}", data_array.shape());
    
    // Create MarketData - we need both returns and features
    // For simplicity, we'll use the same data for both
    let data = MarketData::new(data_array.clone(), data_array.clone());
    
    // Create and use the DeepRiskModel with 16 assets (32 features / 2)
    let model = DeepRiskModel::new(
        16, // n_assets
        2,  // n_factors
        20, // max_seq_len
        32, // d_model
        2,  // n_heads
        64, // d_ff
        2   // n_layers
    ).map_err(|e| Box::new(e) as Box<dyn Error>)?;
    
    // Generate risk factors - this is an async method so we need to await it
    let risk_factors = model.generate_risk_factors(&data).await.map_err(|e| Box::new(e) as Box<dyn Error>)?;
    
    println!("Risk factors shape: {:?}", risk_factors.factors().shape());
    println!("Covariance matrix shape: {:?}", risk_factors.covariance().shape());
    
    println!("Risk factors: {:?}", risk_factors.factors());
    println!("Covariance matrix: {:?}", risk_factors.covariance());
    
    Ok(())
}
