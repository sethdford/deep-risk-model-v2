use deep_risk_model::fallback;
use ndarray::{Array1, Array2};
use deep_risk_model::error::ModelError;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing no-blas fallback implementation");
    
    // Create small matrices that our fallback implementation can handle
    let a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0])?;
    let b = Array2::from_shape_vec((2, 2), vec![5.0, 6.0, 7.0, 8.0])?;
    
    println!("Matrix A:");
    println!("{}", a);
    
    println!("Matrix B:");
    println!("{}", b);
    
    // Test matrix multiplication
    println!("\nTesting matrix multiplication (A * B):");
    let c = fallback::matmul(&a, &b)?;
    println!("{}", c);
    
    // Test matrix inversion (only works for 2x2 and 3x3 in fallback mode)
    println!("\nTesting matrix inversion (inv(A)):");
    let a_inv = fallback::inv(&a)?;
    println!("{}", a_inv);
    
    // Test dot product
    println!("\nTesting dot product of vectors:");
    let v1 = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let v2 = Array1::from_vec(vec![4.0, 5.0, 6.0]);
    let dot_product = fallback::dot(&v1, &v2);
    println!("Dot product: {}", dot_product);
    
    println!("\nAll fallback operations completed successfully!");
    Ok(())
} 