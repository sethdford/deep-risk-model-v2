use deep_risk_model::fallback;
use ndarray::{Array, Array2};
use anyhow::Result;

fn main() -> Result<()> {
    println!("Deep Risk Model - Matrix Operations Example");
    println!("=================================");
    println!("This example demonstrates matrix operations that work");
    println!("with both BLAS-optimized and pure Rust fallback implementations.");
    println!("");

    // Matrix multiplication example
    println!("Matrix Multiplication Example:");
    let a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0])?;
    let b = Array2::from_shape_vec((2, 2), vec![5.0, 6.0, 7.0, 8.0])?;
    
    println!("Matrix A:");
    print_matrix(&a);
    println!("Matrix B:");
    print_matrix(&b);
    
    let c = fallback::matmul(&a, &b)?;
    println!("A × B:");
    print_matrix(&c);
    println!("");
    
    // Matrix-vector multiplication example
    println!("Matrix-Vector Multiplication Example:");
    let a = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    let x = Array::from_vec(vec![7.0, 8.0, 9.0]);
    
    println!("Matrix A:");
    print_matrix(&a);
    println!("Vector x: {:?}", x);
    
    let y = fallback::matvec(&a, &x)?;
    println!("A × x: {:?}", y);
    println!("");
    
    // Matrix inversion example
    println!("Matrix Inversion Example:");
    let a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0])?;
    
    println!("Matrix A:");
    print_matrix(&a);
    
    let a_inv = fallback::inv(&a)?;
    println!("A⁻¹:");
    print_matrix(&a_inv);
    
    let identity = fallback::matmul(&a, &a_inv)?;
    println!("A × A⁻¹ (should be identity matrix):");
    print_matrix(&identity);
    println!("");
    
    // Dot product example
    println!("Dot Product Example:");
    let v1 = Array::from_vec(vec![1.0, 2.0, 3.0]);
    let v2 = Array::from_vec(vec![4.0, 5.0, 6.0]);
    
    println!("v1: {:?}", v1);
    println!("v2: {:?}", v2);
    
    // dot returns f32, not Result
    let dot = fallback::dot(&v1, &v2);
    println!("v1 · v2: {}", dot);
    println!("");
    
    // Transpose example
    println!("Matrix Transpose Example:");
    let a = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    
    println!("Matrix A:");
    print_matrix(&a);
    
    // transpose returns Array2, not Result
    let a_t = fallback::transpose(&a);
    println!("A^T:");
    print_matrix(&a_t);
    println!("");
    
    // Trace example
    println!("Matrix Trace Example:");
    let a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0])?;
    
    println!("Matrix A:");
    print_matrix(&a);
    
    let trace = fallback::trace(&a)?;
    println!("Trace(A): {}", trace);
    println!("");
    
    // Frobenius norm example
    println!("Frobenius Norm Example:");
    let a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0])?;
    
    println!("Matrix A:");
    print_matrix(&a);
    
    // frobenius_norm returns f32, not Result
    let norm = fallback::frobenius_norm(&a);
    println!("||A||_F: {}", norm);
    println!("");
    
    println!("All matrix operations completed successfully!");
    
    Ok(())
}

fn print_matrix(matrix: &Array2<f32>) {
    for i in 0..matrix.shape()[0] {
        print!("[ ");
        for j in 0..matrix.shape()[1] {
            print!("{:.4} ", matrix[[i, j]]);
        }
        println!("]");
    }
} 