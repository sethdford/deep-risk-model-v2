use deep_risk_model::fallback;
use ndarray::{array, Array1, Array2};

fn main() {
    println!("Deep Risk Model - No-BLAS Example");
    println!("=================================");
    println!("This example demonstrates the pure Rust fallback implementations");
    println!("for matrix operations when BLAS is not available.");
    println!();

    // Matrix multiplication example
    println!("Matrix Multiplication Example:");
    let a = array![[1.0, 2.0], [3.0, 4.0]];
    let b = array![[5.0, 6.0], [7.0, 8.0]];
    
    println!("Matrix A:");
    print_matrix(&a);
    
    println!("Matrix B:");
    print_matrix(&b);
    
    match fallback::matmul(&a, &b) {
        Ok(c) => {
            println!("A × B:");
            print_matrix(&c);
        },
        Err(e) => println!("Error: {}", e),
    }
    println!();

    // Matrix-vector multiplication example
    println!("Matrix-Vector Multiplication Example:");
    let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let x = array![7.0, 8.0, 9.0];
    
    println!("Matrix A:");
    print_matrix(&a);
    
    println!("Vector x: {:?}", x);
    
    match fallback::matvec(&a, &x) {
        Ok(result) => println!("A × x: {:?}", result),
        Err(e) => println!("Error: {}", e),
    }
    println!();

    // Matrix inversion example
    println!("Matrix Inversion Example:");
    let a = array![[1.0, 2.0], [3.0, 4.0]];
    
    println!("Matrix A:");
    print_matrix(&a);
    
    match fallback::inv(&a) {
        Ok(inv_a) => {
            println!("A⁻¹:");
            print_matrix(&inv_a);
            
            // Verify A × A⁻¹ = I
            match fallback::matmul(&a, &inv_a) {
                Ok(identity) => {
                    println!("A × A⁻¹ (should be identity matrix):");
                    print_matrix(&identity);
                },
                Err(e) => println!("Error: {}", e),
            }
        },
        Err(e) => println!("Error: {}", e),
    }
    println!();

    // Dot product example
    println!("Dot Product Example:");
    let v1 = array![1.0, 2.0, 3.0];
    let v2 = array![4.0, 5.0, 6.0];
    
    println!("v1: {:?}", v1);
    println!("v2: {:?}", v2);
    println!("v1 · v2: {}", fallback::dot(&v1, &v2));
    println!();

    // Transpose example
    println!("Matrix Transpose Example:");
    let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    
    println!("Matrix A:");
    print_matrix(&a);
    
    let a_t = fallback::transpose(&a);
    println!("A^T:");
    print_matrix(&a_t);
    println!();

    // Trace example
    println!("Matrix Trace Example:");
    let a = array![[1.0, 2.0], [3.0, 4.0]];
    
    println!("Matrix A:");
    print_matrix(&a);
    
    match fallback::trace(&a) {
        Ok(tr) => println!("Trace(A): {}", tr),
        Err(e) => println!("Error: {}", e),
    }
    println!();

    // Frobenius norm example
    println!("Frobenius Norm Example:");
    let a = array![[1.0, 2.0], [3.0, 4.0]];
    
    println!("Matrix A:");
    print_matrix(&a);
    
    println!("||A||_F: {}", fallback::frobenius_norm(&a));
    println!();

    println!("All fallback operations completed successfully!");
}

// Helper function to print matrices
fn print_matrix(matrix: &Array2<f32>) {
    for i in 0..matrix.shape()[0] {
        print!("[ ");
        for j in 0..matrix.shape()[1] {
            print!("{:.4} ", matrix[[i, j]]);
        }
        println!("]");
    }
} 