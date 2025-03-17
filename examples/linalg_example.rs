use deep_risk_model::linalg;
use ndarray::{array, Array1, Array2};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Linear Algebra Example");
    println!("=====================");
    
    // Create some test matrices
    let a = array![
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ];
    
    let b = array![
        [9.0, 8.0, 7.0],
        [6.0, 5.0, 4.0],
        [3.0, 2.0, 1.0]
    ];
    
    let x = array![1.0, 2.0, 3.0];
    
    // Matrix multiplication
    println!("\nMatrix Multiplication:");
    let c = linalg::matmul(&a, &b);
    println!("A * B =\n{}", c);
    
    // Matrix-vector multiplication
    println!("\nMatrix-Vector Multiplication:");
    let y = linalg::matvec(&a, &x);
    println!("A * x = {}", y);
    
    // Dot product
    println!("\nDot Product:");
    let dot_product = linalg::dot(&x, &x);
    println!("x · x = {}", dot_product);
    
    // Transpose
    println!("\nTranspose:");
    let a_t = linalg::transpose(&a);
    println!("A^T =\n{}", a_t);
    
    // SVD decomposition
    println!("\nSingular Value Decomposition:");
    match linalg::svd(&a) {
        Ok((u, s, v)) => {
            println!("U =\n{}", u);
            println!("S = {}", s);
            println!("V =\n{}", v);
            
            // Verify SVD: A ≈ U * diag(S) * V^T
            let s_diag = Array2::from_diag(&s);
            let reconstructed = linalg::matmul(&linalg::matmul(&u, &s_diag), &linalg::transpose(&v));
            println!("Reconstructed A =\n{}", reconstructed);
        },
        Err(e) => println!("SVD failed: {}", e),
    }
    
    // QR decomposition
    println!("\nQR Decomposition:");
    match linalg::qr(&a) {
        Ok((q, r)) => {
            println!("Q =\n{}", q);
            println!("R =\n{}", r);
            
            // Verify QR: A = Q * R
            let reconstructed = linalg::matmul(&q, &r);
            println!("Reconstructed A =\n{}", reconstructed);
        },
        Err(e) => println!("QR decomposition failed: {}", e),
    }
    
    // Create a symmetric positive definite matrix for Cholesky
    let spd = array![
        [4.0, 1.0, 1.0],
        [1.0, 5.0, 2.0],
        [1.0, 2.0, 6.0]
    ];
    
    // Cholesky decomposition
    println!("\nCholesky Decomposition:");
    match linalg::cholesky(&spd) {
        Ok(l) => {
            println!("L =\n{}", l);
            
            // Verify Cholesky: A = L * L^T
            let reconstructed = linalg::matmul(&l, &linalg::transpose(&l));
            println!("Reconstructed A =\n{}", reconstructed);
        },
        Err(e) => println!("Cholesky decomposition failed: {}", e),
    }
    
    // Linear system solving
    println!("\nSolving Linear System Ax = b:");
    let b_vec = array![6.0, 15.0, 24.0];
    match linalg::solve(&a, &b_vec) {
        Ok(solution) => {
            println!("x = {}", solution);
            
            // Verify solution: A * x ≈ b
            let reconstructed = linalg::matvec(&a, &solution);
            println!("A * x = {}", reconstructed);
        },
        Err(e) => println!("Linear system solving failed: {}", e),
    }
    
    // Matrix inverse
    println!("\nMatrix Inverse:");
    // Use a non-singular matrix
    let non_singular = array![
        [1.0, 2.0, 3.0],
        [0.0, 1.0, 4.0],
        [5.0, 6.0, 0.0]
    ];
    match linalg::inv(&non_singular) {
        Ok(inv_a) => {
            println!("A^-1 =\n{}", inv_a);
            
            // Verify inverse: A * A^-1 ≈ I
            let reconstructed = linalg::matmul(&non_singular, &inv_a);
            println!("A * A^-1 =\n{}", reconstructed);
        },
        Err(e) => println!("Matrix inversion failed: {}", e),
    }
    
    // Determinant
    println!("\nDeterminant:");
    match linalg::det(&non_singular) {
        Ok(det_a) => println!("det(A) = {}", det_a),
        Err(e) => println!("Determinant calculation failed: {}", e),
    }
    
    // Eigenvalues and eigenvectors of a symmetric matrix
    println!("\nEigenvalues and Eigenvectors:");
    let symmetric = array![
        [2.0, 1.0, 1.0],
        [1.0, 3.0, 1.0],
        [1.0, 1.0, 4.0]
    ];
    match linalg::eigh(&symmetric) {
        Ok((eigenvalues, eigenvectors)) => {
            println!("Eigenvalues = {}", eigenvalues);
            println!("Eigenvectors =\n{}", eigenvectors);
            
            // Verify: A * v = λ * v for the first eigenvector
            let first_eigenvector = eigenvectors.column(0).to_owned();
            let first_eigenvalue = eigenvalues[0];
            let av = linalg::matvec(&symmetric, &first_eigenvector);
            let lambda_v = &first_eigenvector * first_eigenvalue;
            println!("A * v = {}", av);
            println!("λ * v = {}", lambda_v);
        },
        Err(e) => println!("Eigendecomposition failed: {}", e),
    }
    
    Ok(())
} 