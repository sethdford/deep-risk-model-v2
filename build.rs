fn main() {
    // Check if the no-blas feature is enabled
    let no_blas = std::env::var("CARGO_FEATURE_NO_BLAS").is_ok();
    
    if no_blas {
        // If no-blas is enabled, we don't want to link against any BLAS libraries
        println!("cargo:rustc-cfg=feature=\"no-blas\"");
        println!("cargo:warning=Building without BLAS support (pure Rust implementation)");
        println!("cargo:warning=Note: Some tests and examples requiring matrix operations on large matrices will fail");
    } else {
        // Let the *-src crates handle the linking
        // We don't need to do anything here as the BLAS implementation crates
        // will handle all the linking details for us
        
        // Print which BLAS implementation we're using for debugging
        if std::env::var("CARGO_FEATURE_OPENBLAS").is_ok() {
            println!("cargo:warning=Building with OpenBLAS support");
        } else if std::env::var("CARGO_FEATURE_NETLIB").is_ok() {
            println!("cargo:warning=Building with Netlib support");
        } else if std::env::var("CARGO_FEATURE_INTEL_MKL").is_ok() {
            println!("cargo:warning=Building with Intel MKL support");
        } else if std::env::var("CARGO_FEATURE_ACCELERATE").is_ok() {
            println!("cargo:warning=Building with Accelerate support");
        } else {
            println!("cargo:warning=Building with default BLAS implementation (OpenBLAS)");
            println!("cargo:warning=If you encounter linking errors, try specifying a BLAS implementation explicitly:");
            println!("cargo:warning=  cargo build --features openblas");
            println!("cargo:warning=  cargo build --features netlib");
            println!("cargo:warning=  cargo build --features intel-mkl");
            println!("cargo:warning=  cargo build --features accelerate (macOS only)");
        }
    }
    
    // Check if we're running tests
    let is_test = std::env::var("CARGO_FEATURE_TEST").is_ok() || 
                 std::env::var("CARGO_CFG_TEST").is_ok();
    
    if is_test {
        println!("cargo:warning=Running in test mode");
        if no_blas {
            println!("cargo:warning=Warning: Some tests requiring BLAS will be skipped");
        }
    }
} 