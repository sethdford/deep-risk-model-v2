fn main() {
    // Check if the no-blas feature is enabled
    let no_blas = std::env::var("CARGO_FEATURE_NO_BLAS").is_ok();
    
    if no_blas {
        // If no-blas is enabled, we don't want to link against any BLAS libraries
        println!("cargo:rustc-cfg=feature=\"no_blas\"");
        println!("cargo:warning=Building without BLAS support");
    } else {
        // If no-blas is not enabled, we want to link against BLAS libraries
        println!("cargo:rustc-cfg=feature=\"with_blas\"");
        
        // Check which BLAS implementation to use
        let openblas_system = std::env::var("CARGO_FEATURE_OPENBLAS_SYSTEM").is_ok();
        let netlib = std::env::var("CARGO_FEATURE_NETLIB").is_ok();
        let intel_mkl = std::env::var("CARGO_FEATURE_INTEL_MKL").is_ok();
        let accelerate = std::env::var("CARGO_FEATURE_ACCELERATE").is_ok();
        
        if openblas_system {
            println!("cargo:warning=Building with system OpenBLAS");
        } else if netlib {
            println!("cargo:warning=Building with Netlib");
        } else if intel_mkl {
            println!("cargo:warning=Building with Intel MKL");
        } else if accelerate {
            println!("cargo:warning=Building with Accelerate");
        } else {
            println!("cargo:warning=Building with default BLAS");
        }
    }
} 