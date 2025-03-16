fn main() {
    // Check if the no-blas feature is enabled
    let no_blas = std::env::var("CARGO_FEATURE_NO_BLAS").is_ok();
    
    if no_blas {
        // If no-blas is enabled, we don't want to link against any BLAS libraries
        println!("cargo:rustc-cfg=feature=\"no-blas\"");
        println!("cargo:warning=Building without BLAS support");
    } else {
        // If no-blas is not enabled, we want to link against BLAS libraries
        println!("cargo:rustc-cfg=feature=\"with_blas\"");
        
        // Check which BLAS implementation to use
        let openblas_system = std::env::var("CARGO_FEATURE_OPENBLAS_SYSTEM").is_ok();
        let netlib = std::env::var("CARGO_FEATURE_NETLIB").is_ok();
        let intel_mkl = std::env::var("CARGO_FEATURE_INTEL_MKL").is_ok();
        let accelerate = std::env::var("CARGO_FEATURE_ACCELERATE").is_ok();
        
        // Common library paths for Linux
        println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
        println!("cargo:rustc-link-search=native=/usr/lib");
        println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu/atlas");
        
        // Check if OPENBLAS_PATH is set in the environment
        if let Ok(openblas_path) = std::env::var("OPENBLAS_PATH") {
            println!("cargo:rustc-link-search=native={}", openblas_path);
        }
        
        if openblas_system {
            println!("cargo:warning=Building with system OpenBLAS");
            
            // Explicitly link to system OpenBLAS and LAPACK libraries
            println!("cargo:rustc-link-lib=openblas");
            
            // Try to link with cblas from different sources
            // First try standard cblas
            println!("cargo:rustc-link-lib=cblas");
            // Then try atlas cblas
            println!("cargo:rustc-link-lib=atlas");
            println!("cargo:rustc-link-lib=satlas");
            
            println!("cargo:rustc-link-lib=lapack");
            println!("cargo:rustc-link-lib=lapacke");
            println!("cargo:rustc-link-lib=gfortran");
            
            // On some systems, BLAS might be a separate library
            println!("cargo:rustc-link-lib=blas");
        } else if netlib {
            println!("cargo:warning=Building with Netlib");
            println!("cargo:rustc-link-lib=blas");
            println!("cargo:rustc-link-lib=cblas");
            println!("cargo:rustc-link-lib=lapack");
        } else if intel_mkl {
            println!("cargo:warning=Building with Intel MKL");
        } else if accelerate {
            println!("cargo:warning=Building with Accelerate");
        } else {
            println!("cargo:warning=Building with default BLAS");
            
            // For default BLAS, also try to link with system libraries
            println!("cargo:rustc-link-lib=openblas");
            println!("cargo:rustc-link-lib=cblas");
            println!("cargo:rustc-link-lib=atlas");
            println!("cargo:rustc-link-lib=blas");
            println!("cargo:rustc-link-lib=lapack");
        }
    }
} 