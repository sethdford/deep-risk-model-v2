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
        println!("cargo:rustc-link-search=native=/usr/lib/gcc/x86_64-linux-gnu/13");
        
        // Check if OPENBLAS_PATH is set in the environment
        if let Ok(openblas_path) = std::env::var("OPENBLAS_PATH") {
            println!("cargo:rustc-link-search=native={}", openblas_path);
        }
        
        if openblas_system {
            println!("cargo:warning=Building with system OpenBLAS");
            
            // Link to OpenBLAS in different ways to ensure all symbols are found
            // Static linking
            println!("cargo:rustc-link-lib=static=openblas");
            // Dynamic linking
            println!("cargo:rustc-link-lib=dylib=openblas");
            // Plain linking
            println!("cargo:rustc-link-lib=openblas");
            
            // Explicitly link to CBLAS
            println!("cargo:rustc-link-lib=cblas");
            
            // Link to BLAS and LAPACK
            println!("cargo:rustc-link-lib=blas");
            println!("cargo:rustc-link-lib=lapack");
            
            // Link to gfortran for Fortran runtime
            println!("cargo:rustc-link-lib=gfortran");
            
            // On Ubuntu, we might need to link to libm and libpthread
            println!("cargo:rustc-link-lib=m");
            println!("cargo:rustc-link-lib=pthread");
        } else if netlib {
            println!("cargo:warning=Building with Netlib");
            println!("cargo:rustc-link-lib=cblas");
            println!("cargo:rustc-link-lib=blas");
            println!("cargo:rustc-link-lib=lapack");
        } else if intel_mkl {
            println!("cargo:warning=Building with Intel MKL");
            // MKL includes CBLAS
            println!("cargo:rustc-link-lib=mkl_rt");
        } else if accelerate {
            println!("cargo:warning=Building with Accelerate");
            // Accelerate includes CBLAS
            println!("cargo:rustc-link-lib=framework=Accelerate");
        } else {
            println!("cargo:warning=Building with default BLAS");
            
            // For default BLAS, try multiple linking approaches
            // Try OpenBLAS first (includes CBLAS)
            println!("cargo:rustc-link-lib=openblas");
            
            // Explicitly try CBLAS
            println!("cargo:rustc-link-lib=cblas");
            
            // Try standard BLAS and LAPACK
            println!("cargo:rustc-link-lib=blas");
            println!("cargo:rustc-link-lib=lapack");
            
            // Try with gfortran runtime
            println!("cargo:rustc-link-lib=gfortran");
            
            // Try with system libraries that might be needed
            println!("cargo:rustc-link-lib=m");
            println!("cargo:rustc-link-lib=pthread");
        }
    }
} 