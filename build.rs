fn main() {
    // Check if the no-blas feature is enabled
    let no_blas = std::env::var("CARGO_FEATURE_NO_BLAS").is_ok();
    
    if no_blas {
        // If no-blas is enabled, we don't want to link against any BLAS libraries
        println!("cargo:rustc-cfg=feature=\"no-blas\"");
        println!("cargo:warning=Building without BLAS support");
    } else {
        // Let the *-src crates handle the linking
        // We don't need to do anything here as the BLAS implementation crates
        // will handle all the linking details for us
        
        // Print which BLAS implementation we're using for debugging
        if std::env::var("CARGO_FEATURE_OPENBLAS").is_ok() {
            println!("cargo:warning=Building with OpenBLAS");
        } else if std::env::var("CARGO_FEATURE_NETLIB").is_ok() {
            println!("cargo:warning=Building with Netlib");
        } else if std::env::var("CARGO_FEATURE_INTEL_MKL").is_ok() {
            println!("cargo:warning=Building with Intel MKL");
        } else if std::env::var("CARGO_FEATURE_ACCELERATE").is_ok() {
            println!("cargo:warning=Building with Accelerate");
        } else {
            println!("cargo:warning=Building with default BLAS implementation");
        }
    }
} 