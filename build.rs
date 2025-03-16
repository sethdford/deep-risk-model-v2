fn main() {
    // Check if the no-blas feature is enabled
    let no_blas = std::env::var("CARGO_FEATURE_NO_BLAS").is_ok();
    
    // Check if any BLAS implementation is enabled
    let blas_enabled = std::env::var("CARGO_FEATURE_BLAS_ENABLED").is_ok();
    
    // Detect the operating system
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let is_windows = target_os == "windows";
    
    // If no-blas is enabled, it takes precedence over any BLAS implementation
    if no_blas {
        // If no-blas is enabled, we don't want to link against any BLAS libraries
        println!("cargo:rustc-cfg=feature=\"no-blas\"");
        println!("cargo:warning=Building without BLAS support (pure Rust implementation)");
        println!("cargo:warning=Note: Some tests and examples requiring matrix operations on large matrices will fail");
        
        // Explicitly disable any BLAS-related linking
        println!("cargo:rustc-cfg=feature=\"no-blas-linking\"");
        
        // If both no-blas and a BLAS implementation are enabled, warn about it
        if blas_enabled {
            println!("cargo:warning=Both no-blas and a BLAS implementation are enabled. Using no-blas mode.");
            println!("cargo:rustc-cfg=feature=\"no-blas-override\"");
        }
    } else if blas_enabled {
        // Check if we're using Accelerate
        let use_accelerate = std::env::var("CARGO_FEATURE_ACCELERATE").is_ok();
        
        // Special handling for Windows
        if is_windows {
            // On Windows, we need to use the system feature with vcpkg or no-blas
            let use_system = std::env::var("CARGO_FEATURE_SYSTEM").is_ok();
            
            if !use_system {
                println!("cargo:warning=On Windows, you must use either the 'system' feature with vcpkg or the 'no-blas' feature");
                println!("cargo:warning=Defaulting to no-blas mode for this build");
                println!("cargo:rustc-cfg=feature=\"no-blas\"");
                println!("cargo:rustc-cfg=feature=\"no-blas-linking\"");
            } else {
                println!("cargo:warning=Using system BLAS via vcpkg on Windows");
            }
        }
        // Special handling for macOS
        else if target_os == "macos" {
            if use_accelerate {
                // Use Apple's Accelerate framework
                println!("cargo:warning=Detected macOS with Accelerate feature, using Accelerate framework");
                println!("cargo:rustc-link-lib=framework=Accelerate");
            } else {
                println!("cargo:warning=Detected macOS, adding explicit OpenBLAS linking");
                
                // Try to find OpenBLAS in common locations
                let homebrew_paths = [
                    "/opt/homebrew/opt/openblas",
                    "/usr/local/opt/openblas",
                    "/opt/homebrew/Cellar/openblas/0.3.29",
                    "/usr/local/Cellar/openblas/0.3.29",
                ];
                
                let mut found_openblas = false;
                
                // Check if OpenBLAS exists in any of the paths
                for path in homebrew_paths.iter() {
                    if std::path::Path::new(&format!("{}/lib", path)).exists() {
                        println!("cargo:rustc-link-search={}/lib", path);
                        println!("cargo:warning=Found OpenBLAS at {}", path);
                        found_openblas = true;
                        break;
                    }
                }
                
                if !found_openblas {
                    println!("cargo:warning=Could not find OpenBLAS in standard locations, using default paths");
                }
                
                // Explicitly link to OpenBLAS, BLAS, and LAPACK
                println!("cargo:rustc-link-lib=openblas");
                println!("cargo:rustc-link-lib=blas");
                println!("cargo:rustc-link-lib=lapack");
                
                // Check if gfortran is available
                let gfortran_paths = [
                    "/opt/homebrew/opt/gfortran/lib",
                    "/usr/local/opt/gfortran/lib",
                    "/opt/homebrew/lib",
                    "/usr/local/lib",
                ];
                
                let mut found_gfortran = false;
                
                for path in gfortran_paths.iter() {
                    if std::path::Path::new(&format!("{}/libgfortran.dylib", path)).exists() {
                        println!("cargo:rustc-link-search={}", path);
                        println!("cargo:warning=Found gfortran at {}", path);
                        found_gfortran = true;
                        break;
                    }
                }
                
                // Only link against gfortran if it's available
                if found_gfortran {
                    println!("cargo:rustc-link-lib=gfortran");
                } else {
                    println!("cargo:warning=gfortran not found, skipping gfortran linking");
                    // On macOS, we can often get away without explicitly linking to gfortran
                    // as it might be linked through OpenBLAS
                }
                
                // Add standard math library
                println!("cargo:rustc-link-lib=m");
            }
        }
        
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
        } else if std::env::var("CARGO_FEATURE_SYSTEM").is_ok() {
            println!("cargo:warning=Building with system BLAS support");
        } else {
            println!("cargo:warning=Building with default BLAS implementation (OpenBLAS)");
            println!("cargo:warning=If you encounter linking errors, try specifying a BLAS implementation explicitly:");
            println!("cargo:warning=  cargo build --features openblas");
            println!("cargo:warning=  cargo build --features netlib");
            println!("cargo:warning=  cargo build --features intel-mkl");
            println!("cargo:warning=  cargo build --features accelerate (macOS only)");
            
            if is_windows {
                println!("cargo:warning=On Windows, you must use the 'system' feature with vcpkg:");
                println!("cargo:warning=  cargo build --features system");
                println!("cargo:warning=Or use the 'no-blas' feature for a pure Rust implementation:");
                println!("cargo:warning=  cargo build --features no-blas");
            }
        }
    } else {
        // Neither no-blas nor any BLAS implementation is enabled
        // This shouldn't happen with our default feature, but just in case
        println!("cargo:warning=No BLAS implementation or no-blas feature is enabled. Defaulting to no-blas mode.");
        println!("cargo:rustc-cfg=feature=\"no-blas\"");
        println!("cargo:rustc-cfg=feature=\"no-blas-linking\"");
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