fn main() {
    // Detect the operating system
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    
    // Check if any BLAS feature is explicitly enabled
    let openblas_enabled = std::env::var("CARGO_FEATURE_OPENBLAS").is_ok();
    let accelerate_enabled = std::env::var("CARGO_FEATURE_ACCELERATE").is_ok();
    let intel_mkl_enabled = std::env::var("CARGO_FEATURE_INTEL_MKL").is_ok();
    let system_enabled = std::env::var("CARGO_FEATURE_SYSTEM").is_ok();
    let no_blas_enabled = std::env::var("CARGO_FEATURE_NO_BLAS").is_ok();
    
    let any_blas_feature_enabled = openblas_enabled || accelerate_enabled || 
                                  intel_mkl_enabled || system_enabled;
    
    // If no BLAS feature is explicitly enabled, enable the platform-specific default
    if !any_blas_feature_enabled && !no_blas_enabled {
        match target_os.as_str() {
            "macos" => {
                // On macOS, use Accelerate framework by default
                println!("cargo:rustc-cfg=feature=\"accelerate\"");
                println!("cargo:rustc-cfg=feature=\"blas-enabled\"");
                println!("cargo:warning=Using Apple's Accelerate framework for BLAS operations");
                println!("cargo:rustc-link-lib=framework=Accelerate");
                
                // Explicitly set the framework path
                println!("cargo:rustc-link-search=framework=/System/Library/Frameworks");
            },
            "windows" => {
                // On Windows, use system BLAS (requires vcpkg)
                println!("cargo:rustc-cfg=feature=\"system\"");
                println!("cargo:rustc-cfg=feature=\"blas-enabled\"");
                println!("cargo:warning=Using system BLAS via vcpkg on Windows");
                println!("cargo:warning=Make sure you have installed OpenBLAS with vcpkg:");
                println!("cargo:warning=  vcpkg install openblas:x64-windows");
            },
            _ => {
                // On Linux and other platforms, use OpenBLAS by default
                println!("cargo:rustc-cfg=feature=\"openblas\"");
                println!("cargo:rustc-cfg=feature=\"blas-enabled\"");
                println!("cargo:warning=Using OpenBLAS for BLAS operations");
                
                // Explicitly link to OpenBLAS on Linux
                println!("cargo:rustc-link-lib=openblas");
            }
        }
    }
    
    // Handle no-blas feature
    if no_blas_enabled {
        println!("cargo:rustc-cfg=feature=\"no-blas\"");
        println!("cargo:warning=Building without BLAS support (pure Rust implementation)");
        println!("cargo:warning=Note: Some operations on large matrices will be slower");
        
        // If both no-blas and a BLAS implementation are enabled, warn about it
        if any_blas_feature_enabled {
            println!("cargo:warning=Both no-blas and a BLAS implementation are enabled. Using no-blas mode.");
            
            // Disable all BLAS features at the rustc level
            if openblas_enabled {
                println!("cargo:rustc-cfg=feature=\"!openblas\"");
            }
            if accelerate_enabled {
                println!("cargo:rustc-cfg=feature=\"!accelerate\"");
            }
            if intel_mkl_enabled {
                println!("cargo:rustc-cfg=feature=\"!intel-mkl\"");
            }
            if system_enabled {
                println!("cargo:rustc-cfg=feature=\"!system\"");
            }
        }
    }
    
    // Platform-specific optimizations and configurations
    match target_os.as_str() {
        "macos" => {
            if accelerate_enabled {
                println!("cargo:warning=Using Apple's Accelerate framework for optimal performance on macOS");
                println!("cargo:rustc-link-lib=framework=Accelerate");
                
                // Explicitly set the framework path
                println!("cargo:rustc-link-search=framework=/System/Library/Frameworks");
            } else if openblas_enabled {
                println!("cargo:warning=Using OpenBLAS on macOS. Consider using the 'accelerate' feature for better performance");
                
                // Try to find OpenBLAS in common locations
                let homebrew_paths = [
                    "/opt/homebrew/opt/openblas",
                    "/usr/local/opt/openblas",
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
                    println!("cargo:warning=Could not find OpenBLAS in standard locations");
                    println!("cargo:warning=You may need to install it: brew install openblas");
                }
            }
        },
        "windows" => {
            if system_enabled {
                println!("cargo:warning=Using system BLAS on Windows (requires vcpkg)");
            } else if !no_blas_enabled {
                println!("cargo:warning=On Windows, it's recommended to use the 'system' feature with vcpkg");
                println!("cargo:warning=Or use the 'no-blas' feature for a pure Rust implementation");
            }
        },
        _ => {
            // Linux-specific configurations
            if openblas_enabled {
                println!("cargo:warning=Using OpenBLAS on Linux");
                // Try to find OpenBLAS in common locations
                let linux_paths = [
                    "/usr/lib",
                    "/usr/lib/x86_64-linux-gnu",
                    "/usr/lib/aarch64-linux-gnu",
                    "/usr/local/lib",
                ];
                
                let mut found_openblas = false;
                
                // Check if OpenBLAS exists in any of the paths
                for path in linux_paths.iter() {
                    if std::path::Path::new(&format!("{}/libopenblas.so", path)).exists() {
                        println!("cargo:rustc-link-search={}", path);
                        println!("cargo:warning=Found OpenBLAS at {}", path);
                        found_openblas = true;
                        break;
                    }
                }
                
                if !found_openblas {
                    println!("cargo:warning=Could not find OpenBLAS in standard locations");
                    println!("cargo:warning=You may need to install it: sudo apt-get install libopenblas-dev");
                }
            }
        }
    }
    
    // Check if we're running tests
    let is_test = std::env::var("CARGO_FEATURE_TEST").is_ok() || 
                 std::env::var("CARGO_CFG_TEST").is_ok();
    
    if is_test {
        println!("cargo:warning=Running in test mode");
        if no_blas_enabled {
            println!("cargo:warning=Warning: Some tests requiring BLAS will be skipped");
        }
    }
} 