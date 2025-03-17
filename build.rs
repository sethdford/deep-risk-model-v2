// Define the configure_linux function without conditional compilation
fn configure_linux() {
    // Check if OpenBLAS is enabled
    if cfg!(feature = "openblas") {
        println!("cargo:warning=deep_risk_model@{}: Using OpenBLAS on Linux", env!("CARGO_PKG_VERSION"));
        
        // Try to find OpenBLAS in standard locations
        let openblas_paths = [
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib/x86_64-linux-gnu/openblas-pthread",
            "/usr/lib/openblas-base",
            "/usr/lib64",
            "/usr/local/lib",
        ];
        
        for path in openblas_paths.iter() {
            if std::path::Path::new(path).exists() {
                println!("cargo:warning=deep_risk_model@{}: Found OpenBLAS at {}", env!("CARGO_PKG_VERSION"), path);
                println!("cargo:rustc-link-search=native={}", path);
                break;
            }
        }
        
        // Check if we're using system or static OpenBLAS
        if cfg!(feature = "system") {
            println!("cargo:warning=deep_risk_model@{}: Using system-provided OpenBLAS", env!("CARGO_PKG_VERSION"));
            println!("cargo:rustc-link-lib=openblas");
        } else {
            // Let openblas-src handle the linking
            println!("cargo:warning=deep_risk_model@{}: Using openblas-src for linking", env!("CARGO_PKG_VERSION"));
        }
        
        // Link against gfortran which is needed for some LAPACK routines
        println!("cargo:rustc-link-lib=gfortran");
    }
    
    // Check if Netlib is enabled
    if cfg!(feature = "netlib") {
        println!("cargo:warning=deep_risk_model@{}: Using Netlib BLAS on Linux", env!("CARGO_PKG_VERSION"));
        println!("cargo:rustc-link-lib=blas");
        println!("cargo:rustc-link-lib=lapack");
    }
    
    // Check if Intel MKL is enabled
    if cfg!(feature = "intel-mkl") {
        println!("cargo:warning=deep_risk_model@{}: Using Intel MKL on Linux", env!("CARGO_PKG_VERSION"));
        println!("cargo:rustc-link-search=native=/opt/intel/mkl/lib/intel64");
        println!("cargo:rustc-link-lib=mkl_rt");
    }
}

// Configure Windows build
fn configure_windows() {
    if cfg!(feature = "openblas") {
        println!("cargo:warning=deep_risk_model@{}: Using OpenBLAS on Windows", env!("CARGO_PKG_VERSION"));
        
        // Check if we're using static OpenBLAS (recommended for Windows)
        if cfg!(feature = "static") {
            println!("cargo:warning=deep_risk_model@{}: Using static OpenBLAS linking", env!("CARGO_PKG_VERSION"));
            // Let openblas-src handle the static linking
        } else {
            println!("cargo:warning=deep_risk_model@{}: Using dynamic OpenBLAS linking", env!("CARGO_PKG_VERSION"));
            // For dynamic linking, we need to ensure the DLL is in the PATH
            println!("cargo:warning=deep_risk_model@{}: Ensure libopenblas.dll is in your PATH", env!("CARGO_PKG_VERSION"));
        }
    }
    
    // Check if Intel MKL is enabled
    if cfg!(feature = "intel-mkl") {
        println!("cargo:warning=deep_risk_model@{}: Using Intel MKL on Windows", env!("CARGO_PKG_VERSION"));
        // Add Intel MKL paths for Windows
        println!("cargo:rustc-link-search=native=C:\\Program Files (x86)\\Intel\\oneAPI\\mkl\\latest\\lib\\intel64");
        println!("cargo:rustc-link-lib=mkl_rt");
    }
}

// Configure macOS build
fn configure_macos() {
    if cfg!(feature = "accelerate") {
        println!("cargo:warning=deep_risk_model@{}: Using Apple's Accelerate framework", env!("CARGO_PKG_VERSION"));
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-search=framework=/System/Library/Frameworks");
    } else if cfg!(feature = "openblas") {
        println!("cargo:warning=deep_risk_model@{}: Using OpenBLAS on macOS", env!("CARGO_PKG_VERSION"));
        
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
        
        // Check if OPENBLAS_DIR environment variable is set
        if let Ok(openblas_dir) = std::env::var("OPENBLAS_DIR") {
            println!("cargo:warning=Using OPENBLAS_DIR={}", openblas_dir);
            println!("cargo:rustc-link-search=native={}/lib", openblas_dir);
        }
    }
}

fn main() {
    // Detect the operating system
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    
    // Check if any BLAS feature is explicitly enabled
    let openblas_enabled = std::env::var("CARGO_FEATURE_OPENBLAS").is_ok();
    let accelerate_enabled = std::env::var("CARGO_FEATURE_ACCELERATE").is_ok();
    let intel_mkl_enabled = std::env::var("CARGO_FEATURE_INTEL_MKL").is_ok();
    let netlib_enabled = std::env::var("CARGO_FEATURE_NETLIB").is_ok();
    let pure_rust_enabled = std::env::var("CARGO_FEATURE_PURE_RUST").is_ok();
    let system_enabled = std::env::var("CARGO_FEATURE_SYSTEM").is_ok();
    let static_enabled = std::env::var("CARGO_FEATURE_STATIC").is_ok();
    
    let any_blas_feature_enabled = openblas_enabled || accelerate_enabled || intel_mkl_enabled || netlib_enabled;
    
    // Check if we're running tests
    let is_test = std::env::var("CARGO_FEATURE_TEST").is_ok() || 
                 std::env::var("CARGO_CFG_TEST").is_ok();
    
    // Handle pure-rust feature first - this takes precedence over other features
    if pure_rust_enabled {
        println!("cargo:warning=Building with pure Rust linear algebra implementation (linfa-linalg)");
        println!("cargo:rustc-cfg=feature=\"pure-rust\"");
        return;
    }
    
    // If no BLAS feature is explicitly enabled, enable the platform-specific default
    if !any_blas_feature_enabled {
        match target_os.as_str() {
            "macos" => {
                // On macOS, use Accelerate framework by default
                println!("cargo:rustc-cfg=feature=\"accelerate\"");
                println!("cargo:rustc-cfg=feature=\"blas-enabled\"");
                println!("cargo:warning=Using Apple's Accelerate framework for BLAS operations");
                configure_macos();
            },
            "linux" => {
                // On Linux, use OpenBLAS by default
                println!("cargo:rustc-cfg=feature=\"openblas\"");
                println!("cargo:rustc-cfg=feature=\"blas-enabled\"");
                configure_linux();
            },
            "windows" => {
                // On Windows, use OpenBLAS with static linking by default
                println!("cargo:rustc-cfg=feature=\"openblas\"");
                println!("cargo:rustc-cfg=feature=\"static\"");
                println!("cargo:rustc-cfg=feature=\"blas-enabled\"");
                configure_windows();
            },
            _ => {
                // On other platforms, use OpenBLAS by default
                println!("cargo:rustc-cfg=feature=\"openblas\"");
                println!("cargo:rustc-cfg=feature=\"blas-enabled\"");
                println!("cargo:warning=Using OpenBLAS for BLAS operations");
            }
        }
    } else {
        // A specific BLAS feature is enabled
        println!("cargo:warning=Building with BLAS-accelerated linear algebra");
        println!("cargo:rustc-cfg=feature=\"blas-enabled\"");
        
        // Configure based on the target OS
        match target_os.as_str() {
            "linux" => configure_linux(),
            "macos" => configure_macos(),
            "windows" => configure_windows(),
            _ => {
                println!("cargo:warning=Unsupported target OS: {}", target_os);
                if openblas_enabled {
                    println!("cargo:warning=Attempting to use OpenBLAS on unsupported platform");
                }
            }
        }
    }
    
    // Set feature flags for conditional compilation
    if system_enabled {
        println!("cargo:rustc-cfg=feature=\"system\"");
    }
    
    if static_enabled {
        println!("cargo:rustc-cfg=feature=\"static\"");
    }
    
    if is_test {
        println!("cargo:warning=Running in test mode");
    }
} 