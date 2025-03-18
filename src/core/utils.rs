use ndarray::{Array1, Array2, Array3, ArrayD, Axis};
use rand_distr::{Distribution, Normal};
use crate::error::ModelError;

/// Convert between array types
pub fn array_to_array3(array: &ArrayD<f64>) -> Result<Array3<f64>, ModelError> {
    if array.ndim() != 3 {
        return Err(ModelError::InvalidDimension(format!(
            "Expected 3D array, got {}D",
            array.ndim()
        )));
    }
    let shape = array.shape();
    let data: Vec<f64> = array.iter().cloned().collect();
    Array3::from_shape_vec((shape[0], shape[1], shape[2]), data)
        .map_err(|e| ModelError::InvalidDimension(e.to_string()))
}

/// Convert Array3 to ArrayD
pub fn array3_to_array(array: &Array3<f64>) -> ArrayD<f64> {
    array.clone().into_dyn()
}

/// Normalize array along specified axis
pub fn normalize_array(array: &ArrayD<f64>, axis: usize) -> Result<ArrayD<f64>, ModelError> {
    let mean = array.mean_axis(ndarray::Axis(axis))
        .ok_or_else(|| ModelError::InvalidDimension("Cannot compute mean".to_string()))?;
    
    let mut normalized = array.clone();
    for mut row in normalized.outer_iter_mut() {
        row -= &mean;
    }
    
    let mut std_array = normalized.clone();
    for mut row in std_array.outer_iter_mut() {
        row.mapv_inplace(|x| x * x);
    }
    
    let std = std_array.mean_axis(ndarray::Axis(axis))
        .ok_or_else(|| ModelError::InvalidDimension("Cannot compute std".to_string()))?
        .mapv(|x| x.sqrt());
    
    for mut row in normalized.outer_iter_mut() {
        row /= &std;
    }
    
    Ok(normalized)
}

/// Compute rolling statistics (mean and std) for array
pub fn rolling_stats(array: &ArrayD<f64>, window: usize) -> Result<(ArrayD<f64>, ArrayD<f64>), ModelError> {
    if window > array.shape()[0] {
        return Err(ModelError::InvalidConfig(format!(
            "Window size {} larger than array length {}",
            window,
            array.shape()[0]
        )));
    }
    
    let n = array.shape()[0];
    let mut means = Vec::with_capacity(n - window + 1);
    let mut stds = Vec::with_capacity(n - window + 1);
    
    for i in 0..=(n - window) {
        let window_data = array.slice(ndarray::s![i..i + window, .., ..]);
        let mean = window_data.mean_axis(ndarray::Axis(0))
            .ok_or_else(|| ModelError::InvalidDimension("Cannot compute mean".to_string()))?;
        
        let mut centered = window_data.to_owned();
        for mut row in centered.outer_iter_mut() {
            row -= &mean;
        }
        
        let mut std_array = centered.clone();
        for mut row in std_array.outer_iter_mut() {
            row.mapv_inplace(|x| x * x);
        }
        
        let std = std_array.mean_axis(ndarray::Axis(0))
            .ok_or_else(|| ModelError::InvalidDimension("Cannot compute std".to_string()))?
            .mapv(|x| x.sqrt());
        
        means.push(mean.to_owned());
        stds.push(std);
    }
    
    let means = ArrayD::from_shape_vec(
        ndarray::IxDyn(&[n - window + 1, array.shape()[1], array.shape()[2]]),
        means.into_iter().flatten().collect(),
    ).map_err(|e| ModelError::InvalidDimension(e.to_string()))?;
    
    let stds = ArrayD::from_shape_vec(
        ndarray::IxDyn(&[n - window + 1, array.shape()[1], array.shape()[2]]),
        stds.into_iter().flatten().collect(),
    ).map_err(|e| ModelError::InvalidDimension(e.to_string()))?;
    
    Ok((means, stds))
}

/// Initialize weights using Xavier/Glorot initialization
pub fn xavier_init(input_size: usize, output_size: usize) -> Result<Array2<f32>, ModelError> {
    let scale = (2.0 / (input_size + output_size) as f32).sqrt();
    let normal = Normal::new(0.0, scale)
        .map_err(|e| ModelError::Other(e.to_string()))?;
    
    let mut rng = rand::thread_rng();
    let mut weights = Array2::zeros((input_size, output_size));
    
    for elem in weights.iter_mut() {
        *elem = normal.sample(&mut rng);
    }
    
    Ok(weights)
}

/// Compute mean along specified axis
pub fn mean_axis(arr: &Array2<f32>, axis: Axis) -> Array2<f32> {
    arr.mean_axis(axis).unwrap().insert_axis(axis)
}

/// Compute standard deviation along specified axis
pub fn std_axis(arr: &Array2<f32>, axis: Axis) -> Array2<f32> {
    arr.std_axis(axis, 0.0).insert_axis(axis)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Normal;

    #[test]
    fn test_array_conversion() {
        let array = Array3::random((2, 3, 4), Normal::new(0.0, 1.0).unwrap());
        let dyn_array = array3_to_array(&array);
        let array3 = array_to_array3(&dyn_array).unwrap();
        assert_eq!(array, array3);
    }

    #[test]
    fn test_normalization() {
        let array = Array3::random((10, 5, 3), Normal::new(0.0, 1.0).unwrap()).into_dyn();
        let normalized = normalize_array(&array, 0).unwrap();
        let mean = normalized.mean_axis(ndarray::Axis(0))
            .expect("Failed to compute mean");
        
        let mut std_array = normalized.clone();
        for mut row in std_array.outer_iter_mut() {
            row.mapv_inplace(|x| x * x);
        }
        let std = (std_array.mean_axis(ndarray::Axis(0))
            .expect("Failed to compute std")
            .mapv(|x| x.sqrt()));
        
        assert!(mean.iter().all(|&x| x.abs() < 1e-10));
        assert!(std.iter().all(|&x| (x - 1.0).abs() < 1e-10));
    }

    #[test]
    fn test_rolling_stats() {
        let array = Array3::random((10, 5, 3), Normal::new(0.0, 1.0).unwrap()).into_dyn();
        let (means, stds) = rolling_stats(&array, 3).unwrap();
        
        // For a 3D array with shape (10, 5, 3) and window size 3,
        // we expect output shapes of (8, 5, 3) for both means and stds
        assert_eq!(means.shape(), &[8, 5, 3]);
        assert_eq!(stds.shape(), &[8, 5, 3]);
        
        // Verify that means and stds are within expected ranges
        assert!(means.iter().all(|&x| x.abs() < 5.0));
        assert!(stds.iter().all(|&x| x > 0.0 && x < 5.0));
    }
} 