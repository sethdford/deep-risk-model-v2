//! Random array generation utilities

use ndarray::{Array1, Array2, Array3, ArrayD, Dimension, ShapeBuilder};
use ndarray_rand::rand_distr::{Distribution, Standard, StandardNormal, Uniform, Normal};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use crate::error::ModelError;
use anyhow::Result;

/// Generate a random array with values from the given distribution
pub fn random<D>(shape: (usize, usize), dist: D) -> Array2<f32>
where
    D: Distribution<f32>,
{
    let (rows, cols) = shape;
    let n_elements = rows * cols;
    let mut rng = rand::thread_rng();
    let values: Vec<f32> = (0..n_elements).map(|_| dist.sample(&mut rng)).collect();
    Array2::from_shape_vec((rows, cols), values).unwrap()
}

/// Generate a random array with values from a normal distribution
pub fn random_normal(shape: (usize, usize), mean: f32, std_dev: f32) -> Result<Array2<f32>> {
    let normal = Normal::new(mean, std_dev)?;
    Ok(random(shape, normal))
}

/// Generate a random array with values from a uniform distribution
pub fn random_uniform(shape: (usize, usize), low: f32, high: f32) -> Array2<f32> {
    random(shape, Uniform::new(low, high))
}

/// Generate a random array with values from a standard normal distribution
pub fn random_standard_normal(shape: (usize, usize)) -> Array2<f32> {
    random(shape, StandardNormal)
}

/// Generate a random array with a specific random seed
pub fn random_with_seed<D>(shape: (usize, usize), dist: D, seed: u64) -> Array2<f32>
where
    D: Distribution<f32>,
{
    let (rows, cols) = shape;
    let n_elements = rows * cols;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let values: Vec<f32> = (0..n_elements).map(|_| dist.sample(&mut rng)).collect();
    Array2::from_shape_vec((rows, cols), values).unwrap()
}

/// Generate a dropout mask with the given probability
pub fn dropout_mask(shape: (usize, usize), keep_prob: f32) -> Array2<f32> {
    let uniform = Uniform::new(0.0, 1.0);
    let mask = random(shape, uniform);
    mask.mapv(|x| if x < keep_prob { 1.0 / keep_prob } else { 0.0 })
}

/// Initialize weights using Xavier/Glorot initialization
pub fn xavier_init(in_dim: usize, out_dim: usize) -> Result<Array2<f32>> {
    let shape = (in_dim, out_dim);
    let std_dev = (2.0 / (in_dim + out_dim) as f32).sqrt();
    let normal = Normal::new(0.0, std_dev)?;
    Ok(random(shape, normal))
}

/// Initialize weights using He/Kaiming initialization
pub fn he_init(in_dim: usize, out_dim: usize) -> Result<Array2<f32>> {
    let shape = (in_dim, out_dim);
    let std_dev = (2.0 / in_dim as f32).sqrt();
    let normal = Normal::new(0.0, std_dev)?;
    Ok(random(shape, normal))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_basic_random_functions() {
        // Test random_normal function
        let shape = (5, 10);
        let arr = random_normal(shape, 0.0, 1.0).unwrap();
        assert_eq!(arr.shape()[0], shape.0);
        assert_eq!(arr.shape()[1], shape.1);
        
        // Test random_uniform function
        let arr = random_uniform(shape, -1.0, 1.0);
        assert_eq!(arr.shape()[0], shape.0);
        assert_eq!(arr.shape()[1], shape.1);
        
        // Check that we can create arrays with different dimensions 
        let shape2 = (2, 20);
        let arr = random_normal(shape2, 0.0, 1.0).unwrap();
        assert_eq!(arr.shape()[0], shape2.0);
        assert_eq!(arr.shape()[1], shape2.1);
    }

    #[test]
    fn test_random_array_shapes() {
        let shape = (5, 10);
        let arr = random(shape, StandardNormal);
        assert_eq!(arr.shape()[0], shape.0);
        assert_eq!(arr.shape()[1], shape.1);
        
        let arr = random_uniform(shape, -1.0, 1.0);
        assert_eq!(arr.shape()[0], shape.0);
        assert_eq!(arr.shape()[1], shape.1);
        
        let arr = random_normal(shape, 0.0, 1.0).unwrap();
        assert_eq!(arr.shape()[0], shape.0);
        assert_eq!(arr.shape()[1], shape.1);
    }

    #[test]
    fn test_seeded_random_arrays() {
        let shape = (3, 3);
        let seed = 42;
        
        // Two arrays with the same seed should be identical
        let arr1 = random_with_seed(shape, StandardNormal, seed);
        let arr2 = random_with_seed(shape, StandardNormal, seed);
        
        assert_abs_diff_eq!(arr1, arr2, epsilon = 1e-10);
        
        // Arrays with different seeds should be different
        let arr3 = random_with_seed(shape, StandardNormal, seed + 1);
        assert!(arr1 != arr3);
    }

    #[test]
    fn test_dropout_mask() {
        let shape = (10, 10);
        let keep_prob = 0.3;
        
        let mask = dropout_mask(shape, keep_prob);
        
        // Check that all values are either 0.0 or scaled correctly
        let scale_factor = 1.0 / keep_prob;
        for &val in mask.iter() {
            assert!(val == 0.0 || (val - scale_factor).abs() < 1e-6);
        }
    }
} 