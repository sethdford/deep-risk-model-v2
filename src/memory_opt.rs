//! Memory optimization module for handling large models and datasets.
//!
//! This module provides utilities for optimizing memory usage when working with
//! large models and datasets, including:
//! - Sparse tensor representations
//! - Chunked processing for large datasets
//! - Gradient checkpointing for reducing memory during computation
//! - Memory-mapped arrays for out-of-core computation

use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, Axis, s};
use std::collections::HashMap;
use std::sync::Arc;
use std::path::Path;
use crate::error::ModelError;
use std::fs::File;

/// Configuration for memory optimization
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Whether to use sparse tensors for weights with many zeros
    pub use_sparse_tensors: bool,
    /// Sparsity threshold for converting dense to sparse (0.0-1.0)
    pub sparsity_threshold: f32,
    /// Whether to use chunked processing for large datasets
    pub use_chunked_processing: bool,
    /// Chunk size for processing large datasets
    pub chunk_size: usize,
    /// Whether to use gradient checkpointing
    pub use_checkpointing: bool,
    /// Number of segments for gradient checkpointing
    pub checkpoint_segments: usize,
    /// Whether to use memory-mapped arrays for large datasets
    pub use_mmap: bool,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            use_sparse_tensors: false,
            sparsity_threshold: 0.7,
            use_chunked_processing: false,
            chunk_size: 1000,
            use_checkpointing: false,
            checkpoint_segments: 4,
            use_mmap: false,
        }
    }
}

/// Sparse tensor representation for memory-efficient storage
#[derive(Debug, Clone)]
pub struct SparseTensor {
    /// Non-zero values
    pub values: Array1<f32>,
    /// Indices of non-zero values (row, col)
    pub indices: Vec<(usize, usize)>,
    /// Original tensor shape
    pub shape: (usize, usize),
    /// Density (fraction of non-zero elements)
    pub density: f32,
}

impl SparseTensor {
    /// Create a new sparse tensor from a dense tensor
    pub fn from_dense(tensor: &ArrayView2<f32>, threshold: f32) -> Self {
        let (rows, cols) = tensor.dim();
        let mut values = Vec::new();
        let mut indices = Vec::new();
        
        for i in 0..rows {
            for j in 0..cols {
                let val = tensor[[i, j]];
                if val.abs() > threshold {
                    values.push(val);
                    indices.push((i, j));
                }
            }
        }
        
        let density = values.len() as f32 / (rows * cols) as f32;
        
        Self {
            values: Array1::from_vec(values),
            indices,
            shape: (rows, cols),
            density,
        }
    }
    
    /// Convert sparse tensor back to dense
    pub fn to_dense(&self) -> Array2<f32> {
        let (rows, cols) = self.shape;
        let mut dense = Array2::zeros((rows, cols));
        
        for (idx, &(i, j)) in self.indices.iter().enumerate() {
            dense[[i, j]] = self.values[idx];
        }
        
        dense
    }
    
    /// Perform sparse matrix multiplication
    pub fn dot(&self, rhs: &ArrayView2<f32>) -> Result<Array2<f32>, ModelError> {
        let (m, k) = self.shape;
        let (k2, n) = rhs.dim();
        
        if k != k2 {
            return Err(ModelError::DimensionMismatch(
                format!("Incompatible dimensions for matrix multiplication: ({}, {}) x ({}, {})", 
                        m, k, k2, n)
            ));
        }
        
        let mut result = Array2::zeros((m, n));
        
        for (idx, &(i, j)) in self.indices.iter().enumerate() {
            let val = self.values[idx];
            for col in 0..n {
                result[[i, col]] += val * rhs[[j, col]];
            }
        }
        
        Ok(result)
    }
    
    /// Calculate memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        // Values (f32)
        let values_size = self.values.len() * std::mem::size_of::<f32>();
        
        // Indices (usize, usize)
        let indices_size = self.indices.len() * 2 * std::mem::size_of::<usize>();
        
        // Shape and density
        let metadata_size = 2 * std::mem::size_of::<usize>() + std::mem::size_of::<f32>();
        
        values_size + indices_size + metadata_size
    }
}

/// Chunked processor for handling large datasets
pub struct ChunkedProcessor {
    /// Configuration for chunked processing
    config: MemoryConfig,
    /// Current chunk index
    current_chunk: usize,
    /// Total number of chunks
    total_chunks: usize,
}

impl ChunkedProcessor {
    /// Create a new chunked processor
    pub fn new(config: MemoryConfig, total_samples: usize) -> Self {
        let total_chunks = (total_samples + config.chunk_size - 1) / config.chunk_size;
        
        Self {
            config,
            current_chunk: 0,
            total_chunks,
        }
    }
    
    /// Process data in chunks
    pub fn process_in_chunks<F, T>(&mut self, data: &ArrayView2<f32>, processor: F) -> Result<Vec<T>, ModelError>
    where
        F: Fn(&ArrayView2<f32>) -> Result<T, ModelError>,
    {
        let (total_samples, features) = data.dim();
        let mut results = Vec::with_capacity(self.total_chunks);
        
        for chunk_idx in 0..self.total_chunks {
            let start = chunk_idx * self.config.chunk_size;
            let end = std::cmp::min(start + self.config.chunk_size, total_samples);
            
            let chunk = data.slice(s![start..end, ..]);
            let result = processor(&chunk)?;
            results.push(result);
            
            self.current_chunk = chunk_idx;
        }
        
        Ok(results)
    }
    
    /// Get the current progress (0.0-1.0)
    pub fn progress(&self) -> f32 {
        self.current_chunk as f32 / self.total_chunks as f32
    }
}

/// Gradient checkpointing for memory-efficient computation
pub struct GradientCheckpointer {
    /// Configuration for checkpointing
    config: MemoryConfig,
}

impl GradientCheckpointer {
    /// Create a new gradient checkpointer
    pub fn new(config: MemoryConfig) -> Self {
        Self { config }
    }
    
    /// Process sequence data with checkpointing
    pub fn process_sequence<F, T>(&self, data: &ArrayView2<f32>, processor: F) -> Result<T, ModelError>
    where
        F: Fn(&ArrayView2<f32>) -> Result<T, ModelError>,
    {
        if !self.config.use_checkpointing {
            // If checkpointing is disabled, process the entire sequence at once
            return processor(data);
        }
        
        let (seq_len, features) = data.dim();
        let segment_size = (seq_len + self.config.checkpoint_segments - 1) / self.config.checkpoint_segments;
        
        // Process each segment independently
        let mut segment_results = Vec::with_capacity(self.config.checkpoint_segments);
        
        for segment_idx in 0..self.config.checkpoint_segments {
            let start = segment_idx * segment_size;
            let end = std::cmp::min(start + segment_size, seq_len);
            
            if start >= seq_len {
                break;
            }
            
            let segment = data.slice(s![start..end, ..]);
            let result = processor(&segment)?;
            segment_results.push(result);
        }
        
        // Combine segment results
        self.combine_segment_results(segment_results)
    }
    
    /// Combine results from different segments
    fn combine_segment_results<T>(&self, segment_results: Vec<T>) -> Result<T, ModelError> {
        // This is a placeholder - actual implementation would depend on the type T
        // and how to combine results from different segments
        Err(ModelError::NotImplemented("Combining segment results must be implemented for specific types".to_string()))
    }
}

/// Memory-mapped array for out-of-core computation
pub struct MemoryMappedArray {
    /// Path to the memory-mapped file
    path: Arc<Path>,
    /// Shape of the array
    shape: Vec<usize>,
    /// Data type size in bytes
    element_size: usize,
}

impl MemoryMappedArray {
    /// Create a new memory-mapped array
    pub fn new<P: AsRef<Path>>(path: P, shape: Vec<usize>, element_size: usize) -> Result<Self, ModelError> {
        let path: Arc<Path> = Arc::from(path.as_ref());
        
        // Check if file exists
        if !path.exists() {
            // Create file with appropriate size
            let total_elements: usize = shape.iter().product();
            let file_size = total_elements * element_size;
            
            let file = File::create(&*path)
                .map_err(|e| ModelError::IO(e))?;
            
            file.set_len(file_size as u64)
                .map_err(|e| ModelError::IO(e))?;
        }
        
        Ok(Self {
            path,
            shape,
            element_size,
        })
    }
    
    /// Read a slice of the memory-mapped array
    pub fn read_slice(&self, start: &[usize], end: &[usize]) -> Result<Array2<f32>, ModelError> {
        if start.len() != self.shape.len() || end.len() != self.shape.len() {
            return Err(ModelError::InvalidDimension(
                "Start and end indices must have the same dimensionality as the array".to_string()
            ));
        }
        
        for (i, (&s, &e)) in start.iter().zip(end.iter()).enumerate() {
            if s >= e || e > self.shape[i] {
                return Err(ModelError::InvalidDimension(
                    format!("Invalid slice indices: start={:?}, end={:?}, shape={:?}", start, end, self.shape)
                ));
            }
        }
        
        // Calculate slice dimensions
        let slice_shape: Vec<usize> = start.iter().zip(end.iter())
            .map(|(&s, &e)| e - s)
            .collect();
        
        // For simplicity, we only support 2D arrays for now
        if slice_shape.len() != 2 {
            return Err(ModelError::NotImplemented("Only 2D arrays are supported for memory mapping".to_string()));
        }
        
        let (rows, cols) = (slice_shape[0], slice_shape[1]);
        let mut result = Array2::zeros((rows, cols));
        
        // Open the file and read the slice
        let file = std::fs::File::open(&self.path)
            .map_err(|e| ModelError::IO(e))?;
        let mut reader = std::io::BufReader::new(file);
        
        // Calculate offset
        let mut offset = 0;
        let mut stride = 1;
        for i in (0..self.shape.len()).rev() {
            offset += start[i] * stride;
            stride *= self.shape[i];
        }
        offset *= self.element_size;
        
        // Read data
        use std::io::{Read, Seek, SeekFrom};
        reader.seek(SeekFrom::Start(offset as u64))
            .map_err(|e| ModelError::IO(e))?;
        
        // For simplicity, we assume f32 data
        if self.element_size != 4 {
            return Err(ModelError::NotImplemented("Only f32 data is supported for memory mapping".to_string()));
        }
        
        for i in 0..rows {
            for j in 0..cols {
                let mut buf = [0u8; 4];
                reader.read_exact(&mut buf)
                    .map_err(|e| ModelError::IO(e))?;
                let val = f32::from_le_bytes(buf);
                result[[i, j]] = val;
            }
            
            // Skip to the next row if there's a gap
            if cols < self.shape[1] {
                let skip = (self.shape[1] - cols) * self.element_size;
                reader.seek(SeekFrom::Current(skip as i64))
                    .map_err(|e| ModelError::IO(e))?;
            }
        }
        
        Ok(result)
    }
    
    /// Write a slice to the memory-mapped array
    pub fn write_slice(&self, start: &[usize], data: &ArrayView2<f32>) -> Result<(), ModelError> {
        if start.len() != self.shape.len() {
            return Err(ModelError::InvalidDimension(
                "Start indices must have the same dimensionality as the array".to_string()
            ));
        }
        
        let (rows, cols) = data.dim();
        let end = vec![start[0] + rows, start[1] + cols];
        
        for (i, (&s, &e)) in start.iter().zip(end.iter()).enumerate() {
            if e > self.shape[i] {
                return Err(ModelError::InvalidDimension(
                    format!("Data doesn't fit: start={:?}, data_shape={:?}, array_shape={:?}", 
                            start, (rows, cols), self.shape)
                ));
            }
        }
        
        // Open the file and write the slice
        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(&self.path)
            .map_err(|e| ModelError::IO(e))?;
        let mut writer = std::io::BufWriter::new(file);
        
        // Calculate offset
        let mut offset = 0;
        let mut stride = 1;
        for i in (0..self.shape.len()).rev() {
            offset += start[i] * stride;
            stride *= self.shape[i];
        }
        offset *= self.element_size;
        
        // Write data
        use std::io::{Seek, SeekFrom, Write};
        writer.seek(SeekFrom::Start(offset as u64))
            .map_err(|e| ModelError::IO(e))?;
        
        for i in 0..rows {
            for j in 0..cols {
                let val = data[[i, j]];
                let bytes = val.to_le_bytes();
                writer.write_all(&bytes)
                    .map_err(|e| ModelError::IO(e))?;
            }
            
            // Skip to the next row if there's a gap
            if cols < self.shape[1] {
                let skip = (self.shape[1] - cols) * self.element_size;
                writer.seek(SeekFrom::Current(skip as i64))
                    .map_err(|e| ModelError::IO(e))?;
            }
        }
        
        writer.flush().map_err(|e| ModelError::IO(e))?;
        
        Ok(())
    }
}

/// Memory pool for efficient tensor allocation and reuse
pub struct MemoryPool {
    /// Available tensors by shape
    available: HashMap<Vec<usize>, Vec<Array2<f32>>>,
    /// Total memory usage in bytes
    memory_usage: usize,
    /// Maximum memory usage in bytes
    max_memory: usize,
}

impl MemoryPool {
    /// Create a new memory pool with the specified maximum memory
    pub fn new(max_memory: usize) -> Self {
        Self {
            available: HashMap::new(),
            memory_usage: 0,
            max_memory,
        }
    }
    
    /// Allocate a tensor of the specified shape
    pub fn allocate(&mut self, shape: &[usize]) -> Result<Array2<f32>, ModelError> {
        let shape_key = shape.to_vec();
        
        // Check if we have an available tensor of the right shape
        if let Some(tensors) = self.available.get_mut(&shape_key) {
            if let Some(tensor) = tensors.pop() {
                return Ok(tensor);
            }
        }
        
        // Calculate memory required
        let size = shape.iter().product::<usize>() * std::mem::size_of::<f32>();
        
        // Check if we have enough memory
        if self.memory_usage + size > self.max_memory {
            return Err(ModelError::Other("Memory pool exhausted".to_string()));
        }
        
        // Allocate new tensor
        let tensor = Array2::zeros((shape[0], shape[1]));
        self.memory_usage += size;
        
        Ok(tensor)
    }
    
    /// Release a tensor back to the pool
    pub fn release(&mut self, tensor: Array2<f32>) {
        let shape_key = tensor.shape().to_vec();
        
        self.available.entry(shape_key)
            .or_insert_with(Vec::new)
            .push(tensor);
    }
    
    /// Clear the memory pool
    pub fn clear(&mut self) {
        self.available.clear();
        self.memory_usage = 0;
    }
    
    /// Get current memory usage in bytes
    pub fn get_memory_usage(&self) -> usize {
        self.memory_usage
    }
    
    /// Get maximum memory limit in bytes
    pub fn get_max_memory(&self) -> usize {
        self.max_memory
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;
    
    #[test]
    fn test_sparse_tensor() {
        // Create a dense tensor with some sparsity
        let mut dense = Array2::zeros((5, 5));
        dense[[0, 0]] = 1.0;
        dense[[1, 2]] = 2.0;
        dense[[3, 4]] = 3.0;
        
        // Convert to sparse
        let sparse = SparseTensor::from_dense(&dense.view(), 0.1);
        
        // Check properties
        assert_eq!(sparse.shape, (5, 5));
        assert_eq!(sparse.values.len(), 3);
        assert_eq!(sparse.indices.len(), 3);
        assert_eq!(sparse.density, 3.0 / 25.0);
        
        // Convert back to dense
        let reconstructed = sparse.to_dense();
        
        // Check equality
        for i in 0..5 {
            for j in 0..5 {
                assert_eq!(dense[[i, j]], reconstructed[[i, j]]);
            }
        }
    }
    
    #[test]
    fn test_chunked_processor() {
        // Create test data
        let data = Array2::ones((100, 10));
        
        // Create chunked processor
        let config = MemoryConfig {
            chunk_size: 30,
            ..Default::default()
        };
        let mut processor = ChunkedProcessor::new(config, 100);
        
        // Process in chunks
        let results = processor.process_in_chunks(&data.view(), |chunk| {
            Ok(chunk.sum())
        }).unwrap();
        
        // Check results
        assert_eq!(results.len(), 4);
        assert_eq!(results[0], 30.0 * 10.0); // 30 rows * 10 columns * 1.0
        assert_eq!(results[1], 30.0 * 10.0);
        assert_eq!(results[2], 30.0 * 10.0);
        assert_eq!(results[3], 10.0 * 10.0); // 10 rows * 10 columns * 1.0
    }
    
    #[test]
    fn test_memory_pool() {
        // Create memory pool
        let max_memory = 1000 * std::mem::size_of::<f32>();
        let mut pool = MemoryPool::new(max_memory);
        
        // Allocate tensors
        let t1 = pool.allocate(&[10, 10]).unwrap();
        let t2 = pool.allocate(&[5, 5]).unwrap();
        
        // Check memory usage
        let expected_usage = 10 * 10 * std::mem::size_of::<f32>() + 5 * 5 * std::mem::size_of::<f32>();
        assert_eq!(pool.get_memory_usage(), expected_usage);
        
        // Release tensors
        pool.release(t1);
        pool.release(t2);
        
        // Memory usage remains the same
        assert_eq!(pool.get_memory_usage(), expected_usage);
        
        // Allocate again - should reuse
        let _t3 = pool.allocate(&[10, 10]).unwrap();
        
        // Memory usage should not increase
        assert_eq!(pool.get_memory_usage(), expected_usage);
        
        // Clear pool
        pool.clear();
        
        // Memory usage should be zero
        assert_eq!(pool.get_memory_usage(), 0);
    }
} 