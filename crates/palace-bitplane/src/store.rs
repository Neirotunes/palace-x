//! Tiered storage abstraction for bit-plane vectors.
//!
//! Manages vectors across RAM (coarse planes) and disk-tier (fine planes).
//! Coarse planes (sign + exponent) are kept hot in memory for rapid filtering.
//! Mantissa planes are stored separately and loaded on demand.

use crate::planes::BitPlaneVector;
use palace_core::NodeId;
use std::collections::HashMap;

/// Coarse planes stored in RAM (sign + exponent).
#[derive(Clone, Debug)]
pub struct CoarsePlanes {
    pub sign: Vec<u8>,
    pub exponent: Vec<Vec<u8>>,
}

/// Fine planes stored on disk (mantissa).
#[derive(Clone, Debug)]
pub struct FinePlanes {
    pub mantissa: Vec<Vec<u8>>,
}

/// Manages vectors across RAM (coarse) and disk-tier (fine) storage.
///
/// In production, `fine_planes` would be memory-mapped from NVMe.
/// For now, it's simulated in memory.
pub struct BitPlaneStore {
    coarse_planes: HashMap<NodeId, CoarsePlanes>,
    fine_planes: HashMap<NodeId, FinePlanes>,
    dimensions: usize,
}

impl BitPlaneStore {
    /// Create a new store for vectors with the given dimensionality.
    pub fn new(dimensions: usize) -> Self {
        BitPlaneStore {
            coarse_planes: HashMap::new(),
            fine_planes: HashMap::new(),
            dimensions,
        }
    }

    /// Insert a vector, decomposing it into coarse and fine planes.
    pub fn insert(&mut self, id: u64, vector: &[f32]) -> Result<(), String> {
        if vector.len() != self.dimensions {
            return Err(format!(
                "Vector dimension mismatch: expected {}, got {}",
                self.dimensions,
                vector.len()
            ));
        }

        let bpv = BitPlaneVector::from_f32(vector);

        self.coarse_planes.insert(
            NodeId(id),
            CoarsePlanes {
                sign: bpv.sign_plane.clone(),
                exponent: bpv.exponent_planes.clone(),
            },
        );

        self.fine_planes.insert(
            NodeId(id),
            FinePlanes {
                mantissa: bpv.mantissa_planes.clone(),
            },
        );

        Ok(())
    }

    /// Retrieve coarse vector from RAM only (sign + exponent).
    ///
    /// This is fast (no disk access) and suitable for approximate search.
    pub fn get_coarse(&self, id: NodeId) -> Option<Vec<f32>> {
        self.coarse_planes.get(&id).map(|coarse| {
            let _bytes_per_plane = (self.dimensions + 7) / 8;
            let mut result = Vec::with_capacity(self.dimensions);

            for dim in 0..self.dimensions {
                let byte_idx = dim / 8;
                let bit_idx = dim % 8;

                let sign_bit = (coarse.sign[byte_idx] >> bit_idx) & 1;
                let mut exponent_bits = 0u8;

                for exp_plane in 0..8 {
                    let bit = (coarse.exponent[exp_plane][byte_idx] >> bit_idx) & 1;
                    exponent_bits |= (bit as u8) << exp_plane;
                }

                let reconstructed_bits =
                    ((sign_bit as u32) << 31) | ((exponent_bits as u32) << 23);
                result.push(f32::from_bits(reconstructed_bits));
            }

            result
        })
    }

    /// Retrieve full vector from RAM + disk (all planes).
    pub fn get_full(&self, id: NodeId) -> Option<Vec<f32>> {
        self.get_partial(id, 23)
    }

    /// Retrieve vector with N mantissa bits (0 = coarse, 23 = full precision).
    ///
    /// # Arguments
    /// * `id` - Vector identifier
    /// * `mantissa_bits` - Number of mantissa bits to include (0-23)
    ///
    /// # Returns
    /// None if the vector is not found, Some(vector) otherwise.
    pub fn get_partial(&self, id: NodeId, mantissa_bits: u8) -> Option<Vec<f32>> {
        let coarse = self.coarse_planes.get(&id)?;
        let fine = self.fine_planes.get(&id)?;

        let mut result = Vec::with_capacity(self.dimensions);

        for dim in 0..self.dimensions {
            let byte_idx = dim / 8;
            let bit_idx = dim % 8;

            // Reconstruct sign
            let sign_bit = (coarse.sign[byte_idx] >> bit_idx) & 1;

            // Reconstruct exponent
            let mut exponent_bits = 0u8;
            for exp_plane in 0..8 {
                let bit = (coarse.exponent[exp_plane][byte_idx] >> bit_idx) & 1;
                exponent_bits |= (bit as u8) << exp_plane;
            }

            // Reconstruct mantissa (partial)
            let mut mantissa_bits_val = 0u32;
            for mant_plane in 0..mantissa_bits.min(23) as usize {
                let bit = (fine.mantissa[mant_plane][byte_idx] >> bit_idx) & 1;
                mantissa_bits_val |= (bit as u32) << mant_plane;
            }

            let reconstructed_bits = ((sign_bit as u32) << 31)
                | ((exponent_bits as u32) << 23)
                | mantissa_bits_val;
            result.push(f32::from_bits(reconstructed_bits));
        }

        Some(result)
    }

    /// Remove a vector from the store.
    ///
    /// # Returns
    /// true if the vector was present and removed, false otherwise.
    pub fn remove(&mut self, id: NodeId) -> bool {
        let coarse_removed = self.coarse_planes.remove(&id).is_some();
        let fine_removed = self.fine_planes.remove(&id).is_some();
        coarse_removed || fine_removed
    }

    /// Calculate RAM footprint (coarse planes only).
    pub fn memory_usage_bytes(&self) -> usize {
        let bytes_per_plane = (self.dimensions + 7) / 8;
        let coarse_per_vector = (1 + 8) * bytes_per_plane; // sign + exponent planes
        self.coarse_planes.len() * coarse_per_vector
            + self.fine_planes.len() * (23 * bytes_per_plane) // disk footprint (for accounting)
    }

    /// Number of vectors stored.
    pub fn len(&self) -> usize {
        self.coarse_planes.len()
    }

    /// Check if store is empty.
    pub fn is_empty(&self) -> bool {
        self.coarse_planes.is_empty()
    }

    /// Get the dimensionality of vectors in this store.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Calculate coarse storage per vector in bytes.
    pub fn coarse_size_per_vector(&self) -> usize {
        let bytes_per_plane = (self.dimensions + 7) / 8;
        (1 + 8) * bytes_per_plane
    }

    /// Calculate total storage per vector in bytes (coarse + fine).
    pub fn total_size_per_vector(&self) -> usize {
        let bytes_per_plane = (self.dimensions + 7) / 8;
        (1 + 8 + 23) * bytes_per_plane
    }
}

#[cfg(test)]
#[allow(clippy::approx_constant)]
mod tests {
    use super::*;
    use palace_core::NodeId;

    #[test]
    fn test_insert_retrieve_coarse() {
        let mut store = BitPlaneStore::new(4);
        let vector = vec![1.0, 2.5, -3.14, 0.0];

        store.insert(1, &vector).unwrap();
        let coarse = store.get_coarse(NodeId(1)).unwrap();

        assert_eq!(coarse.len(), 4);
        // Coarse should preserve sign
        for (orig, coarse_val) in vector.iter().zip(coarse.iter()) {
            assert_eq!(orig.is_sign_positive(), coarse_val.is_sign_positive());
        }
    }

    #[test]
    fn test_insert_retrieve_full() {
        let mut store = BitPlaneStore::new(5);
        let vector = vec![1.0, 2.5, -3.14, 0.0, 100.0];

        store.insert(1, &vector).unwrap();
        let full = store.get_full(NodeId(1)).unwrap();

        for (orig, retrieved) in vector.iter().zip(full.iter()) {
            assert!((orig - retrieved).abs() < 1e-6);
        }
    }

    #[test]
    fn test_get_partial() {
        let mut store = BitPlaneStore::new(3);
        let vector = vec![1.5, -2.75, 3.14159];

        store.insert(42, &vector).unwrap();

        let coarse = store.get_partial(NodeId(42), 0).unwrap();
        let partial_8 = store.get_partial(NodeId(42), 8).unwrap();
        let full = store.get_partial(NodeId(42), 23).unwrap();

        // Coarse should have lowest error
        let coarse_error: f32 = vector
            .iter()
            .zip(coarse.iter())
            .map(|(o, r)| (o - r).abs())
            .sum();

        let partial_error: f32 = vector
            .iter()
            .zip(partial_8.iter())
            .map(|(o, r)| (o - r).abs())
            .sum();

        let full_error: f32 = vector
            .iter()
            .zip(full.iter())
            .map(|(o, r)| (o - r).abs())
            .sum();

        assert!(coarse_error > partial_error);
        assert!(partial_error > full_error);
        assert!(full_error < 1e-5);
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut store = BitPlaneStore::new(4);
        let vector = vec![1.0, 2.0, 3.0]; // Wrong dimension

        let result = store.insert(1, &vector);
        assert!(result.is_err());
    }

    #[test]
    fn test_nonexistent_vector() {
        let store = BitPlaneStore::new(4);
        assert!(store.get_coarse(NodeId(999)).is_none());
        assert!(store.get_full(NodeId(999)).is_none());
    }

    #[test]
    fn test_remove() {
        let mut store = BitPlaneStore::new(3);
        let vector = vec![1.0, 2.0, 3.0];

        store.insert(1, &vector).unwrap();
        assert_eq!(store.len(), 1);

        assert!(store.remove(NodeId(1)));
        assert_eq!(store.len(), 0);
        assert!(store.get_coarse(NodeId(1)).is_none());

        assert!(!store.remove(NodeId(1))); // Already removed
    }

    #[test]
    fn test_memory_usage() {
        let mut store = BitPlaneStore::new(256);
        let vector = vec![1.0; 256];

        store.insert(1, &vector).unwrap();
        store.insert(2, &vector).unwrap();

        // For 256 dimensions: 32 bytes per plane
        // Coarse: 9 planes = 288 bytes per vector
        // Fine: 23 planes = 736 bytes per vector
        // Total per vector: 1024 bytes
        // For 2 vectors: 2048 bytes
        let expected = 2 * (9 + 23) * 32;
        assert_eq!(store.memory_usage_bytes(), expected);
    }

    #[test]
    fn test_multiple_vectors() {
        let mut store = BitPlaneStore::new(4);

        let v1 = vec![1.0, 2.0, 3.0, 4.0];
        let v2 = vec![5.0, 6.0, 7.0, 8.0];
        let v3 = vec![-1.0, -2.0, -3.0, -4.0];

        store.insert(10, &v1).unwrap();
        store.insert(20, &v2).unwrap();
        store.insert(30, &v3).unwrap();

        assert_eq!(store.len(), 3);

        let r1 = store.get_full(NodeId(10)).unwrap();
        let r2 = store.get_full(NodeId(20)).unwrap();
        let r3 = store.get_full(NodeId(30)).unwrap();

        for (o, r) in v1.iter().zip(r1.iter()) {
            assert!((o - r).abs() < 1e-6);
        }
        for (o, r) in v2.iter().zip(r2.iter()) {
            assert!((o - r).abs() < 1e-6);
        }
        for (o, r) in v3.iter().zip(r3.iter()) {
            assert!((o - r).abs() < 1e-6);
        }
    }

    #[test]
    fn test_size_calculations() {
        let store = BitPlaneStore::new(1024);

        let coarse_size = store.coarse_size_per_vector();
        let total_size = store.total_size_per_vector();

        // For 1024 dimensions: 128 bytes per plane
        assert_eq!(coarse_size, 128 * 9);
        assert_eq!(total_size, 128 * 32);
        assert!(coarse_size < total_size);
    }
}
