//! Graceful degradation for precision-aware retrieval.
//!
//! Defines precision levels and tracks confidence in retrieved results
//! based on how much data was available vs. how much was requested.

use std::fmt;

/// Precision level for vector retrieval.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PrecisionLevel {
    /// Only sign and exponent (coarse magnitude/sign information)
    CoarseOnly,
    /// N bits of mantissa precision
    Partial { bits: u8 },
    /// Full 23-bit mantissa precision
    Full,
}

impl PrecisionLevel {
    /// Get the number of mantissa bits in this precision level.
    pub fn mantissa_bits(&self) -> u8 {
        match self {
            PrecisionLevel::CoarseOnly => 0,
            PrecisionLevel::Partial { bits } => *bits,
            PrecisionLevel::Full => 23,
        }
    }

    /// Whether this precision level is acceptable for a given requirement.
    pub fn meets_requirement(&self, required: PrecisionLevel) -> bool {
        self >= &required
    }
}

impl fmt::Display for PrecisionLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PrecisionLevel::CoarseOnly => write!(f, "Coarse (0 bits)"),
            PrecisionLevel::Partial { bits } => write!(f, "Partial ({} bits)", bits),
            PrecisionLevel::Full => write!(f, "Full (23 bits)"),
        }
    }
}

/// Result of a potentially degraded retrieval.
///
/// Tracks what precision was actually achieved vs. what was requested,
/// along with a confidence score reflecting the precision level achieved.
#[derive(Clone)]
pub struct DegradedResult<T> {
    /// The retrieved value
    pub value: T,
    /// The precision level achieved
    pub precision: PrecisionLevel,
    /// Confidence from 0.0 (coarse only) to 1.0 (full precision)
    pub confidence: f32,
}

impl<T: fmt::Debug> fmt::Debug for DegradedResult<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DegradedResult")
            .field("value", &self.value)
            .field("precision", &self.precision)
            .field("confidence", &self.confidence)
            .finish()
    }
}

impl<T> DegradedResult<T> {
    /// Create a new degraded result with computed confidence.
    ///
    /// Confidence is computed as `mantissa_bits / 23.0` for a smooth scale
    /// from 0.0 (coarse) to 1.0 (full).
    pub fn new(value: T, precision: PrecisionLevel) -> Self {
        let confidence = (precision.mantissa_bits() as f32) / 23.0;
        DegradedResult {
            value,
            precision,
            confidence,
        }
    }

    /// Create a result with full precision (highest confidence).
    pub fn full(value: T) -> Self {
        DegradedResult {
            value,
            precision: PrecisionLevel::Full,
            confidence: 1.0,
        }
    }

    /// Create a result with coarse precision only (lowest confidence).
    pub fn coarse(value: T) -> Self {
        DegradedResult {
            value,
            precision: PrecisionLevel::CoarseOnly,
            confidence: 0.0,
        }
    }

    /// Create a result with partial precision.
    pub fn partial(value: T, mantissa_bits: u8) -> Self {
        assert!(mantissa_bits <= 23, "mantissa_bits must be <= 23");
        DegradedResult::new(
            value,
            PrecisionLevel::Partial {
                bits: mantissa_bits,
            },
        )
    }

    /// Transform the contained value with a function.
    pub fn map<U, F>(self, op: F) -> DegradedResult<U>
    where
        F: FnOnce(T) -> U,
    {
        DegradedResult {
            value: op(self.value),
            precision: self.precision,
            confidence: self.confidence,
        }
    }

    /// Check if the achieved precision meets a requirement.
    pub fn meets_requirement(&self, required: PrecisionLevel) -> bool {
        self.precision.meets_requirement(required)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_level_ordering() {
        let coarse = PrecisionLevel::CoarseOnly;
        let partial_8 = PrecisionLevel::Partial { bits: 8 };
        let partial_16 = PrecisionLevel::Partial { bits: 16 };
        let full = PrecisionLevel::Full;

        assert!(coarse < partial_8);
        assert!(partial_8 < partial_16);
        assert!(partial_16 < full);
        assert!(coarse < full);
    }

    #[test]
    fn test_precision_level_bits() {
        assert_eq!(PrecisionLevel::CoarseOnly.mantissa_bits(), 0);
        assert_eq!(PrecisionLevel::Partial { bits: 8 }.mantissa_bits(), 8);
        assert_eq!(PrecisionLevel::Partial { bits: 23 }.mantissa_bits(), 23);
        assert_eq!(PrecisionLevel::Full.mantissa_bits(), 23);
    }

    #[test]
    fn test_degraded_result_confidence() {
        let coarse = DegradedResult::<Vec<f32>>::coarse(vec![]);
        assert_eq!(coarse.confidence, 0.0);

        let partial = DegradedResult::<Vec<f32>>::partial(vec![], 8);
        assert!((partial.confidence - 8.0 / 23.0).abs() < 0.001);

        let full = DegradedResult::<Vec<f32>>::full(vec![]);
        assert_eq!(full.confidence, 1.0);
    }

    #[test]
    fn test_degraded_result_meets_requirement() {
        let coarse = DegradedResult::<Vec<f32>>::coarse(vec![]);
        let partial = DegradedResult::<Vec<f32>>::partial(vec![], 12);
        let full = DegradedResult::<Vec<f32>>::full(vec![]);

        // Full meets all requirements
        assert!(full.meets_requirement(PrecisionLevel::CoarseOnly));
        assert!(full.meets_requirement(PrecisionLevel::Partial { bits: 12 }));
        assert!(full.meets_requirement(PrecisionLevel::Full));

        // Partial meets coarse and lower partial requirements
        assert!(partial.meets_requirement(PrecisionLevel::CoarseOnly));
        assert!(partial.meets_requirement(PrecisionLevel::Partial { bits: 8 }));
        assert!(partial.meets_requirement(PrecisionLevel::Partial { bits: 12 }));
        assert!(!partial.meets_requirement(PrecisionLevel::Partial { bits: 16 }));
        assert!(!partial.meets_requirement(PrecisionLevel::Full));

        // Coarse only meets coarse requirement
        assert!(coarse.meets_requirement(PrecisionLevel::CoarseOnly));
        assert!(!coarse.meets_requirement(PrecisionLevel::Partial { bits: 1 }));
        assert!(!coarse.meets_requirement(PrecisionLevel::Full));
    }

    #[test]
    fn test_degraded_result_map() {
        let result = DegradedResult::partial(vec![1.0_f32, 2.0, 3.0], 16);
        let confidence = result.confidence;
        let mapped = result.map(|v| v.len());

        assert_eq!(mapped.value, 3);
        assert_eq!(mapped.precision, PrecisionLevel::Partial { bits: 16 });
        assert_eq!(mapped.confidence, confidence);
    }

    #[test]
    fn test_degraded_result_debug() {
        let result = DegradedResult::partial(vec![1.0_f32], 10);
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("DegradedResult"));
        assert!(debug_str.contains("precision"));
        assert!(debug_str.contains("confidence"));
    }

    #[test]
    fn test_precision_level_display() {
        let coarse_display = format!("{}", PrecisionLevel::CoarseOnly);
        assert!(coarse_display.contains("Coarse"));
        assert!(coarse_display.contains("0"));

        let partial_display = format!("{}", PrecisionLevel::Partial { bits: 16 });
        assert!(partial_display.contains("Partial"));
        assert!(partial_display.contains("16"));

        let full_display = format!("{}", PrecisionLevel::Full);
        assert!(full_display.contains("Full"));
        assert!(full_display.contains("23"));
    }

    #[test]
    fn test_precision_level_equality() {
        let p1 = PrecisionLevel::Partial { bits: 8 };
        let p2 = PrecisionLevel::Partial { bits: 8 };
        let p3 = PrecisionLevel::Partial { bits: 16 };

        assert_eq!(p1, p2);
        assert_ne!(p1, p3);
    }
}
