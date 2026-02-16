// SPDX-License-Identifier: Apache-2.0

//! Dynamic-rank array view over flattened Liberty timing table payloads.
//!
//! `TimingTable` stores values in a flattened row-major vector plus a dynamic
//! dimensions vector. This wrapper validates shape consistency and offers safe
//! N-D indexing.

use crate::liberty_proto::TimingTable;
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TimingTableArrayError {
    InvalidValueCount {
        dimensions: Vec<u32>,
        expected_values: usize,
        actual_values: usize,
    },
    DimensionProductOverflow {
        dimensions: Vec<u32>,
    },
}

impl fmt::Display for TimingTableArrayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TimingTableArrayError::InvalidValueCount {
                dimensions,
                expected_values,
                actual_values,
            } => write!(
                f,
                "timing table shape {:?} expects {} values, found {}",
                dimensions, expected_values, actual_values
            ),
            TimingTableArrayError::DimensionProductOverflow { dimensions } => write!(
                f,
                "timing table dimensions {:?} overflowed element-count computation",
                dimensions
            ),
        }
    }
}

impl std::error::Error for TimingTableArrayError {}

#[derive(Clone, Copy, Debug)]
pub struct TimingTableArrayView<'a> {
    dimensions: &'a [u32],
    values: &'a [f64],
}

impl<'a> TimingTableArrayView<'a> {
    pub fn from_timing_table(table: &'a TimingTable) -> Result<Self, TimingTableArrayError> {
        Self::from_parts(&table.dimensions, &table.values)
    }

    pub fn from_parts(
        dimensions: &'a [u32],
        values: &'a [f64],
    ) -> Result<Self, TimingTableArrayError> {
        let expected_values = expected_value_count(dimensions)?;
        if expected_values != values.len() {
            return Err(TimingTableArrayError::InvalidValueCount {
                dimensions: dimensions.to_vec(),
                expected_values,
                actual_values: values.len(),
            });
        }
        Ok(Self { dimensions, values })
    }

    pub fn rank(&self) -> usize {
        self.dimensions.len()
    }

    pub fn dimensions(&self) -> &'a [u32] {
        self.dimensions
    }

    pub fn values(&self) -> &'a [f64] {
        self.values
    }

    pub fn linear_index(&self, indices: &[usize]) -> Option<usize> {
        if self.dimensions.is_empty() {
            return if indices.is_empty() { Some(0) } else { None };
        }
        if indices.len() != self.dimensions.len() {
            return None;
        }

        let mut linear = 0usize;
        for (axis, index) in indices.iter().enumerate() {
            let dim = self.dimensions[axis] as usize;
            if *index >= dim {
                return None;
            }
            linear = linear.checked_mul(dim)?;
            linear = linear.checked_add(*index)?;
        }
        if linear < self.values.len() {
            Some(linear)
        } else {
            None
        }
    }

    pub fn get(&self, indices: &[usize]) -> Option<f64> {
        let linear = self.linear_index(indices)?;
        self.values.get(linear).copied()
    }
}

fn expected_value_count(dimensions: &[u32]) -> Result<usize, TimingTableArrayError> {
    if dimensions.is_empty() {
        return Ok(1);
    }
    let mut count = 1usize;
    for dimension in dimensions {
        count = count.checked_mul(*dimension as usize).ok_or_else(|| {
            TimingTableArrayError::DimensionProductOverflow {
                dimensions: dimensions.to_vec(),
            }
        })?;
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn array_allows_dynamic_rank_indexing() {
        let table = TimingTable {
            dimensions: vec![2, 3, 2],
            values: (0..12).map(f64::from).collect(),
            ..Default::default()
        };
        let array = TimingTableArrayView::from_timing_table(&table).unwrap();
        assert_eq!(array.rank(), 3);
        assert_eq!(array.dimensions(), &[2, 3, 2]);
        assert_eq!(array.linear_index(&[1, 0, 1]), Some(7));
        assert_eq!(array.get(&[1, 0, 1]), Some(7.0));
    }

    #[test]
    fn array_preserves_singleton_axes() {
        let table = TimingTable {
            dimensions: vec![1, 2],
            values: vec![0.10, 0.20],
            ..Default::default()
        };
        let array = TimingTableArrayView::from_timing_table(&table).unwrap();
        assert_eq!(array.rank(), 2);
        assert_eq!(array.get(&[0, 0]), Some(0.10));
        assert_eq!(array.get(&[0, 1]), Some(0.20));
        assert_eq!(array.get(&[1, 0]), None);
    }

    #[test]
    fn array_rejects_shape_mismatch() {
        let table = TimingTable {
            dimensions: vec![2, 2],
            values: vec![0.10, 0.20, 0.30],
            ..Default::default()
        };
        let err = TimingTableArrayView::from_timing_table(&table).unwrap_err();
        assert_eq!(
            err,
            TimingTableArrayError::InvalidValueCount {
                dimensions: vec![2, 2],
                expected_values: 4,
                actual_values: 3
            }
        );
    }

    #[test]
    fn array_handles_scalar_tables() {
        let table = TimingTable {
            dimensions: vec![],
            values: vec![42.0],
            ..Default::default()
        };
        let array = TimingTableArrayView::from_timing_table(&table).unwrap();
        assert_eq!(array.rank(), 0);
        assert_eq!(array.get(&[]), Some(42.0));
        assert_eq!(array.get(&[0]), None);
    }
}
