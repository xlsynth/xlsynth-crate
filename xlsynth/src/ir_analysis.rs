// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, RwLock};

use crate::ir_package::IrPackagePtr;
use crate::lib_support::{
    xls_interval_set_get_interval_bounds, xls_ir_analysis_at_least_one_bit_true,
    xls_ir_analysis_at_most_one_bit_true, xls_ir_analysis_create_from_package_with_options,
    xls_ir_analysis_exactly_one_bit_true, xls_ir_analysis_get_intervals_for_node_id,
    xls_ir_analysis_get_known_bits_for_node_id, xls_ir_analysis_implies,
    xls_ir_analysis_known_not_equals,
};
use crate::xlsynth_error::XlsynthError;
use crate::IrBits;
use xlsynth_sys::{CIrAnalysis, CIrIntervalSet};

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum IrAnalysisLevel {
    Fast,
    RangeWithContext,
}

impl IrAnalysisLevel {
    fn to_sys(self) -> xlsynth_sys::XlsIrAnalysisLevel {
        match self {
            IrAnalysisLevel::Fast => xlsynth_sys::XLS_IR_ANALYSIS_LEVEL_FAST,
            IrAnalysisLevel::RangeWithContext => {
                xlsynth_sys::XLS_IR_ANALYSIS_LEVEL_RANGE_WITH_CONTEXT
            }
        }
    }
}

pub struct KnownBits {
    pub mask: IrBits,
    pub value: IrBits,
}

pub struct Interval {
    pub lo: IrBits,
    pub hi: IrBits,
}

pub struct IntervalSet {
    pub(crate) ptr: *mut CIrIntervalSet,
}

impl Drop for IntervalSet {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                xlsynth_sys::xls_interval_set_free(self.ptr);
            }
            self.ptr = std::ptr::null_mut();
        }
    }
}

impl IntervalSet {
    pub fn interval_count(&self) -> usize {
        let count = unsafe { xlsynth_sys::xls_interval_set_get_interval_count(self.ptr) };
        assert!(count >= 0);
        count as usize
    }

    pub fn intervals(&self) -> Result<Vec<Interval>, XlsynthError> {
        let count = self.interval_count();
        let mut result: Vec<Interval> = Vec::with_capacity(count);
        for i in 0..count {
            let (lo, hi) = xls_interval_set_get_interval_bounds(self.ptr, i as i64)?;
            result.push(Interval { lo, hi });
        }
        Ok(result)
    }
}

pub struct IrAnalysis {
    pub(crate) ptr: *mut CIrAnalysis,
    pub(crate) package: Arc<RwLock<IrPackagePtr>>,
}

unsafe impl Send for IrAnalysis {}
unsafe impl Sync for IrAnalysis {}

impl Drop for IrAnalysis {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                xlsynth_sys::xls_ir_analysis_free(self.ptr);
            }
            self.ptr = std::ptr::null_mut();
        }
    }
}

impl IrAnalysis {
    pub(crate) fn create_from_package_ptr(
        package: Arc<RwLock<IrPackagePtr>>,
    ) -> Result<Self, XlsynthError> {
        Self::create_from_package_ptr_with_level(package, IrAnalysisLevel::RangeWithContext)
    }

    pub(crate) fn create_from_package_ptr_with_level(
        package: Arc<RwLock<IrPackagePtr>>,
        level: IrAnalysisLevel,
    ) -> Result<Self, XlsynthError> {
        let guard = package.write().unwrap();
        let options = xlsynth_sys::XlsIrAnalysisOptions {
            level: level.to_sys(),
        };
        let analysis_ptr =
            xls_ir_analysis_create_from_package_with_options(guard.mut_c_ptr(), Some(&options))?;
        drop(guard);
        Ok(Self {
            ptr: analysis_ptr,
            package,
        })
    }

    pub fn get_known_bits_for_node_id(&self, node_id: i64) -> Result<KnownBits, XlsynthError> {
        let _guard = self.package.read().unwrap();
        let (mask, value) = xls_ir_analysis_get_known_bits_for_node_id(self.ptr, node_id)?;
        Ok(KnownBits { mask, value })
    }

    pub fn get_intervals_for_node_id(&self, node_id: i64) -> Result<IntervalSet, XlsynthError> {
        let _guard = self.package.read().unwrap();
        let intervals_ptr = xls_ir_analysis_get_intervals_for_node_id(self.ptr, node_id)?;
        Ok(IntervalSet { ptr: intervals_ptr })
    }

    pub fn at_most_one_bit_true(&self, node_id: i64) -> Result<bool, XlsynthError> {
        let _guard = self.package.read().unwrap();
        xls_ir_analysis_at_most_one_bit_true(self.ptr, node_id)
    }

    pub fn at_least_one_bit_true(&self, node_id: i64) -> Result<bool, XlsynthError> {
        let _guard = self.package.read().unwrap();
        xls_ir_analysis_at_least_one_bit_true(self.ptr, node_id)
    }

    pub fn exactly_one_bit_true(&self, node_id: i64) -> Result<bool, XlsynthError> {
        let _guard = self.package.read().unwrap();
        xls_ir_analysis_exactly_one_bit_true(self.ptr, node_id)
    }

    pub fn known_not_equals(
        &self,
        lhs_node_id: i64,
        lhs_bit_index: i64,
        rhs_node_id: i64,
        rhs_bit_index: i64,
    ) -> Result<bool, XlsynthError> {
        let _guard = self.package.read().unwrap();
        xls_ir_analysis_known_not_equals(
            self.ptr,
            lhs_node_id,
            lhs_bit_index,
            rhs_node_id,
            rhs_bit_index,
        )
    }

    pub fn implies(
        &self,
        lhs_node_id: i64,
        lhs_bit_index: i64,
        rhs_node_id: i64,
        rhs_bit_index: i64,
    ) -> Result<bool, XlsynthError> {
        let _guard = self.package.read().unwrap();
        xls_ir_analysis_implies(
            self.ptr,
            lhs_node_id,
            lhs_bit_index,
            rhs_node_id,
            rhs_bit_index,
        )
    }
}
