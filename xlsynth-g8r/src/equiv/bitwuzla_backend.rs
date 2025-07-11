// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "has-bitwuzla")]

use std::{
    ffi::{CStr, CString},
    io,
    sync::Arc,
};

use bitwuzla_sys::{
    BITWUZLA_KIND_BV_ADD, BITWUZLA_KIND_BV_AND, BITWUZLA_KIND_BV_ASHR, BITWUZLA_KIND_BV_CONCAT,
    BITWUZLA_KIND_BV_EXTRACT, BITWUZLA_KIND_BV_MUL, BITWUZLA_KIND_BV_NAND, BITWUZLA_KIND_BV_NEG,
    BITWUZLA_KIND_BV_NOR, BITWUZLA_KIND_BV_NOT, BITWUZLA_KIND_BV_OR, BITWUZLA_KIND_BV_SDIV,
    BITWUZLA_KIND_BV_SGE, BITWUZLA_KIND_BV_SGT, BITWUZLA_KIND_BV_SHL, BITWUZLA_KIND_BV_SHR,
    BITWUZLA_KIND_BV_SIGN_EXTEND, BITWUZLA_KIND_BV_SLE, BITWUZLA_KIND_BV_SLT,
    BITWUZLA_KIND_BV_SREM, BITWUZLA_KIND_BV_SUB, BITWUZLA_KIND_BV_UDIV, BITWUZLA_KIND_BV_UGE,
    BITWUZLA_KIND_BV_UGT, BITWUZLA_KIND_BV_ULE, BITWUZLA_KIND_BV_ULT, BITWUZLA_KIND_BV_UREM,
    BITWUZLA_KIND_BV_XOR, BITWUZLA_KIND_BV_ZERO_EXTEND, BITWUZLA_KIND_EQUAL, BITWUZLA_KIND_ITE,
    BITWUZLA_KIND_NOT, BITWUZLA_OPT_ABSTRACTION, BITWUZLA_OPT_ABSTRACTION_ASSERT,
    BITWUZLA_OPT_ABSTRACTION_ASSERT_REFS, BITWUZLA_OPT_ABSTRACTION_BV_ADD,
    BITWUZLA_OPT_ABSTRACTION_BV_MUL, BITWUZLA_OPT_ABSTRACTION_BV_SIZE,
    BITWUZLA_OPT_ABSTRACTION_BV_UDIV, BITWUZLA_OPT_ABSTRACTION_BV_UREM,
    BITWUZLA_OPT_ABSTRACTION_EAGER_REFINE, BITWUZLA_OPT_ABSTRACTION_EQUAL,
    BITWUZLA_OPT_ABSTRACTION_INC_BITBLAST, BITWUZLA_OPT_ABSTRACTION_INITIAL_LEMMAS,
    BITWUZLA_OPT_ABSTRACTION_ITE, BITWUZLA_OPT_ABSTRACTION_VALUE_LIMIT,
    BITWUZLA_OPT_ABSTRACTION_VALUE_ONLY, BITWUZLA_OPT_BV_SOLVER, BITWUZLA_OPT_DBG_CHECK_MODEL,
    BITWUZLA_OPT_DBG_CHECK_UNSAT_CORE, BITWUZLA_OPT_DBG_PP_NODE_THRESH,
    BITWUZLA_OPT_DBG_RW_NODE_THRESH, BITWUZLA_OPT_LOGLEVEL, BITWUZLA_OPT_MEMORY_LIMIT,
    BITWUZLA_OPT_NTHREADS, BITWUZLA_OPT_PP_CONTRADICTING_ANDS, BITWUZLA_OPT_PP_ELIM_BV_EXTRACTS,
    BITWUZLA_OPT_PP_ELIM_BV_UDIV, BITWUZLA_OPT_PP_EMBEDDED_CONSTR, BITWUZLA_OPT_PP_FLATTEN_AND,
    BITWUZLA_OPT_PP_NORMALIZE, BITWUZLA_OPT_PP_SKELETON_PREPROC, BITWUZLA_OPT_PP_VARIABLE_SUBST,
    BITWUZLA_OPT_PP_VARIABLE_SUBST_NORM_BV_INEQ, BITWUZLA_OPT_PP_VARIABLE_SUBST_NORM_DISEQ,
    BITWUZLA_OPT_PP_VARIABLE_SUBST_NORM_EQ, BITWUZLA_OPT_PREPROCESS, BITWUZLA_OPT_PRODUCE_MODELS,
    BITWUZLA_OPT_PRODUCE_UNSAT_ASSUMPTIONS, BITWUZLA_OPT_PRODUCE_UNSAT_CORES,
    BITWUZLA_OPT_PROP_CONST_BITS, BITWUZLA_OPT_PROP_INFER_INEQ_BOUNDS, BITWUZLA_OPT_PROP_NPROPS,
    BITWUZLA_OPT_PROP_NUPDATES, BITWUZLA_OPT_PROP_OPT_LT_CONCAT_SEXT, BITWUZLA_OPT_PROP_PATH_SEL,
    BITWUZLA_OPT_PROP_PROB_RANDOM_INPUT, BITWUZLA_OPT_PROP_PROB_USE_INV_VALUE,
    BITWUZLA_OPT_PROP_SEXT, BITWUZLA_OPT_REWRITE_LEVEL, BITWUZLA_OPT_SAT_SOLVER, BITWUZLA_OPT_SEED,
    BITWUZLA_OPT_TIME_LIMIT_PER, BITWUZLA_OPT_VERBOSITY, BITWUZLA_OPT_WRITE_AIGER,
    BITWUZLA_OPT_WRITE_CNF, BITWUZLA_SAT, BITWUZLA_UNKNOWN, BITWUZLA_UNSAT, BitwuzlaKind,
    bitwuzla_assert, bitwuzla_check_sat, bitwuzla_delete, bitwuzla_get_value, bitwuzla_mk_bv_one,
    bitwuzla_mk_bv_sort, bitwuzla_mk_bv_value, bitwuzla_mk_bv_value_uint64, bitwuzla_mk_bv_zero,
    bitwuzla_mk_const, bitwuzla_mk_term1, bitwuzla_mk_term1_indexed1, bitwuzla_mk_term1_indexed2,
    bitwuzla_mk_term2, bitwuzla_mk_term3, bitwuzla_new, bitwuzla_options_delete,
    bitwuzla_options_new, bitwuzla_pop, bitwuzla_push, bitwuzla_set_option,
    bitwuzla_set_option_mode, bitwuzla_term_manager_delete, bitwuzla_term_manager_new,
    bitwuzla_term_to_string, bitwuzla_term_value_get_str,
};

use crate::{
    equiv::solver_interface::{BitVec, Response, Solver},
    ir_value_utils::{ir_bits_from_lsb_is_0, ir_value_from_bits_with_type},
    xls_ir::ir,
};

struct RawBitwuzla {
    term_manager: *mut bitwuzla_sys::BitwuzlaTermManager,
    raw: *mut bitwuzla_sys::Bitwuzla,
}

impl RawBitwuzla {
    pub fn new(options: *const bitwuzla_sys::BitwuzlaOptions) -> Self {
        let term_manager = unsafe { bitwuzla_term_manager_new() };
        let raw = unsafe { bitwuzla_new(term_manager, options) };
        RawBitwuzla { raw, term_manager }
    }
}

impl Drop for RawBitwuzla {
    fn drop(&mut self) {
        unsafe {
            bitwuzla_delete(self.raw);
            // Bitwuzla doesn't require us to release the terms prior to calling the delete.
            bitwuzla_term_manager_delete(self.term_manager);
        };
    }
}

/// A Bitwuzla-rawd SMT solver implementing the `Solver` trait.
pub struct Bitwuzla {
    bitwuzla: Arc<RawBitwuzla>,
}

impl Bitwuzla {
    pub fn raw_term_manager(&self) -> *mut bitwuzla_sys::BitwuzlaTermManager {
        self.bitwuzla.term_manager
    }

    pub fn raw_bitwuzla(&self) -> *mut bitwuzla_sys::Bitwuzla {
        self.bitwuzla.raw
    }
}

struct RawBitwuzlaOptions {
    raw: *mut bitwuzla_sys::BitwuzlaOptions,
}

impl RawBitwuzlaOptions {
    pub fn new() -> Self {
        let raw = unsafe { bitwuzla_options_new() };
        unsafe {
            bitwuzla_set_option(raw, BITWUZLA_OPT_PRODUCE_MODELS, 1);
        }
        RawBitwuzlaOptions { raw }
    }
}

impl Drop for RawBitwuzlaOptions {
    fn drop(&mut self) {
        unsafe { bitwuzla_options_delete(self.raw) };
    }
}

pub struct BitwuzlaOptions {
    options: Arc<RawBitwuzlaOptions>,
}

unsafe impl Send for BitwuzlaOptions {}
unsafe impl Sync for BitwuzlaOptions {}

pub enum BitwuzlaBVSolver {
    BitBlast,
    Prop,
    PreProp,
}

pub enum BitwuzlaRewriteLevel {
    Off,
    CheapOnly,
    Full,
}

pub enum BitwuzlaSatSolver {
    CaDiCaL,
    CryptoMiniSat,
    Kissat,
}

impl BitwuzlaOptions {
    pub fn set_option(&mut self, option: bitwuzla_sys::BitwuzlaOption, value: u64) -> &mut Self {
        unsafe {
            bitwuzla_set_option(self.options.raw, option, value);
        }
        self
    }

    pub fn set_option_mode(
        &mut self,
        option: bitwuzla_sys::BitwuzlaOption,
        value: &str,
    ) -> &mut Self {
        unsafe {
            bitwuzla_set_option_mode(
                self.options.raw,
                option,
                CString::new(value).unwrap().as_ptr(),
            );
        }
        self
    }

    pub fn new() -> Self {
        BitwuzlaOptions {
            options: Arc::new(RawBitwuzlaOptions::new()),
        }
    }

    pub fn set_loglevel(&mut self, level: u64) -> &mut Self {
        self.set_option(BITWUZLA_OPT_LOGLEVEL, level)
    }

    pub fn enable_produce_models(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_PRODUCE_MODELS, 1)
    }

    pub fn disable_produce_models(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_PRODUCE_MODELS, 0)
    }

    pub fn enable_produce_unsat_assumptions(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_PRODUCE_UNSAT_ASSUMPTIONS, 1)
    }

    pub fn disable_produce_unsat_assumptions(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_PRODUCE_UNSAT_ASSUMPTIONS, 0)
    }

    pub fn enable_produce_unsat_cores(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_PRODUCE_UNSAT_CORES, 1)
    }

    pub fn disable_produce_unsat_cores(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_PRODUCE_UNSAT_CORES, 0)
    }

    pub fn set_seed(&mut self, seed: u64) -> &mut Self {
        self.set_option(BITWUZLA_OPT_SEED, seed)
    }

    pub fn set_verbosity(&mut self, verbosity: u64) -> &mut Self {
        self.set_option(BITWUZLA_OPT_VERBOSITY, verbosity)
    }

    pub fn set_time_limit_per(&mut self, time_limit_per: u64) -> &mut Self {
        self.set_option(BITWUZLA_OPT_TIME_LIMIT_PER, time_limit_per)
    }

    pub fn set_memory_limit(&mut self, memory_limit: u64) -> &mut Self {
        self.set_option(BITWUZLA_OPT_MEMORY_LIMIT, memory_limit)
    }

    pub fn set_nthreads(&mut self, nthreads: u64) -> &mut Self {
        self.set_option(BITWUZLA_OPT_NTHREADS, nthreads)
    }

    pub fn set_bv_solver(&mut self, solver: BitwuzlaBVSolver) -> &mut Self {
        match solver {
            BitwuzlaBVSolver::BitBlast => self.set_option_mode(BITWUZLA_OPT_BV_SOLVER, "bitblast"),
            BitwuzlaBVSolver::Prop => self.set_option_mode(BITWUZLA_OPT_BV_SOLVER, "prop"),
            BitwuzlaBVSolver::PreProp => self.set_option_mode(BITWUZLA_OPT_BV_SOLVER, "preprop"),
        }
    }

    pub fn set_rewrite_level(&mut self, level: BitwuzlaRewriteLevel) -> &mut Self {
        match level {
            BitwuzlaRewriteLevel::Off => self.set_option(BITWUZLA_OPT_REWRITE_LEVEL, 0),
            BitwuzlaRewriteLevel::CheapOnly => self.set_option(BITWUZLA_OPT_REWRITE_LEVEL, 1),
            BitwuzlaRewriteLevel::Full => self.set_option(BITWUZLA_OPT_REWRITE_LEVEL, 2),
        }
    }

    pub fn set_sat_solver(&mut self, solver: BitwuzlaSatSolver) -> &mut Self {
        match solver {
            BitwuzlaSatSolver::CaDiCaL => self.set_option_mode(BITWUZLA_OPT_SAT_SOLVER, "cadical"),
            BitwuzlaSatSolver::CryptoMiniSat => {
                self.set_option_mode(BITWUZLA_OPT_SAT_SOLVER, "cms")
            }
            BitwuzlaSatSolver::Kissat => self.set_option_mode(BITWUZLA_OPT_SAT_SOLVER, "kissat"),
        }
    }

    pub fn set_write_aiger(&mut self, file: &str) -> &mut Self {
        self.set_option_mode(BITWUZLA_OPT_WRITE_AIGER, file)
    }

    pub fn set_write_cnf(&mut self, file: &str) -> &mut Self {
        self.set_option_mode(BITWUZLA_OPT_WRITE_CNF, file)
    }

    pub fn enable_prop_const_bits(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_PROP_CONST_BITS, 1)
    }

    pub fn disable_prop_const_bits(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_PROP_CONST_BITS, 0)
    }

    pub fn enable_prop_infer_ineq_bounds(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_PROP_INFER_INEQ_BOUNDS, 1)
    }

    pub fn disable_prop_infer_ineq_bounds(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_PROP_INFER_INEQ_BOUNDS, 0)
    }

    pub fn set_prop_nprops(&mut self, nprops: u64) -> &mut Self {
        self.set_option(BITWUZLA_OPT_PROP_NPROPS, nprops)
    }

    pub fn set_prop_nupdates(&mut self, nupdates: u64) -> &mut Self {
        self.set_option(BITWUZLA_OPT_PROP_NUPDATES, nupdates)
    }

    pub fn enable_prop_opt_lt_concat_sext(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_PROP_OPT_LT_CONCAT_SEXT, 1)
    }

    pub fn disable_prop_opt_lt_concat_sext(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_PROP_OPT_LT_CONCAT_SEXT, 0)
    }

    pub fn enable_prop_path_sel(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_PROP_PATH_SEL, 1)
    }

    pub fn disable_prop_path_sel(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_PROP_PATH_SEL, 0)
    }

    pub fn set_prop_prob_random_input(&mut self, prob: u64) -> &mut Self {
        self.set_option(BITWUZLA_OPT_PROP_PROB_RANDOM_INPUT, prob)
    }

    pub fn set_prop_prob_use_inv_value(&mut self, prob: u64) -> &mut Self {
        self.set_option(BITWUZLA_OPT_PROP_PROB_USE_INV_VALUE, prob)
    }

    pub fn enable_prop_sext(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_PROP_SEXT, 1)
    }

    pub fn disable_prop_sext(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_PROP_SEXT, 0)
    }

    pub fn enable_abstraction(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_ABSTRACTION, 1)
    }

    pub fn disable_abstraction(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_ABSTRACTION, 0)
    }

    pub fn set_abstraction_bv_size(&mut self, size: u64) -> &mut Self {
        self.set_option(BITWUZLA_OPT_ABSTRACTION_BV_SIZE, size)
    }

    pub fn enable_abstraction_eager_refine(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_ABSTRACTION_EAGER_REFINE, 1)
    }

    pub fn disable_abstraction_eager_refine(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_ABSTRACTION_EAGER_REFINE, 0)
    }

    pub fn set_abstraction_value_limit(&mut self, limit: u64) -> &mut Self {
        self.set_option(BITWUZLA_OPT_ABSTRACTION_VALUE_LIMIT, limit)
    }

    pub fn enable_abstraction_value_only(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_ABSTRACTION_VALUE_ONLY, 1)
    }

    pub fn disable_abstraction_value_only(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_ABSTRACTION_VALUE_ONLY, 0)
    }

    pub fn enable_abstraction_assert(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_ABSTRACTION_ASSERT, 1)
    }

    pub fn disable_abstraction_assert(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_ABSTRACTION_ASSERT, 0)
    }

    pub fn set_abstraction_assert_refs(&mut self, refs: u64) -> &mut Self {
        self.set_option(BITWUZLA_OPT_ABSTRACTION_ASSERT_REFS, refs)
    }
    pub fn enable_abstraction_initial_lemmas(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_ABSTRACTION_INITIAL_LEMMAS, 1)
    }

    pub fn disable_abstraction_initial_lemmas(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_ABSTRACTION_INITIAL_LEMMAS, 0)
    }

    pub fn enable_abstraction_inc_bitblast(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_ABSTRACTION_INC_BITBLAST, 1)
    }

    pub fn disable_abstraction_inc_bitblast(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_ABSTRACTION_INC_BITBLAST, 0)
    }

    pub fn enable_abstraction_bv_add(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_ABSTRACTION_BV_ADD, 1)
    }

    pub fn disable_abstraction_bv_add(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_ABSTRACTION_BV_ADD, 0)
    }

    pub fn enable_abstraction_bv_mul(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_ABSTRACTION_BV_MUL, 1)
    }

    pub fn disable_abstraction_bv_mul(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_ABSTRACTION_BV_MUL, 0)
    }

    pub fn enable_abstraction_bv_udiv(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_ABSTRACTION_BV_UDIV, 1)
    }

    pub fn disable_abstraction_bv_udiv(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_ABSTRACTION_BV_UDIV, 0)
    }

    pub fn enable_abstraction_bv_urem(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_ABSTRACTION_BV_UREM, 1)
    }

    pub fn disable_abstraction_bv_urem(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_ABSTRACTION_BV_UREM, 0)
    }

    pub fn enable_abstraction_equal(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_ABSTRACTION_EQUAL, 1)
    }

    pub fn disable_abstraction_equal(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_ABSTRACTION_EQUAL, 0)
    }

    pub fn enable_abstraction_ite(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_ABSTRACTION_ITE, 1)
    }

    pub fn disable_abstraction_ite(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_ABSTRACTION_ITE, 0)
    }

    pub fn enable_preprocess(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_PREPROCESS, 1)
    }

    pub fn disable_preprocess(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_PREPROCESS, 0)
    }

    pub fn enable_pp_contradicting_ands(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_PP_CONTRADICTING_ANDS, 1)
    }

    pub fn disable_pp_contradicting_ands(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_PP_CONTRADICTING_ANDS, 0)
    }

    pub fn enable_pp_elim_bv_extracts(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_PP_ELIM_BV_EXTRACTS, 1)
    }

    pub fn disable_pp_elim_bv_extracts(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_PP_ELIM_BV_EXTRACTS, 0)
    }

    pub fn enable_pp_elim_bv_udiv(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_PP_ELIM_BV_UDIV, 1)
    }

    pub fn disable_pp_elim_bv_udiv(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_PP_ELIM_BV_UDIV, 0)
    }

    pub fn enable_pp_embedded_constr(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_PP_EMBEDDED_CONSTR, 1)
    }

    pub fn disable_pp_embedded_constr(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_PP_EMBEDDED_CONSTR, 0)
    }

    pub fn enable_pp_flatten_and(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_PP_FLATTEN_AND, 1)
    }

    pub fn disable_pp_flatten_and(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_PP_FLATTEN_AND, 0)
    }

    pub fn enable_pp_normalize(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_PP_NORMALIZE, 1)
    }

    pub fn disable_pp_normalize(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_PP_NORMALIZE, 0)
    }

    pub fn enable_pp_skeleton_preproc(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_PP_SKELETON_PREPROC, 1)
    }

    pub fn disable_pp_skeleton_preproc(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_PP_SKELETON_PREPROC, 0)
    }

    pub fn enable_pp_variable_subst(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_PP_VARIABLE_SUBST, 1)
    }

    pub fn disable_pp_variable_subst(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_PP_VARIABLE_SUBST, 0)
    }

    pub fn enable_pp_variable_subst_norm_eq(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_PP_VARIABLE_SUBST_NORM_EQ, 1)
    }

    pub fn disable_pp_variable_subst_norm_eq(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_PP_VARIABLE_SUBST_NORM_EQ, 0)
    }

    pub fn enable_pp_variable_subst_norm_diseq(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_PP_VARIABLE_SUBST_NORM_DISEQ, 1)
    }

    pub fn disable_pp_variable_subst_norm_diseq(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_PP_VARIABLE_SUBST_NORM_DISEQ, 0)
    }

    pub fn enable_pp_variable_subst_norm_bv_ineq(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_PP_VARIABLE_SUBST_NORM_BV_INEQ, 1)
    }

    pub fn disable_pp_variable_subst_norm_bv_ineq(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_PP_VARIABLE_SUBST_NORM_BV_INEQ, 0)
    }

    pub fn set_dbg_rw_node_thresh(&mut self, thresh: u64) -> &mut Self {
        self.set_option(BITWUZLA_OPT_DBG_RW_NODE_THRESH, thresh)
    }

    pub fn set_dbg_pp_node_thresh(&mut self, thresh: u64) -> &mut Self {
        self.set_option(BITWUZLA_OPT_DBG_PP_NODE_THRESH, thresh)
    }

    pub fn enable_dbg_check_model(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_DBG_CHECK_MODEL, 1)
    }

    pub fn disable_dbg_check_model(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_DBG_CHECK_MODEL, 0)
    }

    pub fn enable_dbg_check_unsat_core(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_DBG_CHECK_UNSAT_CORE, 1)
    }

    pub fn disable_dbg_check_unsat_core(&mut self) -> &mut Self {
        self.set_option(BITWUZLA_OPT_DBG_CHECK_UNSAT_CORE, 0)
    }
}

impl Bitwuzla {
    fn bool_to_bv(&mut self, cond: BitwuzlaTerm) -> BitVec<BitwuzlaTerm> {
        let sort1 = unsafe { bitwuzla_mk_bv_sort(self.raw_term_manager(), 1) };
        let one = unsafe { bitwuzla_mk_bv_one(self.raw_term_manager(), sort1) };
        let zero = unsafe { bitwuzla_mk_bv_zero(self.raw_term_manager(), sort1) };
        let rep = unsafe {
            bitwuzla_mk_term3(
                self.raw_term_manager(),
                BITWUZLA_KIND_ITE,
                cond.raw,
                one,
                zero,
            )
        };
        BitVec::BitVec {
            width: 1,
            rep: BitwuzlaTerm { raw: rep },
        }
    }

    fn bv_to_bool(&mut self, bv: &BitVec<BitwuzlaTerm>) -> BitwuzlaTerm {
        match bv {
            BitVec::BitVec { width: 1, rep } => {
                let sort1 = unsafe { bitwuzla_mk_bv_sort(self.raw_term_manager(), 1) };
                let one = unsafe { bitwuzla_mk_bv_one(self.raw_term_manager(), sort1) };
                BitwuzlaTerm {
                    raw: unsafe {
                        bitwuzla_mk_term2(
                            self.raw_term_manager(),
                            BITWUZLA_KIND_EQUAL,
                            rep.raw,
                            one,
                        )
                    },
                }
            }
            _ => panic!("Invalid bitvector width for boolean: {:?}", bv.get_width()),
        }
    }

    fn unary_op(&mut self, bv: &BitVec<BitwuzlaTerm>, kind: BitwuzlaKind) -> BitVec<BitwuzlaTerm> {
        match bv {
            BitVec::ZeroWidth => BitVec::ZeroWidth,
            BitVec::BitVec { width, rep } => {
                let rep2 = unsafe { bitwuzla_mk_term1(self.raw_term_manager(), kind, rep.raw) };
                BitVec::BitVec {
                    width: *width,
                    rep: BitwuzlaTerm { raw: rep2 },
                }
            }
        }
    }

    fn bin_op(
        &mut self,
        lhs: &BitVec<BitwuzlaTerm>,
        rhs: &BitVec<BitwuzlaTerm>,
        kind: BitwuzlaKind,
    ) -> BitVec<BitwuzlaTerm> {
        match (lhs, rhs) {
            (BitVec::BitVec { width: w1, rep: r1 }, BitVec::BitVec { width: w2, rep: r2 }) => {
                assert_eq!(w1, w2, "Bitvector width mismatch");
                let rep2 =
                    unsafe { bitwuzla_mk_term2(self.raw_term_manager(), kind, r1.raw, r2.raw) };
                BitVec::BitVec {
                    width: *w1,
                    rep: BitwuzlaTerm { raw: rep2 },
                }
            }
            (BitVec::ZeroWidth, BitVec::ZeroWidth) => BitVec::ZeroWidth,
            _ => panic!("Bitvector width mismatch"),
        }
    }

    fn bin_bool_op(
        &mut self,
        lhs: &BitVec<BitwuzlaTerm>,
        rhs: &BitVec<BitwuzlaTerm>,
        kind: BitwuzlaKind,
        zero_width_result: BitVec<BitwuzlaTerm>,
    ) -> BitVec<BitwuzlaTerm> {
        match (lhs, rhs) {
            (BitVec::BitVec { width: w1, rep: r1 }, BitVec::BitVec { width: w2, rep: r2 }) => {
                assert_eq!(w1, w2, "Bitvector width mismatch");
                let cond =
                    unsafe { bitwuzla_mk_term2(self.raw_term_manager(), kind, r1.raw, r2.raw) };
                self.bool_to_bv(BitwuzlaTerm { raw: cond })
            }
            (BitVec::ZeroWidth, BitVec::ZeroWidth) => zero_width_result,
            _ => panic!("Bitvector width mismatch"),
        }
    }

    fn reduce_op(&mut self, bv: &BitVec<BitwuzlaTerm>, kind: BitwuzlaKind) -> BitVec<BitwuzlaTerm> {
        match bv {
            BitVec::BitVec { width, rep } => {
                let mut acc = unsafe {
                    bitwuzla_mk_term1_indexed2(
                        self.raw_term_manager(),
                        BITWUZLA_KIND_BV_EXTRACT,
                        rep.raw,
                        0,
                        0,
                    )
                };
                for i in 1..*width {
                    let bit = unsafe {
                        bitwuzla_mk_term1_indexed2(
                            self.raw_term_manager(),
                            BITWUZLA_KIND_BV_EXTRACT,
                            rep.raw,
                            i as u64,
                            i as u64,
                        )
                    };
                    acc = unsafe { bitwuzla_mk_term2(self.raw_term_manager(), kind, acc, bit) };
                }
                BitVec::BitVec {
                    width: 1,
                    rep: BitwuzlaTerm { raw: acc },
                }
            }
            BitVec::ZeroWidth => panic!("Cannot reduce zero-width bitvector"),
        }
    }
}

#[derive(Clone)]
pub struct BitwuzlaTerm {
    raw: bitwuzla_sys::BitwuzlaTerm,
}

impl Solver for Bitwuzla {
    type Term = BitwuzlaTerm;
    type Config = BitwuzlaOptions;

    fn new(config: &Self::Config) -> io::Result<Self> {
        let instance = RawBitwuzla::new(config.options.raw);
        Ok(Bitwuzla {
            bitwuzla: Arc::new(instance),
        })
    }

    fn declare(&mut self, name: &str, width: usize) -> io::Result<BitVec<Self::Term>> {
        if width == 0 {
            return Ok(BitVec::ZeroWidth);
        }
        let sort = unsafe { bitwuzla_mk_bv_sort(self.raw_term_manager(), width as u64) };
        let c_name = CString::new(name).unwrap();
        let rep = unsafe { bitwuzla_mk_const(self.raw_term_manager(), sort, c_name.as_ptr()) };
        Ok(BitVec::BitVec {
            width,
            rep: BitwuzlaTerm { raw: rep },
        })
    }

    fn numerical(&mut self, width: usize, mut value: u64) -> BitVec<Self::Term> {
        assert!(width > 0, "Width must be positive");

        // Clamp the value so that it fits in the requested bit-width.
        if width < 64 {
            let mask = (1u64 << width) - 1;
            value &= mask;
        }

        let sort = unsafe { bitwuzla_mk_bv_sort(self.raw_term_manager(), width as u64) };

        // Build the Bitwuzla bit-vector constant.
        let rep = if width <= 64 {
            unsafe { bitwuzla_mk_bv_value_uint64(self.raw_term_manager(), sort, value) }
        } else {
            // Pad / truncate the binary string so it matches the desired width.
            let bitstr = format!("{:0width$b}", value, width = width);
            let c = CString::new(bitstr).unwrap();
            unsafe { bitwuzla_mk_bv_value(self.raw_term_manager(), sort, c.as_ptr(), 2) }
        };

        BitVec::BitVec {
            width,
            rep: BitwuzlaTerm { raw: rep },
        }
    }

    fn from_raw_str(&mut self, width: usize, value: &str) -> BitVec<Self::Term> {
        assert!(width > 0, "Width must be positive");
        let sort = unsafe { bitwuzla_mk_bv_sort(self.raw_term_manager(), width as u64) };
        let rep = if let Some(str) = value.strip_prefix("#b") {
            let c = CString::new(str).unwrap();
            unsafe { bitwuzla_mk_bv_value(self.raw_term_manager(), sort, c.as_ptr(), 2) }
        } else if let Some(str) = value.strip_prefix("#x") {
            let c = CString::new(str).unwrap();
            unsafe { bitwuzla_mk_bv_value(self.raw_term_manager(), sort, c.as_ptr(), 16) }
        } else {
            panic!("Invalid atom: {}", value);
        };
        BitVec::BitVec {
            width,
            rep: BitwuzlaTerm { raw: rep },
        }
    }

    fn get_value(
        &mut self,
        bit_vec: &BitVec<Self::Term>,
        ty: &ir::Type,
    ) -> io::Result<xlsynth::IrValue> {
        match bit_vec {
            BitVec::BitVec { rep, .. } => unsafe {
                let val = bitwuzla_get_value(self.bitwuzla.raw, rep.raw);
                let bitstr = CStr::from_ptr(bitwuzla_term_value_get_str(val))
                    .to_str()
                    .unwrap();
                let bits: Vec<bool> = bitstr.chars().rev().map(|c| c == '1').collect();
                Ok(ir_value_from_bits_with_type(
                    &ir_bits_from_lsb_is_0(&bits),
                    ty,
                ))
            },
            BitVec::ZeroWidth => panic!("Cannot get value of zero-width bitvector"),
        }
    }

    fn extract(&mut self, bit_vec: &BitVec<Self::Term>, high: i32, low: i32) -> BitVec<Self::Term> {
        if high < low {
            return BitVec::ZeroWidth;
        }
        match bit_vec {
            BitVec::ZeroWidth => BitVec::ZeroWidth,
            BitVec::BitVec { width, rep } => {
                assert!((high as usize) < *width, "High index out of bounds");
                assert!(low >= 0, "Low index out of bounds");
                assert!(
                    low <= high,
                    "Low index must be less than or equal to high index"
                );
                let rep2 = unsafe {
                    bitwuzla_mk_term1_indexed2(
                        self.raw_term_manager(),
                        BITWUZLA_KIND_BV_EXTRACT,
                        rep.raw,
                        high as u64,
                        low as u64,
                    )
                };
                BitVec::BitVec {
                    width: (high - low + 1) as usize,
                    rep: BitwuzlaTerm { raw: rep2 },
                }
            }
        }
    }

    fn not(&mut self, bv: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        self.unary_op(bv, BITWUZLA_KIND_BV_NOT)
    }

    fn neg(&mut self, bv: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        self.unary_op(bv, BITWUZLA_KIND_BV_NEG)
    }

    fn reverse(&mut self, bv: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        match bv {
            BitVec::ZeroWidth => BitVec::ZeroWidth,
            BitVec::BitVec { width, rep } => {
                let mut acc = unsafe {
                    bitwuzla_mk_term1_indexed2(
                        self.raw_term_manager(),
                        BITWUZLA_KIND_BV_EXTRACT,
                        rep.raw,
                        0,
                        0,
                    )
                };
                for i in 1..*width {
                    let bit = unsafe {
                        bitwuzla_mk_term1_indexed2(
                            self.raw_term_manager(),
                            BITWUZLA_KIND_BV_EXTRACT,
                            rep.raw,
                            i as u64,
                            i as u64,
                        )
                    };
                    acc = unsafe {
                        bitwuzla_mk_term2(
                            self.raw_term_manager(),
                            BITWUZLA_KIND_BV_CONCAT,
                            acc,
                            bit,
                        )
                    };
                }
                BitVec::BitVec {
                    width: *width,
                    rep: BitwuzlaTerm { raw: acc },
                }
            }
        }
    }

    fn or_reduce(&mut self, bv: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        self.reduce_op(bv, BITWUZLA_KIND_BV_OR)
    }

    fn and_reduce(&mut self, bv: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        self.reduce_op(bv, BITWUZLA_KIND_BV_AND)
    }

    fn xor_reduce(&mut self, bv: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        self.reduce_op(bv, BITWUZLA_KIND_BV_XOR)
    }

    fn add(&mut self, x: &BitVec<Self::Term>, y: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        self.bin_op(x, y, BITWUZLA_KIND_BV_ADD)
    }

    fn sub(&mut self, x: &BitVec<Self::Term>, y: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        self.bin_op(x, y, BITWUZLA_KIND_BV_SUB)
    }

    fn mul(&mut self, x: &BitVec<Self::Term>, y: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        self.bin_op(x, y, BITWUZLA_KIND_BV_MUL)
    }

    fn udiv(&mut self, x: &BitVec<Self::Term>, y: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        self.bin_op(x, y, BITWUZLA_KIND_BV_UDIV)
    }

    fn urem(&mut self, x: &BitVec<Self::Term>, y: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        self.bin_op(x, y, BITWUZLA_KIND_BV_UREM)
    }

    fn srem(&mut self, x: &BitVec<Self::Term>, y: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        self.bin_op(x, y, BITWUZLA_KIND_BV_SREM)
    }

    fn sdiv(&mut self, x: &BitVec<Self::Term>, y: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        self.bin_op(x, y, BITWUZLA_KIND_BV_SDIV)
    }

    fn shl(&mut self, x: &BitVec<Self::Term>, y: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        self.bin_op(x, y, BITWUZLA_KIND_BV_SHL)
    }

    fn lshr(&mut self, x: &BitVec<Self::Term>, y: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        self.bin_op(x, y, BITWUZLA_KIND_BV_SHR)
    }

    fn ashr(&mut self, x: &BitVec<Self::Term>, y: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        self.bin_op(x, y, BITWUZLA_KIND_BV_ASHR)
    }

    fn concat(&mut self, a: &BitVec<Self::Term>, b: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        match (&a, &b) {
            (BitVec::BitVec { width: w1, rep: r1 }, BitVec::BitVec { width: w2, rep: r2 }) => {
                let rep2 = unsafe {
                    bitwuzla_mk_term2(
                        self.raw_term_manager(),
                        BITWUZLA_KIND_BV_CONCAT,
                        r1.raw,
                        r2.raw,
                    )
                };
                BitVec::BitVec {
                    width: w1 + w2,
                    rep: BitwuzlaTerm { raw: rep2 },
                }
            }
            (BitVec::ZeroWidth, _) => b.clone(),
            (_, BitVec::ZeroWidth) => a.clone(),
        }
    }

    fn or(&mut self, x: &BitVec<Self::Term>, y: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        self.bin_op(x, y, BITWUZLA_KIND_BV_OR)
    }

    fn and(&mut self, x: &BitVec<Self::Term>, y: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        self.bin_op(x, y, BITWUZLA_KIND_BV_AND)
    }

    fn xor(&mut self, x: &BitVec<Self::Term>, y: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        self.bin_op(x, y, BITWUZLA_KIND_BV_XOR)
    }

    fn nor(&mut self, x: &BitVec<Self::Term>, y: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        self.bin_op(x, y, BITWUZLA_KIND_BV_NOR)
    }

    fn nand(&mut self, x: &BitVec<Self::Term>, y: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        self.bin_op(x, y, BITWUZLA_KIND_BV_NAND)
    }

    fn extend(&mut self, bv: &BitVec<Self::Term>, ext: usize, signed: bool) -> BitVec<Self::Term> {
        match bv {
            BitVec::ZeroWidth => panic!("Cannot extend zero-width bitvector"),
            BitVec::BitVec { width, rep } => {
                if ext == 0 {
                    return BitVec::BitVec {
                        width: *width,
                        rep: BitwuzlaTerm { raw: rep.raw },
                    };
                }
                let kind = if signed {
                    BITWUZLA_KIND_BV_SIGN_EXTEND
                } else {
                    BITWUZLA_KIND_BV_ZERO_EXTEND
                };
                let rep2 = unsafe {
                    bitwuzla_mk_term1_indexed1(self.raw_term_manager(), kind, rep.raw, ext as u64)
                };
                BitVec::BitVec {
                    width: *width + ext,
                    rep: BitwuzlaTerm { raw: rep2 },
                }
            }
        }
    }

    fn ite(
        &mut self,
        c: &BitVec<Self::Term>,
        t: &BitVec<Self::Term>,
        e: &BitVec<Self::Term>,
    ) -> BitVec<Self::Term> {
        match (c.clone(), t, e) {
            (
                BitVec::BitVec { rep: _, width: wc },
                BitVec::BitVec { rep: rt, width: wt },
                BitVec::BitVec { rep: re, width: we },
            ) => {
                assert_eq!(wc, 1, "Condition must be 1-bit");
                assert_eq!(wt, we, "Then and else must have same width");
                let cond = self.bv_to_bool(&c);
                let rep2 = unsafe {
                    bitwuzla_mk_term3(
                        self.raw_term_manager(),
                        BITWUZLA_KIND_ITE,
                        cond.raw,
                        rt.raw,
                        re.raw,
                    )
                };
                BitVec::BitVec {
                    width: *wt,
                    rep: BitwuzlaTerm { raw: rep2 },
                }
            }
            (BitVec::BitVec { width: wc, .. }, BitVec::ZeroWidth, BitVec::ZeroWidth) => {
                assert_eq!(wc, 1, "Condition must be 1-bit");
                BitVec::ZeroWidth
            }
            _ => panic!("Bitvector width mismatch in ite"),
        }
    }

    fn eq(&mut self, x: &BitVec<Self::Term>, y: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        let result = self.numerical(1, 1);
        self.bin_bool_op(x, y, BITWUZLA_KIND_EQUAL, result)
    }

    fn ne(&mut self, x: &BitVec<Self::Term>, y: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        match (x, y) {
            (BitVec::BitVec { width: w1, rep: r1 }, BitVec::BitVec { width: w2, rep: r2 }) => {
                assert_eq!(w1, w2, "Bitvector width mismatch");
                let cond = unsafe {
                    bitwuzla_mk_term1(
                        self.raw_term_manager(),
                        BITWUZLA_KIND_NOT,
                        bitwuzla_mk_term2(
                            self.raw_term_manager(),
                            BITWUZLA_KIND_EQUAL,
                            r1.raw,
                            r2.raw,
                        ),
                    )
                };
                self.bool_to_bv(BitwuzlaTerm { raw: cond })
            }
            (BitVec::ZeroWidth, BitVec::ZeroWidth) => self.numerical(1, 0),
            _ => panic!("Bitvector width mismatch"),
        }
    }

    fn slt(&mut self, x: &BitVec<Self::Term>, y: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        let result = self.numerical(1, 0);
        self.bin_bool_op(x, y, BITWUZLA_KIND_BV_SLT, result)
    }

    fn sgt(&mut self, x: &BitVec<Self::Term>, y: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        let result = self.numerical(1, 0);
        self.bin_bool_op(x, y, BITWUZLA_KIND_BV_SGT, result)
    }

    fn sle(&mut self, x: &BitVec<Self::Term>, y: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        let result = self.numerical(1, 1);
        self.bin_bool_op(x, y, BITWUZLA_KIND_BV_SLE, result)
    }

    fn sge(&mut self, x: &BitVec<Self::Term>, y: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        let result = self.numerical(1, 1);
        self.bin_bool_op(x, y, BITWUZLA_KIND_BV_SGE, result)
    }

    fn ult(&mut self, x: &BitVec<Self::Term>, y: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        let result = self.numerical(1, 0);
        self.bin_bool_op(x, y, BITWUZLA_KIND_BV_ULT, result)
    }

    fn ugt(&mut self, x: &BitVec<Self::Term>, y: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        let result = self.numerical(1, 0);
        self.bin_bool_op(x, y, BITWUZLA_KIND_BV_UGT, result)
    }

    fn ule(&mut self, x: &BitVec<Self::Term>, y: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        let result = self.numerical(1, 1);
        self.bin_bool_op(x, y, BITWUZLA_KIND_BV_ULE, result)
    }

    fn uge(&mut self, x: &BitVec<Self::Term>, y: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        let result = self.numerical(1, 1);
        self.bin_bool_op(x, y, BITWUZLA_KIND_BV_UGE, result)
    }

    fn push(&mut self) -> io::Result<()> {
        unsafe { bitwuzla_push(self.raw_bitwuzla(), 1) };
        Ok(())
    }

    fn pop(&mut self) -> io::Result<()> {
        unsafe { bitwuzla_pop(self.bitwuzla.raw, 1) };
        Ok(())
    }

    fn check(&mut self) -> io::Result<Response> {
        let r = unsafe { bitwuzla_check_sat(self.bitwuzla.raw) };
        match r {
            BITWUZLA_SAT => Ok(Response::Sat),
            BITWUZLA_UNSAT => Ok(Response::Unsat),
            BITWUZLA_UNKNOWN => Ok(Response::Unknown),
            _ => Err(io::Error::new(
                io::ErrorKind::Other,
                "bitwuzla_check_sat failed",
            )),
        }
    }

    fn assert(&mut self, bv: &BitVec<Self::Term>) -> io::Result<()> {
        let cond = self.bv_to_bool(&bv);
        unsafe { bitwuzla_assert(self.bitwuzla.raw, cond.raw) };
        Ok(())
    }

    fn render(&mut self, bv: &BitVec<Self::Term>) -> String {
        match bv {
            BitVec::ZeroWidth => "<zero-width>".to_string(),
            BitVec::BitVec { rep, .. } => unsafe {
                let s = CStr::from_ptr(bitwuzla_term_to_string(rep.raw));
                s.to_string_lossy().into_owned()
            },
        }
    }
}

#[cfg(test)]
use crate::test_solver;

#[cfg(test)]
test_solver!(
    bitwuzla_tests,
    super::Bitwuzla::new(&super::BitwuzlaOptions::new()).unwrap()
);
