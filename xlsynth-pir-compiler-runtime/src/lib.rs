// SPDX-License-Identifier: Apache-2.0

//! Runtime ABI and observable-event collection for compiled PIR functions.

use std::ffi::c_void;
use std::marker::PhantomData;
use std::ptr;

/// Native compiled-function entrypoint shared by in-memory and AOT execution.
pub type CompiledEntrypoint = unsafe extern "C" fn(
    inputs: *const *const u8,
    output: *mut u8,
    scratch: *mut u8,
    context: *mut RawExecutionContext,
) -> i32;

/// Opaque execution context forwarded by compiled code to runtime callbacks.
#[repr(C)]
pub struct RawExecutionContext {
    private_state: *mut c_void,
}

/// Kind of observable PIR event described by an event site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventKind {
    Assert,
    Cover,
    Trace,
}

/// Native data description sufficient for immediate trace-value decoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TraceValueLayout {
    Bits {
        bit_count: usize,
        byte_count: usize,
    },
    Array {
        element: Box<TraceValueLayout>,
        element_count: usize,
    },
    Tuple {
        fields: Vec<TraceTupleFieldLayout>,
        byte_count: usize,
    },
    Token,
}

/// Description of one tuple field supplied as a trace operand.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceTupleFieldLayout {
    pub layout: Box<TraceValueLayout>,
    pub offset: usize,
}

/// Static information attached to one observable node in compiled code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EventSiteMetadata {
    pub node_text_id: usize,
    pub kind: EventKind,
    pub label: Option<String>,
    pub message: Option<String>,
    pub format: Option<String>,
    pub operand_layouts: Vec<TraceValueLayout>,
}

/// Runtime metadata for all observable sites in one compiled function.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CompiledFunctionMetadata {
    pub event_sites: Vec<EventSiteMetadata>,
}

/// A failed compiled assertion, resolved to its source-site metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssertionFailure {
    pub node_text_id: usize,
    pub message: String,
    pub label: String,
}

/// One emitted compiled trace statement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceMessage {
    pub node_text_id: usize,
    pub message: String,
    pub verbosity: i64,
}

/// Accumulated execution count for a compiled `cover` site.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoverCount {
    pub node_text_id: usize,
    pub label: String,
    pub count: u64,
}

/// Rust-owned observable results recorded while executing compiled code.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExecutionResult {
    pub assertion_failures: Vec<AssertionFailure>,
    pub trace_messages: Vec<TraceMessage>,
    pub cover_counts: Vec<CoverCount>,
}

struct ContextState {
    metadata: *const CompiledFunctionMetadata,
    assertion_failures: Vec<AssertionFailure>,
    trace_messages: Vec<TraceMessage>,
    event_counts: Vec<u64>,
}

/// Rust-owned event collector used for one or more compiled executions.
///
/// Cover counts accumulate until [`Self::clear`] is called. Assertion and
/// trace messages also accumulate, permitting callers to consume a batch of
/// invocations through one context.
pub struct ExecutionContext<'metadata> {
    state: Box<ContextState>,
    marker: PhantomData<&'metadata CompiledFunctionMetadata>,
}

impl<'metadata> ExecutionContext<'metadata> {
    /// Creates an empty collector for the supplied function metadata.
    pub fn new(metadata: &'metadata CompiledFunctionMetadata) -> Self {
        Self {
            state: Box::new(ContextState {
                metadata,
                assertion_failures: Vec::new(),
                trace_messages: Vec::new(),
                event_counts: vec![0; metadata.event_sites.len()],
            }),
            marker: PhantomData,
        }
    }

    /// Returns an opaque ABI object valid while this context is mutably
    /// borrowed.
    pub fn raw_context(&mut self) -> RawExecutionContext {
        RawExecutionContext {
            private_state: ptr::from_mut(self.state.as_mut()).cast(),
        }
    }

    /// Resolves all currently recorded events into ordinary Rust values.
    pub fn result(&self) -> ExecutionResult {
        let metadata = self.metadata();
        let cover_counts = metadata
            .event_sites
            .iter()
            .zip(&self.state.event_counts)
            .filter(|(site, _)| site.kind == EventKind::Cover)
            .map(|(site, count)| CoverCount {
                node_text_id: site.node_text_id,
                label: site.label.clone().unwrap_or_default(),
                count: *count,
            })
            .collect();
        ExecutionResult {
            assertion_failures: self.state.assertion_failures.clone(),
            trace_messages: self.state.trace_messages.clone(),
            cover_counts,
        }
    }

    /// Clears all event records and accumulated cover counters.
    pub fn clear(&mut self) {
        self.state.assertion_failures.clear();
        self.state.trace_messages.clear();
        self.state.event_counts.fill(0);
    }

    fn metadata(&self) -> &CompiledFunctionMetadata {
        // SAFETY: the context's lifetime guarantees metadata remains alive.
        unsafe { &*self.state.metadata }
    }
}

unsafe fn state_from_raw<'a>(context: *mut RawExecutionContext) -> &'a mut ContextState {
    assert!(
        !context.is_null(),
        "compiled PIR callback requires an execution context"
    );
    // SAFETY: generated entrypoints receive a `RawExecutionContext` produced
    // by `ExecutionContext::raw_context` for the duration of the call.
    unsafe {
        (*context)
            .private_state
            .cast::<ContextState>()
            .as_mut()
            .expect("compiled PIR callback received an invalid execution context")
    }
}

fn site(state: &ContextState, site_id: u32, kind: EventKind) -> Option<&EventSiteMetadata> {
    // SAFETY: the owning `ExecutionContext` keeps metadata alive.
    let metadata = unsafe { state.metadata.as_ref()? };
    let site = metadata.event_sites.get(site_id as usize)?;
    (site.kind == kind).then_some(site)
}

/// Records a failed assertion from generated code.
///
/// # Safety
///
/// `context` must point to an active raw context created by
/// [`ExecutionContext::raw_context`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn xlsynth_pir_record_assert(
    context: *mut RawExecutionContext,
    site_id: u32,
) {
    // SAFETY: forwarded from the caller's ABI contract.
    let state = unsafe { state_from_raw(context) };
    let Some(site) = site(state, site_id, EventKind::Assert).cloned() else {
        return;
    };
    state.assertion_failures.push(AssertionFailure {
        node_text_id: site.node_text_id,
        message: site.message.unwrap_or_default(),
        label: site.label.unwrap_or_default(),
    });
}

/// Records one active cover occurrence from generated code.
///
/// # Safety
///
/// `context` must point to an active raw context created by
/// [`ExecutionContext::raw_context`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn xlsynth_pir_record_cover(context: *mut RawExecutionContext, site_id: u32) {
    // SAFETY: forwarded from the caller's ABI contract.
    let state = unsafe { state_from_raw(context) };
    if site(state, site_id, EventKind::Cover).is_some() {
        if let Some(count) = state.event_counts.get_mut(site_id as usize) {
            *count = count.saturating_add(1);
        }
    }
}

/// Records and formats one active trace occurrence from generated code.
///
/// # Safety
///
/// `context` must point to an active raw context created by
/// [`ExecutionContext::raw_context`]. Each operand pointer must describe
/// readable native storage matching the corresponding site's operand layout
/// for the duration of this callback.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn xlsynth_pir_record_trace(
    context: *mut RawExecutionContext,
    site_id: u32,
    operand_ptrs: *const *const u8,
) {
    // SAFETY: forwarded from the caller's ABI contract.
    let state = unsafe { state_from_raw(context) };
    let Some(site) = site(state, site_id, EventKind::Trace).cloned() else {
        return;
    };
    if !site.operand_layouts.is_empty() && operand_ptrs.is_null() {
        return;
    }
    let values = site
        .operand_layouts
        .iter()
        .enumerate()
        .map(|(index, layout)| {
            // SAFETY: the generated caller supplies one pointer per metadata operand.
            let pointer = unsafe { *operand_ptrs.add(index) };
            // SAFETY: the pointer and recursively described native layout match.
            unsafe { format_native_value(pointer, layout) }
        })
        .collect::<Vec<_>>();
    state.trace_messages.push(TraceMessage {
        node_text_id: site.node_text_id,
        message: substitute_trace_values(site.format.as_deref().unwrap_or(""), &values),
        verbosity: 0,
    });
}

fn substitute_trace_values(format: &str, values: &[String]) -> String {
    let mut output = String::new();
    let mut remainder = format;
    let mut value_index = 0usize;
    while let Some(index) = remainder.find("{}") {
        output.push_str(&remainder[..index]);
        if let Some(value) = values.get(value_index) {
            output.push_str(value);
        }
        value_index += 1;
        remainder = &remainder[index + 2..];
    }
    output.push_str(remainder);
    if value_index < values.len() {
        output.push_str(" [");
        output.push_str(&values[value_index..].join(", "));
        output.push(']');
    }
    output
}

unsafe fn format_native_value(pointer: *const u8, layout: &TraceValueLayout) -> String {
    match layout {
        TraceValueLayout::Bits {
            bit_count,
            byte_count,
        } => {
            let mut bytes = [0u8; std::mem::size_of::<u64>()];
            // SAFETY: callback ABI provides native scalar storage of this size.
            unsafe { ptr::copy_nonoverlapping(pointer, bytes.as_mut_ptr(), *byte_count) };
            let mut value = u64::from_ne_bytes(bytes);
            if *bit_count < 64 {
                value &= (1u64 << *bit_count) - 1;
            }
            format!("bits[{bit_count}]:{value}")
        }
        TraceValueLayout::Array {
            element,
            element_count,
        } => {
            let elements = (0..*element_count)
                .map(|index| {
                    // SAFETY: each element is within the caller-provided array region.
                    unsafe {
                        format_native_value(
                            pointer.wrapping_add(index * element.byte_count()),
                            element,
                        )
                    }
                })
                .collect::<Vec<_>>();
            format!("[{}]", elements.join(", "))
        }
        TraceValueLayout::Tuple { fields, .. } => {
            let fields = fields
                .iter()
                .map(|field| {
                    // SAFETY: each field offset is prescribed by native tuple metadata.
                    unsafe {
                        format_native_value(pointer.wrapping_add(field.offset), &field.layout)
                    }
                })
                .collect::<Vec<_>>();
            format!("({})", fields.join(", "))
        }
        TraceValueLayout::Token => "token".to_string(),
    }
}

impl TraceValueLayout {
    fn byte_count(&self) -> usize {
        match self {
            Self::Bits { byte_count, .. } => *byte_count,
            Self::Array {
                element,
                element_count,
            } => element.byte_count() * element_count,
            Self::Tuple { byte_count, .. } => *byte_count,
            Self::Token => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metadata() -> CompiledFunctionMetadata {
        CompiledFunctionMetadata {
            event_sites: vec![
                EventSiteMetadata {
                    node_text_id: 10,
                    kind: EventKind::Cover,
                    label: Some("covered".to_string()),
                    message: None,
                    format: None,
                    operand_layouts: Vec::new(),
                },
                EventSiteMetadata {
                    node_text_id: 11,
                    kind: EventKind::Assert,
                    label: Some("assert_label".to_string()),
                    message: Some("failed".to_string()),
                    format: None,
                    operand_layouts: Vec::new(),
                },
                EventSiteMetadata {
                    node_text_id: 12,
                    kind: EventKind::Trace,
                    label: None,
                    message: None,
                    format: Some("x={} arr={}".to_string()),
                    operand_layouts: vec![
                        TraceValueLayout::Bits {
                            bit_count: 8,
                            byte_count: 1,
                        },
                        TraceValueLayout::Array {
                            element: Box::new(TraceValueLayout::Bits {
                                bit_count: 8,
                                byte_count: 1,
                            }),
                            element_count: 2,
                        },
                    ],
                },
            ],
        }
    }

    #[test]
    fn cover_and_assert_callbacks_collect_rust_owned_results() {
        let metadata = metadata();
        let mut context = ExecutionContext::new(&metadata);
        let mut raw = context.raw_context();
        // SAFETY: `raw` points into `context` for these immediate calls.
        unsafe {
            xlsynth_pir_record_cover(&mut raw, 0);
            xlsynth_pir_record_cover(&mut raw, 0);
            xlsynth_pir_record_assert(&mut raw, 1);
        }
        let result = context.result();
        assert_eq!(result.cover_counts[0].count, 2);
        assert_eq!(result.cover_counts[0].label, "covered");
        assert_eq!(result.assertion_failures[0].message, "failed");
        assert_eq!(result.assertion_failures[0].label, "assert_label");
    }

    #[test]
    fn trace_callback_decodes_values_before_native_storage_changes() {
        let metadata = metadata();
        let mut context = ExecutionContext::new(&metadata);
        let mut raw = context.raw_context();
        let mut scalar = 7u8;
        let mut array = [2u8, 3u8];
        let operands = [
            ptr::from_ref(&scalar).cast::<u8>(),
            ptr::from_ref(&array).cast::<u8>(),
        ];
        // SAFETY: operands use the native layouts specified by trace metadata.
        unsafe { xlsynth_pir_record_trace(&mut raw, 2, operands.as_ptr()) };
        scalar = 90;
        array[0] = 91;
        assert_eq!(scalar, 90);
        assert_eq!(array[0], 91);
        assert_eq!(
            context.result().trace_messages[0].message,
            "x=bits[8]:7 arr=[bits[8]:2, bits[8]:3]"
        );
    }

    #[test]
    fn clear_resets_accumulated_event_results() {
        let metadata = metadata();
        let mut context = ExecutionContext::new(&metadata);
        let mut raw = context.raw_context();
        // SAFETY: `raw` points into `context` for this immediate call.
        unsafe { xlsynth_pir_record_cover(&mut raw, 0) };
        context.clear();
        let result = context.result();
        assert!(result.assertion_failures.is_empty());
        assert!(result.trace_messages.is_empty());
        assert_eq!(result.cover_counts[0].count, 0);
    }
}
