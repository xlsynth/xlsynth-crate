// SPDX-License-Identifier: Apache-2.0

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::ffi::c_void;
use std::ptr::NonNull;

use crate::aot_entrypoint_metadata::AotElementLayout;
use crate::aot_lib::{AotEntrypointMetadata, AotExecContext, AotResult};
use crate::ir_package::TraceMessage;
use crate::xlsynth_error::XlsynthError;

#[derive(Debug, Clone)]
pub struct AotEntrypointDescriptor<'a> {
    entrypoints_proto: &'a [u8],
    function_ptr: usize,
    metadata: AotEntrypointMetadata,
}

impl<'a> AotEntrypointDescriptor<'a> {
    /// Creates an AOT entrypoint descriptor from raw parts.
    ///
    /// This is intended for generated wrapper code and other trusted callers
    /// that obtain the function pointer from a linked XLS AOT symbol.
    ///
    /// # Safety
    ///
    /// `function_ptr` must be a valid pointer to an XLS AOT entrypoint
    /// function with the ABI expected by `xls_aot_entrypoint_trampoline`, and
    /// it must remain valid for the lifetime of the returned descriptor.
    #[doc(hidden)]
    pub unsafe fn from_raw_parts_unchecked(
        entrypoints_proto: &'a [u8],
        function_ptr: usize,
        metadata: AotEntrypointMetadata,
    ) -> Self {
        Self {
            entrypoints_proto,
            function_ptr,
            metadata,
        }
    }

    pub fn entrypoints_proto(&self) -> &'a [u8] {
        self.entrypoints_proto
    }

    pub fn metadata(&self) -> &AotEntrypointMetadata {
        &self.metadata
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AotRunResult<T> {
    pub output: T,
    pub trace_messages: Vec<TraceMessage>,
    pub assert_messages: Vec<String>,
}

pub fn write_leaf_element(dst: &mut [u8], layout: &AotElementLayout, src: &[u8]) {
    debug_assert!(
        layout.padded_size >= layout.data_size,
        "AOT invalid layout in runner helper: padded_size {} is smaller than data_size {}",
        layout.padded_size,
        layout.data_size
    );
    let end = layout.offset + layout.padded_size;
    debug_assert!(
        end <= dst.len(),
        "AOT invalid layout in runner helper: write range [{}, {}) exceeds buffer size {}",
        layout.offset,
        end,
        dst.len()
    );
    debug_assert!(
        src.len() >= layout.data_size,
        "AOT invalid layout in runner helper: source byte length {} is smaller than required data_size {}",
        src.len(),
        layout.data_size
    );
    let copy_len = layout.data_size;
    dst[layout.offset..end].fill(0);
    if copy_len > 0 {
        let data_end = layout.offset + copy_len;
        dst[layout.offset..data_end].copy_from_slice(&src[..copy_len]);
    }
}

pub fn read_leaf_element(src: &[u8], layout: &AotElementLayout, dst: &mut [u8]) {
    debug_assert!(
        layout.padded_size >= layout.data_size,
        "AOT invalid layout in runner helper: padded_size {} is smaller than data_size {}",
        layout.padded_size,
        layout.data_size
    );
    let end = layout.offset + layout.padded_size;
    debug_assert!(
        end <= src.len(),
        "AOT invalid layout in runner helper: read range [{}, {}) exceeds buffer size {}",
        layout.offset,
        end,
        src.len()
    );
    debug_assert!(
        dst.len() >= layout.data_size,
        "AOT invalid layout in runner helper: destination byte length {} is smaller than required data_size {}",
        dst.len(),
        layout.data_size
    );
    let copy_len = layout.data_size;
    dst.fill(0);
    if copy_len > 0 {
        let data_end = layout.offset + copy_len;
        dst[..copy_len].copy_from_slice(&src[layout.offset..data_end]);
    }
}

pub struct AotRunner<'a> {
    descriptor: AotEntrypointDescriptor<'a>,
    context: AotExecContext,
    input_buffers: Vec<AlignedBuffer>,
    output_buffers: Vec<AlignedBuffer>,
    input_ptrs: Vec<*const u8>,
    output_ptrs: Vec<*mut u8>,
    temp_buffer: AlignedBuffer,
    clear_events_before_run: bool,
}

impl<'a> AotRunner<'a> {
    pub fn new(descriptor: AotEntrypointDescriptor<'a>) -> AotResult<Self> {
        if descriptor.function_ptr == 0 {
            return Err(XlsynthError(
                "AOT invalid argument: entrypoint function pointer must be non-zero".to_string(),
            ));
        }
        if descriptor.metadata.input_buffer_sizes.len()
            != descriptor.metadata.input_buffer_alignments.len()
        {
            return Err(XlsynthError(format!(
                "AOT invalid argument: input buffer metadata length mismatch: sizes={} alignments={}",
                descriptor.metadata.input_buffer_sizes.len(),
                descriptor.metadata.input_buffer_alignments.len()
            )));
        }
        if descriptor.metadata.output_buffer_sizes.len()
            != descriptor.metadata.output_buffer_alignments.len()
        {
            return Err(XlsynthError(format!(
                "AOT invalid argument: output buffer metadata length mismatch: sizes={} alignments={}",
                descriptor.metadata.output_buffer_sizes.len(),
                descriptor.metadata.output_buffer_alignments.len()
            )));
        }

        let context = AotExecContext::create(descriptor.entrypoints_proto)?;

        let mut input_buffers = Vec::with_capacity(descriptor.metadata.input_buffer_sizes.len());
        for (size, align) in descriptor
            .metadata
            .input_buffer_sizes
            .iter()
            .copied()
            .zip(descriptor.metadata.input_buffer_alignments.iter().copied())
        {
            input_buffers.push(AlignedBuffer::new(size, align)?);
        }

        let mut output_buffers = Vec::with_capacity(descriptor.metadata.output_buffer_sizes.len());
        for (size, align) in descriptor
            .metadata
            .output_buffer_sizes
            .iter()
            .copied()
            .zip(descriptor.metadata.output_buffer_alignments.iter().copied())
        {
            output_buffers.push(AlignedBuffer::new(size, align)?);
        }

        let temp_buffer = AlignedBuffer::new(
            descriptor.metadata.temp_buffer_size,
            descriptor.metadata.temp_buffer_alignment,
        )?;

        let input_ptrs = input_buffers
            .iter()
            .map(AlignedBuffer::as_const_ptr)
            .collect::<Vec<_>>();
        let output_ptrs = output_buffers
            .iter()
            .map(AlignedBuffer::as_mut_ptr)
            .collect::<Vec<_>>();

        Ok(Self {
            descriptor,
            context,
            input_buffers,
            output_buffers,
            input_ptrs,
            output_ptrs,
            temp_buffer,
            clear_events_before_run: true,
        })
    }

    pub fn set_clear_events_before_run(&mut self, value: bool) {
        self.clear_events_before_run = value;
    }

    pub fn clear_events(&mut self) {
        self.context.clear_events();
    }

    pub fn input_count(&self) -> usize {
        self.input_buffers.len()
    }

    pub fn input_size(&self, index: usize) -> usize {
        self.input_buffers
            .get(index)
            .map(AlignedBuffer::len)
            .unwrap_or_else(|| {
                panic!(
                    "AOT invalid argument: input index {index} out of range (count={})",
                    self.input_buffers.len()
                )
            })
    }

    pub fn input(&self, index: usize) -> &[u8] {
        self.input_buffers
            .get(index)
            .map(AlignedBuffer::as_slice)
            .unwrap_or_else(|| {
                panic!(
                    "AOT invalid argument: input index {index} out of range (count={})",
                    self.input_buffers.len()
                )
            })
    }

    pub fn input_mut(&mut self, index: usize) -> &mut [u8] {
        let input_count = self.input_buffers.len();
        self.input_buffers
            .get_mut(index)
            .map(AlignedBuffer::as_mut_slice)
            .unwrap_or_else(|| {
                panic!(
                    "AOT invalid argument: input index {} out of range (count={})",
                    index, input_count
                );
            })
    }

    pub fn copy_input_from(&mut self, index: usize, src: &[u8]) {
        let dst = self.input_mut(index);
        assert_eq!(
            src.len(),
            dst.len(),
            "AOT invalid argument: input copy size mismatch at index {index}: src={} dst={}",
            src.len(),
            dst.len()
        );
        dst.copy_from_slice(src);
    }

    pub fn output_count(&self) -> usize {
        self.output_buffers.len()
    }

    pub fn output_size(&self, index: usize) -> usize {
        self.output_buffers
            .get(index)
            .map(AlignedBuffer::len)
            .unwrap_or_else(|| {
                panic!(
                    "AOT invalid argument: output index {index} out of range (count={})",
                    self.output_buffers.len()
                )
            })
    }

    pub fn output(&self, index: usize) -> &[u8] {
        self.output_buffers
            .get(index)
            .map(AlignedBuffer::as_slice)
            .unwrap_or_else(|| {
                panic!(
                    "AOT invalid argument: output index {index} out of range (count={})",
                    self.output_buffers.len()
                )
            })
    }

    pub fn copy_output_to(&self, index: usize, dst: &mut [u8]) {
        let src = self.output(index);
        assert_eq!(
            dst.len(),
            src.len(),
            "AOT invalid argument: output copy size mismatch at index {index}: dst={} src={}",
            dst.len(),
            src.len()
        );
        dst.copy_from_slice(src);
    }

    fn run_trampoline(&mut self) -> (usize, usize) {
        if self.clear_events_before_run {
            self.context.clear_events();
        }

        let mut trace_count = 0usize;
        let mut assert_count = 0usize;
        let _continuation = unsafe {
            xlsynth_sys::xls_aot_entrypoint_trampoline(
                self.descriptor.function_ptr,
                self.input_ptrs.as_ptr(),
                self.output_ptrs.as_ptr(),
                self.temp_buffer.as_mut_ptr() as *mut c_void,
                self.context.as_ptr(),
                0,
                &mut trace_count,
                &mut assert_count,
            )
        };

        (trace_count, assert_count)
    }

    pub fn run(&mut self) -> AotResult<()> {
        let (_trace_count, assert_count) = self.run_trampoline();
        if assert_count > 0 {
            // The AOT engine ran successfully, but the compiled function surfaced an
            // assertion. Surface the first assertion message as an error for
            // the "simple run" API.
            let first = self.assert_message(0)?;
            return Err(XlsynthError(format!("XLS assertion failed: {first}")));
        }
        Ok(())
    }

    pub fn run_with_events<T>(
        &mut self,
        make_output: impl FnOnce(&Self) -> T,
    ) -> AotResult<AotRunResult<T>> {
        let (trace_count, assert_count) = self.run_trampoline();

        let output = make_output(self);

        let mut trace_messages = Vec::with_capacity(trace_count);
        for index in 0..trace_count {
            trace_messages.push(self.trace_message(index)?);
        }

        let mut assert_messages = Vec::with_capacity(assert_count);
        for index in 0..assert_count {
            assert_messages.push(self.assert_message(index)?);
        }

        Ok(AotRunResult {
            output,
            trace_messages,
            assert_messages,
        })
    }

    pub fn trace_message(&self, index: usize) -> AotResult<TraceMessage> {
        self.context.trace_message(index)
    }

    pub fn assert_message(&self, index: usize) -> AotResult<String> {
        self.context.assert_message(index)
    }
}

struct AlignedBuffer {
    ptr: NonNull<u8>,
    len: usize,
    alloc_len: usize,
    align: usize,
}

impl AlignedBuffer {
    fn new(len: usize, align: usize) -> AotResult<Self> {
        let align = if align == 0 { 1 } else { align };
        if !align.is_power_of_two() {
            return Err(XlsynthError(format!(
                "AOT allocation failed: alignment must be a power of two, got: {align}"
            )));
        }

        // Use a minimum allocation size of 1 byte so we always have a valid,
        // non-null pointer to pass into the C ABI.
        let alloc_len = if len == 0 { 1 } else { len };
        let layout = Layout::from_size_align(alloc_len, align).map_err(|e| {
            XlsynthError(format!(
                "AOT allocation failed: invalid layout with size={alloc_len} align={align}: {e}"
            ))
        })?;

        let ptr = unsafe { alloc_zeroed(layout) };
        let ptr = NonNull::new(ptr).ok_or_else(|| {
            XlsynthError(format!(
                "AOT allocation failed: allocation returned null for size={alloc_len}"
            ))
        })?;

        Ok(Self {
            ptr,
            len,
            alloc_len,
            align,
        })
    }

    fn len(&self) -> usize {
        self.len
    }

    fn as_const_ptr(&self) -> *const u8 {
        self.ptr.as_ptr() as *const u8
    }

    fn as_mut_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.alloc_len, self.align)
            .expect("AlignedBuffer drop should have a valid layout");
        unsafe {
            dealloc(self.ptr.as_ptr(), layout);
        }
    }
}
