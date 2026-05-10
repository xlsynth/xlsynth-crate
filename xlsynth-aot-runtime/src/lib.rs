// SPDX-License-Identifier: Apache-2.0
//! Artifact-agnostic runtime support for standalone xlsynth AOT wrappers.
//!
//! Generated wrappers keep artifact-specific symbols, layouts, and typed
//! encode/decode logic. This crate owns the reusable ABI support that those
//! wrappers statically link into final consumer binaries.

/// ABI version understood by the standalone runtime support.
pub const SUPPORTED_ARTIFACT_ABI_VERSION: u32 = 1;
const FEATURE_ASSERTIONS: u32 = 1 << 0;
const FEATURE_TRACES: u32 = 1 << 1;
const STATUS_OK: u32 = 0;
const STATUS_UNSUPPORTED_ABI_VERSION: u32 = 1;
const STATUS_UNSUPPORTED_FEATURE: u32 = 2;
const STATUS_ALLOCATION_FAILED: u32 = 3;

#[derive(Debug, Clone, PartialEq, Eq)]
/// Error returned by standalone generated runners.
pub struct AotError(pub String);

impl std::fmt::Display for AotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for AotError {}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Decoded output plus runtime assertion messages captured for one invocation.
pub struct AotRunResult<T> {
    /// Decoded return value produced by the compiled entrypoint.
    pub output: T,
    /// Assertion messages recorded while the entrypoint executed.
    pub assert_messages: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Runtime-visible metadata describing one generated standalone artifact.
pub struct AotArtifactMetadata {
    /// Standalone ABI version expected by the generated artifact.
    pub abi_version: u32,
    /// Linked AOT entrypoint symbol exported by the compiled object.
    pub entrypoint_symbol: &'static str,
    /// Whether the artifact can emit runtime assertion events.
    pub has_asserts: bool,
    /// Whether the artifact requires trace callbacks.
    pub has_traces: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Width-preserving unsigned bits value used by standalone generated wrappers.
pub struct UBits<const BIT_COUNT: usize> {
    bytes: Vec<u8>,
}

impl<const BIT_COUNT: usize> UBits<BIT_COUNT> {
    /// Builds an unsigned bits value from a `u64` when it fits the declared
    /// width.
    pub fn from_u64(value: u64) -> Result<Self, AotError> {
        if BIT_COUNT < 64 && (value >> BIT_COUNT) != 0 {
            Err(AotError(format!(
                "AOT bits overflow: value {value} does not fit in {BIT_COUNT} unsigned bits"
            )))
        } else {
            let byte_count = BIT_COUNT.div_ceil(8);
            let mut bytes = vec![0u8; byte_count];
            bytes[..byte_count.min(8)].copy_from_slice(&value.to_le_bytes()[..byte_count.min(8)]);
            Ok(Self { bytes })
        }
    }

    /// Builds an unsigned bits value from little-endian bytes of the exact
    /// width.
    pub fn from_le_bytes(bytes: &[u8]) -> Result<Self, AotError> {
        let byte_count = BIT_COUNT.div_ceil(8);
        if bytes.len() != byte_count {
            Err(AotError(format!(
                "AOT bits width mismatch: expected {byte_count} byte(s), got {}",
                bytes.len()
            )))
        } else if !BIT_COUNT.is_multiple_of(8)
            && bytes
                .last()
                .is_some_and(|last| (last >> (BIT_COUNT % 8)) != 0)
        {
            Err(AotError(format!(
                "AOT bits overflow: encoded value does not fit in {BIT_COUNT} unsigned bits"
            )))
        } else {
            Ok(Self {
                bytes: bytes.to_vec(),
            })
        }
    }

    /// Returns this value as width-preserving little-endian bytes.
    pub fn to_le_bytes(&self) -> Result<Vec<u8>, AotError> {
        Ok(self.bytes.clone())
    }

    /// Converts this value to `u64` when the declared width allows it.
    pub fn to_u64(&self) -> Result<u64, AotError> {
        if BIT_COUNT > 64 {
            Err(AotError(format!(
                "AOT bits conversion overflow: cannot convert {BIT_COUNT} unsigned bits to u64"
            )))
        } else {
            let mut bytes = [0u8; 8];
            bytes[..self.bytes.len()].copy_from_slice(&self.bytes);
            Ok(u64::from_le_bytes(bytes))
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Width-preserving signed bits value used by standalone generated wrappers.
pub struct SBits<const BIT_COUNT: usize> {
    bytes: Vec<u8>,
}

impl<const BIT_COUNT: usize> SBits<BIT_COUNT> {
    /// Builds a signed bits value from an `i64` when it fits the declared
    /// width.
    pub fn from_i64(value: i64) -> Result<Self, AotError> {
        if BIT_COUNT == 0 {
            Err(AotError(
                "AOT signed bits width must be non-zero".to_string(),
            ))
        } else if BIT_COUNT < 64 {
            let min = -(1i128 << (BIT_COUNT - 1));
            let max = (1i128 << (BIT_COUNT - 1)) - 1;
            let value = i128::from(value);
            if value < min || value > max {
                Err(AotError(format!(
                    "AOT bits overflow: value {value} does not fit in {BIT_COUNT} signed bits"
                )))
            } else {
                let byte_count = BIT_COUNT.div_ceil(8);
                let mut bytes = (value as i64).to_le_bytes()[..byte_count].to_vec();
                if !BIT_COUNT.is_multiple_of(8) {
                    let value_mask = (1u8 << (BIT_COUNT % 8)) - 1;
                    if let Some(last) = bytes.last_mut() {
                        *last &= value_mask;
                    }
                }
                Ok(Self { bytes })
            }
        } else {
            let byte_count = BIT_COUNT.div_ceil(8);
            let mut bytes = vec![if value < 0 { 0xff } else { 0u8 }; byte_count];
            bytes[..8].copy_from_slice(&value.to_le_bytes());
            Ok(Self { bytes })
        }
    }

    /// Builds a signed bits value from little-endian bytes of the exact width.
    pub fn from_le_bytes(bytes: &[u8]) -> Result<Self, AotError> {
        let byte_count = BIT_COUNT.div_ceil(8);
        if bytes.len() != byte_count {
            Err(AotError(format!(
                "AOT bits width mismatch: expected {byte_count} byte(s), got {}",
                bytes.len()
            )))
        } else if !BIT_COUNT.is_multiple_of(8)
            && bytes
                .last()
                .is_some_and(|last| (last >> (BIT_COUNT % 8)) != 0)
        {
            Err(AotError(format!(
                "AOT bits overflow: encoded value does not fit in {BIT_COUNT} signed bits"
            )))
        } else {
            Ok(Self {
                bytes: bytes.to_vec(),
            })
        }
    }

    /// Returns this value as width-preserving little-endian bytes.
    pub fn to_le_bytes(&self) -> Result<Vec<u8>, AotError> {
        Ok(self.bytes.clone())
    }

    /// Converts this value to `i64` when the declared width allows it.
    pub fn to_i64(&self) -> Result<i64, AotError> {
        if BIT_COUNT == 0 {
            Err(AotError(
                "AOT signed bits width must be non-zero".to_string(),
            ))
        } else if BIT_COUNT > 64 {
            Err(AotError(format!(
                "AOT bits conversion overflow: cannot convert {BIT_COUNT} signed bits to i64"
            )))
        } else {
            let is_negative = self
                .bytes
                .last()
                .is_some_and(|last| (last >> ((BIT_COUNT - 1) % 8)) & 1 == 1);
            let mut bytes = if is_negative { [0xff; 8] } else { [0u8; 8] };
            bytes[..self.bytes.len()].copy_from_slice(&self.bytes);
            if is_negative && !BIT_COUNT.is_multiple_of(8) {
                let value_mask = (1u8 << (BIT_COUNT % 8)) - 1;
                bytes[self.bytes.len() - 1] |= !value_mask;
            }
            Ok(i64::from_le_bytes(bytes))
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Byte layout for one flattened AOT leaf value inside an ABI buffer.
pub struct AotElementLayout {
    /// Byte offset of the flattened leaf within its containing ABI buffer.
    pub offset: usize,
    /// Number of payload bytes occupied by the flattened leaf.
    pub data_size: usize,
    /// Number of bytes reserved for the flattened leaf including ABI padding.
    pub padded_size: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Buffer geometry needed to construct one standalone artifact runner.
pub struct AotRunnerLayout {
    /// Logical byte lengths for each input ABI buffer.
    pub input_buffer_sizes: &'static [usize],
    /// Required byte alignments for each input ABI buffer.
    pub input_buffer_alignments: &'static [usize],
    /// Logical byte lengths for each output ABI buffer.
    pub output_buffer_sizes: &'static [usize],
    /// Required byte alignments for each output ABI buffer.
    pub output_buffer_alignments: &'static [usize],
    /// Logical byte length for the shared temporary buffer.
    pub temp_buffer_size: usize,
    /// Required byte alignment for the shared temporary buffer.
    pub temp_buffer_alignment: usize,
}

#[allow(dead_code)]
/// Writes one flattened leaf value and clears any ABI padding bytes.
pub fn write_leaf_element(dst: &mut [u8], layout: &AotElementLayout, src: &[u8]) {
    let end = layout.offset + layout.padded_size;
    dst[layout.offset..end].fill(0);
    if layout.data_size > 0 {
        dst[layout.offset..layout.offset + layout.data_size]
            .copy_from_slice(&src[..layout.data_size]);
    }
}

#[allow(dead_code)]
/// Reads one flattened leaf value while ignoring ABI padding bytes.
pub fn read_leaf_element(src: &[u8], layout: &AotElementLayout, dst: &mut [u8]) {
    dst.fill(0);
    if layout.data_size > 0 {
        dst[..layout.data_size]
            .copy_from_slice(&src[layout.offset..layout.offset + layout.data_size]);
    }
}

/// Heap allocation with the alignment required by one standalone ABI buffer.
struct AlignedBuffer {
    ptr: std::ptr::NonNull<u8>,
    layout: std::alloc::Layout,
    logical_len: usize,
}

impl AlignedBuffer {
    /// Allocates one zeroed ABI buffer with the requested logical size and
    /// alignment.
    fn new(size: usize, align: usize) -> Result<Self, AotError> {
        let layout = std::alloc::Layout::from_size_align(size.max(1), align.max(1))
            .map_err(|error| AotError(format!("AOT invalid buffer layout: {error}")))?;
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        let ptr = std::ptr::NonNull::new(ptr)
            .ok_or_else(|| AotError("AOT buffer allocation failed".to_string()))?;
        Ok(Self {
            ptr,
            layout,
            logical_len: size,
        })
    }

    /// Returns the caller-visible byte length, excluding internal alignment
    /// padding.
    fn len(&self) -> usize {
        self.logical_len
    }

    /// Borrows the caller-visible buffer contents.
    fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len()) }
    }

    /// Borrows the caller-visible buffer contents for mutation.
    fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len()) }
    }

    /// Returns the buffer address passed to input ABI slots.
    fn as_const_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    /// Returns the buffer address passed to output ABI slots.
    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.ptr.as_ptr(), self.layout) }
    }
}

#[repr(C)]
struct XlsStandaloneAotRuntime {
    _private: [u8; 0],
}

#[link(name = "xls_aot_runtime", kind = "static")]
unsafe extern "C" {
    fn xls_standalone_aot_runtime_create(
        abi_version: u32,
        required_features: u32,
        out: *mut *mut XlsStandaloneAotRuntime,
    ) -> u32;
    fn xls_standalone_aot_runtime_free(runtime: *mut XlsStandaloneAotRuntime);
    fn xls_standalone_aot_runtime_clear_events(runtime: *mut XlsStandaloneAotRuntime);
    fn xls_standalone_aot_runtime_get_assert_message(
        runtime: *const XlsStandaloneAotRuntime,
        index: usize,
    ) -> *const std::ffi::c_char;
    fn xls_standalone_aot_entrypoint_trampoline(
        function_ptr: usize,
        inputs: *const *const u8,
        outputs: *const *mut u8,
        temp_buffer: *mut std::ffi::c_void,
        runtime: *mut XlsStandaloneAotRuntime,
        continuation_point: i64,
        assert_messages_count_out: *mut usize,
    ) -> i64;
}

/// Owns reusable ABI buffers and callback state for one standalone artifact.
pub struct StandaloneRunner {
    entrypoint: usize,
    input_buffers: Vec<AlignedBuffer>,
    output_buffers: Vec<AlignedBuffer>,
    input_ptrs: Vec<*const u8>,
    output_ptrs: Vec<*mut u8>,
    temp_buffer: AlignedBuffer,
    runtime: std::ptr::NonNull<XlsStandaloneAotRuntime>,
    assert_messages: Vec<String>,
}

impl StandaloneRunner {
    /// Builds reusable buffers and callback state for one compiled artifact.
    ///
    /// # Safety
    ///
    /// `entrypoint` must point to a linked standalone AOT entrypoint with the
    /// ABI expected by the artifact metadata.
    pub unsafe fn new(
        artifact_metadata: &AotArtifactMetadata,
        entrypoint: *const (),
        layout: &AotRunnerLayout,
    ) -> Result<Self, AotError> {
        if artifact_metadata.abi_version != SUPPORTED_ARTIFACT_ABI_VERSION {
            return Err(AotError(format!(
                "AOT artifact ABI mismatch: artifact={} runtime={}",
                artifact_metadata.abi_version, SUPPORTED_ARTIFACT_ABI_VERSION
            )));
        } else if layout.input_buffer_sizes.len() != layout.input_buffer_alignments.len() {
            return Err(AotError(format!(
                "AOT invalid input metadata: sizes={} alignments={}",
                layout.input_buffer_sizes.len(),
                layout.input_buffer_alignments.len()
            )));
        } else if layout.output_buffer_sizes.len() != layout.output_buffer_alignments.len() {
            return Err(AotError(format!(
                "AOT invalid output metadata: sizes={} alignments={}",
                layout.output_buffer_sizes.len(),
                layout.output_buffer_alignments.len()
            )));
        }

        let input_buffers = layout
            .input_buffer_sizes
            .iter()
            .copied()
            .zip(layout.input_buffer_alignments.iter().copied())
            .map(|(size, align)| AlignedBuffer::new(size, align))
            .collect::<Result<Vec<_>, _>>()?;
        let mut output_buffers = layout
            .output_buffer_sizes
            .iter()
            .copied()
            .zip(layout.output_buffer_alignments.iter().copied())
            .map(|(size, align)| AlignedBuffer::new(size, align))
            .collect::<Result<Vec<_>, _>>()?;
        let input_ptrs = input_buffers
            .iter()
            .map(AlignedBuffer::as_const_ptr)
            .collect::<Vec<_>>();
        let output_ptrs = output_buffers
            .iter_mut()
            .map(AlignedBuffer::as_mut_ptr)
            .collect::<Vec<_>>();
        let temp_buffer =
            AlignedBuffer::new(layout.temp_buffer_size, layout.temp_buffer_alignment)?;

        let required_features = if artifact_metadata.has_asserts {
            FEATURE_ASSERTIONS
        } else {
            0
        } | if artifact_metadata.has_traces {
            FEATURE_TRACES
        } else {
            0
        };
        let mut runtime = std::ptr::null_mut();
        let create_status = unsafe {
            xls_standalone_aot_runtime_create(
                artifact_metadata.abi_version,
                required_features,
                &mut runtime,
            )
        };
        if create_status != STATUS_OK {
            return Err(AotError(format!(
                "AOT runtime creation failed: {}",
                describe_runtime_status(create_status)
            )));
        }
        let runtime = std::ptr::NonNull::new(runtime)
            .ok_or_else(|| AotError("AOT runtime creation returned a null object".to_string()))?;

        Ok(Self {
            entrypoint: entrypoint as usize,
            input_buffers,
            output_buffers,
            input_ptrs,
            output_ptrs,
            temp_buffer,
            runtime,
            assert_messages: Vec::new(),
        })
    }

    /// Returns one mutable input buffer for argument packing.
    pub fn input_mut(&mut self, index: usize) -> &mut [u8] {
        self.input_buffers[index].as_mut_slice()
    }

    /// Returns one output buffer after entrypoint execution.
    pub fn output(&self, index: usize) -> &[u8] {
        self.output_buffers[index].as_slice()
    }

    /// Executes the linked entrypoint using the cached standalone ABI buffers.
    pub fn run_raw(&mut self) {
        self.assert_messages.clear();
        unsafe {
            xls_standalone_aot_runtime_clear_events(self.runtime.as_ptr());
            let mut assert_messages_count = 0usize;
            xls_standalone_aot_entrypoint_trampoline(
                self.entrypoint,
                self.input_ptrs.as_ptr(),
                self.output_ptrs.as_ptr(),
                self.temp_buffer.as_mut_ptr() as *mut std::ffi::c_void,
                self.runtime.as_ptr(),
                0,
                &mut assert_messages_count,
            );
            self.assert_messages = (0..assert_messages_count)
                .map(|index| {
                    let message =
                        xls_standalone_aot_runtime_get_assert_message(self.runtime.as_ptr(), index);
                    std::ffi::CStr::from_ptr(message)
                        .to_string_lossy()
                        .into_owned()
                })
                .collect();
        }
    }

    /// Returns assertion messages captured during the most recent invocation.
    pub fn assert_messages(&self) -> &[String] {
        &self.assert_messages
    }
}

impl Drop for StandaloneRunner {
    fn drop(&mut self) {
        unsafe { xls_standalone_aot_runtime_free(self.runtime.as_ptr()) }
    }
}

fn describe_runtime_status(status: u32) -> String {
    match status {
        STATUS_OK => "runtime returned success without an object".to_string(),
        STATUS_UNSUPPORTED_ABI_VERSION => "unsupported ABI version".to_string(),
        STATUS_UNSUPPORTED_FEATURE => "unsupported runtime feature".to_string(),
        STATUS_ALLOCATION_FAILED => "allocation failed".to_string(),
        other => format!("unknown status {other}"),
    }
}
