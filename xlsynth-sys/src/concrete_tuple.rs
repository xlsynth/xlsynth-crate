// SPDX-License-Identifier: Apache-2.0

//! Optional concrete DSLX tuple accessors for opaque imported or generic types.

use std::ffi::CStr;
use std::sync::OnceLock;

use crate::CDslxType;

/// Optional symbols resolved together so callers can check the capability.
pub const SYMBOLS: [&CStr; 2] = [
    c"xls_dslx_type_tuple_get_member_count",
    c"xls_dslx_type_tuple_get_member_type",
];

type TupleCount = unsafe extern "C" fn(*const CDslxType) -> i64;
type TupleMember = unsafe extern "C" fn(*const CDslxType, i64) -> *mut CDslxType;

/// Entry points for inspecting concrete tuple members.
pub struct ConcreteTupleApi {
    pub tuple_count: TupleCount,
    pub tuple_member: TupleMember,
}

/// Looks up the optional ABI once in the linked XLS DSO. Ordinary DSLX
/// tuples with written annotations use the published v0.61.0 API.
pub fn api() -> Option<&'static ConcreteTupleApi> {
    static API: OnceLock<Option<ConcreteTupleApi>> = OnceLock::new();
    API.get_or_init(|| {
        // Resolve in the DSO that owns the opaque DSLX handles, even if the
        // process loaded another version of libxls. Referencing the required
        // parse entry point also keeps the linked DSO alive under --as-needed.
        unsafe {
            let mut info = std::mem::MaybeUninit::<libc::Dl_info>::uninit();
            let parse = crate::xls_dslx_parse_and_typecheck as *const ();
            if libc::dladdr(parse.cast(), info.as_mut_ptr()) == 0 {
                return None;
            }
            let handle = libc::dlopen(
                info.assume_init().dli_fname,
                libc::RTLD_LAZY | libc::RTLD_NOLOAD,
            );
            if handle.is_null() {
                return None;
            }
            let [tuple_count, tuple_member] =
                SYMBOLS.map(|symbol| libc::dlsym(handle, symbol.as_ptr()));
            libc::dlclose(handle);
            if tuple_count.is_null() || tuple_member.is_null() {
                return None;
            }
            Some(ConcreteTupleApi {
                tuple_count: std::mem::transmute::<*mut libc::c_void, TupleCount>(tuple_count),
                tuple_member: std::mem::transmute::<*mut libc::c_void, TupleMember>(tuple_member),
            })
        }
    })
    .as_ref()
}
