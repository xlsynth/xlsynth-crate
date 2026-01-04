// SPDX-License-Identifier: Apache-2.0

/// Returns \(\lceil \log_2(n) \rceil\) for `usize` values.
///
/// By convention:
/// - `ceil_log2(0) == 0`
/// - `ceil_log2(1) == 0`
pub fn ceil_log2(n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    // For n > 1: ceil(log2(n)) == floor(log2(n-1)) + 1
    (usize::BITS - (n - 1).leading_zeros()) as usize
}
