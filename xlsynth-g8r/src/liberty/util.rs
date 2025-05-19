// SPDX-License-Identifier: Apache-2.0

pub fn human_readable_size(size: u64) -> String {
    if size > 10 * 1024 * 1024 * 1024 {
        format!("{:.2} GiB", size as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if size > 10 * 1024 * 1024 {
        format!("{:.2} MiB", size as f64 / (1024.0 * 1024.0))
    } else if size > 10 * 1024 {
        format!("{:.2} KiB", size as f64 / 1024.0)
    } else {
        format!("{} bytes", size)
    }
}
