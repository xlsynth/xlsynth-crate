// SPDX-License-Identifier: Apache-2.0

#[derive(Debug)]
pub struct XlsynthError(pub String);

impl std::fmt::Display for XlsynthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "xlsynth error: {}", self.0)
    }
}

impl std::error::Error for XlsynthError {}
