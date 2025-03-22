// SPDX-License-Identifier: Apache-2.0

pub struct AsciiStream<I: Iterator<Item = u8>> {
    iter: I,
    buffer: Vec<u8>,
    raw_pos: usize,
    lineno: usize,
    colno: usize,
}

impl<I: Iterator<Item = u8>> AsciiStream<I> {
    pub fn new(iter: I) -> Self {
        Self {
            iter,
            buffer: Vec::new(),
            raw_pos: 0,
            lineno: 0,
            colno: 0,
        }
    }

    fn get_pos(&self) -> (usize, usize) {
        (self.lineno, self.colno)
    }

    pub fn human_pos(&self) -> String {
        // Since lineno and colno are zero-based.
        format!("{}:{}", self.lineno + 1, self.colno + 1)
    }

    fn ensure_buffer(&mut self, n: usize) {
        while self.buffer.len().saturating_sub(self.raw_pos) < n {
            if let Some(b) = self.iter.next() {
                self.buffer.push(b);
            } else {
                break;
            }
        }
    }

    fn consume(&mut self, n: usize) {
        assert!(self.raw_pos + n <= self.buffer.len());
        for i in 0..n {
            let b = self.buffer[self.raw_pos + i];
            if b == b'\n' {
                self.lineno += 1;
                self.colno = 0;
            } else {
                self.colno += 1;
            }
        }
        self.raw_pos += n;
        if self.raw_pos > 64 * 1024 {
            self.buffer.drain(0..self.raw_pos);
            self.raw_pos = 0;
        }
    }

    pub fn try_consume(&mut self, n: usize) -> bool {
        if self.raw_pos + n <= self.buffer.len() {
            self.consume(n);
            true
        } else {
            false
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.ensure_buffer(1);
        self.buffer.get(self.raw_pos).cloned()
    }

    fn peek_ahead(&mut self, n: usize) -> Option<u8> {
        self.ensure_buffer(n);
        self.buffer.get(self.raw_pos + n).cloned()
    }

    fn skip_whitespace(&mut self) {
        loop {
            if let Some(b) = self.peek() {
                if b.is_ascii_whitespace() {
                    self.consume(1);
                } else {
                    break;
                }
            } else {
                break;
            }
        }
    }

    fn skip_comment(&mut self) -> Result<(), String> {
        if self.peek_is_noskip(b"/*") {
            self.consume(2);
            loop {
                if self.peek_is_noskip(b"*/") {
                    self.consume(2);
                    break;
                }
                self.ensure_buffer(1);
                if self.peek().is_none() {
                    return Err("Unterminated comment".to_string());
                }
                self.consume(1);
            }
            Ok(())
        } else {
            Ok(())
        }
    }

    fn skip_line_continuation(&mut self) {
        if self.peek_is_noskip(b"\\") {
            self.consume(1);
        }
    }

    fn skip_whitespace_and_comments(&mut self) -> Result<(), String> {
        loop {
            let start_pos = self.get_pos();
            self.skip_whitespace();
            self.skip_comment()?;
            self.skip_line_continuation();
            if self.get_pos() == start_pos {
                return Ok(());
            }
        }
    }

    fn peek_is_noskip(&mut self, expected: &[u8]) -> bool {
        self.ensure_buffer(expected.len());
        for (i, &expected_byte) in expected.iter().enumerate() {
            if let Some(b) = self.peek_ahead(i) {
                if b != expected_byte {
                    return false;
                }
            } else {
                return false;
            }
        }
        true
    }

    pub fn peek_is(&mut self, expected: &[u8]) -> Result<bool, String> {
        self.skip_whitespace_and_comments()?;
        Ok(self.peek_is_noskip(expected))
    }

    pub fn pop_or_error(&mut self, expected: &[u8], context: &str) -> Result<(), String> {
        if self.peek_is(expected)? {
            self.consume(expected.len());
            Ok(())
        } else {
            Err(format!(
                "Expected: {:?} at {}, rest: {:?}",
                String::from_utf8_lossy(expected),
                context,
                self.peek_line()
            ))
        }
    }

    pub fn pop_semi_or_newline(&mut self, context: &str) -> Result<(), String> {
        loop {
            self.ensure_buffer(1);
            if let Some(b) = self.peek() {
                if b == b';' || b == b'\n' {
                    self.consume(1);
                    return Ok(());
                } else if b == b' ' || b == b'\t' {
                    self.consume(1);
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        Err(format!(
            "Expected `;` or newline in {} @ {} rest: {:?}",
            context,
            self.human_pos(),
            self.peek_line()
        ))
    }

    // This scans forward through whitespace and/or comments until we see either an
    // EOL or another character.
    pub fn peek_char_or_eol(&mut self) -> Result<Option<u8>, String> {
        loop {
            self.ensure_buffer(2);
            if self.peek_is_noskip(b"/*") {
                self.skip_comment()?;
            } else if self.peek() == Some(b'\n') {
                return Ok(Some(b'\n'));
            } else if self.peek().is_some() && self.peek().unwrap().is_ascii_whitespace() {
                self.consume(1);
            } else if let Some(b) = self.peek() {
                return Ok(Some(b));
            } else {
                return Ok(None);
            }
        }
    }

    pub fn try_pop(&mut self, expected: &[u8]) -> Result<bool, String> {
        if self.peek_is(expected)? {
            self.consume(expected.len());
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub fn peek_line(&mut self) -> String {
        let slice = &self.buffer[self.raw_pos..];
        // Since contents are ASCII, this conversion is safe.
        String::from_utf8_lossy(slice).into_owned()
    }

    pub fn pop_identifier_or_error(&mut self, context: &str) -> Result<String, String> {
        self.skip_whitespace_and_comments()?;
        let mut ident = Vec::new();
        while {
            self.ensure_buffer(1);
            if let Some(b) = self.peek() {
                b.is_ascii_alphanumeric() || b == b'_'
            } else {
                false
            }
        } {
            ident.push(self.peek().unwrap());
            self.consume(1);
        }
        if ident.is_empty() {
            Err(format!(
                "Expected identifier in {} @ {} rest: {:?}",
                context,
                self.human_pos(),
                self.peek_line()
            ))
        } else {
            // Since we assume ASCII, converting vector of bytes into String is safe.
            Ok(String::from_utf8_lossy(&ident).into_owned())
        }
    }

    pub fn pop_number(&mut self) -> Result<f64, String> {
        self.skip_whitespace_and_comments()?;
        let mut num_str = String::new();
        let mut saw_dot = false;
        let mut saw_e = false;
        loop {
            self.ensure_buffer(1);
            if let Some(b) = self.peek() {
                if num_str.is_empty() && b == b'-' {
                    num_str.push(b as char);
                    self.consume(1);
                } else if b.is_ascii_digit() {
                    num_str.push(b as char);
                    self.consume(1);
                } else if b == b'.' {
                    if saw_dot {
                        return Err("Multiple dots in number".to_string());
                    }
                    saw_dot = true;
                    num_str.push(b as char);
                    self.consume(1);
                } else if b == b'e' || b == b'E' {
                    if saw_e {
                        return Err("Multiple e's in number".to_string());
                    }
                    saw_e = true;
                    num_str.push(b as char);
                    self.consume(1);

                    // If the peek is a minus sign that's acceptable here.
                    self.ensure_buffer(1);
                    if let Some(b) = self.peek() {
                        if b == b'-' {
                            num_str.push(b as char);
                            self.consume(1);
                        }
                    }
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        num_str
            .parse::<f64>()
            .map_err(|e| format!("Failed to parse number: {} error: {}", num_str, e))
    }

    pub fn pop_string(&mut self) -> Result<String, String> {
        self.skip_whitespace_and_comments()?;
        self.pop_or_error(b"\"", "string value start")?;
        let mut s = String::new();
        loop {
            self.ensure_buffer(1);
            if let Some(b) = self.peek() {
                if b == b'"' {
                    self.consume(1);
                    break;
                } else {
                    s.push(b as char);
                    self.consume(1);
                }
            } else {
                break;
            }
        }
        Ok(s)
    }

    pub fn peek_is_numeric(&mut self) -> Result<bool, String> {
        self.skip_whitespace_and_comments()?;
        self.ensure_buffer(1);
        if let Some(b) = self.peek() {
            Ok(b.is_ascii_digit())
        } else {
            Ok(false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_char_stream_basic_iteration() {
        let input = "hello";
        let mut stream = AsciiStream::new(input.bytes());
        // Verify that peek returns the first character without consuming it.
        assert_eq!(stream.peek(), Some(b'h'));
        // Ensure that consecutive calls to peek yield the same result.
        assert_eq!(stream.peek(), Some(b'h'));

        // Calling consume(1) should consume the character.
        stream.consume(1);

        // Now peek should report the next character.
        assert_eq!(stream.peek(), Some(b'e'));

        // Consume the remaining characters.
        assert!(stream.peek_is_noskip(b"ello"));
        assert!(stream.try_pop(b"ello").unwrap());

        // After consuming everything, peek() and next() should return None.
        assert_eq!(stream.peek(), None);
    }

    #[test]
    fn test_char_stream_empty() {
        let input = "";
        let mut stream = AsciiStream::new(input.bytes());
        // For an empty stream, both peek and next should immediately return None.
        assert_eq!(stream.peek(), None);
    }

    #[test]
    fn test_char_stream_peek_with_whitespace_skipping() {
        let input = " hello there";
        let mut stream = AsciiStream::new(input.bytes());
        assert!(!stream.peek_is_noskip(b"hello"));
        assert!(stream.peek_is(b"hello").unwrap());
        assert!(stream.try_pop(b"hello").unwrap());
        assert!(stream.try_pop(b"there").unwrap());
        assert_eq!(stream.peek(), None);
    }

    #[test]
    fn test_pop_negative_number() {
        let input = "-1.23e-4";
        let mut stream = AsciiStream::new(input.bytes());
        assert_eq!(stream.pop_number().unwrap(), -1.23e-4);
    }
}
