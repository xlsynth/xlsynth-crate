use std::collections::HashMap;

struct LuTableTemplate {
    variables: Vec<String>,
    indices: Vec<Vec<f64>>,
}

struct LibertyLibrary {
    name: String,
    lu_table_templates: HashMap<String, LuTableTemplate>,
    cells: Vec<LibertyCell>,
}

struct LibertyCell {
    name: String,
    area: f64,
    pins: Vec<LibertyPin>,
}

struct LibertyPinTiming {
    related_pin_name: String,
}

struct LibertyPin {
    name: String,
    direction: PinDirection,
    function: Option<String>,
    timing: LibertyPinTiming,
}

#[derive(Debug, PartialEq)]
enum PinDirection {
    Input,
    Output,
}

struct LibertyParser {
    chars: Vec<char>,
    pos: usize,
}

impl LibertyParser {
    pub fn new(text: &str) -> Self {
        Self {
            chars: text.chars().collect(),
            pos: 0,
        }
    }

    fn skip_whitespace(&mut self) {
        while self.pos < self.chars.len() && self.chars[self.pos].is_whitespace() {
            self.pos += 1;
        }
    }

    /// Note: skips whitespace before attempting to pop the value, as whitespace
    /// insensitivity is the most common case in liberty file reading.
    fn pop_or_error(&mut self, expected: &str) -> Result<(), String> {
        if self.peek_is(expected) {
            self.pos += expected.len();
            Ok(())
        } else {
            Err(format!(
                "Expected: {:?}, rest: {:?}",
                expected,
                self.peek_line()
            ))
        }
    }

    fn peek_is(&mut self, expected: &str) -> bool {
        self.skip_whitespace();
        for (i, c) in expected.chars().enumerate() {
            if self.chars[self.pos + i] != c {
                return false;
            }
        }
        true
    }

    fn peek_line(&mut self) -> String {
        let mut line = String::new();
        let mut pos = self.pos;
        while pos < self.chars.len() && self.chars[pos] != '\n' {
            line.push(self.chars[pos]);
            pos += 1;
        }
        line
    }

    fn pop_identifier_or_error(&mut self) -> Result<String, String> {
        self.skip_whitespace();
        let mut chars = String::new();
        while self.pos < self.chars.len() {
            let c = self.chars[self.pos];
            if c.is_alphanumeric() || c == '_' {
                chars.push(c);
                self.pos += 1;
            } else {
                break;
            }
        }
        Ok(chars)
    }

    fn pop_number(&mut self) -> Result<f64, String> {
        self.skip_whitespace();
        let mut chars = String::new();
        let mut saw_dot = false;
        while self.pos < self.chars.len() {
            let c = self.chars[self.pos];
            if c.is_digit(10) {
                chars.push(c);
                self.pos += 1;
            } else if c == '.' {
                if saw_dot {
                    return Err("Multiple dots in number".to_string());
                }
                saw_dot = true;
                chars.push(c);
                self.pos += 1;
            } else {
                break;
            }
        }
        Ok(chars.parse::<f64>().unwrap())
    }

    fn pop_string(&mut self) -> Result<String, String> {
        self.skip_whitespace();
        self.pop_or_error("\"")?;
        let mut chars = String::new();
        while self.pos < self.chars.len() {
            let c = self.chars[self.pos];
            if c == '"' {
                self.pos += 1;
                break;
            } else {
                chars.push(c);
                self.pos += 1;
            }
        }
        Ok(chars)
    }

    fn parse_pin(&mut self) -> Result<LibertyPin, String> {
        self.pop_or_error("pin")?;
        self.pop_or_error("(")?;
        let pin_name = self.pop_identifier_or_error()?;
        self.pop_or_error(")")?;
        self.pop_or_error("{")?;

        let mut direction = None;
        let mut function = None;
        let mut max_capacitance = None;
        let mut capacitance = None;
        while !self.peek_is("}") {
            let attribute = self.pop_identifier_or_error()?;
            self.pop_or_error(":")?;
            match attribute.as_str() {
                "direction" => {
                    let direction_str = self.pop_identifier_or_error()?;
                    direction = match direction_str.as_str() {
                        "input" => Some(PinDirection::Input),
                        "output" => Some(PinDirection::Output),
                        _ => return Err(format!("Invalid pin direction: {}", direction_str)),
                    };
                }
                "function" => {
                    function = Some(self.pop_string()?);
                }
                "max_capacitance" => {
                    max_capacitance = Some(self.pop_number()?);
                }
                "capacitance" => {
                    capacitance = Some(self.pop_number()?);
                }
                _ => return Err(format!("Invalid pin attribute: {}", attribute)),
            }
            self.pop_or_error(";")?;
        }
        self.pop_or_error("}")?;
        let timing = LibertyPinTiming {
            related_pin_name: String::new(),
        };
        let direction = if let Some(direction) = direction {
            direction
        } else {
            return Err("Pin direction not specified".to_string());
        };
        Ok(LibertyPin {
            name: pin_name,
            direction,
            function,
            timing,
        })
    }

    fn parse_cell(&mut self) -> Result<LibertyCell, String> {
        self.pop_or_error("cell")?;
        self.skip_whitespace();
        self.pop_or_error("(")?;
        self.skip_whitespace();
        let name = self.pop_identifier_or_error()?;
        self.skip_whitespace();
        self.pop_or_error(")")?;
        self.skip_whitespace();
        self.pop_or_error("{")?;
        self.pop_or_error("area")?;
        self.pop_or_error(":")?;
        let area_value: f64 = self.pop_number()?;
        self.pop_or_error(";")?;
        let mut pins = Vec::new();
        while !self.peek_is("}") {
            if self.peek_is("pin") {
                pins.push(self.parse_pin()?);
            } else {
                return Err(format!("Unexpected token; rest: {}", self.peek_line()));
            }
        }
        Ok(LibertyCell {
            name,
            area: area_value,
            pins,
        })
    }

    pub fn parse(&mut self) -> Result<LibertyLibrary, String> {
        self.pop_or_error("library")?;
        self.skip_whitespace();
        self.pop_or_error("(")?;
        self.skip_whitespace();
        let name = self.pop_identifier_or_error()?;
        self.skip_whitespace();
        self.pop_or_error(")")?;
        self.skip_whitespace();
        self.pop_or_error("{")?;
        self.skip_whitespace();
        let mut cells = Vec::new();
        while !self.peek_is("}") {
            if self.peek_is("cell") {
                cells.push(self.parse_cell()?);
            } else {
                return Err(format!("Unexpected token; rest: {}", self.peek_line()));
            }
        }

        self.pop_or_error("}")?;
        Ok(LibertyLibrary {
            name,
            lu_table_templates: HashMap::new(),
            cells,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_fake_liberty_file() {
        let text = r#"
        library (my_library) {
            cell (my_and) {
                area: 1.0;
                pin (Y) {
                    direction: output;
                    function: "(A * B)";
                    max_capacitance: 12.34;
                }
                pin (A) {
                    direction: input;
                    capacitance: 0.1;
                }
                pin (B) {
                    direction: input;
                    capacitance: 0.2;
                }
            }
        }
        "#;
        let mut parser = LibertyParser::new(text);
        let library = parser.parse().unwrap();
        assert_eq!(library.name, "my_library");
        assert_eq!(library.cells.len(), 1);
        assert_eq!(library.cells[0].name, "my_and");
        assert_eq!(library.cells[0].area, 1.0);
        assert_eq!(library.cells[0].pins.len(), 3);
        assert_eq!(library.cells[0].pins[0].name, "Y");
        assert_eq!(library.cells[0].pins[0].direction, PinDirection::Output);
        assert_eq!(
            library.cells[0].pins[0].function,
            Some("(A * B)".to_string())
        );
        assert_eq!(library.cells[0].pins[1].name, "A");
        assert_eq!(library.cells[0].pins[1].direction, PinDirection::Input);
        assert_eq!(library.cells[0].pins[2].name, "B");
        assert_eq!(library.cells[0].pins[2].direction, PinDirection::Input);
    }
}
