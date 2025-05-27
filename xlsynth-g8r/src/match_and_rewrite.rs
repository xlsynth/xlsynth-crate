// SPDX-License-Identifier: Apache-2.0

#[derive(Debug, PartialEq, Eq)]
pub enum GatePattern {
    And {
        lhs: Box<Pattern>,
        rhs: Box<Pattern>,
        bind_name: String,
    },
    Any {
        bind_name: String,
    },
}

#[derive(Debug, PartialEq, Eq)]
pub enum NegatedPattern {
    Yes,
    No,
    Any,
}

#[derive(Debug, PartialEq, Eq)]
pub struct OperandPattern {
    pub negated: NegatedPattern,
    pub gate: Box<GatePattern>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Pattern {
    Gate(GatePattern),
    Operand(OperandPattern),
}

#[derive(Debug, PartialEq, Eq)]
pub enum Rewrite {
    And {
        lhs: String,
        negate_lhs: bool,
        rhs: String,
        negate_rhs: bool,
    },
    Identity {
        which: String,
    },
}

pub struct PatternRewrite {
    pub pattern: Pattern,
    pub rewrite: Rewrite,
}

pub fn parse(input: &str) -> Result<PatternRewrite, String> {
    let parts: Vec<&str> = input.split("=>").collect();
    if parts.len() != 2 {
        return Err("expected => separating pattern and rewrite".to_string());
    }
    let pattern = parse_operand_pattern(parts[0].trim())?;
    let rewrite = parse_rewrite(parts[1].trim())?;
    Ok(PatternRewrite { pattern, rewrite })
}

fn parse_operand_pattern(s: &str) -> Result<Pattern, String> {
    let mut rest = s.trim();
    let negated = if let Some(stripped) = rest.strip_prefix('~') {
        rest = stripped.trim();
        NegatedPattern::Yes
    } else {
        NegatedPattern::No
    };
    let gate = parse_gate_pattern(rest)?;
    Ok(Pattern::Operand(OperandPattern {
        negated,
        gate: Box::new(gate),
    }))
}

fn parse_gate_pattern(s: &str) -> Result<GatePattern, String> {
    let s = s.trim();
    if s.starts_with("AND(") && s.ends_with(')') {
        let inner = &s[4..s.len() - 1];
        let (lhs_str, rhs_str) = split_top_level_comma(inner).ok_or("expected comma in AND")?;
        let lhs = parse_operand_pattern(lhs_str.trim())?;
        let rhs = parse_operand_pattern(rhs_str.trim())?;
        Ok(GatePattern::And {
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
            bind_name: String::new(),
        })
    } else if is_ident(s) {
        Ok(GatePattern::Any {
            bind_name: s.to_string(),
        })
    } else {
        Err(format!("unable to parse gate pattern: {s}"))
    }
}

fn parse_rewrite(s: &str) -> Result<Rewrite, String> {
    let s = s.trim();
    if s.starts_with("AND(") && s.ends_with(')') {
        let inner = &s[4..s.len() - 1];
        let (lhs_str, rhs_str) = split_top_level_comma(inner).ok_or("expected comma in AND")?;
        let (lhs, negate_lhs) = parse_rewrite_operand(lhs_str.trim())?;
        let (rhs, negate_rhs) = parse_rewrite_operand(rhs_str.trim())?;
        Ok(Rewrite::And {
            lhs,
            negate_lhs,
            rhs,
            negate_rhs,
        })
    } else if is_ident(s) {
        Ok(Rewrite::Identity {
            which: s.to_string(),
        })
    } else {
        Err(format!("unable to parse rewrite: {s}"))
    }
}

fn parse_rewrite_operand(s: &str) -> Result<(String, bool), String> {
    let mut rest = s.trim();
    let negate = if let Some(stripped) = rest.strip_prefix('~') {
        rest = stripped.trim();
        true
    } else {
        false
    };
    if is_ident(rest) {
        Ok((rest.to_string(), negate))
    } else {
        Err(format!("invalid operand: {s}"))
    }
}

fn split_top_level_comma(s: &str) -> Option<(&str, &str)> {
    let mut depth = 0i32;
    for (i, ch) in s.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => depth -= 1,
            ',' if depth == 0 => return Some((&s[..i], &s[i + 1..])),
            _ => {}
        }
    }
    None
}

fn is_ident(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_negated_and() {
        let pr = parse("~AND(a, b) => AND(~a, ~b)").unwrap();
        assert!(matches!(pr.pattern, Pattern::Operand(_)));
        match pr.rewrite {
            Rewrite::And {
                lhs,
                negate_lhs,
                rhs,
                negate_rhs,
            } => {
                assert_eq!(lhs, "a");
                assert!(negate_lhs);
                assert_eq!(rhs, "b");
                assert!(negate_rhs);
            }
            _ => panic!("expected Rewrite::And"),
        }
    }

    #[test]
    fn test_parse_and_idempotent() {
        let pr = parse("AND(x, x) => x").unwrap();
        match pr.rewrite {
            Rewrite::Identity { which } => assert_eq!(which, "x"),
            _ => panic!("expected identity rewrite"),
        }
    }

    #[test]
    fn test_parse_nested_and() {
        let pr = parse("AND(AND(x, x), AND(x, x)) => x").unwrap();
        match pr.rewrite {
            Rewrite::Identity { which } => assert_eq!(which, "x"),
            _ => panic!("expected identity rewrite"),
        }
    }
}
