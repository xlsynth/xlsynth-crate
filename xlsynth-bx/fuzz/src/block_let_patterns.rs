// SPDX-License-Identifier: Apache-2.0

//! Compare BX block let patterns against the same bindings in native DSLX.

use crate::eval::expr_eval::{bx_source, dslx_source};
use crate::function_oracle::{DslxExecution, FunctionProgram, check_sources, compare_bx_program};
use crate::input_seed;
use xlsynth_bx::ast::Type;

/// Generate a bounded tuple shape and an independently chosen pattern for it.
struct Generator<'a> {
    bytes: std::slice::Iter<'a, u8>,
    bound: Vec<(String, Type)>,
}

impl Generator<'_> {
    fn byte(&mut self) -> u8 {
        self.bytes.next().copied().unwrap_or(0)
    }

    fn tuple_type(&mut self, depth: usize) -> Type {
        let arity = [2, 1, 3, 0][usize::from(self.byte() % 4)];
        Type::Tuple((0..arity).map(|_| self.member_type(depth)).collect())
    }

    fn member_type(&mut self, depth: usize) -> Type {
        let kind = self.byte();
        if depth < 2 && kind % 4 == 1 {
            return self.tuple_type(depth + 1);
        }
        if kind % 4 == 2 {
            return Type::Bool;
        }
        let bits = Type::Bits {
            signed: kind & 4 != 0,
            width: 1 + usize::from(self.byte() % 96),
        };
        if kind % 4 == 3 {
            Type::Array {
                element: Box::new(bits),
                size: 1 + usize::from(self.byte() % 3),
            }
        } else {
            bits
        }
    }

    /// Emit one name per selected member. A rest can cover an arbitrary
    /// contiguous span, including zero members at either end of the tuple.
    fn pattern(&mut self, ty: &Type) -> String {
        let mode = self.byte();
        if let Type::Tuple(fields) = ty
            && mode & 3 < 2
        {
            let mut parts = Vec::new();
            let rest = if mode & 4 != 0 {
                let start = usize::from(self.byte()) % (fields.len() + 1);
                let count = usize::from(self.byte()) % (fields.len() - start + 1);
                Some(start..start + count)
            } else {
                None
            };
            for (index, field) in fields.iter().enumerate() {
                if let Some(rest) = &rest {
                    if rest.start == index {
                        parts.push("..".to_owned());
                    }
                    if rest.contains(&index) {
                        continue;
                    }
                }
                parts.push(self.pattern(field));
            }
            if rest.is_some_and(|r| r.start == fields.len()) {
                parts.push("..".to_owned());
            }
            return tuple(&parts);
        }
        if mode & 3 == 3 {
            return "_".to_owned();
        }
        let name = format!("piece{}", self.bound.len());
        self.bound.push((name.clone(), ty.clone()));
        name
    }
}

fn tuple(members: &[String]) -> String {
    let comma = if members.len() == 1 { "," } else { "" };
    format!("({}{comma})", members.join(", "))
}

/// Compare acceptance, lowered IR, and concretely evaluated DSLX results.
pub fn check_block_let_patterns(data: &[u8]) -> Result<(), String> {
    let mut g = Generator {
        bytes: data.iter(),
        bound: Vec::new(),
    };
    let flags = g.byte();
    let case = g.byte();
    if flags & 128 != 0 {
        if case & 128 != 0 {
            return compare_inferred_members(data, flags, g.byte(), g.byte(), case & 64 != 0);
        }
        return literal_pattern(data, flags, g.byte());
    }
    let mutation = case % 12;
    let payload = g.tuple_type(0);
    let pattern = g.pattern(&payload);
    let params = format!("tag:u8, data:{payload}");
    let direct = flags & 64 != 0 && mutation < 7;
    let shape = if direct {
        payload.to_string()
    } else {
        format!("(u8, {payload}, u8)")
    };
    // Both outer members depend on the previous binding of the name that
    // the first member can shadow. An early scope update changes the result.
    let (prefix, prior, first) = match flags & 3 {
        1 => ("", "tag", "tag"),
        2 => ("let prior = !tag;", "prior", "prior"),
        _ => ("", "tag", "first"),
    };
    let mut pattern = if direct {
        pattern
    } else {
        format!("({first}, {pattern}, last)")
    };
    let mut annotation = if flags & 4 == 0 {
        String::new()
    } else if case & 64 != 0 && pattern.starts_with('(') {
        format!(": {}", shape.replace(",)", ")"))
    } else {
        format!(": {shape}")
    };
    match mutation {
        7 => pattern = format!("({first}, _, last, extra)"),
        8 => pattern = format!("({first}, _, {first})"),
        9 => pattern = format!("({first}, .., .., last)"),
        10 => pattern = format!("({first}, _, (last,))"),
        11 => annotation = format!(": (s8, {payload}, u8)"),
        _ => { /* Keep the generated valid pattern and annotation. */ }
    }
    let pack = |input: &str| {
        if direct {
            "data".to_owned()
        } else {
            format!("({input} + u8:1, data, {input})")
        }
    };
    let helpers = format!("fn pack({params}) -> {shape} {{ {} }}", pack("tag"));
    let child = format!("block pack(clk, {params}) -> {shape} {{ {} }}", pack("tag"));
    let (bx_helpers, dslx_helpers, rhs) = match flags >> 3 & 3 {
        1 => (helpers.clone(), helpers, format!("pack({prior}, data)")),
        2 => (child, helpers, format!("pack({prior}, data)")),
        _ => (String::new(), String::new(), pack(prior)),
    };
    // Invalid cases have an independently well-typed result. A spurious
    // acceptance cannot be hidden by a second error from an absent binder.
    let (result, value) = if mutation < 7 {
        let mut types = vec!["u8".to_owned()];
        let first = if direct { "tag" } else { first };
        let mut values = vec![if flags & 32 != 0 {
            format!("acc.0 + {first} + u8:1")
        } else {
            first.to_owned()
        }];
        if !direct {
            types.push("u8".into());
            values.push("last".into());
        }
        for (name, ty) in g.bound {
            types.push(ty.to_string());
            values.push(name);
        }
        (tuple(&types), tuple(&values))
    } else {
        ("u8".to_owned(), "u8:0".to_owned())
    };
    let bindings = format!("{prefix} let {pattern}{annotation} = {rhs};");
    let body = if flags & 32 != 0 {
        format!("unroll_for! (i, acc) in u32:0..u32:2 {{ {bindings} {value} }}(zero!<{result}>())")
    } else {
        format!("{bindings} {value}")
    };
    let bx = format!("{bx_helpers}\n{}", bx_source(&params, &result, &body));
    compare_bx_program(
        &bx,
        FunctionProgram {
            functions: dslx_helpers,
            params,
            result,
            body,
            valid: Some(mutation < 7),
            dslx_ir: true,
            dynamic_index: None,
        },
        input_seed(data),
        DslxExecution::ConcreteConstants,
        Vec::new(),
        &[],
    )
}

/// Compare two destructured values whose use in equality gives their literal
/// members a common type, including in an unrolled iteration.
fn compare_inferred_members(
    data: &[u8],
    flags: u8,
    left: u8,
    right: u8,
    projected: bool,
) -> Result<(), String> {
    let literals = [
        "1",
        "17",
        "-1",
        "0xff",
        "255",
        "0b11111111",
        "-8",
        "0x8",
        "0",
    ];
    let (left_literals, right_literals): (&[&str], &[&str]) = if projected {
        (
            &["s8:-1", "s8:-128", "s8:17"],
            &["0xff", "0x80", "0b11111111", "17"],
        )
    } else {
        (&literals, &literals)
    };
    let left = left_literals[usize::from(left) % left_literals.len()];
    let right = right_literals[usize::from(right) % right_literals.len()];
    let shape = |value: &str| match flags & 3 {
        0 => value.to_owned(),
        1 => format!("({value},)"),
        2 => format!("[{value}]"),
        _ if flags & 32 != 0 => format!("(!flag, [{value}])"),
        _ => format!("(flag, [{value}])"),
    };
    let body = |left: &str, right: &str| {
        let (binding, rhs) = if projected {
            // The selected tuple's member inherits the concrete signed type
            // from the opposite operand, including equal-width hex literals.
            let selected = if flags & 1 == 0 {
                format!("(({right},), flag).0")
            } else {
                format!("[({right},)][u1:0]")
            };
            (
                format!("let (lo,) = ((1, ({left},)),);"),
                format!("(1, {selected})"),
            )
        } else if flags & 16 == 0 {
            (
                format!("let (lo, hi) = ({}, {});", shape(left), shape(right)),
                "hi".to_owned(),
            )
        } else {
            let left = shape(left);
            let right = shape(right);
            let rhs = match flags & 96 {
                64 => format!("(if flag {{ {right} }} else {{ {left} }})"),
                96 => format!("(match flag {{ true => {right}, _ => {left} }})"),
                _ => right,
            };
            (format!("let (lo,) = ({left},);"), rhs)
        };
        let comparison = if flags & 4 == 0 {
            format!("lo == {rhs}")
        } else {
            format!("{rhs} == lo")
        };
        if flags & 8 == 0 {
            format!("{binding} flag != ({comparison})")
        } else {
            format!(
                "unroll_for! (_, acc) in u8:0..u8:2 {{ {binding} acc || (flag != ({comparison})) }}(false)"
            )
        }
    };
    let params = "flag: bool";
    let original = body(left, right);
    let bx = bx_source(params, "bool", &original);
    check_sources(
        &bx,
        &dslx_source(params, "bool", &original),
        Some(true),
        &[],
    )?;
    // Pinned DSLX can retain the original widths during conversion of the
    // equality. Let native DSLX unify the same literals in an array, whose
    // indexed elements already have the common width and signedness.
    compare_bx_program(
        &bx,
        FunctionProgram {
            functions: format!("const VALUES = [{left}, {right}];"),
            params: params.into(),
            result: "bool".into(),
            body: body("VALUES[u32:0]", "VALUES[u32:1]"),
            valid: Some(true),
            dslx_ir: true,
            dynamic_index: None,
        },
        input_seed(data),
        DslxExecution::ConcreteConstants,
        Vec::new(),
        &[],
    )
}

/// Let DSLX decide whether an inferred integer field fits a later function
/// argument, including signed high-bit literals and explicit annotations.
fn literal_pattern(data: &[u8], flags: u8, byte: u8) -> Result<(), String> {
    let width = 1 + usize::from(byte % 32);
    let sign = if flags & 1 == 0 { 'u' } else { 's' };
    let ty = format!("{sign}N[{width}]");
    let (raw, other, minimum_width) = [
        ("1", "0", 1),
        ("0x7f", "0x41", 7),
        ("0x80", "0x81", 8),
        ("0xff", "0xf0", 8),
        ("0x80 + 1", "0x82", 8),
        ("-1", "-1", 1),
    ][usize::from(flags >> 2 & 7) % 6];
    let cast_array = |name: &str| format!("[{name}[u32:0] as {ty}, {name}[u32:1] as {ty}]");
    let (field_type, field_expr, normalized) = match flags >> 5 & 3 {
        0 => (ty.clone(), raw.to_owned(), format!("raw as {ty}")),
        1 => (
            format!("({ty}, u8)"),
            format!("({raw}, x)"),
            format!("(raw.0 as {ty}, raw.1)"),
        ),
        2 => (
            format!("{ty}[2]"),
            format!("[{raw}, {other}]"),
            cast_array("raw"),
        ),
        _ => (
            format!("(u8, {ty}[2])"),
            format!("(x, [{raw}, {other}])"),
            format!("(raw.0, {})", cast_array("raw.1")),
        ),
    };
    let annotation = if flags & 2 != 0 {
        format!(": (({field_type}, u8), u8)")
    } else {
        String::new()
    };
    let body = format!(
        "let _ = x; let _ = !x; let ((raw, _), live){annotation} = (({field_expr}, x), !x); (use_part(raw), live)"
    );
    let mut program = FunctionProgram {
        functions: format!("fn use_part(x:{field_type})->{field_type} {{ x }}"),
        params: "x:u8".into(),
        result: format!("({field_type}, u8)"),
        body,
        valid: None,
        dslx_ir: true,
        dynamic_index: None,
    };
    let bx = format!(
        "{}\n{}",
        program.functions,
        bx_source(&program.params, &program.result, &program.body)
    );
    if annotation.is_empty() && width != minimum_width {
        let reference = format!(
            "{}\n{}",
            program.functions,
            dslx_source(&program.params, &program.result, &program.body)
        );
        // Shared rejection completes this sample before execution.
        if check_sources(&bx, &reference, None, &[])?.is_none() {
            return Ok(());
        }
        // Pinned DSLX typechecks widening a destructured literal, but its
        // converter can abort and bytecode can return the minimum-width value.
        // The cast makes the already accepted width explicit. These literals
        // and the sum fit their initial widths, so it preserves the value.
        program.body = program
            .body
            .replace("use_part(raw)", &format!("use_part({normalized})"));
        program.valid = Some(true);
    }
    compare_bx_program(
        &bx,
        program,
        input_seed(data),
        DslxExecution::ConcreteConstants,
        Vec::new(),
        &[],
    )
}

#[cfg(test)]
mod tests {
    use super::check_block_let_patterns;

    #[test]
    fn compare_generated_block_let_patterns() -> Result<(), String> {
        for flags in [0, 5, 10, 17, 35, 50, 64, 84, 112] {
            for mutation in [0, 7, 8, 9, 10, 11] {
                check_block_let_patterns(&[
                    flags, mutation, 2, 1, 1, 7, 64, 2, 0, 7, 2, 0, 7, 4, 1, 1, 0, 0, 2,
                ])?;
            }
        }
        for flags in [4, 68, 100] {
            check_block_let_patterns(&[flags, 72, 1, 0, 7, 0, 0])?;
        }
        for literal in 0..6 {
            for mode in 0..4 {
                for width in [1, 7, 8, 17] {
                    for shape in 0..4 {
                        check_block_let_patterns(&[
                            128 | (shape << 5) | (literal << 2) | mode,
                            0,
                            width - 1,
                        ])?;
                    }
                }
            }
        }
        for (left, right) in [(0, 1), (0, 2), (3, 2), (4, 2), (5, 2), (7, 6), (8, 8)] {
            for mode in 0..32 {
                check_block_let_patterns(&[128 | mode, 128, left, right])?;
                if mode & 16 != 0 {
                    for branch in [64, 96] {
                        check_block_let_patterns(&[128 | mode | branch, 128, left, right])?;
                    }
                    if mode & 3 == 3 {
                        check_block_let_patterns(&[128 | mode | 32, 128, left, right])?;
                    }
                }
            }
        }
        for (left, right) in [(0, 0), (0, 1), (1, 1), (0, 2), (2, 3)] {
            for mode in [0, 1, 4, 5, 8, 9, 12, 13] {
                check_block_let_patterns(&[128 | mode, 192, left, right])?;
            }
        }
        Ok(())
    }
}
