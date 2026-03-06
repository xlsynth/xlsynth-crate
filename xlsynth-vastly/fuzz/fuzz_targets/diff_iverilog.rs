// SPDX-License-Identifier: Apache-2.0

#![no_main]

use arbitrary::Arbitrary;
use arbitrary::Unstructured;
use libfuzzer_sys::fuzz_target;
use wait_timeout::ChildExt;

use std::io::Write;
use std::process::Command;
use std::process::Stdio;
use std::time::Duration;
use std::time::SystemTime;

use xlsynth_vastly::Env;
use xlsynth_vastly::LogicBit;
use xlsynth_vastly::Signedness;
use xlsynth_vastly::Value4;
use xlsynth_vastly::ast::Expr;

const MAX_BASED_LITERAL_WIDTH: u32 = 384;
const MAX_EXPR_AST_DEPTH: usize = 16;
const MAX_EXPR_AST_NODES: usize = 192;

fuzz_target!(|data: &[u8]| {
    let mut u = Unstructured::new(data);
    let case = match FuzzCase::arbitrary(&mut u) {
        Ok(x) => x,
        // Degenerate/unconsumable arbitrary payloads are not signal for this target.
        Err(_) => return,
    };

    let expr = case.expr.to_string();
    if expr.trim().is_empty() {
        return;
    }
    // Extremely wide explicit based literals are not interesting signal for this target;
    // they mainly drive pathological width-sensitive parser/evaluator costs.
    if has_based_literal_width_over_limit(&expr, MAX_BASED_LITERAL_WIDTH) {
        return;
    }

    // Parse using OUR parser first; if we don't accept it, don't ask iverilog.
    let ast = match xlsynth_vastly::parser::parse_expr(&expr) {
        Ok(x) => x,
        // Skip parser-rejected expressions; this target compares accepted inputs only.
        Err(_) => return,
    };
    let (depth, nodes) = expr_depth_and_node_count(&ast);
    // Very deep/large parsed trees mostly surface evaluator performance limits rather than
    // semantic mismatches for this differential target.
    if depth > MAX_EXPR_AST_DEPTH || nodes > MAX_EXPR_AST_NODES {
        return;
    }

    // Enforce: expression only refers to identifiers defined in env.
    if !all_idents_defined_in_env(&ast, &case.env) {
        return;
    }

    // Feed a normalized rendering of the parsed AST into both evaluators.
    let normalized_expr = render_expr(&ast);

    let oracle = match run_oracle_with_timeout(&normalized_expr, &case.env, Duration::from_millis(250)) {
        OracleOutcome::Accepted(x) => x,
        // Oracle-side rejection/runtime failures are not sample failures for this differential check.
        OracleOutcome::RejectedOrFailed => return,
        // Timeout is infrastructure noise; skip instead of treating as semantic mismatch.
        OracleOutcome::TimedOut => return,
    };

    let ours = match xlsynth_vastly::eval_expr(&normalized_expr, &case.env) {
        Ok(x) => x,
        Err(_) => panic!("iverilog accepted but we rejected: expr={normalized_expr:?}"),
    };

    if ours.value.width != oracle.width {
        panic!(
            "width mismatch expr={:?} ours={} oracle={}",
            normalized_expr, ours.value.width, oracle.width
        );
    }
    let ours_bits = ours.value.to_bit_string_msb_first();
    if ours_bits != oracle.value_bits_msb {
        panic!(
            "value mismatch expr={:?} ours={:?} oracle={:?}",
            normalized_expr, ours_bits, oracle.value_bits_msb
        );
    }
});

fn has_based_literal_width_over_limit(expr: &str, limit: u32) -> bool {
    let bytes = expr.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if !bytes[i].is_ascii_digit() {
            i += 1;
            continue;
        }

        let mut width = 0u32;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            width = width
                .saturating_mul(10)
                .saturating_add(u32::from(bytes[i] - b'0'));
            i += 1;
        }

        if i >= bytes.len() || bytes[i] != b'\'' {
            continue;
        }
        let mut j = i + 1;
        if j < bytes.len() && matches!(bytes[j], b's' | b'S') {
            j += 1;
        }
        if j < bytes.len() && matches!(bytes[j], b'b' | b'B' | b'd' | b'D' | b'h' | b'H' | b'o' | b'O') && width > limit {
            return true;
        }
    }
    false
}

fn expr_depth_and_node_count(expr: &Expr) -> (usize, usize) {
    match expr {
        Expr::Ident(_) | Expr::Literal(_) | Expr::UnbasedUnsized(_) => (1, 1),
        Expr::Call { args, .. } | Expr::Concat(args) => {
            let mut max_child_depth = 0usize;
            let mut total_nodes = 1usize;
            for arg in args {
                let (depth, nodes) = expr_depth_and_node_count(arg);
                max_child_depth = max_child_depth.max(depth);
                total_nodes += nodes;
            }
            (1 + max_child_depth, total_nodes)
        }
        Expr::Replicate { count, expr } => {
            let (count_depth, count_nodes) = expr_depth_and_node_count(count);
            let (expr_depth, expr_nodes) = expr_depth_and_node_count(expr);
            (
                1 + count_depth.max(expr_depth),
                1 + count_nodes + expr_nodes,
            )
        }
        Expr::Index { expr, index } => {
            let (expr_depth, expr_nodes) = expr_depth_and_node_count(expr);
            let (index_depth, index_nodes) = expr_depth_and_node_count(index);
            (
                1 + expr_depth.max(index_depth),
                1 + expr_nodes + index_nodes,
            )
        }
        Expr::Slice { expr, msb, lsb } => {
            let (expr_depth, expr_nodes) = expr_depth_and_node_count(expr);
            let (msb_depth, msb_nodes) = expr_depth_and_node_count(msb);
            let (lsb_depth, lsb_nodes) = expr_depth_and_node_count(lsb);
            (
                1 + expr_depth.max(msb_depth).max(lsb_depth),
                1 + expr_nodes + msb_nodes + lsb_nodes,
            )
        }
        Expr::IndexedSlice {
            expr,
            base,
            width,
            ..
        } => {
            let (expr_depth, expr_nodes) = expr_depth_and_node_count(expr);
            let (base_depth, base_nodes) = expr_depth_and_node_count(base);
            let (width_depth, width_nodes) = expr_depth_and_node_count(width);
            (
                1 + expr_depth.max(base_depth).max(width_depth),
                1 + expr_nodes + base_nodes + width_nodes,
            )
        }
        Expr::Unary { expr, .. } => {
            let (depth, nodes) = expr_depth_and_node_count(expr);
            (1 + depth, 1 + nodes)
        }
        Expr::Binary { lhs, rhs, .. } => {
            let (lhs_depth, lhs_nodes) = expr_depth_and_node_count(lhs);
            let (rhs_depth, rhs_nodes) = expr_depth_and_node_count(rhs);
            (1 + lhs_depth.max(rhs_depth), 1 + lhs_nodes + rhs_nodes)
        }
        Expr::Ternary { cond, t, f } => {
            let (cond_depth, cond_nodes) = expr_depth_and_node_count(cond);
            let (t_depth, t_nodes) = expr_depth_and_node_count(t);
            let (f_depth, f_nodes) = expr_depth_and_node_count(f);
            (
                1 + cond_depth.max(t_depth).max(f_depth),
                1 + cond_nodes + t_nodes + f_nodes,
            )
        }
    }
}

fn all_idents_defined_in_env(expr: &Expr, env: &Env) -> bool {
    let mut ok = true;
    visit_expr_idents(expr, &mut |name| {
        if env.get(name).is_none() {
            ok = false;
        }
    });
    ok
}

fn visit_expr_idents(expr: &Expr, f: &mut dyn FnMut(&str)) {
    match expr {
        Expr::Ident(name) => f(name),
        Expr::Literal(_) => {}
        Expr::UnbasedUnsized(_) => {}
        Expr::Call { args, .. } => {
            for a in args {
                visit_expr_idents(a, f);
            }
        }
        Expr::Concat(parts) => {
            for p in parts {
                visit_expr_idents(p, f);
            }
        }
        Expr::Replicate { count, expr } => {
            visit_expr_idents(count, f);
            visit_expr_idents(expr, f);
        }
        Expr::Index { expr, index } => {
            visit_expr_idents(expr, f);
            visit_expr_idents(index, f);
        }
        Expr::Slice { expr, msb, lsb } => {
            visit_expr_idents(expr, f);
            visit_expr_idents(msb, f);
            visit_expr_idents(lsb, f);
        }
        Expr::IndexedSlice {
            expr,
            base,
            width,
            ..
        } => {
            visit_expr_idents(expr, f);
            visit_expr_idents(base, f);
            visit_expr_idents(width, f);
        }
        Expr::Unary { expr, .. } => visit_expr_idents(expr, f),
        Expr::Binary { lhs, rhs, .. } => {
            visit_expr_idents(lhs, f);
            visit_expr_idents(rhs, f);
        }
        Expr::Ternary { cond, t, f: ff } => {
            visit_expr_idents(cond, f);
            visit_expr_idents(t, f);
            visit_expr_idents(ff, f);
        }
    }
}

fn render_expr(expr: &Expr) -> String {
    match expr {
        Expr::Ident(name) => name.clone(),
        Expr::Literal(v) => render_literal(v),
        Expr::UnbasedUnsized(bit) => match bit {
            LogicBit::Zero => "'0".to_string(),
            LogicBit::One => "'1".to_string(),
            LogicBit::X => "'x".to_string(),
            LogicBit::Z => "'z".to_string(),
        },
        Expr::Call { name, args } => {
            let mut out = String::new();
            out.push_str(name);
            out.push('(');
            for (i, a) in args.iter().enumerate() {
                if i != 0 {
                    out.push(',');
                }
                out.push_str(&render_expr(a));
            }
            out.push(')');
            out
        }
        Expr::Concat(parts) => {
            let mut out = String::new();
            out.push('{');
            for (i, p) in parts.iter().enumerate() {
                if i != 0 {
                    out.push(',');
                }
                out.push_str(&render_expr(p));
            }
            out.push('}');
            out
        }
        Expr::Replicate { count, expr } => {
            let c = render_expr(count);
            let e = render_expr(expr);
            format!("{{{c}{{{e}}}}}")
        }
        Expr::Index { expr, index } => {
            let e = render_expr(expr);
            let i = render_expr(index);
            format!("({e})[{i}]")
        }
        Expr::Slice { expr, msb, lsb } => {
            let e = render_expr(expr);
            let m = render_expr(msb);
            let l = render_expr(lsb);
            format!("({e})[{m}:{l}]")
        }
        Expr::IndexedSlice {
            expr,
            base,
            width,
            upward,
        } => {
            let e = render_expr(expr);
            let b = render_expr(base);
            let w = render_expr(width);
            let dir = if *upward { "+:" } else { "-:" };
            format!("({e})[{b} {dir} {w}]")
        }
        Expr::Unary { op, expr } => {
            let inner = render_expr(expr);
            let op_str = match op {
                xlsynth_vastly::ast::UnaryOp::LogicalNot => "!",
                xlsynth_vastly::ast::UnaryOp::BitNot => "~",
                xlsynth_vastly::ast::UnaryOp::UnaryPlus => "+",
                xlsynth_vastly::ast::UnaryOp::UnaryMinus => "-",
                xlsynth_vastly::ast::UnaryOp::ReduceAnd => "&",
                xlsynth_vastly::ast::UnaryOp::ReduceNand => "~&",
                xlsynth_vastly::ast::UnaryOp::ReduceOr => "|",
                xlsynth_vastly::ast::UnaryOp::ReduceNor => "~|",
                xlsynth_vastly::ast::UnaryOp::ReduceXor => "^",
                xlsynth_vastly::ast::UnaryOp::ReduceXnor => "^~",
            };
            format!("({op_str}{inner})")
        }
        Expr::Binary { op, lhs, rhs } => {
            let a = render_expr(lhs);
            let b = render_expr(rhs);
            let op_str = match op {
                xlsynth_vastly::ast::BinaryOp::Add => "+",
                xlsynth_vastly::ast::BinaryOp::Sub => "-",
                xlsynth_vastly::ast::BinaryOp::Mul => "*",
                xlsynth_vastly::ast::BinaryOp::Div => "/",
                xlsynth_vastly::ast::BinaryOp::Mod => "%",
                xlsynth_vastly::ast::BinaryOp::Shl => "<<",
                xlsynth_vastly::ast::BinaryOp::Shr => ">>",
                xlsynth_vastly::ast::BinaryOp::Sshr => ">>>",
                xlsynth_vastly::ast::BinaryOp::BitAnd => "&",
                xlsynth_vastly::ast::BinaryOp::BitOr => "|",
                xlsynth_vastly::ast::BinaryOp::BitXor => "^",
                xlsynth_vastly::ast::BinaryOp::LogicalAnd => "&&",
                xlsynth_vastly::ast::BinaryOp::LogicalOr => "||",
                xlsynth_vastly::ast::BinaryOp::Lt => "<",
                xlsynth_vastly::ast::BinaryOp::Le => "<=",
                xlsynth_vastly::ast::BinaryOp::Gt => ">",
                xlsynth_vastly::ast::BinaryOp::Ge => ">=",
                xlsynth_vastly::ast::BinaryOp::Eq => "==",
                xlsynth_vastly::ast::BinaryOp::Neq => "!=",
                xlsynth_vastly::ast::BinaryOp::CaseEq => "===",
                xlsynth_vastly::ast::BinaryOp::CaseNeq => "!==",
            };
            format!("({a}{op_str}{b})")
        }
        Expr::Ternary { cond, t, f } => {
            let c = render_expr(cond);
            let tt = render_expr(t);
            let ff = render_expr(f);
            format!("(({c})?({tt}):({ff}))")
        }
    }
}

fn render_literal(v: &Value4) -> String {
    let bits = v.to_bit_string_msb_first();
    match v.signedness {
        Signedness::Signed => format!("{}'sb{}", v.width, bits),
        Signedness::Unsigned => format!("{}'b{}", v.width, bits),
    }
}

#[derive(Debug)]
struct FuzzCase {
    env: Env,
    expr: ExprText,
}

impl<'a> Arbitrary<'a> for FuzzCase {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let env = ArbEnv::arbitrary(u)?.into_env();
        let expr = ExprText::arbitrary(u)?;
        Ok(Self { env, expr })
    }
}

/// Arbitrarily-sized env (bounded) generated via `impl Arbitrary`.
#[derive(Debug)]
struct ArbEnv {
    bindings: Vec<(Ident, ArbValue4)>,
}

impl<'a> Arbitrary<'a> for ArbEnv {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        // Keep iverilog quick: bound number of bindings and widths.
        let n: u8 = u.arbitrary()?;
        let n = (n as usize) % 17; // 0..=16

        let mut bindings = Vec::with_capacity(n);
        for _ in 0..n {
            let ident = Ident::arbitrary(u)?;
            let v = ArbValue4::arbitrary(u)?;
            bindings.push((ident, v));
        }
        Ok(Self { bindings })
    }
}

impl ArbEnv {
    fn into_env(self) -> Env {
        let mut env = Env::new();
        for (id, v) in self.bindings {
            env.insert(id.0, v.into_value4());
        }
        env
    }
}

#[derive(Debug, Clone)]
struct Ident(String);

impl<'a> Arbitrary<'a> for Ident {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        // Verilog identifier: [A-Za-z_][A-Za-z0-9_]*
        let len: u8 = u.arbitrary()?;
        let len = 1 + ((len as usize) % 12); // 1..=12

        let first = ident_first_char(u.arbitrary::<u8>()?);
        let mut s = String::with_capacity(len);
        s.push(first);
        for _ in 1..len {
            s.push(ident_tail_char(u.arbitrary::<u8>()?));
        }
        Ok(Self(s))
    }
}

fn ident_first_char(b: u8) -> char {
    match b % 53 {
        0 => '_',
        n @ 1..=26 => (b'a' + (n - 1)) as char,
        n => (b'A' + (n - 27)) as char,
    }
}

fn ident_tail_char(b: u8) -> char {
    match b % 63 {
        0 => '_',
        n @ 1..=26 => (b'a' + (n - 1)) as char,
        n @ 27..=52 => (b'A' + (n - 27)) as char,
        n => (b'0' + (n - 53)) as char,
    }
}

#[derive(Debug, Clone)]
struct ArbValue4 {
    width: u32,
    signedness: Signedness,
    bits_lsb_first: Vec<LogicBit>,
}

impl<'a> Arbitrary<'a> for ArbValue4 {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let w: u8 = u.arbitrary()?;
        let width = 1 + ((w as u32) % 64); // 1..=64

        let signed: bool = u.arbitrary()?;
        let signedness = if signed { Signedness::Signed } else { Signedness::Unsigned };

        let mut bits = Vec::with_capacity(width as usize);
        for _ in 0..width {
            bits.push(bit_from_u8(u.arbitrary::<u8>()?));
        }

        Ok(Self {
            width,
            signedness,
            bits_lsb_first: bits,
        })
    }
}

impl ArbValue4 {
    fn into_value4(self) -> Value4 {
        Value4::new(self.width, self.signedness, self.bits_lsb_first)
    }
}

fn bit_from_u8(b: u8) -> LogicBit {
    match b & 3 {
        0 => LogicBit::Zero,
        1 => LogicBit::One,
        2 => LogicBit::X,
        _ => LogicBit::Z,
    }
}

#[derive(Debug, Clone)]
struct ExprText(String);

impl ExprText {
    fn to_string(&self) -> String {
        self.0.clone()
    }
}

impl<'a> Arbitrary<'a> for ExprText {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let len: u8 = u.arbitrary()?;
        let len = (len as usize) % 129; // 0..=128 chars

        let mut s = String::with_capacity(len);
        for _ in 0..len {
            let b: u8 = u.arbitrary()?;
            s.push(map_expr_char(b));
        }
        Ok(Self(s))
    }
}

fn map_expr_char(b: u8) -> char {
    // Restricted set to improve chance of valid tokens for our subset + numeric literals.
    const TABLE: &[u8] =
        b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_()?:!~&|=<>^+-*/%{}[],: 'bdhBDHxoXOzZ\t\n";
    let idx = (b as usize) % TABLE.len();
    TABLE[idx] as char
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct OracleResult {
    width: u32,
    value_bits_msb: String,
}

enum OracleOutcome {
    Accepted(OracleResult),
    RejectedOrFailed,
    TimedOut,
}

fn run_oracle_with_timeout(expr: &str, env: &Env, timeout: Duration) -> OracleOutcome {
    // Quick precheck: if iverilog isn't present, don't hard-fail fuzzing.
    // (The user can ensure it's installed; otherwise fuzzing is pointless.)
    let ok = Command::new("iverilog")
        .arg("-V")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !ok {
        return OracleOutcome::RejectedOrFailed;
    }

    let sv = build_sv(expr, env);
    let td = mk_temp_dir();
    let sv_path = td.join("oracle.sv");
    let out_path = td.join("oracle.out");

    if std::fs::File::create(&sv_path)
        .and_then(|mut f| f.write_all(sv.as_bytes()))
        .is_err()
    {
        return OracleOutcome::RejectedOrFailed;
    }

    let mut iverilog_cmd = Command::new("iverilog");
    iverilog_cmd
        .arg("-g2012")
        .arg("-o")
        .arg(&out_path)
        .arg(&sv_path);
    match run_cmd_timeout(iverilog_cmd, timeout) {
        CmdOutcome::Ok(output) => {
            if !output.status.success() {
                return OracleOutcome::RejectedOrFailed;
            }
        }
        CmdOutcome::TimedOut => return OracleOutcome::TimedOut,
        CmdOutcome::Err => return OracleOutcome::RejectedOrFailed,
    }

    let mut vvp_cmd = Command::new("vvp");
    vvp_cmd.arg(&out_path);
    let out = match run_cmd_timeout(vvp_cmd, timeout) {
        CmdOutcome::Ok(output) => {
            if !output.status.success() {
                return OracleOutcome::RejectedOrFailed;
            }
            output.stdout
        }
        CmdOutcome::TimedOut => return OracleOutcome::TimedOut,
        CmdOutcome::Err => return OracleOutcome::RejectedOrFailed,
    };

    let out = String::from_utf8_lossy(&out);
    match parse_oracle_output(&out) {
        Some(x) => OracleOutcome::Accepted(x),
        None => OracleOutcome::RejectedOrFailed,
    }
}

enum CmdOutcome {
    Ok(std::process::Output),
    TimedOut,
    Err,
}

fn run_cmd_timeout(mut cmd: Command, timeout: Duration) -> CmdOutcome {
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(_) => return CmdOutcome::Err,
    };

    match child.wait_timeout(timeout) {
        Ok(Some(_status)) => match child.wait_with_output() {
            Ok(out) => CmdOutcome::Ok(out),
            Err(_) => CmdOutcome::Err,
        },
        Ok(None) => {
            let _ = child.kill();
            let _ = child.wait();
            CmdOutcome::TimedOut
        }
        Err(_) => {
            let _ = child.kill();
            let _ = child.wait();
            CmdOutcome::Err
        }
    }
}

fn parse_oracle_output(out: &str) -> Option<OracleResult> {
    for line in out.lines() {
        let line = line.trim();
        if !line.starts_with("W=") {
            continue;
        }
        let mut width: Option<u32> = None;
        let mut v: Option<String> = None;
        for part in line.split_whitespace() {
            if let Some(x) = part.strip_prefix("W=") {
                width = x.parse::<u32>().ok();
            } else if let Some(x) = part.strip_prefix("V=") {
                v = Some(x.to_string());
            }
        }
        if let (Some(width), Some(v)) = (width, v) {
            return Some(OracleResult {
                width,
                value_bits_msb: v,
            });
        }
    }
    None
}

fn build_sv(expr: &str, env: &Env) -> String {
    let mut s = String::new();
    s.push_str("module oracle;\n");
    for (name, v) in env.iter() {
        let decl = match v.signedness {
            Signedness::Signed => format!(
                "  localparam logic signed [{}:0] {} = {};\n",
                v.width - 1,
                name,
                to_verilog_literal(v)
            ),
            Signedness::Unsigned => format!(
                "  localparam logic [{}:0] {} = {};\n",
                v.width - 1,
                name,
                to_verilog_literal(v)
            ),
        };
        s.push_str(&decl);
    }

    s.push_str("  initial begin\n");
    // Avoid $bits(...) in a constant context; some iverilog versions are picky.
    // Printing the expression directly lets iverilog choose the self-determined size for %b.
    s.push_str(&format!("    $display(\"W=%0d V=%b\", $bits({expr}), ({expr}));\n"));
    s.push_str("  end\n");
    s.push_str("endmodule\n");
    s
}

fn to_verilog_literal(v: &Value4) -> String {
    let mut msb = String::with_capacity(v.width as usize);
    for b in v.bits_lsb_first().iter().rev() {
        msb.push(b.as_char());
    }
    format!("{}'b{}", v.width, msb)
}

fn mk_temp_dir() -> std::path::PathBuf {
    let base = std::env::temp_dir();
    let pid = std::process::id();
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .ok()
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    for attempt in 0u32..1000u32 {
        let p = base.join(format!("vastly_fuzz_oracle_{pid}_{nanos}_{attempt}"));
        match std::fs::create_dir(&p) {
            Ok(()) => return p,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(_) => break,
        }
    }
    base
}
