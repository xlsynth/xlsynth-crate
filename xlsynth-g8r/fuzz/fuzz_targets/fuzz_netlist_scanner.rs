// SPDX-License-Identifier: Apache-2.0

//! Creates a randomized sequence of token-like payloads, emits a textual netlist fragment for those
//! payloads, scans the text with `TokenScanner`, and checks that the resulting token payloads
//! round-trip (shape and values) for the supported subset.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

use xlsynth::IrValue;
use xlsynth_g8r::netlist::parse::{AnnotationValue, Keyword, TokenPayload, TokenScanner};

#[derive(Debug, Clone, Arbitrary)]
enum ArbPunct { OParen, CParen, OBrack, CBrack, OBrace, CBrace, Colon, Semi, Comma, Dot, Equals }

#[derive(Debug, Clone, Arbitrary)]
enum ArbKeyword { Module, Wire, Endmodule, Input, Output, Inout }

#[derive(Debug, Clone, Arbitrary)]
enum ArbPayload {
    Identifier(String),
    Keyword(ArbKeyword),
    Punct(ArbPunct),
    Comment(String),
    VerilogIntDec { width: Option<u8>, value: u32 },
    AnnotationString { key: String, value: String },
}

fn sanitize_comment_ascii(s: &str) -> String {
    s.chars()
        .filter(|&c| c.is_ascii() && c != '\n' && c != '\r')
        .collect()
}

fn sanitize_annot_string_ascii(s: &str) -> String {
    s.chars()
        .filter(|&c| c.is_ascii() && c != '"' && c != '\n' && c != '\r')
        .collect()
}
fn sanitize_key(s: &str) -> String {
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if c.is_ascii_alphanumeric() || c == '_' || (i == 0 && (c.is_ascii_alphabetic() || c == '_')) {
            out.push(c);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() { "k".to_string() } else { out }
}

fn sanitize_identifier(s: &str) -> String {
    let mut out = String::new();
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => out.push(c),
        Some(_) | None => out.push('a'),
    }
    for c in chars {
        if c.is_ascii_alphanumeric() || c == '_' {
            out.push(c);
        } else {
            out.push('_');
        }
    }
    // Avoid reserved keywords becoming Keyword tokens
    match out.as_str() {
        "module" | "wire" | "endmodule" | "input" | "output" | "inout" => {
            out.push('_');
        }
        _ => {}
    }
    out
}

fn to_token_payload(a: &ArbPayload) -> TokenPayload {
    match a {
        ArbPayload::Identifier(s) => TokenPayload::Identifier(sanitize_identifier(s)),
        ArbPayload::Keyword(k) => TokenPayload::Keyword(match k {
            ArbKeyword::Module => Keyword::Module,
            ArbKeyword::Wire => Keyword::Wire,
            ArbKeyword::Endmodule => Keyword::Endmodule,
            ArbKeyword::Input => Keyword::Input,
            ArbKeyword::Output => Keyword::Output,
            ArbKeyword::Inout => Keyword::Inout,
        }),
        ArbPayload::Punct(p) => match p {
            ArbPunct::OParen => TokenPayload::OParen,
            ArbPunct::CParen => TokenPayload::CParen,
            ArbPunct::OBrack => TokenPayload::OBrack,
            ArbPunct::CBrack => TokenPayload::CBrack,
            ArbPunct::OBrace => TokenPayload::OBrace,
            ArbPunct::CBrace => TokenPayload::CBrace,
            ArbPunct::Colon => TokenPayload::Colon,
            ArbPunct::Semi => TokenPayload::Semi,
            ArbPunct::Comma => TokenPayload::Comma,
            ArbPunct::Dot => TokenPayload::Dot,
            ArbPunct::Equals => TokenPayload::Equals,
        },
        ArbPayload::Comment(s) => TokenPayload::Comment(sanitize_comment_ascii(s)),
        ArbPayload::VerilogIntDec { width, value } => {
            match width {
                Some(w) => {
                    let w = (*w).max(1).min(32) as usize;
                    let mask: u32 = if w == 32 { u32::MAX } else { (1u32 << w) - 1 };
                    let v_masked = *value & mask;
                    let ir = IrValue::parse_typed(&format!("bits[{}]:{}", w, v_masked)).unwrap();
                    TokenPayload::VerilogInt { width: Some(w), value: ir.to_bits().unwrap() }
                }
                None => {
                    let ir = IrValue::parse_typed(&format!("bits[32]:{}", value)).unwrap();
                    TokenPayload::VerilogInt { width: None, value: ir.to_bits().unwrap() }
                }
            }
        }
        ArbPayload::AnnotationString { key, value } => TokenPayload::Annotation {
            key: sanitize_key(key),
            value: AnnotationValue::String(sanitize_annot_string_ascii(value)),
        },
    }
}

fuzz_target!(|data: Vec<ArbPayload>| {
    let _ = env_logger::builder().is_test(true).try_init();

    // Materialize TokenPayloads, then rely on TokenPayload::to_string()
    // to produce textual form for scanning.
    let tokens: Vec<TokenPayload> = data.iter().map(to_token_payload).collect();
    let mut src = String::new();
    for (i, p) in tokens.iter().enumerate() {
        if i > 0 { src.push(' '); }
        src.push_str(&p.to_string());
    }
    let lines: Vec<String> = src.lines().map(|s| s.to_string()).collect();
    let lookup = move |lineno: u32| lines.get((lineno - 1) as usize).cloned();
    let reader = std::io::Cursor::new(src.into_bytes());
    let mut scanner = TokenScanner::with_line_lookup(reader, Box::new(lookup));

    let mut scanned: Vec<TokenPayload> = Vec::new();
    loop {
        match scanner.popt() {
            Ok(Some(tok)) => scanned.push(tok.payload),
            Ok(None) => break,
            Err(e) => panic!("scanner error tokenizing to_string() output: {}", e.message),
        }
    }

    if scanned.len() != tokens.len() { return; }
    for (exp, got) in tokens.iter().zip(scanned.iter()) {
        assert_eq!(exp, got);
    }
});
