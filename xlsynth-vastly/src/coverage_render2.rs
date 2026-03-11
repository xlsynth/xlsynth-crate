// SPDX-License-Identifier: Apache-2.0

use crate::coverage::SpanKey;
use crate::sv_lexer::TokKind;
use crate::CoverabilityMap;
use crate::CoverageCounters;
use crate::SourceText;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum LineClass {
    Hit,
    Miss,
    NonCoverable,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum SegClass {
    TrueHit,
    TrueMiss,
    FalseHit,
    FalseMiss,
    ArmHit,
    ArmMiss,
    ZeroToggle,
}

#[derive(Debug, Copy, Clone)]
struct Segment {
    start: usize,
    end: usize,
    class: SegClass,
    priority: u8,
    // Smaller is higher priority (used to prefer inner segments in overlaps).
    span_len: usize,
}

pub fn render_annotated_source(src: &SourceText, cov: &CoverageCounters, ansi: bool) -> String {
    let cover = crate::compute_coverability_or_fallback_with_defines(src, &cov.defines);
    let fn_line_ann = build_function_line_annotations(src, cov);
    let zero_toggle_by_line = if ansi {
        build_zero_toggle_token_spans(src, cov)
    } else {
        std::collections::BTreeMap::new()
    };

    let mut out = String::new();
    let mut line_start: usize = 0;
    for (idx0, line) in src.text.lines().enumerate() {
        let line_1b = (idx0 + 1) as u32;
        let line_end = line_start + line.len();
        let class = classify_line(&cover, cov, line_1b);
        let mut segs = segments_for_line(cov, line_start, line_end);
        if let Some(spans) = zero_toggle_by_line.get(&line_1b) {
            for (s, e) in spans {
                // Lower priority than branch coloring, higher than base line bg.
                segs.push(Segment {
                    start: *s,
                    end: *e,
                    class: SegClass::ZeroToggle,
                    priority: 1,
                    span_len: e.saturating_sub(*s),
                });
            }
        }
        let suffix = match fn_line_ann.get(&line_1b) {
            Some(s) => Some(s.as_str()),
            None => None,
        };
        if ansi {
            out.push_str(&render_line_ansi(
                line_1b, line, class, &segs, line_start, suffix,
            ));
        } else {
            out.push_str(&render_line_plain(
                line_1b, line, class, &segs, line_start, suffix,
            ));
        }
        out.push('\n');
        // Advance by line plus newline.
        line_start = line_end + 1;
    }
    out
}

fn classify_line(cover: &CoverabilityMap, cov: &CoverageCounters, line_1b: u32) -> LineClass {
    if !cover.is_coverable(line_1b) {
        return LineClass::NonCoverable;
    }
    if cov.line_hits.contains_key(&line_1b) {
        LineClass::Hit
    } else {
        LineClass::Miss
    }
}

fn segments_for_line(cov: &CoverageCounters, line_start: usize, line_end: usize) -> Vec<Segment> {
    let mut segs: Vec<Segment> = Vec::new();
    for (_id, c) in &cov.ternary_branches {
        let t = c.t_span;
        let f = c.f_span;
        let t_hit = c.t_taken != 0;
        let f_hit = c.f_taken != 0;

        add_segment_if_intersects(
            &mut segs,
            t,
            line_start,
            line_end,
            if t_hit {
                SegClass::TrueHit
            } else {
                SegClass::TrueMiss
            },
        );
        add_segment_if_intersects(
            &mut segs,
            f,
            line_start,
            line_end,
            if f_hit {
                SegClass::FalseHit
            } else {
                SegClass::FalseMiss
            },
        );
    }

    for (k, count) in &cov.selected_arm_spans {
        add_segment_if_intersects(
            &mut segs,
            *k,
            line_start,
            line_end,
            if *count != 0 {
                SegClass::ArmHit
            } else {
                SegClass::ArmMiss
            },
        );
    }
    segs
}

fn add_segment_if_intersects(
    out: &mut Vec<Segment>,
    span: SpanKey,
    line_start: usize,
    line_end: usize,
    class: SegClass,
) {
    let s = span.start.max(line_start);
    let e = span.end.min(line_end);
    if s < e {
        out.push(Segment {
            start: s,
            end: e,
            class,
            priority: 0,
            span_len: span.end.saturating_sub(span.start),
        });
    }
}

fn render_line_ansi(
    line_1b: u32,
    line: &str,
    class: LineClass,
    segs: &[Segment],
    line_start: usize,
    suffix: Option<&str>,
) -> String {
    let gutter = format!("{:4} | ", line_1b);

    // Base background.
    let base_bg = match class {
        LineClass::Miss => "\x1b[48;5;224m", // light red
        LineClass::Hit => "",
        LineClass::NonCoverable => "\x1b[2m", // dim
    };
    let reset = "\x1b[0m";

    if segs.is_empty() {
        if base_bg.is_empty() {
            return match suffix {
                None => format!("{gutter}{line}"),
                Some(s) => format!("{gutter}{line}{s}"),
            };
        }
        return match suffix {
            None => format!("{gutter}{base_bg}{line}{reset}"),
            Some(s) => format!("{gutter}{base_bg}{line}{reset}{s}"),
        };
    }

    // Build boundaries for piecewise coloring.
    let mut bounds: Vec<usize> = Vec::new();
    bounds.push(line_start);
    bounds.push(line_start + line.len());
    for s in segs {
        bounds.push(s.start);
        bounds.push(s.end);
    }
    bounds.sort_unstable();
    bounds.dedup();

    let mut out = String::new();
    out.push_str(&gutter);
    if !base_bg.is_empty() {
        out.push_str(base_bg);
    }

    for w in bounds.windows(2) {
        let a = w[0];
        let b = w[1];
        if a == b {
            continue;
        }
        let rel_a = a - line_start;
        let rel_b = b - line_start;
        if !line.is_char_boundary(rel_a) || !line.is_char_boundary(rel_b) {
            continue;
        }
        let slice = &line[rel_a..rel_b];

        if let Some(seg_bg) = pick_segment_bg(segs, a, b) {
            out.push_str(seg_bg);
            out.push_str(slice);
            // Restore base background.
            out.push_str(reset);
            if !base_bg.is_empty() {
                out.push_str(base_bg);
            }
        } else {
            out.push_str(slice);
        }
    }

    if !base_bg.is_empty() {
        out.push_str(reset);
    }
    if let Some(s) = suffix {
        out.push_str(s);
    }
    out
}

fn pick_segment_bg(segs: &[Segment], a: usize, b: usize) -> Option<&'static str> {
    // Choose the most specific segment (smallest span_len) that covers [a,b).
    let mut best: Option<&Segment> = None;
    for s in segs {
        if s.start <= a && s.end >= b {
            match best {
                None => best = Some(s),
                Some(cur) => {
                    if s.priority < cur.priority
                        || (s.priority == cur.priority && s.span_len < cur.span_len)
                    {
                        best = Some(s);
                    }
                }
            }
        }
    }
    let best = match best {
        Some(b) => b,
        None => return None,
    };
    Some(match best.class {
        SegClass::TrueHit => "\x1b[48;5;120m",
        SegClass::TrueMiss => "\x1b[48;5;217m",
        SegClass::FalseHit => "\x1b[48;5;120m",
        SegClass::FalseMiss => "\x1b[48;5;217m",
        SegClass::ArmHit => "\x1b[48;5;159m",
        SegClass::ArmMiss => "\x1b[48;5;224m",
        SegClass::ZeroToggle => "\x1b[48;5;229m",
    })
}

fn render_line_plain(
    line_1b: u32,
    line: &str,
    class: LineClass,
    segs: &[Segment],
    line_start: usize,
    suffix: Option<&str>,
) -> String {
    let prefix = match class {
        LineClass::Hit => "HIT ",
        LineClass::Miss => "MIS ",
        LineClass::NonCoverable => "SKP ",
    };
    let gutter = format!("{prefix}{:4} | ", line_1b);

    if segs.is_empty() {
        return match suffix {
            None => format!("{gutter}{line}"),
            Some(s) => format!("{gutter}{line}{s}"),
        };
    }

    // Same boundary splitting as ANSI mode, but insert markers instead of colors.
    let mut bounds: Vec<usize> = Vec::new();
    bounds.push(line_start);
    bounds.push(line_start + line.len());
    for s in segs {
        bounds.push(s.start);
        bounds.push(s.end);
    }
    bounds.sort_unstable();
    bounds.dedup();

    let mut out = String::new();
    out.push_str(&gutter);

    for w in bounds.windows(2) {
        let a = w[0];
        let b = w[1];
        if a == b {
            continue;
        }
        let rel_a = a - line_start;
        let rel_b = b - line_start;
        if !line.is_char_boundary(rel_a) || !line.is_char_boundary(rel_b) {
            continue;
        }
        let slice = &line[rel_a..rel_b];
        if let Some(marker) = pick_segment_marker(segs, a, b) {
            out.push_str(marker.0);
            out.push_str(slice);
            out.push_str(marker.1);
        } else {
            out.push_str(slice);
        }
    }
    if let Some(s) = suffix {
        out.push_str(s);
    }
    out
}

fn pick_segment_marker(
    segs: &[Segment],
    a: usize,
    b: usize,
) -> Option<(&'static str, &'static str)> {
    let mut best: Option<&Segment> = None;
    for s in segs {
        if s.start <= a && s.end >= b {
            match best {
                None => best = Some(s),
                Some(cur) => {
                    if s.priority < cur.priority
                        || (s.priority == cur.priority && s.span_len < cur.span_len)
                    {
                        best = Some(s);
                    }
                }
            }
        }
    }
    let best = match best {
        Some(b) => b,
        None => return None,
    };
    Some(match best.class {
        SegClass::TrueHit => ("[T+]", "[/T]"),
        SegClass::TrueMiss => ("[T-]", "[/T]"),
        SegClass::FalseHit => ("[F+]", "[/F]"),
        SegClass::FalseMiss => ("[F-]", "[/F]"),
        SegClass::ArmHit => ("[A+]", "[/A]"),
        SegClass::ArmMiss => ("[A-]", "[/A]"),
        SegClass::ZeroToggle => ("[Z0]", "[/Z]"),
    })
}

fn build_function_line_annotations(
    src: &SourceText,
    cov: &CoverageCounters,
) -> std::collections::BTreeMap<u32, String> {
    let mut m: std::collections::BTreeMap<u32, String> = std::collections::BTreeMap::new();
    for (name, meta) in &cov.functions {
        let l = src.line_of_offset(meta.def_span.start);
        let calls = cov.function_calls.get(name).copied().unwrap_or(0);
        m.insert(l, format!(" // calls={}", calls));
    }
    m
}

fn build_zero_toggle_token_spans(
    src: &SourceText,
    cov: &CoverageCounters,
) -> std::collections::BTreeMap<u32, Vec<(usize, usize)>> {
    let mut zero: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for (name, per_bit) in &cov.toggle_counts {
        let sum: u64 = per_bit.iter().copied().sum();
        if sum == 0 {
            zero.insert(name.clone());
        }
    }
    if zero.is_empty() {
        return std::collections::BTreeMap::new();
    }
    let toks = match crate::sv_lexer::lex_all(&src.text) {
        Ok(t) => t,
        Err(_) => return std::collections::BTreeMap::new(),
    };
    let mut out: std::collections::BTreeMap<u32, Vec<(usize, usize)>> =
        std::collections::BTreeMap::new();
    for t in toks {
        if let TokKind::Ident(name) = &t.kind {
            if zero.contains(name) {
                let line = src.line_of_offset(t.start);
                out.entry(line).or_default().push((t.start, t.end));
            }
        }
    }
    out
}
