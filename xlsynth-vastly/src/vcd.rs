// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::collections::BTreeSet;

use crate::Error;
use crate::Result;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VcdVar {
    pub id: String,
    pub width: u32,
    pub name: String, // fully-qualified (scope1.scope2.var)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Vcd {
    pub timescale: Option<String>,
    pub vars: Vec<VcdVar>,
    /// Events: time -> (var_name -> value_string_msb)
    pub events: BTreeMap<u64, BTreeMap<String, String>>,
}

impl Vcd {
    pub fn parse(text: &str) -> Result<Vcd> {
        let mut timescale: Option<String> = None;
        let mut in_timescale_block = false;
        let mut timescale_buf: String = String::new();
        let mut scope: Vec<String> = Vec::new();
        let mut id_to_targets: BTreeMap<String, Vec<IdTarget>> = BTreeMap::new();
        let mut vars: Vec<VcdVar> = Vec::new();

        let mut cur_time: u64 = 0;
        let mut events: BTreeMap<u64, BTreeMap<String, String>> = BTreeMap::new();

        let mut in_definitions = true;
        for raw_line in text.lines() {
            let line = raw_line.trim();
            if line.is_empty() {
                continue;
            }
            if in_definitions {
                if in_timescale_block {
                    if line.starts_with("$end") {
                        timescale = Some(timescale_buf.trim().to_string());
                        timescale_buf.clear();
                        in_timescale_block = false;
                    } else {
                        if !timescale_buf.is_empty() {
                            timescale_buf.push(' ');
                        }
                        timescale_buf.push_str(line);
                    }
                    continue;
                }
                if line.starts_with("$timescale") {
                    // Accept both:
                    // - "$timescale 1ns $end"
                    // - "$timescale" <newline> "1ns" <newline> "$end"
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 3 && parts.last() == Some(&"$end") {
                        // $timescale <val> $end (maybe extra whitespace)
                        timescale = Some(parts[1..parts.len() - 1].join(" "));
                    } else if line == "$timescale" {
                        in_timescale_block = true;
                        timescale_buf.clear();
                    } else {
                        // "$timescale" with tokens but no $end (rare); capture remainder and enter
                        // block.
                        let rest = line.trim_start_matches("$timescale").trim();
                        in_timescale_block = true;
                        timescale_buf = rest.to_string();
                    }
                    continue;
                }
                if line.starts_with("$scope") {
                    // $scope module <name> $end
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 3 {
                        scope.push(parts[2].to_string());
                    }
                    continue;
                }
                if line.starts_with("$upscope") {
                    scope.pop();
                    continue;
                }
                if line.starts_with("$var") {
                    // $var <type> <width> <id> <ref> $end
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() < 5 {
                        return Err(Error::Parse(format!("bad $var line: {line}")));
                    }
                    let width = parts[2]
                        .parse::<u32>()
                        .map_err(|_| Error::Parse(format!("bad $var width: {line}")))?;
                    let id = parts[3].to_string();
                    let ref_name = parts[4].to_string();
                    let full_name = if scope.is_empty() {
                        ref_name
                    } else {
                        format!("{}.{}", scope.join("."), ref_name)
                    };
                    let v = VcdVar {
                        id: id.clone(),
                        width,
                        name: full_name,
                    };
                    id_to_targets.entry(id.clone()).or_default().push(IdTarget {
                        name: v.name.clone(),
                        width: v.width,
                    });
                    vars.push(v);
                    continue;
                }
                if line.starts_with("$enddefinitions") {
                    in_definitions = false;
                    continue;
                }
                continue;
            }

            // Value change section.
            if let Some(t) = line.strip_prefix('#') {
                cur_time = t
                    .parse::<u64>()
                    .map_err(|_| Error::Parse(format!("bad timestamp: {line}")))?;
                continue;
            }

            // Vector: b1010 <id>
            if let Some(rest) = line.strip_prefix('b') {
                let mut it = rest.split_whitespace();
                let bits = it
                    .next()
                    .ok_or_else(|| Error::Parse(format!("bad vector change: {line}")))?;
                let id = it
                    .next()
                    .ok_or_else(|| Error::Parse(format!("bad vector change: {line}")))?;
                let v = normalize_bits(bits);
                let targets = id_to_targets
                    .get(id)
                    .ok_or_else(|| Error::Parse(format!("unknown id {id} in change")))?;
                let e = events.entry(cur_time).or_default();
                for t in targets {
                    e.insert(t.name.clone(), pad_to_width(&v, t.width));
                }
                continue;
            }

            // Scalar: 0<id>, 1<id>, x<id>, z<id>
            let mut chars = line.chars();
            let vch = chars.next().unwrap();
            if matches!(vch, '0' | '1' | 'x' | 'X' | 'z' | 'Z') {
                let id: String = chars.collect();
                let targets = id_to_targets
                    .get(&id)
                    .ok_or_else(|| Error::Parse(format!("unknown id {id} in change")))?;
                let v = normalize_bits(&vch.to_string());
                let e = events.entry(cur_time).or_default();
                for t in targets {
                    e.insert(t.name.clone(), pad_to_width(&v, t.width));
                }
                continue;
            }
        }

        Ok(Vcd {
            timescale: timescale.map(normalize_timescale),
            vars,
            events,
        })
    }

    pub fn var_names(&self) -> BTreeSet<String> {
        self.vars.iter().map(|v| v.name.clone()).collect()
    }

    /// Returns the set of timestamps present in the file (sorted).
    pub fn times(&self) -> Vec<u64> {
        self.events.keys().copied().collect()
    }

    /// Computes a value timeline: for each time in `times`, returns full
    /// var->value map after applying changes at that time.
    pub fn materialize(&self) -> Result<Vec<(u64, BTreeMap<String, String>)>> {
        let mut cur: BTreeMap<String, String> = BTreeMap::new();
        // Initialize unknowns for all vars.
        for v in &self.vars {
            if v.width <= 1 {
                cur.insert(v.name.clone(), "x".to_string());
            } else {
                cur.insert(v.name.clone(), "x".repeat(v.width as usize));
            }
        }

        let mut out: Vec<(u64, BTreeMap<String, String>)> = Vec::new();
        for (t, changes) in &self.events {
            for (name, val) in changes {
                cur.insert(name.clone(), val.clone());
            }
            out.push((*t, cur.clone()));
        }
        Ok(out)
    }
}

fn normalize_bits(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '0' => '0',
            '1' => '1',
            'x' | 'X' => 'x',
            'z' | 'Z' => 'z',
            _ => 'x',
        })
        .collect()
}

fn normalize_timescale(s: String) -> String {
    // Keep minimal normalization: strip any surrounding tokens/whitespace, collapse
    // spaces. Common forms: "1ns", "1 ns", "10ps".
    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.len() == 2 {
        format!("{}{}", parts[0], parts[1])
    } else {
        parts.join("")
    }
}

#[derive(Debug, Clone)]
struct IdTarget {
    name: String,
    width: u32,
}

fn pad_to_width(bits_msb: &str, width: u32) -> String {
    if width <= 1 {
        return bits_msb.chars().next().unwrap_or('x').to_string();
    }
    let mut s = bits_msb.to_string();
    if s.len() > width as usize {
        // Truncate MSBs if somehow longer.
        s = s[s.len() - width as usize..].to_string();
    }
    if s.len() < width as usize {
        let pad_ch = match s.chars().next().unwrap_or('x') {
            // Preserve compact unknown/high-impedance vector encodings like
            // `bx` / `bz` from simulators (e.g. Icarus): these represent all
            // unknown/high-Z high bits, not zero-extension.
            'x' | 'z' => s.chars().next().unwrap(),
            _ => '0',
        };
        let pad = pad_ch.to_string().repeat(width as usize - s.len());
        s = format!("{pad}{s}");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::Vcd;

    #[test]
    fn compact_unknown_vector_changes_preserve_unknown_high_bits() {
        let text = r#"$date
    today
$end
$version
    case-study
$end
$timescale 1s $end
$scope module t $end
$var reg 4 ! a [3:0] $end
$upscope $end
$enddefinitions $end
#0
bx !
#1
bz !
#2
b1010 !
"#;
        let vcd = Vcd::parse(text).expect("parse vcd");
        let materialized = vcd.materialize().expect("materialize vcd");
        let name = vcd.vars[0].name.clone();
        let at0 = &materialized[0].1;
        let at1 = &materialized[1].1;
        let at2 = &materialized[2].1;
        assert_eq!(at0.get(&name).unwrap(), "xxxx");
        assert_eq!(at1.get(&name).unwrap(), "zzzz");
        assert_eq!(at2.get(&name).unwrap(), "1010");
    }
}
