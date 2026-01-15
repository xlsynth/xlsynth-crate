// SPDX-License-Identifier: Apache-2.0

use crate::gate_builder::GateBuilder;

/// Selects the prefix network to use for prefix scans.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefixScanStrategy {
    Linear,
    BrentKung,
    KoggeStone,
}

impl Default for PrefixScanStrategy {
    fn default() -> Self {
        PrefixScanStrategy::Linear
    }
}

impl std::fmt::Display for PrefixScanStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PrefixScanStrategy::Linear => write!(f, "linear"),
            PrefixScanStrategy::BrentKung => write!(f, "brent-kung"),
            PrefixScanStrategy::KoggeStone => write!(f, "kogge-stone"),
        }
    }
}

pub fn prefix_scan_exclusive<T: Clone>(inclusive: &[T], identity: T) -> Vec<T> {
    if inclusive.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(inclusive.len());
    out.push(identity);
    for i in 1..inclusive.len() {
        out.push(inclusive[i - 1].clone());
    }
    out
}

pub fn prefix_scan_inclusive<T, F>(
    gb: &mut GateBuilder,
    inputs: &[T],
    strategy: PrefixScanStrategy,
    identity: T,
    apply_op: F,
) -> Vec<T>
where
    T: Clone,
    F: Fn(&mut GateBuilder, T, T) -> T + Copy,
{
    match strategy {
        PrefixScanStrategy::Linear => prefix_scan_inclusive_linear(gb, inputs, apply_op),
        PrefixScanStrategy::BrentKung => {
            prefix_scan_inclusive_brent_kung(gb, inputs, identity, apply_op)
        }
        PrefixScanStrategy::KoggeStone => prefix_scan_inclusive_kogge_stone(gb, inputs, apply_op),
    }
}

fn prefix_scan_inclusive_linear<T, F>(gb: &mut GateBuilder, inputs: &[T], apply_op: F) -> Vec<T>
where
    T: Clone,
    F: Fn(&mut GateBuilder, T, T) -> T + Copy,
{
    if inputs.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(inputs.len());
    let mut acc = inputs[0].clone();
    out.push(acc.clone());
    for input in inputs.iter().skip(1) {
        acc = apply_op(gb, acc, input.clone());
        out.push(acc.clone());
    }
    out
}

fn prefix_scan_inclusive_kogge_stone<T, F>(
    gb: &mut GateBuilder,
    inputs: &[T],
    apply_op: F,
) -> Vec<T>
where
    T: Clone,
    F: Fn(&mut GateBuilder, T, T) -> T + Copy,
{
    let n = inputs.len();
    if n == 0 {
        return Vec::new();
    }
    let mut y: Vec<T> = inputs.to_vec();
    let mut step = 1usize;
    while step < n {
        let mut y2 = y.clone();
        for i in step..n {
            y2[i] = apply_op(gb, y[i - step].clone(), y[i].clone());
        }
        y = y2;
        step <<= 1;
    }
    y
}

fn prefix_scan_inclusive_brent_kung<T, F>(
    gb: &mut GateBuilder,
    inputs: &[T],
    identity: T,
    apply_op: F,
) -> Vec<T>
where
    T: Clone,
    F: Fn(&mut GateBuilder, T, T) -> T + Copy,
{
    let n = inputs.len();
    if n == 0 {
        return Vec::new();
    }

    let m = next_pow2(n);
    let mut y: Vec<T> = Vec::with_capacity(m);
    for i in 0..m {
        if i < n {
            y.push(inputs[i].clone());
        } else {
            y.push(identity.clone());
        }
    }

    let mut levels = 0usize;
    let mut size = m;
    while size > 1 {
        levels += 1;
        size >>= 1;
    }

    for d in 0..levels {
        let step = 1usize << (d + 1);
        let half = step >> 1;
        let mut i = step - 1;
        while i < m {
            y[i] = apply_op(gb, y[i - half].clone(), y[i].clone());
            i += step;
        }
    }

    if levels >= 2 {
        for d in (0..=levels - 2).rev() {
            let step = 1usize << (d + 1);
            let half = step >> 1;
            let mut i = step + half - 1;
            while i < m {
                y[i] = apply_op(gb, y[i - half].clone(), y[i].clone());
                i += step;
            }
        }
    }

    y.truncate(n);
    y
}

fn next_pow2(n: usize) -> usize {
    let mut p = 1usize;
    while p < n {
        p <<= 1;
    }
    p
}
