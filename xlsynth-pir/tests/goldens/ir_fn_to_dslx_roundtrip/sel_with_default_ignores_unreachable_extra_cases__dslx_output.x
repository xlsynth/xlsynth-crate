fn f(s: uN[1], a: uN[8], b: uN[8]) -> uN[8] {
  let c: uN[8] = uN[8]:7;
  let d: uN[8] = uN[8]:9;
  let out_v: uN[8] = if s { b } else { a };
  out_v
}
