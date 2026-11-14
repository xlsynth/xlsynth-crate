// SPDX-License-Identifier: Apache-2.0

use xlsynth_pir::ir_parser::Parser;

#[test]
fn block_nodes_reject_repeated_keyword_arguments() {
    let cases: &[(&str, &str, &str, &[&str])] = &[
        (
            "register_write",
            "()",
            "en, register=r, load_enable=en, reset=rst, id=3",
            &["load_enable=rst", "reset=en", "register=r", "id=4"],
        ),
        (
            "instantiation_input",
            "()",
            "en, instantiation=i, port_name=x, id=3",
            &["instantiation=i", "port_name=y", "id=4"],
        ),
        (
            "instantiation_output",
            "bits[1]",
            "instantiation=i, port_name=y, id=3",
            &["instantiation=i", "port_name=x", "id=4"],
        ),
    ];
    for &(operator, ty, args, repeats) in cases {
        let block = |args: &str| {
            format!(
                r#"block b(clk: clock, en: bits[1], rst: bits[1]) {{
  #![reset(port="rst", asynchronous=false, active_low=false)]
  reg r(bits[1], reset_value=0)
  instantiation i(block=child, kind=block)
  en: bits[1] = input_port(name=en, id=1)
  rst: bits[1] = input_port(name=rst, id=2)
  {operator}.3: {ty} = {operator}({args})
}}"#
            )
        };
        Parser::new(&block(args)).parse_block().unwrap();
        for &repeat in repeats {
            let (attribute, _) = repeat.split_once('=').unwrap();
            let with_duplicate =
                args.replace(&format!("{attribute}="), &format!("{repeat}, {attribute}="));
            let error = Parser::new(&block(&with_duplicate))
                .parse_block()
                .expect_err(repeat);
            assert_eq!(
                error.to_string(),
                format!("ParseError: duplicate {operator} {attribute} attribute")
            );
        }
    }
}

#[test]
fn extension_nodes_reject_repeated_keyword_arguments() {
    for (operator, attribute, args, repeated) in [
        (
            "ext_prio_encode",
            "lsb_prio",
            "x, lsb_prio=true, id=2",
            "lsb_prio=false",
        ),
        (
            "ext_nary_add",
            "id",
            "x, signed=[false], negated=[false], id=2",
            "id=3",
        ),
    ] {
        let function = |args: &str| {
            format!(
                "fn f(x: bits[1] id=1) -> bits[1] {{ ret {operator}.2: bits[1] = {operator}({args}) }}"
            )
        };
        Parser::new(&function(args)).parse_fn().unwrap();
        let with_duplicate = args.replace(
            &format!("{attribute}="),
            &format!("{repeated}, {attribute}="),
        );
        let error = Parser::new(&function(&with_duplicate))
            .parse_fn()
            .expect_err(repeated);
        assert_eq!(
            error.to_string(),
            format!("ParseError: duplicate {operator} {attribute} attribute")
        );
    }
}
