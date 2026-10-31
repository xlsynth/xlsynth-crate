// SPDX-License-Identifier: Apache-2.0

use std::fmt::Write;
use std::sync::LazyLock;

use prost_reflect::{DescriptorPool, DynamicMessage, MessageDescriptor, Value, text_format};

use crate::ir::ForeignFunctionData;

static FOREIGN_FUNCTION_DATA: LazyLock<MessageDescriptor> = LazyLock::new(|| {
    DescriptorPool::decode(include_bytes!("../proto/foreign_function_data.bin").as_slice())
        .expect("valid embedded foreign function descriptor")
        .get_message_by_name("xlsynth.pir.ForeignFunctionDataBytes")
        .expect("embedded descriptor defines xlsynth.pir.ForeignFunctionDataBytes")
});

/// Parses the textproto in an XLS FFI attribute, preserving field presence.
pub(crate) fn parse_foreign_function_data(
    text: &str,
) -> Result<ForeignFunctionData, text_format::ParseError> {
    let parse =
        |source: &str| DynamicMessage::parse_text_format(FOREIGN_FUNCTION_DATA.clone(), source);
    let mut proto = parse(text).or_else(|error| match normalize_xls_escapes(text) {
        Some(normalized) => parse(&normalized),
        None => Err(error),
    })?;
    let code_template = proto.take_field_by_name("code_template").map(|value| {
        let Value::Bytes(template) = value else {
            unreachable!("the embedded descriptor declares code_template as bytes");
        };
        template.into()
    });
    let delay_ps = proto.take_field_by_name("delay_ps").map(|value| {
        value
            .as_i32()
            .expect("the embedded descriptor declares delay_ps as int32")
    });
    Ok(ForeignFunctionData {
        code_template,
        delay_ps,
    })
}

/// Rewrites exceptional C++ TextFormat escapes in a temporary parser input.
/// Every replacement remains an escape, so it cannot introduce a delimiter,
/// end a comment, or make a stray backslash outside a string into a valid
/// token.
fn normalize_xls_escapes(mut rest: &str) -> Option<String> {
    let mut output = String::new();
    let mut changed = false;
    while let Some(index) = rest.find('\\') {
        output.push_str(&rest[..index]);
        let escape = &rest[index..];
        let consumed = if escape.starts_with(r"\\") {
            output.push_str(r"\\");
            2
        } else if let Some(len) = append_xls_escape(escape, &mut output) {
            changed = true;
            len
        } else {
            output.push('\\');
            1
        };
        rest = &escape[consumed..];
    }
    if changed {
        output.push_str(rest);
        Some(output)
    } else {
        None
    }
}

/// Encodes the escapes where protobuf's C++ io::Tokenizer differs from
/// prost-reflect. Returns the original escape length when a rewrite applies.
fn append_xls_escape(escape: &str, output: &mut String) -> Option<usize> {
    if let [b'\\', high @ b'4'..=b'7', b'0'..=b'7', b'0'..=b'7', ..] = escape.as_bytes() {
        // XLS takes the low byte of three octal digits.
        output.push('\\');
        output.push(char::from(high - 4));
        output.push_str(&escape[2..4]);
        return Some(4);
    }
    let len = match escape.as_bytes().get(1)? {
        b'u' => 6,
        b'U' => 10,
        _ => return None,
    };
    let point = hex_digits(escape.get(2..len)?)?;
    match point {
        0xd800..=0xdfff => {
            // A trail surrogate must use the short \u spelling.
            if point < 0xdc00
                && let Some(trail) = escape[len..]
                    .strip_prefix(r"\u")
                    .and_then(|rest| hex_digits(rest.get(..4)?))
                    .filter(|value| (0xdc00..=0xdfff).contains(value))
            {
                let scalar = 0x10000 + ((point - 0xd800) << 10) + (trail - 0xdc00);
                write!(output, r"\U{scalar:08x}").unwrap();
                return Some(len + 6);
            }
            // XLS encodes a lone surrogate with the ordinary three-byte UTF-8
            // bit layout even though the resulting sequence is not valid UTF-8.
            let bytes = [
                0xe0 | (point >> 12),
                0x80 | ((point >> 6) & 0x3f),
                0x80 | (point & 0x3f),
            ];
            for byte in bytes {
                write!(output, r"\x{byte:02x}").unwrap();
            }
        }
        // ConsumeString accepts eight hex digits up to 1fffff. AppendUTF8
        // renders the out-of-range values as literal, lowercase escape text.
        0x110000..=0x1fffff => write!(output, r"\\U{point:08x}").unwrap(),
        _ => return None,
    }
    Some(len)
}

/// Reads exactly the supplied ASCII hex digits, without accepting a sign.
fn hex_digits(text: &str) -> Option<u32> {
    text.bytes().try_fold(0, |value, byte| {
        Some((value << 4) | char::from(byte).to_digit(16)?)
    })
}
