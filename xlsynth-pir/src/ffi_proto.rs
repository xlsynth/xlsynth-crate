// SPDX-License-Identifier: Apache-2.0

use std::sync::LazyLock;

use prost_reflect::{DescriptorPool, DynamicMessage, MessageDescriptor, Value, text_format};

use crate::ir::ForeignFunctionData;

static FOREIGN_FUNCTION_DATA: LazyLock<MessageDescriptor> = LazyLock::new(|| {
    DescriptorPool::decode(include_bytes!("../proto/foreign_function_data.bin").as_slice())
        .expect("valid embedded foreign function descriptor")
        .get_message_by_name("xls.ForeignFunctionData")
        .expect("embedded descriptor defines xls.ForeignFunctionData")
});

/// Parses the textproto in an XLS FFI attribute, preserving field presence.
pub(crate) fn parse_foreign_function_data(
    text: &str,
) -> Result<ForeignFunctionData, text_format::ParseError> {
    let mut proto = DynamicMessage::parse_text_format(FOREIGN_FUNCTION_DATA.clone(), text)?;
    let code_template = proto.take_field_by_name("code_template").map(|value| {
        let Value::String(template) = value else {
            unreachable!("the embedded descriptor declares code_template as a string");
        };
        template
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
