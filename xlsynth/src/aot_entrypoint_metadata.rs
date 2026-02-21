// SPDX-License-Identifier: Apache-2.0

use std::convert::TryFrom;

use prost::Message;

use crate::xlsynth_error::XlsynthError;

#[derive(Debug, Clone)]
pub struct AotEntrypointMetadata {
    pub symbol: String,
    pub input_buffer_sizes: Vec<usize>,
    pub input_buffer_alignments: Vec<usize>,
    pub output_buffer_sizes: Vec<usize>,
    pub output_buffer_alignments: Vec<usize>,
    pub temp_buffer_size: usize,
    pub temp_buffer_alignment: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AotType {
    Bits { bit_count: usize },
    Tuple { elements: Vec<AotType> },
    Array { size: usize, element: Box<AotType> },
    Token,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AotFunctionParameter {
    pub name: String,
    pub ty: AotType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AotFunctionSignature {
    pub function_name: String,
    pub params: Vec<AotFunctionParameter>,
    pub return_type: AotType,
    pub input_layouts: Vec<AotTypeLayout>,
    pub output_layouts: Vec<AotTypeLayout>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AotElementLayout {
    pub offset: usize,
    pub data_size: usize,
    pub padded_size: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AotTypeLayout {
    pub size: usize,
    pub elements: Vec<AotElementLayout>,
}

#[derive(Clone, PartialEq, Message)]
struct AotPackageEntrypointsProto {
    #[prost(message, repeated, tag = "1")]
    entrypoint: Vec<AotEntrypointProto>,
}

#[derive(Clone, PartialEq, Message)]
struct AotEntrypointProto {
    #[prost(string, optional, tag = "3")]
    function_symbol: Option<String>,

    #[prost(int64, repeated, tag = "4")]
    input_buffer_sizes: Vec<i64>,

    #[prost(int64, repeated, tag = "5")]
    input_buffer_alignments: Vec<i64>,

    #[prost(int64, repeated, tag = "7")]
    output_buffer_sizes: Vec<i64>,

    #[prost(int64, repeated, tag = "8")]
    output_buffer_alignments: Vec<i64>,

    #[prost(int64, optional, tag = "13")]
    temp_buffer_size: Option<i64>,

    #[prost(int64, optional, tag = "14")]
    temp_buffer_alignment: Option<i64>,

    #[prost(message, optional, tag = "18")]
    inputs_layout: Option<TypeLayoutsProto>,

    #[prost(message, optional, tag = "19")]
    outputs_layout: Option<TypeLayoutsProto>,

    #[prost(message, optional, tag = "23")]
    function_metadata: Option<FunctionMetadataProto>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, prost::Enumeration)]
enum TypeEnumProto {
    Invalid = 0,
    Bits = 1,
    Tuple = 2,
    Array = 3,
    Token = 4,
}

#[derive(Clone, PartialEq, Message)]
struct TypeProto {
    #[prost(enumeration = "TypeEnumProto", optional, tag = "1")]
    type_enum: Option<i32>,

    #[prost(int64, optional, tag = "2")]
    bit_count: Option<i64>,

    #[prost(message, repeated, tag = "3")]
    tuple_elements: Vec<TypeProto>,

    #[prost(int64, optional, tag = "4")]
    array_size: Option<i64>,

    #[prost(message, optional, boxed, tag = "5")]
    array_element: Option<Box<TypeProto>>,
}

#[derive(Clone, PartialEq, Message)]
struct ElementLayoutProto {
    #[prost(int64, optional, tag = "1")]
    offset: Option<i64>,

    #[prost(int64, optional, tag = "2")]
    data_size: Option<i64>,

    #[prost(int64, optional, tag = "3")]
    padded_size: Option<i64>,
}

#[derive(Clone, PartialEq, Message)]
struct TypeLayoutProto {
    #[prost(string, optional, tag = "1")]
    r#type: Option<String>,

    #[prost(int64, optional, tag = "2")]
    size: Option<i64>,

    #[prost(message, repeated, tag = "3")]
    elements: Vec<ElementLayoutProto>,
}

#[derive(Clone, PartialEq, Message)]
struct TypeLayoutsProto {
    #[prost(message, repeated, tag = "1")]
    layouts: Vec<TypeLayoutProto>,
}

#[derive(Clone, PartialEq, Message)]
struct PackageInterfaceFunctionBaseProto {
    #[prost(bool, optional, tag = "1")]
    top: Option<bool>,

    #[prost(string, optional, tag = "2")]
    name: Option<String>,
}

#[derive(Clone, PartialEq, Message)]
struct PackageInterfaceNamedValueProto {
    #[prost(string, optional, tag = "1")]
    name: Option<String>,

    #[prost(message, optional, tag = "2")]
    r#type: Option<TypeProto>,

    #[prost(string, optional, tag = "3")]
    sv_type: Option<String>,
}

#[derive(Clone, PartialEq, Message)]
struct PackageInterfaceFunctionProto {
    #[prost(message, optional, tag = "1")]
    base: Option<PackageInterfaceFunctionBaseProto>,

    #[prost(message, repeated, tag = "2")]
    parameters: Vec<PackageInterfaceNamedValueProto>,

    #[prost(message, optional, tag = "3")]
    result_type: Option<TypeProto>,

    #[prost(string, optional, tag = "4")]
    sv_result_type: Option<String>,
}

#[derive(Clone, PartialEq, Message)]
struct FunctionMetadataProto {
    #[prost(message, optional, tag = "1")]
    function_interface: Option<PackageInterfaceFunctionProto>,
}

pub fn get_entrypoint_metadata(
    entrypoints_proto: &[u8],
) -> Result<AotEntrypointMetadata, XlsynthError> {
    let entrypoint = decode_single_entrypoint(entrypoints_proto)?;
    parse_entrypoint_metadata(&entrypoint)
}

pub fn get_entrypoint_function_signature(
    entrypoints_proto: &[u8],
) -> Result<AotFunctionSignature, XlsynthError> {
    let entrypoint = decode_single_entrypoint(entrypoints_proto)?;
    parse_entrypoint_function_signature(&entrypoint)
}

fn decode_single_entrypoint(entrypoints_proto: &[u8]) -> Result<AotEntrypointProto, XlsynthError> {
    let decoded = AotPackageEntrypointsProto::decode(entrypoints_proto)
        .map_err(|e| XlsynthError(format!("Failed decoding AOT entrypoints proto: {e}")))?;
    if decoded.entrypoint.len() != 1 {
        return Err(XlsynthError(format!(
            "Expected exactly 1 AOT entrypoint; got {}",
            decoded.entrypoint.len()
        )));
    }
    Ok(decoded
        .entrypoint
        .into_iter()
        .next()
        .expect("checked single entrypoint length"))
}

fn parse_entrypoint_metadata(
    entrypoint: &AotEntrypointProto,
) -> Result<AotEntrypointMetadata, XlsynthError> {
    let symbol = entrypoint.function_symbol.clone().ok_or_else(|| {
        XlsynthError("Entrypoint metadata has no unpacked function symbol".to_string())
    })?;

    let input_buffer_sizes = parse_sizes(&entrypoint.input_buffer_sizes, "input_buffer_sizes")?;
    let raw_input_buffer_alignments = parse_sizes(
        &entrypoint.input_buffer_alignments,
        "input_buffer_alignments",
    )?;
    let output_buffer_sizes = parse_sizes(&entrypoint.output_buffer_sizes, "output_buffer_sizes")?;
    let raw_output_buffer_alignments = parse_sizes(
        &entrypoint.output_buffer_alignments,
        "output_buffer_alignments",
    )?;
    let input_buffer_alignments =
        validate_alignments(&input_buffer_sizes, raw_input_buffer_alignments, "input")?;
    let output_buffer_alignments =
        validate_alignments(&output_buffer_sizes, raw_output_buffer_alignments, "output")?;

    Ok(AotEntrypointMetadata {
        symbol,
        input_buffer_sizes,
        input_buffer_alignments,
        output_buffer_sizes,
        output_buffer_alignments,
        temp_buffer_size: parse_optional_size(entrypoint.temp_buffer_size, "temp_buffer_size")?
            .unwrap_or(0),
        temp_buffer_alignment: parse_optional_size(
            entrypoint.temp_buffer_alignment,
            "temp_buffer_alignment",
        )?
        .unwrap_or(1)
        .max(1),
    })
}

fn parse_entrypoint_function_signature(
    entrypoint: &AotEntrypointProto,
) -> Result<AotFunctionSignature, XlsynthError> {
    let function_metadata = entrypoint
        .function_metadata
        .as_ref()
        .ok_or_else(|| XlsynthError("Entrypoint metadata missing function_metadata".to_string()))?;
    let function_interface = function_metadata
        .function_interface
        .as_ref()
        .ok_or_else(|| {
            XlsynthError("Entrypoint metadata missing function_interface".to_string())
        })?;

    let function_name = function_interface
        .base
        .as_ref()
        .and_then(|base| base.name.clone())
        .or_else(|| entrypoint.function_symbol.clone())
        .unwrap_or_else(|| "run".to_string());

    let mut params = Vec::with_capacity(function_interface.parameters.len());
    for (index, param) in function_interface.parameters.iter().enumerate() {
        let name = param.name.clone().unwrap_or_else(|| format!("arg{index}"));
        let type_proto = param
            .r#type
            .as_ref()
            .ok_or_else(|| XlsynthError(format!("Parameter {index} missing type information")))?;
        params.push(AotFunctionParameter {
            name,
            ty: parse_type_proto(type_proto)?,
        });
    }

    let return_type_proto = function_interface
        .result_type
        .as_ref()
        .ok_or_else(|| XlsynthError("Function interface missing return type".to_string()))?;
    let return_type = parse_type_proto(return_type_proto)?;

    let input_layouts = match entrypoint.inputs_layout.as_ref() {
        Some(layouts) => parse_type_layouts(layouts)?,
        None if params.is_empty() => Vec::new(),
        None => {
            return Err(XlsynthError(
                "Entrypoint metadata missing inputs_layout".to_string(),
            ));
        }
    };
    if input_layouts.len() != params.len() {
        return Err(XlsynthError(format!(
            "inputs_layout count mismatch: params={} layouts={}",
            params.len(),
            input_layouts.len()
        )));
    }

    let output_layouts = match entrypoint.outputs_layout.as_ref() {
        Some(layouts) => parse_type_layouts(layouts)?,
        None if is_empty_tuple(&return_type) => vec![AotTypeLayout {
            size: 0,
            elements: Vec::new(),
        }],
        None => {
            return Err(XlsynthError(
                "Entrypoint metadata missing outputs_layout".to_string(),
            ));
        }
    };
    if output_layouts.len() != 1 {
        return Err(XlsynthError(format!(
            "Expected exactly 1 output layout; got {}",
            output_layouts.len()
        )));
    }

    Ok(AotFunctionSignature {
        function_name,
        params,
        return_type,
        input_layouts,
        output_layouts,
    })
}

fn parse_type_proto(type_proto: &TypeProto) -> Result<AotType, XlsynthError> {
    let Some(type_enum_raw) = type_proto.type_enum else {
        return Err(XlsynthError("Type proto missing type_enum".to_string()));
    };
    let type_enum = TypeEnumProto::try_from(type_enum_raw).map_err(|_| {
        XlsynthError(format!(
            "Type proto had unknown type_enum value: {type_enum_raw}"
        ))
    })?;

    match type_enum {
        TypeEnumProto::Invalid => Err(XlsynthError("Type proto had INVALID type_enum".to_string())),
        TypeEnumProto::Bits => {
            let bit_count = type_proto
                .bit_count
                .ok_or_else(|| XlsynthError("BITS type missing bit_count".to_string()))?;
            if bit_count < 0 {
                return Err(XlsynthError(format!(
                    "BITS type had negative bit_count: {bit_count}"
                )));
            }
            let bit_count = usize::try_from(bit_count).map_err(|_| {
                XlsynthError(format!(
                    "BITS type bit_count {bit_count} does not fit in usize"
                ))
            })?;
            Ok(AotType::Bits { bit_count })
        }
        TypeEnumProto::Tuple => {
            let mut elements = Vec::with_capacity(type_proto.tuple_elements.len());
            for element in &type_proto.tuple_elements {
                elements.push(parse_type_proto(element)?);
            }
            Ok(AotType::Tuple { elements })
        }
        TypeEnumProto::Array => {
            let array_size = type_proto
                .array_size
                .ok_or_else(|| XlsynthError("ARRAY type missing array_size".to_string()))?;
            if array_size < 0 {
                return Err(XlsynthError(format!(
                    "ARRAY type had negative array_size: {array_size}"
                )));
            }
            let size = usize::try_from(array_size).map_err(|_| {
                XlsynthError(format!(
                    "ARRAY type array_size {array_size} does not fit in usize"
                ))
            })?;
            let element = type_proto
                .array_element
                .as_ref()
                .ok_or_else(|| XlsynthError("ARRAY type missing array_element".to_string()))?;
            Ok(AotType::Array {
                size,
                element: Box::new(parse_type_proto(element)?),
            })
        }
        TypeEnumProto::Token => Ok(AotType::Token),
    }
}

fn parse_type_layouts(
    layouts_proto: &TypeLayoutsProto,
) -> Result<Vec<AotTypeLayout>, XlsynthError> {
    let mut out = Vec::with_capacity(layouts_proto.layouts.len());
    for layout in &layouts_proto.layouts {
        out.push(parse_type_layout(layout)?);
    }
    Ok(out)
}

fn is_empty_tuple(ty: &AotType) -> bool {
    matches!(ty, AotType::Tuple { elements } if elements.is_empty())
}

fn parse_type_layout(layout_proto: &TypeLayoutProto) -> Result<AotTypeLayout, XlsynthError> {
    let size = parse_optional_size(layout_proto.size, "type_layout.size")?
        .ok_or_else(|| XlsynthError("Type layout missing size".to_string()))?;
    let mut elements = Vec::with_capacity(layout_proto.elements.len());
    for element in &layout_proto.elements {
        elements.push(parse_element_layout(element)?);
    }
    Ok(AotTypeLayout { size, elements })
}

fn parse_element_layout(
    element_proto: &ElementLayoutProto,
) -> Result<AotElementLayout, XlsynthError> {
    let offset = parse_optional_size(element_proto.offset, "element_layout.offset")?
        .ok_or_else(|| XlsynthError("Element layout missing offset".to_string()))?;
    let data_size = parse_optional_size(element_proto.data_size, "element_layout.data_size")?
        .ok_or_else(|| XlsynthError("Element layout missing data_size".to_string()))?;
    let padded_size = parse_optional_size(element_proto.padded_size, "element_layout.padded_size")?
        .ok_or_else(|| XlsynthError("Element layout missing padded_size".to_string()))?;
    if padded_size < data_size {
        return Err(XlsynthError(format!(
            "Element layout padded_size must be >= data_size; got padded_size={padded_size} data_size={data_size}"
        )));
    }
    Ok(AotElementLayout {
        offset,
        data_size,
        padded_size,
    })
}

fn parse_sizes(values: &[i64], field_name: &str) -> Result<Vec<usize>, XlsynthError> {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        out.push(parse_size(*value, field_name)?);
    }
    Ok(out)
}

fn parse_optional_size(
    value: Option<i64>,
    field_name: &str,
) -> Result<Option<usize>, XlsynthError> {
    match value {
        Some(v) => Ok(Some(parse_size(v, field_name)?)),
        None => Ok(None),
    }
}

fn parse_size(value: i64, field_name: &str) -> Result<usize, XlsynthError> {
    if value < 0 {
        return Err(XlsynthError(format!(
            "Field '{field_name}' had negative value: {value}"
        )));
    }
    usize::try_from(value).map_err(|_| {
        XlsynthError(format!(
            "Field '{field_name}' value {value} does not fit in usize on this platform"
        ))
    })
}

fn validate_alignments(
    sizes: &[usize],
    alignments: Vec<usize>,
    kind: &str,
) -> Result<Vec<usize>, XlsynthError> {
    if sizes.len() != alignments.len() {
        return Err(XlsynthError(format!(
            "{kind} alignment count mismatch: sizes={} alignments={}",
            sizes.len(),
            alignments.len()
        )));
    }

    for (index, alignment) in alignments.iter().enumerate() {
        if *alignment < 1 {
            return Err(XlsynthError(format!(
                "{kind} alignment at index {index} must be >= 1, got {alignment}"
            )));
        }
    }
    Ok(alignments)
}

#[cfg(test)]
mod tests {
    use prost::Message;

    use super::*;

    #[test]
    fn get_entrypoint_metadata_rejects_empty_proto() {
        let encoded = AotPackageEntrypointsProto {
            entrypoint: Vec::new(),
        }
        .encode_to_vec();

        let err = get_entrypoint_metadata(&encoded).unwrap_err();
        assert!(
            err.to_string()
                .contains("Expected exactly 1 AOT entrypoint"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn get_entrypoint_metadata_parses_simple_entrypoint() {
        let encoded = AotPackageEntrypointsProto {
            entrypoint: vec![AotEntrypointProto {
                function_symbol: Some("foo".to_string()),
                input_buffer_sizes: vec![1, 2],
                input_buffer_alignments: vec![1, 8],
                output_buffer_sizes: vec![4],
                output_buffer_alignments: vec![8],
                temp_buffer_size: Some(16),
                temp_buffer_alignment: Some(8),
                inputs_layout: None,
                outputs_layout: None,
                function_metadata: None,
            }],
        }
        .encode_to_vec();

        let metadata = get_entrypoint_metadata(&encoded).unwrap();
        assert_eq!(metadata.symbol, "foo");
        assert_eq!(metadata.input_buffer_sizes, vec![1, 2]);
        assert_eq!(metadata.input_buffer_alignments, vec![1, 8]);
        assert_eq!(metadata.output_buffer_sizes, vec![4]);
        assert_eq!(metadata.output_buffer_alignments, vec![8]);
        assert_eq!(metadata.temp_buffer_size, 16);
        assert_eq!(metadata.temp_buffer_alignment, 8);
    }

    #[test]
    fn get_entrypoint_metadata_rejects_zero_alignment() {
        let encoded = AotPackageEntrypointsProto {
            entrypoint: vec![AotEntrypointProto {
                function_symbol: Some("foo".to_string()),
                input_buffer_sizes: vec![1],
                input_buffer_alignments: vec![1],
                output_buffer_sizes: vec![1],
                output_buffer_alignments: vec![0],
                temp_buffer_size: Some(0),
                temp_buffer_alignment: Some(1),
                inputs_layout: None,
                outputs_layout: None,
                function_metadata: None,
            }],
        }
        .encode_to_vec();

        let err = get_entrypoint_metadata(&encoded).unwrap_err();
        assert!(
            err.to_string().contains("must be >= 1"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn get_entrypoint_metadata_rejects_alignment_count_mismatch() {
        let encoded = AotPackageEntrypointsProto {
            entrypoint: vec![AotEntrypointProto {
                function_symbol: Some("foo".to_string()),
                input_buffer_sizes: vec![1, 2],
                input_buffer_alignments: vec![8],
                output_buffer_sizes: vec![1],
                output_buffer_alignments: vec![1],
                temp_buffer_size: Some(0),
                temp_buffer_alignment: Some(1),
                inputs_layout: None,
                outputs_layout: None,
                function_metadata: None,
            }],
        }
        .encode_to_vec();

        let err = get_entrypoint_metadata(&encoded).unwrap_err();
        assert!(
            err.to_string().contains("alignment count mismatch"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn get_entrypoint_metadata_rejects_multiple_entrypoints() {
        let encoded = AotPackageEntrypointsProto {
            entrypoint: vec![
                AotEntrypointProto {
                    function_symbol: Some("foo".to_string()),
                    input_buffer_sizes: vec![1],
                    input_buffer_alignments: vec![1],
                    output_buffer_sizes: vec![1],
                    output_buffer_alignments: vec![1],
                    temp_buffer_size: Some(0),
                    temp_buffer_alignment: Some(1),
                    inputs_layout: None,
                    outputs_layout: None,
                    function_metadata: None,
                },
                AotEntrypointProto {
                    function_symbol: Some("bar".to_string()),
                    input_buffer_sizes: vec![1],
                    input_buffer_alignments: vec![1],
                    output_buffer_sizes: vec![1],
                    output_buffer_alignments: vec![1],
                    temp_buffer_size: Some(0),
                    temp_buffer_alignment: Some(1),
                    inputs_layout: None,
                    outputs_layout: None,
                    function_metadata: None,
                },
            ],
        }
        .encode_to_vec();

        let err = get_entrypoint_metadata(&encoded).unwrap_err();
        assert!(
            err.to_string()
                .contains("Expected exactly 1 AOT entrypoint"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn get_entrypoint_function_signature_allows_missing_inputs_layout_for_zero_params() {
        let encoded = AotPackageEntrypointsProto {
            entrypoint: vec![AotEntrypointProto {
                function_symbol: Some("make_empty_tuple".to_string()),
                input_buffer_sizes: vec![],
                input_buffer_alignments: vec![],
                output_buffer_sizes: vec![0],
                output_buffer_alignments: vec![1],
                temp_buffer_size: Some(0),
                temp_buffer_alignment: Some(1),
                inputs_layout: None,
                outputs_layout: Some(TypeLayoutsProto {
                    layouts: vec![TypeLayoutProto {
                        r#type: Some("()".to_string()),
                        size: Some(0),
                        elements: vec![],
                    }],
                }),
                function_metadata: Some(FunctionMetadataProto {
                    function_interface: Some(PackageInterfaceFunctionProto {
                        base: Some(PackageInterfaceFunctionBaseProto {
                            top: Some(true),
                            name: Some("make_empty_tuple".to_string()),
                        }),
                        parameters: vec![],
                        result_type: Some(TypeProto {
                            type_enum: Some(TypeEnumProto::Tuple as i32),
                            bit_count: None,
                            tuple_elements: vec![],
                            array_size: None,
                            array_element: None,
                        }),
                        sv_result_type: None,
                    }),
                }),
            }],
        }
        .encode_to_vec();

        let signature = get_entrypoint_function_signature(&encoded).unwrap();
        assert!(signature.params.is_empty());
        assert!(signature.input_layouts.is_empty());
        assert_eq!(signature.output_layouts.len(), 1);
        assert_eq!(signature.output_layouts[0].size, 0);
        assert!(signature.output_layouts[0].elements.is_empty());
    }

    #[test]
    fn get_entrypoint_function_signature_allows_missing_outputs_layout_for_empty_tuple() {
        let encoded = AotPackageEntrypointsProto {
            entrypoint: vec![AotEntrypointProto {
                function_symbol: Some("make_empty_tuple".to_string()),
                input_buffer_sizes: vec![],
                input_buffer_alignments: vec![],
                output_buffer_sizes: vec![0],
                output_buffer_alignments: vec![1],
                temp_buffer_size: Some(0),
                temp_buffer_alignment: Some(1),
                inputs_layout: None,
                outputs_layout: None,
                function_metadata: Some(FunctionMetadataProto {
                    function_interface: Some(PackageInterfaceFunctionProto {
                        base: Some(PackageInterfaceFunctionBaseProto {
                            top: Some(true),
                            name: Some("make_empty_tuple".to_string()),
                        }),
                        parameters: vec![],
                        result_type: Some(TypeProto {
                            type_enum: Some(TypeEnumProto::Tuple as i32),
                            bit_count: None,
                            tuple_elements: vec![],
                            array_size: None,
                            array_element: None,
                        }),
                        sv_result_type: None,
                    }),
                }),
            }],
        }
        .encode_to_vec();

        let signature = get_entrypoint_function_signature(&encoded).unwrap();
        assert!(signature.params.is_empty());
        assert!(signature.input_layouts.is_empty());
        assert_eq!(signature.output_layouts.len(), 1);
        assert_eq!(signature.output_layouts[0].size, 0);
        assert!(signature.output_layouts[0].elements.is_empty());
    }
}
