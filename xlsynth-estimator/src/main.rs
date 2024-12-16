// SPDX-License-Identifier: Apache-2.0

//! Command line tool that takes an `xls.estimator_model.EstimatorModel`
//! textproto and a node description and emits the predicted delay for the node.
//!
//! ```
//! <binary> <estimator_model_path> <node_description_str>
//! ```
//!
//! Sample invocation of the command line:
//! ```shell
//! $ time xlsynth-estimator xls/estimators/delay_model/models/asap7.textproto \
//!   "op: \"kAdd\" result_bit_count: 16 operand_bit_count: [16, 16] operand_count: 2"
//! delay: 517.48
//!
//! real    0m0.010s
//! ```

mod estimator;

use prost::Message;
use prost_reflect::DescriptorPool;

mod proto {
    include!(concat!(env!("OUT_DIR"), "/xlsynth.estimator.rs"));
}

fn parse_sample_node_textproto(
    textproto: &str,
) -> Result<proto::SampleNode, Box<dyn std::error::Error>> {
    // Load the descriptor set generated during build
    let descriptor_bytes = include_bytes!(concat!(env!("OUT_DIR"), "/descriptors.bin"));
    let pool = DescriptorPool::decode(descriptor_bytes.as_ref())?;

    // Get the message descriptor for EstimatorModel
    let message_descriptor: prost_reflect::MessageDescriptor = pool
        .get_message_by_name("xlsynth.estimator.SampleNode")
        .ok_or("Message descriptor not found")?;

    let dynamic_message =
        prost_reflect::DynamicMessage::parse_text_format(message_descriptor, textproto)?;

    // Encode the dynamic message and then decode it into the non-dynamic form.
    let buf = dynamic_message.encode_to_vec();
    let sample_node = proto::SampleNode::decode(&*buf)?;

    Ok(sample_node)
}

struct SampleNode {
    wrapped: proto::SampleNode,
}

impl estimator::Node for SampleNode {
    fn op(&self) -> &str {
        &self.wrapped.op
    }
    fn result_bit_count(&self) -> u64 {
        self.wrapped.result_bit_count
    }
    fn operand_bit_count(&self, operand_number: usize) -> u64 {
        self.wrapped.operand_bit_count[operand_number]
    }
    fn operand_count(&self) -> u64 {
        self.wrapped.operand_count
    }
    fn operand_element_count(&self, operand_number: usize) -> Option<u64> {
        if self.wrapped.operand_element_count.is_empty() {
            None
        } else {
            Some(self.wrapped.operand_element_count[operand_number])
        }
    }
    fn operand_element_bit_count(&self, operand_number: usize) -> Option<u64> {
        if self.wrapped.operand_element_bit_count.is_empty() {
            None
        } else {
            Some(self.wrapped.operand_element_bit_count[operand_number])
        }
    }
    fn all_operands_identical(&self) -> bool {
        self.wrapped.all_operands_identical
    }
    fn has_literal_operand(&self) -> bool {
        self.wrapped.has_literal_operand
    }
    fn literal_operands(&self) -> Vec<bool> {
        self.wrapped.literal_operands.clone()
    }
}

fn main() {
    let _ = env_logger::builder().init();
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        println!("Usage: <binary> <estimator_model_path> <node_description_str>");
        std::process::exit(1);
    }

    let estimator_model_path = std::path::PathBuf::from(&args[1]);
    let node_description_str = &args[2];

    // Read in the estimator model from the given path.
    let estimator_model_text = std::fs::read_to_string(estimator_model_path).unwrap();
    let estimator_model =
        estimator::parse_estimator_model_textproto(&estimator_model_text).unwrap();

    // Read in the node description from the given string.
    let node_proto = parse_sample_node_textproto(&node_description_str).unwrap();
    let node = SampleNode {
        wrapped: node_proto,
    };

    match estimator::eval_estimator_model(&estimator_model, &node) {
        Ok(delay) => println!("delay: {:.2}", delay),
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(1);
        }
    }
}
