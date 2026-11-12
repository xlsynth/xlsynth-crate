// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/block_evaluator.h"
#include "xls/interpreter/block_interpreter.h"
#include "xls/ir/block.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"

namespace {

using ValueMap = absl::flat_hash_map<std::string, xls::Value>;

// Names in the generated blocks contain neither tabs nor '='.
absl::StatusOr<ValueMap> ParseMap(std::string_view line) {
  ValueMap values;
  if (line.empty()) {
    return values;
  }
  for (std::string_view field : absl::StrSplit(line, '\t')) {
    const size_t separator = field.find('=');
    if (separator == 0 || separator == std::string_view::npos) {
      return absl::InvalidArgumentError(
          absl::StrCat("expected name=value: ", field));
    }
    std::string name(field.substr(0, separator));
    XLS_ASSIGN_OR_RETURN(xls::Value value, xls::Parser::ParseTypedValue(
                                               field.substr(separator + 1)));
    if (!values.emplace(name, std::move(value)).second) {
      return absl::InvalidArgumentError(absl::StrCat("duplicate name: ", name));
    }
  }
  return values;
}

void PrintMap(const ValueMap& values) {
  std::map<std::string, std::string> ordered;
  for (const auto& [name, value] : values) {
    ordered.emplace(name, value.ToString());
  }
  bool first = true;
  for (const auto& [name, value] : ordered) {
    if (!first) {
      std::cout << '\t';
    }
    std::cout << name << '=' << value;
    first = false;
  }
  std::cout << '\n';
}

absl::Status Evaluate(const char* path, const char* top) {
  XLS_ASSIGN_OR_RETURN(std::string ir, xls::GetFileContents(path));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<xls::Package> package,
                       xls::Parser::ParsePackage(ir));
  XLS_ASSIGN_OR_RETURN(xls::Block * block, package->GetBlock(top));
  std::string line;
  if (!std::getline(std::cin, line)) {
    return absl::InvalidArgumentError("expected initial register map");
  }
  XLS_ASSIGN_OR_RETURN(ValueMap initial, ParseMap(line));
  XLS_ASSIGN_OR_RETURN(
      auto continuation,
      xls::kInterpreterBlockEvaluator.NewContinuation(
          block, initial,
          xls::BlockEvaluator::OutputPortSampleTime::kAtLastPosEdgeClock));
  while (std::getline(std::cin, line)) {
    XLS_ASSIGN_OR_RETURN(ValueMap inputs, ParseMap(line));
    XLS_RETURN_IF_ERROR(continuation->RunOneCycle(inputs));
    PrintMap(continuation->output_ports());
    PrintMap(continuation->registers());
  }
  if (!std::cin.eof() || !std::cout) {
    return absl::InternalError("failed to read input or write output");
  }
  return absl::OkStatus();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: xls_block_oracle <package.ir> <top>\n";
    return 2;
  }
  absl::Status status = Evaluate(argv[1], argv[2]);
  if (!status.ok()) {
    std::cerr << status << '\n';
    return 1;
  }
  return 0;
}
