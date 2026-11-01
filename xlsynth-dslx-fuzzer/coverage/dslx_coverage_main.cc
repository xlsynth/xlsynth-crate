// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "xls/dslx/bytecode/bytecode_emitter.h"
#include "xls/dslx/bytecode/bytecode_interpreter.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/virtualizable_file_system.h"
#include "xls/dslx/warning_kind.h"

namespace {

// Parses, typechecks, emits bytecode, and interprets one generated environment.
absl::Status RunSample(const std::filesystem::path& path,
                       const std::filesystem::path& stdlib,
                       bool typecheck_only) {
  std::ifstream source(path);
  if (!source) {
    return absl::NotFoundError("cannot open sample: " + path.string());
  }
  const std::string text((std::istreambuf_iterator<char>(source)),
                         std::istreambuf_iterator<char>());
  const std::vector<std::filesystem::path> search_paths = {path.parent_path()};
  auto import_data = xls::dslx::CreateImportData(
      stdlib, search_paths, xls::dslx::kNoWarningsSet,
      std::make_unique<xls::dslx::RealFilesystem>());
  auto typed = xls::dslx::ParseAndTypecheck(text, path.string(),
                                            path.stem().string(), &import_data);
  if (!typed.ok()) {
    return typed.status();
  }
  if (typecheck_only) {
    // Constexpr evaluation during typechecking is the entire measured workload.
    return absl::OkStatus();
  }
  auto function =
      typed->module->GetMemberOrError<xls::dslx::Function>("coverage_entry");
  if (!function.ok()) {
    return function.status();
  }
  auto bytecode = xls::dslx::BytecodeEmitter::Emit(
      &import_data, typed->type_info, **function, xls::dslx::ParametricEnv());
  if (!bytecode.ok()) {
    return bytecode.status();
  }
  auto result = xls::dslx::BytecodeInterpreter::Interpret(
      &import_data, bytecode->get(), std::vector<xls::dslx::InterpValue>());
  return result.status();
}

}  // namespace

int main(int argc, char** argv) {
  if ((argc != 3 && argc != 4) ||
      (argc == 4 && std::string(argv[3]) != "--typecheck-only")) {
    std::cerr
        << "usage: dslx_coverage_main <stdlib-directory> <sample-manifest> "
           "[--typecheck-only]\n";
    return 2;
  }
  std::ifstream manifest(argv[2]);
  if (!manifest) {
    std::cerr << "cannot open sample manifest: " << argv[2] << '\n';
    return 2;
  }
  std::string path;
  size_t count = 0;
  while (std::getline(manifest, path)) {
    if (path.empty()) {
      continue;  // Empty manifest lines do not identify samples.
    }
    const absl::Status status = RunSample(path, argv[1], argc == 4);
    if (!status.ok()) {
      std::cerr << path << ": " << status << '\n';
      return 1;
    }
    ++count;
  }
  if (!manifest.eof() || count == 0) {
    std::cerr << "manifest is unreadable or contains no samples\n";
    return 2;
  }
  std::cout << "Parsed and typechecked " << count << " samples"
            << (argc == 3 ? " with explicit interpretation"
                          : " with constexpr evaluation only")
            << '\n';
  return 0;
}
