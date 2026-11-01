// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/virtualizable_file_system.h"
#include "xls/dslx/warning_kind.h"

// Clang's ELF sections are also registered with libFuzzer by its generated
// module constructor. Weak references let the probe diagnose an uninstrumented
// build instead of making it impossible to load.
extern "C" {
extern unsigned char __start___sancov_cntrs[] __attribute__((weak));
extern unsigned char __stop___sancov_cntrs[] __attribute__((weak));
}

namespace {

// Fresh import state prevents an earlier input from changing later typechecks.
absl::Status Check(const std::filesystem::path& entry,
                   const std::filesystem::path& stdlib) {
  std::ifstream source(entry);
  if (!source) {
    return absl::NotFoundError("cannot open sample: " + entry.string());
  }
  const std::string text((std::istreambuf_iterator<char>(source)),
                         std::istreambuf_iterator<char>());
  const std::vector<std::filesystem::path> paths = {entry.parent_path()};
  auto imports = xls::dslx::CreateImportData(
      stdlib, paths, xls::dslx::kNoWarningsSet,
      std::make_unique<xls::dslx::RealFilesystem>());
  return xls::dslx::ParseAndTypecheck(text, entry.string(),
                                      entry.stem().string(), &imports)
      .status();
}

}  // namespace

// All C++ symbols are local to this DSO; only this C ABI is public. This keeps
// the local checkout's C++ objects separate from the released libxls's objects.
extern "C" int dslx_fuzz_native_check(const char* stdlib, const char* entry,
                                      char* error, size_t error_capacity) {
  const absl::Status status = Check(entry, stdlib);
  if (error_capacity != 0) {
    const std::string message = status.ok() ? "" : status.ToString();
    const size_t length = std::min(message.size(), error_capacity - 1);
    std::memcpy(error, message.data(), length);
    error[length] = '\0';
  }
  return status.ok() ? 0 : 1;
}

// Copies a counter snapshot for typed probes. Normal fuzz iterations rely on
// libFuzzer reading and resetting these exact counters directly.
extern "C" __attribute__((no_sanitize("coverage"))) size_t
dslx_fuzz_native_counters(unsigned char* output, size_t capacity) {
  if (__start___sancov_cntrs == nullptr || __stop___sancov_cntrs == nullptr) {
    return 0;
  }
  const size_t count = reinterpret_cast<uintptr_t>(__stop___sancov_cntrs) -
                       reinterpret_cast<uintptr_t>(__start___sancov_cntrs);
  if (output != nullptr) {
    std::memcpy(output, __start___sancov_cntrs, std::min(count, capacity));
  }
  return count;
}
