# SPDX-License-Identifier: Apache-2.0

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

cc_binary(
    name = "libdslx_fuzz_native.so",
    srcs = ["dslx_fuzz_native.cc"],
    additional_linker_inputs = ["exports.lds"],
    linkopts = [
        "-Wl,--version-script=$(location exports.lds)",
        "-Wl,-Bsymbolic",
        "-Wl,-z,start-stop-visibility=hidden",
    ],
    linkshared = True,
    linkstatic = True,
    deps = [
        "//xls/dslx:create_import_data",
        "//xls/dslx:import_data",
        "//xls/dslx:parse_and_typecheck",
        "//xls/dslx:virtualizable_file_system",
        "//xls/dslx:warning_kind",
        "@com_google_absl//absl/status",
    ],
)
