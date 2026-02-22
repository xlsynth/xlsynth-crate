#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import os
import pathlib
import shlex
import subprocess
import tomllib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Yank or unyank a specific version of the xlsynth crate constellation on crates.io."
        )
    )
    parser.add_argument(
        "version",
        help="Version to yank/unyank, e.g. v0.32.0 or 0.32.0.",
    )
    parser.add_argument(
        "--workspace-root",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent.parent,
        help="Workspace root containing Cargo.toml.",
    )
    parser.add_argument(
        "--crate-prefix",
        default="xlsynth",
        help="Only include crates whose package name starts with this prefix.",
    )
    parser.add_argument(
        "--include-nonprefix",
        action="store_true",
        help="Include all publishable workspace crates, not just prefix-matching ones.",
    )
    parser.add_argument(
        "--undo",
        action="store_true",
        help="Undo a previous yank (i.e. unyank).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run cargo yank. Without this, only print planned commands.",
    )
    parser.add_argument(
        "--registry",
        default=None,
        help="Optional cargo --registry value.",
    )
    parser.add_argument(
        "--index",
        default=None,
        help="Optional cargo --index URL.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional crates.io API token (otherwise cargo defaults apply).",
    )
    return parser.parse_args()


def normalize_version(raw: str) -> str:
    if raw.startswith("v"):
        return raw[1:]
    return raw


def registry_token_env_var_name(registry: str) -> str:
    normalized = registry.upper().replace("-", "_")
    return f"CARGO_REGISTRIES_{normalized}_TOKEN"


def preflight_auth_check(
    args: argparse.Namespace, workspace_root: pathlib.Path
) -> tuple[bool, str]:
    _ = workspace_root  # kept in signature for call-site clarity
    if args.token:
        return True, "token provided via --token"

    if os.getenv("CARGO_REGISTRY_TOKEN"):
        return True, "token found in CARGO_REGISTRY_TOKEN"

    if args.registry:
        registry_env_var = registry_token_env_var_name(args.registry)
        if os.getenv(registry_env_var):
            return True, f"token found in {registry_env_var}"

    hints = ["--token", "CARGO_REGISTRY_TOKEN"]
    if args.registry:
        hints.append(registry_token_env_var_name(args.registry))
    hint_list = ", ".join(hints)
    return False, (
        "No cargo registry token source found for --execute mode. "
        f"Provide one of: {hint_list}."
    )


def read_toml(path: pathlib.Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def gather_publishable_workspace_crates(
    workspace_root: pathlib.Path, crate_prefix: str, include_nonprefix: bool
) -> list[str]:
    workspace_manifest = read_toml(workspace_root / "Cargo.toml")
    members = workspace_manifest.get("workspace", {}).get("members", [])
    if not members:
        raise ValueError("No workspace members found in root Cargo.toml.")

    crates: list[str] = []
    for member in members:
        member_manifest_path = workspace_root / member / "Cargo.toml"
        if not member_manifest_path.exists():
            continue
        member_manifest = read_toml(member_manifest_path)
        package = member_manifest.get("package")
        if package is None:
            continue
        if package.get("publish") is False:
            continue
        package_name = package.get("name")
        if not isinstance(package_name, str):
            continue
        if include_nonprefix or package_name.startswith(crate_prefix):
            crates.append(package_name)

    crates = sorted(set(crates))
    if not crates:
        raise ValueError("No matching publishable crates found in workspace.")
    return crates


def build_cargo_yank_command(
    crate_name: str,
    version: str,
    undo: bool,
    registry: str | None,
    index: str | None,
    token: str | None,
) -> list[str]:
    cmd = ["cargo", "yank", crate_name, "--version", version]
    if undo:
        cmd.append("--undo")
    if registry:
        cmd.extend(["--registry", registry])
    if index:
        cmd.extend(["--index", index])
    if token:
        cmd.extend(["--token", token])
    return cmd


def main() -> int:
    args = parse_args()
    workspace_root = args.workspace_root.resolve()
    normalized_version = normalize_version(args.version)
    tag_version = f"v{normalized_version}"

    crates = gather_publishable_workspace_crates(
        workspace_root=workspace_root,
        crate_prefix=args.crate_prefix,
        include_nonprefix=args.include_nonprefix,
    )

    action = "unyank" if args.undo else "yank"
    print(f"Workspace root: {workspace_root}")
    print(f"Version: {normalized_version} ({tag_version})")
    print(f"Action: {action}")
    print(f"Execute mode: {args.execute}")
    print(f"Crates to process ({len(crates)}): {', '.join(crates)}")

    if args.execute:
        auth_ok, auth_details = preflight_auth_check(args, workspace_root)
        if not auth_ok:
            print(f"Preflight auth check failed: {auth_details}")
            return 2
        print(f"Preflight auth check passed: {auth_details}")

    failures: list[tuple[str, str]] = []
    for crate_name in crates:
        cmd = build_cargo_yank_command(
            crate_name=crate_name,
            version=normalized_version,
            undo=args.undo,
            registry=args.registry,
            index=args.index,
            token=args.token,
        )
        rendered = shlex.join(cmd)
        if not args.execute:
            print(f"[dry-run] {rendered}")
            continue

        print(f"Running: {rendered}")
        result = subprocess.run(
            cmd,
            cwd=workspace_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            failure_reason = (
                result.stderr.strip() or result.stdout.strip() or "unknown error"
            )
            failures.append((crate_name, failure_reason))
            print(f"  Failed: {crate_name}")
            if result.stderr.strip():
                print(result.stderr.strip())
        else:
            print(f"  Success: {crate_name}")

    if failures:
        print(f"\nSome crates failed to {action}:")
        for crate_name, reason in failures:
            print(f"  - {crate_name}: {reason}")
        return 1

    if not args.execute:
        print("\nDry run only; rerun with --execute to perform the operation.")
    else:
        print(f"\nAll selected crates processed for {action} successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
