#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import os
import re
import sys
import difflib


class VersionBumpError(Exception):
    """Raised when an invalid version bump operation is requested."""


def find_cargo_toml_files():
    cargo_files = []
    for root, _, files in os.walk("."):
        if "Cargo.toml" in files:
            cargo_files.append(os.path.join(root, "Cargo.toml"))
    return cargo_files


def compute_bumped_version(old_version: str, bump_kind: str) -> str:
    parts = old_version.split(".")
    if len(parts) != 3:
        raise VersionBumpError(f"Invalid version format: {old_version}")
    try:
        major = int(parts[0])
        minor = int(parts[1])
        patch = int(parts[2])
    except ValueError:
        raise VersionBumpError(f"Invalid numeric components in version: {old_version}")

    if bump_kind == "minor":
        minor += 1
        patch = 0
    elif bump_kind == "patch":
        patch += 1
    else:
        raise VersionBumpError(f"Unknown bump kind: {bump_kind}")

    return f"{major}.{minor}.{patch}"


def validate_transition(old_version: str, new_version: str, bump_kind: str) -> None:
    try:
        old_parts = [int(p) for p in old_version.split(".")]
        new_parts = [int(p) for p in new_version.split(".")]
    except ValueError:
        raise VersionBumpError(
            f"Invalid version format(s): old={old_version} new={new_version}"
        )
    if len(old_parts) != 3 or len(new_parts) != 3:
        raise VersionBumpError(
            f"Invalid version format(s): old={old_version} new={new_version}"
        )

    if bump_kind == "minor":
        if not (
            old_parts[2] == 0
            and new_parts[2] == 0
            and new_parts[0] == old_parts[0]
            and new_parts[1] == old_parts[1] + 1
        ):
            raise VersionBumpError(
                "Invalid minor bump: expected to go from X.Y.0 to X.(Y+1).0; "
                f"got {old_version} -> {new_version}"
            )
    elif bump_kind == "patch":
        if not (
            new_parts[0] == old_parts[0]
            and new_parts[1] == old_parts[1]
            and new_parts[2] == old_parts[2] + 1
        ):
            raise VersionBumpError(
                "Invalid patch bump: expected to go from X.Y.Z to X.Y.(Z+1); "
                f"got {old_version} -> {new_version}"
            )


def gather_workspace_info(bump_kind: str):
    """Collect package info from all Cargo.toml files in the workspace.
    Returns a dict mapping package names to a tuple of (file_path, old_version, new_version).
    """
    workspace_info = {}
    for file in find_cargo_toml_files():
        with open(file, "r") as f:
            lines = f.readlines()
        in_package = False
        pkg_name = None
        version_line_index = None
        old_version = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped == "[package]":
                in_package = True
                continue
            if in_package and stripped.startswith("[") and stripped != "[package]":
                break
            if in_package:
                m = re.match(r'\s*name\s*=\s*"([^"]+)"', line)
                if m:
                    pkg_name = m.group(1)
                m = re.match(r'\s*version\s*=\s*"([0-9]+\.[0-9]+\.[0-9]+)"', line)
                if m:
                    if version_line_index is not None:
                        raise VersionBumpError(
                            f"Multiple version lines found in [package] section of {file}"
                        )
                    version_line_index = i
                    old_version = m.group(1)
        if pkg_name and old_version:
            new_version = compute_bumped_version(old_version, bump_kind)
            workspace_info[pkg_name] = (file, old_version, new_version)
    return workspace_info


def bump_package_version(file_path, do_bump, bump_kind: str):
    """Update the [package] section's version in a Cargo.toml file."""
    with open(file_path, "r") as f:
        lines = f.readlines()
    in_package = False
    version_line_index = None
    old_version = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "[package]":
            in_package = True
            continue
        if in_package and stripped.startswith("[") and stripped != "[package]":
            break
        if in_package:
            m = re.match(r'\s*version\s*=\s*"([0-9]+\.[0-9]+\.[0-9]+)"', line)
            if m:
                if version_line_index is not None:
                    raise VersionBumpError(
                        f"Multiple version lines found in [package] section of {file_path}"
                    )
                version_line_index = i
                old_version = m.group(1)
    if version_line_index is None:
        print(f"No version found in [package] section of {file_path}.")
        return (False, None, None)
    new_version = compute_bumped_version(old_version, bump_kind)
    new_line = lines[version_line_index].replace(old_version, new_version, 1)
    if do_bump:
        lines[version_line_index] = new_line
        with open(file_path, "w") as f:
            f.writelines(lines)
        print(
            f"Updated {file_path}: bumped [package] version from {old_version} to {new_version}."
        )
    else:
        print(
            f"Would update {file_path}: bump [package] version from {old_version} to {new_version}."
        )
    return (True, old_version, new_version)


def update_dependency_versions(file_path, workspace_info, do_bump):
    """Update dependency version numbers in dependency declarations for local packages.

    For each dependency in the file that has a local path and whose package name is in workspace_info,
    ensure its version matches the expected version. In bump mode, update the version; in check mode, error if mismatch.
    """
    with open(file_path, "r") as f:
        content = f.read()
    updated = False
    # For each package in workspace_info, update dependency declarations
    for pkg, (_, old_version, new_version) in workspace_info.items():
        expected_version = new_version if do_bump else old_version
        # Regex to match a dependency declaration for the package in a single line
        # Example: xlsynth = { path = "../xlsynth", version = "0.0.107" }
        pattern = re.compile(
            r"(^\s*"
            + re.escape(pkg)
            + r"\s*=\s*\{[^}]*version\s*=\s*\")([0-9]+\.[0-9]+\.[0-9]+)(\")",
            re.MULTILINE,
        )

        def dep_repl(m):
            current_version = m.group(2)
            if current_version != expected_version:
                if do_bump:
                    nonlocal updated
                    updated = True
                    return f"{m.group(1)}{expected_version}{m.group(3)}"
                else:
                    raise VersionBumpError(
                        f"In file {file_path}, dependency {pkg} has version {current_version} but expected {expected_version}"
                    )
            else:
                return m.group(0)

        content, sub_count = pattern.subn(dep_repl, content)
        if sub_count > 0:
            updated = True
    if updated and do_bump:
        with open(file_path, "w") as f:
            f.write(content)
        print(f"Updated dependency versions in {file_path}.")
    elif updated and not do_bump:
        print(f"Dependencies in {file_path} would be updated.")
    return updated


def _parse_version_tuple(version_str: str) -> tuple[int, int, int]:
    parts = version_str.split(".")
    if len(parts) != 3:
        raise VersionBumpError(f"Invalid version format: {version_str}")
    try:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError:
        raise VersionBumpError(f"Invalid numeric components in version: {version_str}")


def validate_transition_arbitrary(old_version: str, new_version: str) -> None:
    """Validate that new_version is a strictly greater semver than old_version.

    This allows transitions like 0.3.1 -> 0.4.0 that are not strictly a patch or
    minor bump shape, but still move forward in version order.
    """
    old_t = _parse_version_tuple(old_version)
    new_t = _parse_version_tuple(new_version)
    if new_t <= old_t:
        raise VersionBumpError(
            f"Arbitrary transition must increase version: {old_version} -> {new_version}"
        )


def infer_and_validate_transition(old_version: str, new_version: str) -> str:
    """Infer whether the transition is a valid 'minor' or 'patch' bump and validate.

    Returns the inferred bump kind ('minor' or 'patch') if valid, otherwise raises.
    """
    # Try minor first to strongly enforce the X.Y.0 -> X.(Y+1).0 discipline.
    try:
        expected_minor = compute_bumped_version(old_version, "minor")
        if expected_minor == new_version:
            validate_transition(old_version, new_version, "minor")
            return "minor"
    except VersionBumpError:
        pass

    # Try patch next.
    try:
        expected_patch = compute_bumped_version(old_version, "patch")
        if expected_patch == new_version:
            validate_transition(old_version, new_version, "patch")
            return "patch"
    except VersionBumpError:
        pass

    raise VersionBumpError(
        f"Transition not recognized as minor or patch: {old_version} -> {new_version}"
    )


def check_all_packages_at_version(expected_version: str) -> dict:
    """Ensure all packages in the workspace currently have expected_version.

    Returns a map of package name to (file_path, old_version) for use in follow-on updates.
    Raises VersionBumpError on any mismatch.
    """
    mismatches = []
    pkg_info: dict[str, tuple[str, str]] = {}
    for file in find_cargo_toml_files():
        with open(file, "r") as f:
            lines = f.readlines()
        in_package = False
        pkg_name = None
        found_version = None
        for line in lines:
            stripped = line.strip()
            if stripped == "[package]":
                in_package = True
                continue
            if in_package and stripped.startswith("[") and stripped != "[package]":
                break
            if in_package:
                m = re.match(r'\s*name\s*=\s*"([^"]+)"', line)
                if m:
                    pkg_name = m.group(1)
                m = re.match(r'\s*version\s*=\s*"([0-9]+\.[0-9]+\.[0-9]+)"', line)
                if m:
                    found_version = m.group(1)
                    break
        if pkg_name and found_version:
            pkg_info[pkg_name] = (file, found_version)
            if found_version != expected_version:
                mismatches.append((file, found_version, expected_version))
    if mismatches:
        details = "; ".join(
            [f"{f} has {cur}, expected {exp}" for (f, cur, exp) in mismatches]
        )
        raise VersionBumpError(f"Version mismatch before transition: {details}")
    if not pkg_info:
        raise VersionBumpError("No workspace package info found.")
    return pkg_info


def set_package_version(
    file_path: str, new_version: str, do_write: bool
) -> tuple[bool, str | None, str | None]:
    """Replace the [package] version in a Cargo.toml file with new_version."""
    with open(file_path, "r") as f:
        lines = f.readlines()
    in_package = False
    version_line_index = None
    old_version = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "[package]":
            in_package = True
            continue
        if in_package and stripped.startswith("[") and stripped != "[package]":
            break
        if in_package:
            m = re.match(r'\s*version\s*=\s*"([0-9]+\.[0-9]+\.[0-9]+)"', line)
            if m:
                if version_line_index is not None:
                    raise VersionBumpError(
                        f"Multiple version lines found in [package] section of {file_path}"
                    )
                version_line_index = i
                old_version = m.group(1)
    if version_line_index is None:
        return (False, None, None)
    new_line = re.sub(
        r'(\s*version\s*=\s*")([0-9]+\.[0-9]+\.[0-9]+)(")',
        rf"\g<1>{new_version}\3",
        lines[version_line_index],
        count=1,
    )
    if do_write:
        lines[version_line_index] = new_line
        with open(file_path, "w") as f:
            f.writelines(lines)
        print(
            f"Updated {file_path}: set [package] version from {old_version} to {new_version}."
        )
    else:
        print(
            f"Would update {file_path}: set [package] version from {old_version} to {new_version}."
        )
    return (True, old_version, new_version)


def main():
    def usage() -> str:
        return (
            "Usage: python3 scripts/bump_version_numbers.py <command> [args]\n\n"
            "Commands:\n"
            "  check                                  Validate current Cargo.toml versions are consistent (no write)\n"
            "  bump                                    Alias for bump-patch\n"
            "  bump-patch                              Bump X.Y.Z -> X.Y.(Z+1) across workspace\n"
            "  bump-minor                              Bump X.Y.0 -> X.(Y+1).0 across workspace (enforced)\n"
            "  transition <old> <new>                  Infer minor/patch, validate, then apply old->new\n"
            "  transition-minor <old> <new>            Validate minor shape, then apply old->new\n"
            "  transition-patch <old> <new>            Validate patch shape, then apply old->new\n"
            "  transition-arbitrary <old> <new>        Validate forward-only semver, then apply old->new\n\n"
            "Examples:\n"
            "  python3 scripts/bump_version_numbers.py check\n"
            "  python3 scripts/bump_version_numbers.py bump-minor\n"
            "  python3 scripts/bump_version_numbers.py transition 0.3.0 0.4.0\n"
            "  python3 scripts/bump_version_numbers.py transition-arbitrary 0.3.1 0.4.0\n"
        )

    if len(sys.argv) < 2:
        print(usage())
        sys.exit(1)
    arg = sys.argv[1]

    # Transition validation mode
    if arg in (
        "transition",
        "transition-minor",
        "transition-patch",
        "transition-arbitrary",
    ):
        if len(sys.argv) != 4:
            print(usage())
            sys.exit(1)
        old_version = sys.argv[2]
        new_version = sys.argv[3]
        try:
            if arg == "transition-minor":
                validate_transition(old_version, new_version, "minor")
                inferred = "minor"
            elif arg == "transition-patch":
                validate_transition(old_version, new_version, "patch")
                inferred = "patch"
            elif arg == "transition-arbitrary":
                validate_transition_arbitrary(old_version, new_version)
                inferred = "arbitrary"
            else:
                inferred = infer_and_validate_transition(old_version, new_version)

            # Ensure workspace at old_version before applying transition to new_version.
            pkg_info_current = check_all_packages_at_version(old_version)

            # Build workspace_info with explicit new_version for dependency updates.
            workspace_info_for_update = {
                pkg: (file, old, new_version)
                for pkg, (file, old) in pkg_info_current.items()
            }

            # Update package versions across all Cargo.toml files.
            for file in find_cargo_toml_files():
                set_package_version(file, new_version, True)

            # Update dependency versions to match new_version for local packages.
            for file in find_cargo_toml_files():
                update_dependency_versions(file, workspace_info_for_update, True)

            print(f"OK: applied {inferred} transition {old_version} -> {new_version}")
        except VersionBumpError as e:
            print(str(e))
            sys.exit(1)
        return

    if arg not in ("check", "bump", "bump-patch", "bump-minor"):
        valid = [
            "check",
            "bump",
            "bump-patch",
            "bump-minor",
            "transition",
            "transition-minor",
            "transition-patch",
            "transition-arbitrary",
        ]
        suggestion = difflib.get_close_matches(arg, valid, n=1)
        if suggestion:
            print(f"Unknown command: {arg}. Did you mean: {suggestion[0]}?\n")
        print(usage())
        sys.exit(1)

    do_bump = arg != "check"
    bump_kind = (
        "patch"
        if arg in ("bump", "bump-patch")
        else ("minor" if arg == "bump-minor" else "patch")
    )
    mode = "Bumping" if do_bump else "Checking"
    print(f"{mode} package versions in Cargo.toml files...")
    try:
        cargo_files = find_cargo_toml_files()
        workspace_info = gather_workspace_info(bump_kind)
        if not workspace_info:
            raise VersionBumpError("No workspace package info found.")
        bumped_info = {}
        # First update package versions
        for file in cargo_files:
            bumped, old_version, new_version = bump_package_version(
                file, do_bump, bump_kind
            )
            if bumped:
                bumped_info[file] = (old_version, new_version)
        # Then update dependency versions
        for file in cargo_files:
            update_dependency_versions(file, workspace_info, do_bump)
        if bumped_info:
            old_versions = {info[0] for info in bumped_info.values()}
            new_versions = {info[1] for info in bumped_info.values()}
            if len(old_versions) > 1 or len(new_versions) > 1:
                raise VersionBumpError(
                    f"Inconsistent package version bump across files: {bumped_info}"
                )
            else:
                common_old = old_versions.pop()
                common_new = new_versions.pop()
                # Validate the overall transition shape matches the bump kind.
                validate_transition(common_old, common_new, bump_kind)
                if do_bump:
                    print(
                        f"All packages bumped from {common_old} to {common_new}. Files: {list(bumped_info.keys())}"
                    )
                else:
                    print(
                        f"All packages would be updated from {common_old} to {common_new} if bump is performed. Files: {list(bumped_info.keys())}"
                    )
        else:
            print("No Cargo.toml files modified.")
    except VersionBumpError as e:
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()


# -- Inline tests (pytest will discover these when importing this module)


def test_compute_bumped_version_patch_increments_patch_only():
    assert compute_bumped_version("0.3.0", "patch") == "0.3.1"
    assert compute_bumped_version("0.10.9", "patch") == "0.10.10"


def test_compute_bumped_version_minor_increments_minor_and_resets_patch():
    assert compute_bumped_version("0.3.0", "minor") == "0.4.0"
    # Even if old patch is nonzero, the computation resets patch to 0; the
    # transition validator below ensures we only allow minor bumps from X.Y.0.
    assert compute_bumped_version("0.10.5", "minor") == "0.11.0"


def test_validate_transition_minor_ok_from_x_y_0_to_x_yplus1_0():
    validate_transition("0.3.0", "0.4.0", "minor")


def test_validate_transition_minor_rejects_non_zero_old_patch():
    import pytest

    with pytest.raises(VersionBumpError):
        validate_transition("0.3.5", "0.4.0", "minor")


def test_validate_transition_patch_ok_from_x_y_z_to_x_y_zplus1():
    validate_transition("0.3.4", "0.3.5", "patch")


def test_validate_transition_patch_rejects_cross_minor_or_major():
    import pytest

    with pytest.raises(VersionBumpError):
        validate_transition("0.3.4", "0.4.0", "patch")
    with pytest.raises(VersionBumpError):
        validate_transition("0.3.4", "1.0.0", "patch")


def test_infer_and_validate_transition_identifies_minor_and_patch():
    assert infer_and_validate_transition("0.3.0", "0.4.0") == "minor"
    assert infer_and_validate_transition("0.3.4", "0.3.5") == "patch"


def test_infer_and_validate_transition_rejects_invalid():
    import pytest

    with pytest.raises(VersionBumpError):
        infer_and_validate_transition("0.3.1", "0.4.0")


def test_validate_transition_arbitrary_accepts_forward_and_rejects_equal_or_backward():
    # Forward transitions allowed
    validate_transition_arbitrary("0.3.1", "0.4.0")
    validate_transition_arbitrary("0.3.1", "0.3.2")

    # Equal or backward rejected
    import pytest

    with pytest.raises(VersionBumpError):
        validate_transition_arbitrary("0.3.1", "0.3.1")
    with pytest.raises(VersionBumpError):
        validate_transition_arbitrary("0.3.2", "0.3.1")
