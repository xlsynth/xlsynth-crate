# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import gen_version_compat


def test_release_eligibility_requires_complete_tagged_publish_order(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        gen_version_compat,
        "_get_publish_order_at_tag",
        lambda tag: ["xlsynth", "xlsynth-driver"],
    )
    monkeypatch.setattr(
        gen_version_compat,
        "_published_versions",
        lambda crate: (
            frozenset(["0.52.0", "0.52.1"])
            if crate == "xlsynth"
            else frozenset(["0.52.1"])
        ),
    )

    assert gen_version_compat._get_release_ineligibility_reason(
        "v0.52.0", "0.52.0"
    ) == ("missing crate publications: xlsynth-driver")
    assert (
        gen_version_compat._get_release_ineligibility_reason("v0.52.1", "0.52.1")
        is None
    )


def test_modern_release_without_publish_order_is_ineligible(monkeypatch) -> None:
    monkeypatch.setattr(
        gen_version_compat, "_get_publish_order_at_tag", lambda tag: None
    )
    monkeypatch.setattr(
        gen_version_compat,
        "_published_versions",
        lambda crate: frozenset(["0.48.0"]),
    )

    assert gen_version_compat._get_release_ineligibility_reason(
        "v0.49.0", "0.49.0"
    ) == ("v0.49.0 does not contain publish_order.toml")
    assert (
        gen_version_compat._get_release_ineligibility_reason("v0.48.0", "0.48.0")
        is None
    )


def test_cached_mapping_revalidation_starts_at_v0_49_0(monkeypatch) -> None:
    calls = []

    def get_reason(tag: str, crate_version: str) -> Optional[str]:
        calls.append((tag, crate_version))
        return "incomplete"

    monkeypatch.setattr(
        gen_version_compat, "_get_release_ineligibility_reason", get_reason
    )

    assert (
        gen_version_compat._get_cached_mapping_ineligibility_reason("v0.48.0", "0.48.0")
        is None
    )
    assert (
        gen_version_compat._get_cached_mapping_ineligibility_reason("v0.49.0", "0.49.0")
        == "incomplete"
    )
    assert calls == [("v0.49.0", "0.49.0")]


def test_get_version_mapping_removes_ineligible_cached_modern_row(monkeypatch) -> None:
    existing = {
        "0.48.0": gen_version_compat.VersionMapping("0.48.0", "0.45.0", "old"),
        "0.49.0": gen_version_compat.VersionMapping("0.49.0", "0.45.0", "new"),
    }
    calls = []

    def get_reason(tag: str, crate_version: str) -> Optional[str]:
        calls.append((tag, crate_version))
        return "incomplete"

    monkeypatch.setattr(
        gen_version_compat, "get_all_tags", lambda: ["v0.48.0", "v0.49.0"]
    )
    monkeypatch.setattr(
        gen_version_compat, "_load_existing_mappings", lambda path: existing
    )
    monkeypatch.setattr(
        gen_version_compat, "_get_release_ineligibility_reason", get_reason
    )

    mappings = gen_version_compat.get_version_mapping(recompute_all_entries=False)

    assert mappings == [existing["0.48.0"]]
    assert calls == [("v0.49.0", "0.49.0")]
