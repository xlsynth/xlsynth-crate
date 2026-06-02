# SPDX-License-Identifier: Apache-2.0

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

    assert gen_version_compat._get_release_set_eligibility(
        "v0.52.0", "0.52.0"
    ) == gen_version_compat.ReleaseSetEligibility(
        publish_order_present=True,
        ineligibility_reason="missing crate publications: xlsynth-driver",
    )
    assert gen_version_compat._get_release_set_eligibility(
        "v0.52.1", "0.52.1"
    ) == gen_version_compat.ReleaseSetEligibility(
        publish_order_present=True,
        ineligibility_reason=None,
    )


def test_release_without_publish_order_is_eligible_without_sparse_index_lookup(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        gen_version_compat, "_get_publish_order_at_tag", lambda tag: None
    )

    def published_versions(crate: str):
        raise AssertionError(f"sparse index must not be queried for {crate}")

    monkeypatch.setattr(
        gen_version_compat,
        "_published_versions",
        published_versions,
    )

    assert gen_version_compat._get_release_set_eligibility(
        "v0.49.0", "0.49.0"
    ) == gen_version_compat.ReleaseSetEligibility(
        publish_order_present=False,
        ineligibility_reason=None,
    )
    assert gen_version_compat._get_release_set_eligibility(
        "v0.48.0", "0.48.0"
    ) == gen_version_compat.ReleaseSetEligibility(
        publish_order_present=False,
        ineligibility_reason=None,
    )


def test_tagged_publish_order_is_historical_per_release(monkeypatch) -> None:
    publish_orders = {
        "v0.48.0": ["xlsynth"],
        "v0.52.1": ["xlsynth", "xlsynth-driver"],
    }
    monkeypatch.setattr(
        gen_version_compat,
        "_get_publish_order_at_tag",
        lambda tag: publish_orders[tag],
    )
    monkeypatch.setattr(
        gen_version_compat,
        "_published_versions",
        lambda crate: (
            frozenset(["0.48.0", "0.52.1"])
            if crate == "xlsynth"
            else frozenset(["0.52.1"])
        ),
    )

    assert gen_version_compat._get_release_set_eligibility(
        "v0.48.0", "0.48.0"
    ) == gen_version_compat.ReleaseSetEligibility(
        publish_order_present=True,
        ineligibility_reason=None,
    )
    assert gen_version_compat._get_release_set_eligibility(
        "v0.52.1", "0.52.1"
    ) == gen_version_compat.ReleaseSetEligibility(
        publish_order_present=True,
        ineligibility_reason=None,
    )


def test_get_version_mapping_reuses_cached_row_without_eligibility_check(
    monkeypatch,
) -> None:
    existing = {
        "0.52.1": gen_version_compat.VersionMapping("0.52.1", "0.50.1", "cached"),
    }
    calls = []

    def get_eligibility(tag: str, crate_version: str):
        calls.append((tag, crate_version))
        raise AssertionError("cached rows must not be revalidated")

    monkeypatch.setattr(gen_version_compat, "get_all_tags", lambda: ["v0.52.1"])
    monkeypatch.setattr(
        gen_version_compat, "_load_existing_mappings", lambda path: existing
    )
    monkeypatch.setattr(
        gen_version_compat, "_get_release_set_eligibility", get_eligibility
    )

    mappings = gen_version_compat.get_version_mapping(recompute_all_entries=False)

    assert mappings == [existing["0.52.1"]]
    assert calls == []


def test_get_version_mapping_admits_only_complete_absent_releases(monkeypatch) -> None:
    publish_orders = {
        "v0.52.0": ["xlsynth", "xlsynth-driver"],
        "v0.52.1": ["xlsynth", "xlsynth-driver"],
    }
    monkeypatch.setattr(
        gen_version_compat, "get_all_tags", lambda: ["v0.52.0", "v0.52.1"]
    )
    monkeypatch.setattr(gen_version_compat, "_load_existing_mappings", lambda path: {})
    monkeypatch.setattr(
        gen_version_compat,
        "get_file_content_at_commit",
        lambda tag, path: "v0.50.1",
    )
    monkeypatch.setattr(
        gen_version_compat,
        "_get_publish_order_at_tag",
        lambda tag: publish_orders[tag],
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
    monkeypatch.setattr(gen_version_compat, "get_tag_datetime", lambda tag: "when")

    mappings = gen_version_compat.get_version_mapping(recompute_all_entries=False)

    assert mappings == [
        gen_version_compat.VersionMapping("0.52.1", "0.50.1", "when"),
    ]


def test_get_version_mapping_does_not_backfill_absent_release_without_publish_order(
    monkeypatch,
) -> None:
    monkeypatch.setattr(gen_version_compat, "get_all_tags", lambda: ["v0.4.0"])
    monkeypatch.setattr(gen_version_compat, "_load_existing_mappings", lambda path: {})
    monkeypatch.setattr(
        gen_version_compat,
        "get_file_content_at_commit",
        lambda tag, path: "v0.5.0",
    )
    monkeypatch.setattr(
        gen_version_compat,
        "_get_publish_order_at_tag",
        lambda tag: None,
    )

    mappings = gen_version_compat.get_version_mapping(recompute_all_entries=False)

    assert mappings == []


def test_recompute_all_entries_reconstructs_release_without_publish_order(
    monkeypatch,
) -> None:
    monkeypatch.setattr(gen_version_compat, "get_all_tags", lambda: ["v0.4.0"])
    monkeypatch.setattr(
        gen_version_compat,
        "get_file_content_at_commit",
        lambda tag, path: "v0.5.0",
    )
    monkeypatch.setattr(
        gen_version_compat,
        "_get_publish_order_at_tag",
        lambda tag: None,
    )
    monkeypatch.setattr(gen_version_compat, "get_tag_datetime", lambda tag: "when")

    mappings = gen_version_compat.get_version_mapping(recompute_all_entries=True)

    assert mappings == [
        gen_version_compat.VersionMapping("0.4.0", "0.5.0", "when"),
    ]


def test_recompute_all_entries_ignores_cache_and_reconstructs_from_tags(
    monkeypatch,
) -> None:
    existing = {
        "0.52.1": gen_version_compat.VersionMapping("0.52.1", "stale", "cached"),
    }
    calls = []

    def load_existing(path: str):
        raise AssertionError("recompute mode must not load cached mappings")

    def get_eligibility(tag: str, crate_version: str):
        calls.append((tag, crate_version))
        return gen_version_compat.ReleaseSetEligibility(
            publish_order_present=True,
            ineligibility_reason=None,
        )

    monkeypatch.setattr(gen_version_compat, "get_all_tags", lambda: ["v0.52.1"])
    monkeypatch.setattr(gen_version_compat, "_load_existing_mappings", load_existing)
    monkeypatch.setattr(
        gen_version_compat,
        "get_file_content_at_commit",
        lambda tag, path: "v0.50.1",
    )
    monkeypatch.setattr(
        gen_version_compat, "_get_release_set_eligibility", get_eligibility
    )
    monkeypatch.setattr(gen_version_compat, "get_tag_datetime", lambda tag: "when")

    mappings = gen_version_compat.get_version_mapping(recompute_all_entries=True)

    assert mappings == [
        gen_version_compat.VersionMapping("0.52.1", "0.50.1", "when"),
    ]
    assert mappings != list(existing.values())
    assert calls == [("v0.52.1", "0.52.1")]
