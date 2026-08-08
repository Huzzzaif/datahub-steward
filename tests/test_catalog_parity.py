"""Behavioural tests for the catalog layer.

These run against `FakeCatalog`, so they need no Docker, no network and no API
key — but the fake is built from the same scenario definition that seeds the
live DataHub, so what they assert is true of the real graph too.
"""

from __future__ import annotations

import pytest

from steward import scenario as s
from steward.catalog import Catalog
from steward.fake import FakeCatalog
from steward.models import Finding


@pytest.fixture()
def catalog() -> FakeCatalog:
    return FakeCatalog()


def test_fake_satisfies_the_catalog_protocol(catalog: FakeCatalog) -> None:
    assert isinstance(catalog, Catalog)


def test_entity_hydrates_with_columns_and_owners(catalog: FakeCatalog) -> None:
    entity = catalog.get_entity(s.RAW_CHARGES)

    assert entity is not None
    assert entity.name == "raw.stripe.charges"
    assert "urn:li:corpuser:payments_team" in entity.owners
    assert "amount_cents" in {c.name for c in entity.columns}


def test_unknown_urn_returns_none_rather_than_raising(catalog: FakeCatalog) -> None:
    assert catalog.get_entity("urn:li:dataset:(urn:li:dataPlatform:x,nope,PROD)") is None


class TestBlastRadius:
    """The graph property the whole demo rests on."""

    def test_raw_payments_column_reaches_the_production_model(
        self, catalog: FakeCatalog
    ) -> None:
        reached = {e.entity.urn for e in catalog.lineage(s.RAW_CHARGES, "downstream")}

        assert s.MODEL_CHURN in reached, "production churn model must be in blast radius"
        assert s.DASH_REVENUE in reached
        assert s.CUSTOMER_FEATURES in reached

    def test_the_path_is_genuinely_multi_hop(self, catalog: FakeCatalog) -> None:
        edges = {e.entity.urn: e.degree for e in catalog.lineage(s.RAW_CHARGES, "downstream")}

        # If this collapses to 1 the scenario has lost the property that makes
        # it interesting: the danger is precisely that it is *not* adjacent.
        assert edges[s.MODEL_CHURN] >= 4
        assert edges[s.FCT_ORDERS] == 1

    def test_unrelated_branch_is_not_swept_in(self, catalog: FakeCatalog) -> None:
        reached = {e.entity.urn for e in catalog.lineage(s.RAW_CHARGES, "downstream")}

        # dim_customer descends from CRM, not from payments.
        assert s.DIM_CUSTOMER not in reached


class TestRootCause:
    def test_upstream_from_the_model_reaches_the_raw_source(
        self, catalog: FakeCatalog
    ) -> None:
        reached = {e.entity.urn for e in catalog.lineage(s.MODEL_CHURN, "upstream")}

        assert s.RAW_CHARGES in reached
        assert s.CUSTOMER_FEATURES in reached

    def test_ml_lineage_routes_through_the_training_job(self, catalog: FakeCatalog) -> None:
        edges = catalog.lineage(s.MODEL_CHURN, "upstream")
        nearest = min(edges, key=lambda e: e.degree)

        # DataHub models ML lineage via trainingJobs, not a direct dataset edge.
        assert nearest.entity.urn == s.JOB_TRAIN_CHURN


class TestFindings:
    def _finding(self, **overrides) -> Finding:
        base = dict(
            finding_id="blast-radius-test",
            subject_urn=s.RAW_CHARGES,
            kind="blast_radius",
            severity="critical",
            title="amount_cents feeds a production model",
            body="Four hops downstream.",
            evidence_urns=[s.MODEL_CHURN],
        )
        base.update(overrides)
        return Finding(**base)

    def test_written_findings_are_readable_back(self, catalog: FakeCatalog) -> None:
        catalog.write_finding(self._finding())

        found = catalog.read_findings(s.RAW_CHARGES)
        assert len(found) == 1
        assert found[0].evidence_urns == [s.MODEL_CHURN]

    def test_rewriting_the_same_id_updates_rather_than_duplicates(
        self, catalog: FakeCatalog
    ) -> None:
        catalog.write_finding(self._finding())
        catalog.write_finding(self._finding(title="revised title"))

        found = catalog.read_findings(s.RAW_CHARGES)
        assert len(found) == 1
        assert found[0].title == "revised title"

    def test_recording_a_finding_tags_the_entity(self, catalog: FakeCatalog) -> None:
        catalog.write_finding(self._finding())

        entity = catalog.get_entity(s.RAW_CHARGES)
        assert entity is not None
        assert "urn:li:tag:steward-reviewed" in entity.tags

    def test_entities_start_with_no_findings(self, catalog: FakeCatalog) -> None:
        assert catalog.read_findings(s.CUSTOMER_FEATURES) == []

    def test_findings_survive_a_json_round_trip(self) -> None:
        original = Finding(
            finding_id="x",
            subject_urn=s.RAW_CHARGES,
            kind="root_cause",
            severity="warning",
            title="t",
            body="b",
            evidence_urns=[s.FCT_ORDERS],
            built_on=["earlier-finding"],
        )

        assert Finding.from_json(original.to_json()) == original
