"""Crew behaviour, with a scripted model so the tests are deterministic.

The point of the crew design is that each model step is a bounded choice with a
deterministic fallback. That is only worth claiming if the fallbacks are
actually exercised — so these tests drive the crews with a stub provider that
returns good replies, garbage replies, and nothing at all.

No network, no Ollama, no API key.
"""

from __future__ import annotations

from typing import Any

import pytest

from steward import scenario as s
from steward.config import Config
from steward.crew import BlastRadiusCrew, RootCauseCrew, Stage
from steward.fake import FakeCatalog
from steward.llm import ToolSpec, Turn


class StubProvider:
    """Returns canned replies in order, cycling on the last one."""

    name = "stub"

    def __init__(self, replies: list[str]) -> None:
        self.replies = replies or [""]
        self.calls: list[str] = []

    def complete(
        self, system: str, messages: list[dict[str, Any]], tools: list[ToolSpec]
    ) -> Turn:
        self.calls.append(messages[-1]["content"] if messages else "")
        index = min(len(self.calls) - 1, len(self.replies) - 1)
        return Turn(text=self.replies[index], input_tokens=10, output_tokens=5)

    def format_assistant(self, turn: Turn) -> dict[str, Any]:
        return {"role": "assistant", "content": turn.text}

    def format_tool_result(self, call: Any, result: str) -> dict[str, Any]:
        return {"role": "tool", "content": result}


@pytest.fixture()
def catalog() -> FakeCatalog:
    return FakeCatalog()


def _config() -> Config:
    return Config(provider="ollama")


#: Names the table exactly, so the resolve stage has a single candidate and the
#: subject is pinned regardless of what the stub returns for that step. Tests
#: that care about *later* stages shouldn't be hostage to the first one.
_PINNED_CHANGE = "raw.stripe.charges amount_cents is being renamed to amount_minor_units"


class TestBlastRadiusCrew:
    def test_records_a_finding_naming_the_production_model(
        self, catalog: FakeCatalog
    ) -> None:
        # The exact table name resolves to a single candidate, so the subject is
        # pinned without depending on what the stub says for the resolve step.
        crew = BlastRadiusCrew(
            catalog, _config(), provider=StubProvider(["0", "0,1,2", "A short note."])
        )
        result = crew.run("raw.stripe.charges amount_cents is being renamed")

        assert result.finding is not None
        assert result.finding.kind == "blast_radius"
        assert result.finding.evidence_urns, "a finding must cite what it inspected"
        assert catalog.read_findings(result.finding.subject_urn)

    def test_severity_is_critical_when_an_ml_model_is_downstream(
        self, catalog: FakeCatalog
    ) -> None:
        crew = BlastRadiusCrew(catalog, _config(), provider=StubProvider(["", "", "note"]))
        result = crew.run("what breaks if I change raw.stripe.charges amount_cents")

        # The fallback path surfaces terminal consumers, which includes a model.
        assert result.finding is not None
        assert result.finding.severity == "critical"

    def test_garbled_model_replies_fall_back_instead_of_crashing(
        self, catalog: FakeCatalog
    ) -> None:
        crew = BlastRadiusCrew(
            catalog, _config(), provider=StubProvider(["banana", "not a number", ""])
        )
        result = crew.run("raw.stripe.charges amount_cents change")

        assert result.finding is not None
        names = {stage.name for stage in result.trace}
        assert "traverse" in names and "record" in names

    def test_second_run_reuses_the_first_runs_finding(self, catalog: FakeCatalog) -> None:
        provider = StubProvider(["0", "0,1", "note"])
        BlastRadiusCrew(catalog, _config(), provider=provider).run(_PINNED_CHANGE)
        calls_after_first = len(provider.calls)

        second = BlastRadiusCrew(catalog, _config(), provider=provider).run(_PINNED_CHANGE)

        assert second.stats.prior_findings_reused == 1
        assert second.stats.entities_inspected < 20
        # The expensive stages must not run again.
        assert {stage.name for stage in second.trace} & {"traverse", "assess"} == set()
        assert len(provider.calls) - calls_after_first <= 1

    def test_stage_callback_fires_live(self, catalog: FakeCatalog) -> None:
        seen: list[Stage] = []
        crew = BlastRadiusCrew(
            catalog,
            _config(),
            provider=StubProvider(["0", "0", "note"]),
            on_stage=seen.append,
        )
        result = crew.run(_PINNED_CHANGE)

        assert [st.name for st in seen] == [st.name for st in result.trace]


class TestRootCauseCrew:
    def test_ranks_upstream_candidates_and_records(self, catalog: FakeCatalog) -> None:
        crew = RootCauseCrew(
            catalog, _config(), provider=StubProvider(["0", "1,0,2", "A short note."])
        )
        result = crew.run(s.DEMO_SYMPTOM)

        assert result.finding is not None
        assert result.finding.kind == "root_cause"
        # A ranked shortlist is a lead, never a confirmed defect.
        assert result.finding.severity == "warning"

    def test_walks_upstream_not_downstream(self, catalog: FakeCatalog) -> None:
        crew = RootCauseCrew(catalog, _config(), provider=StubProvider(["0", "0", "note"]))
        result = crew.run(s.DEMO_SYMPTOM)

        assert result.finding is not None
        # Upstream of the churn model must include the raw payments source.
        assert s.RAW_CHARGES in result.finding.evidence_urns
        assert s.DASH_REVENUE not in result.finding.evidence_urns

    def test_falls_back_to_source_first_ordering(self, catalog: FakeCatalog) -> None:
        crew = RootCauseCrew(catalog, _config(), provider=StubProvider(["0", "nonsense", ""]))
        result = crew.run(s.DEMO_SYMPTOM)

        assert result.finding is not None
        assert "raw." in result.finding.title or "raw." in result.finding.body


class TestExpandedScenario:
    def test_churn_model_has_two_independent_upstream_branches(
        self, catalog: FakeCatalog
    ) -> None:
        upstream = {e.entity.urn for e in catalog.lineage(s.MODEL_CHURN, "upstream")}

        # Payments and support are separate causal paths — which is what makes
        # root-cause ranking a real problem rather than a single-path lookup.
        assert s.RAW_CHARGES in upstream
        assert s.RAW_TICKETS in upstream

    def test_finance_dashboard_is_in_the_payments_blast_radius(
        self, catalog: FakeCatalog
    ) -> None:
        downstream = {e.entity.urn for e in catalog.lineage(s.RAW_CHARGES, "downstream")}

        assert s.DASH_FINANCE in downstream
        assert s.RAW_REFUNDS not in downstream, "refunds is a sibling source, not downstream"
