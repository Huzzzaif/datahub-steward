"""The catalog, exposed to Claude as tools.

Each tool returns compact text rather than raw JSON: the agent reasons better
over a readable table than over a wall of braces, and it costs far fewer tokens
on a graph this shape.

`read_prior_findings` is the load-bearing one. It is what makes a later run
inherit an earlier run's conclusions instead of re-deriving them, which is the
whole premise of the project.
"""

from __future__ import annotations

from anthropic import beta_tool

from .catalog import Catalog
from .models import Finding, RunStats


def _fmt_entity(entity, indent: str = "") -> str:
    bits = [f"{indent}{entity.entity_type} | {entity.name}", f"{indent}  urn: {entity.urn}"]
    if entity.description:
        bits.append(f"{indent}  desc: {entity.description}")
    if entity.owners:
        owners = ", ".join(o.split(":")[-1] for o in entity.owners)
        bits.append(f"{indent}  owners: {owners}")
    if entity.tags:
        bits.append(f"{indent}  tags: {', '.join(t.split(':')[-1] for t in entity.tags)}")
    return "\n".join(bits)


def build_tools(catalog: Catalog, stats: RunStats) -> list:
    """Closures over the catalog, so tests can pass a fake with no globals."""

    @beta_tool
    def search_catalog(query: str, entity_type: str | None = None) -> str:
        """Search the data catalog by free text.

        Use this to turn a human description ("the stripe charges table", "the
        churn model") into a URN before calling anything else.

        Args:
            query: Free-text search, e.g. "charges" or "customer features".
            entity_type: Optionally restrict to one of DATASET, DASHBOARD,
                MLMODEL, DATA_JOB.
        """
        stats.tool_calls += 1
        results = catalog.search(query, [entity_type] if entity_type else None, limit=15)
        stats.entities_inspected += len(results)
        if not results:
            return f"No catalog entities matched {query!r}."
        return "\n".join(_fmt_entity(e) for e in results)

    @beta_tool
    def get_entity_details(urn: str) -> str:
        """Full detail for one entity: description, owners, tags, and columns.

        Call this before asserting anything about a specific column.

        Args:
            urn: The entity URN, as returned by search_catalog or a lineage tool.
        """
        stats.tool_calls += 1
        entity = catalog.get_entity(urn)
        if entity is None:
            return f"No entity found for urn {urn}."
        stats.entities_inspected += 1

        out = [_fmt_entity(entity)]
        if entity.columns:
            out.append("  columns:")
            for column in entity.columns:
                null = "" if column.nullable else " NOT NULL"
                desc = f" — {column.description}" if column.description else ""
                out.append(f"    - {column.name}: {column.type}{null}{desc}")
        return "\n".join(out)

    @beta_tool
    def get_downstream(urn: str, max_degree: int = 3) -> str:
        """Everything that consumes this entity, directly or transitively.

        This is the blast radius of changing it. Degrees above 3 are collapsed
        into "3+" by the catalog, so results can include hops beyond 3.

        Args:
            urn: The entity to walk downstream from.
            max_degree: How many hops out to look, 1-3.
        """
        stats.tool_calls += 1
        edges = catalog.lineage(urn, "downstream", max_degree=max_degree)
        stats.entities_inspected += len(edges)
        if not edges:
            return f"Nothing downstream of {urn}."
        return "\n".join(
            f"degree {e.degree} | {e.entity.entity_type} | {e.entity.name} | {e.entity.urn}"
            + (
                f" | owners: {', '.join(o.split(':')[-1] for o in e.entity.owners)}"
                if e.entity.owners
                else ""
            )
            for e in edges
        )

    @beta_tool
    def get_upstream(urn: str, max_degree: int = 3) -> str:
        """Everything this entity depends on, directly or transitively.

        Use this to work backwards from a symptom toward a cause.

        Args:
            urn: The entity to walk upstream from.
            max_degree: How many hops out to look, 1-3.
        """
        stats.tool_calls += 1
        edges = catalog.lineage(urn, "upstream", max_degree=max_degree)
        stats.entities_inspected += len(edges)
        if not edges:
            return f"Nothing upstream of {urn}."
        return "\n".join(
            f"degree {e.degree} | {e.entity.entity_type} | {e.entity.name} | {e.entity.urn}"
            for e in edges
        )

    @beta_tool
    def read_prior_findings(urn: str) -> str:
        """Findings a previous Steward run already recorded on this entity.

        ALWAYS call this before investigating an entity from scratch. If a prior
        finding already answers the question, cite it by its finding_id and move
        on rather than re-deriving it — that is the point of writing them down.

        Args:
            urn: The entity to check for existing recorded knowledge.
        """
        stats.tool_calls += 1
        findings = catalog.read_findings(urn)
        if not findings:
            return f"No prior findings recorded on {urn}."
        stats.prior_findings_reused += len(findings)
        return "\n\n".join(
            f"finding_id: {f.finding_id}\n"
            f"recorded: {f.created_at} by {f.agent}\n"
            f"severity: {f.severity}\n"
            f"title: {f.title}\n"
            f"body: {f.body}\n"
            f"evidence: {', '.join(f.evidence_urns) or 'none recorded'}"
            for f in findings
        )

    @beta_tool
    def record_finding(
        subject_urn: str,
        title: str,
        body: str,
        severity: str,
        kind: str,
        evidence_urns: list[str],
        built_on: list[str] | None = None,
    ) -> str:
        """Write a conclusion back into the catalog, permanently.

        Record one finding per genuinely distinct conclusion, on the entity a
        future reader would be looking at when they need to know it. Include the
        URNs you actually inspected as evidence.

        Args:
            subject_urn: Entity the finding belongs on.
            title: One line, specific. Name the real entities and numbers.
            body: A short paragraph. State what is true and what should be done.
            severity: critical, warning, or info.
            kind: Short slug, e.g. blast_radius or root_cause.
            evidence_urns: URNs you inspected to reach this conclusion.
            built_on: finding_ids of prior findings this builds on, if any.
        """
        stats.tool_calls += 1
        safe_severity = severity if severity in {"critical", "warning", "info"} else "info"
        slug = "".join(ch if ch.isalnum() else "-" for ch in title.lower())[:48].strip("-")
        finding = Finding(
            finding_id=f"{kind}-{slug}",
            subject_urn=subject_urn,
            kind=kind,
            severity=safe_severity,
            title=title,
            body=body,
            evidence_urns=list(evidence_urns or []),
            built_on=list(built_on or []),
        )
        catalog.write_finding(finding)
        return f"Recorded finding {finding.finding_id} on {subject_urn}."

    return [
        search_catalog,
        get_entity_details,
        get_downstream,
        get_upstream,
        read_prior_findings,
        record_finding,
    ]
