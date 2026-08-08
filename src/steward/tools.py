"""The catalog, exposed to a model as tools.

Tools are declared once as provider-neutral `ToolSpec`s and adapted per backend
in `llm.py`, so Ollama and Anthropic run identical tools.

Each returns compact text rather than JSON: models reason better over a readable
table than a wall of braces, and it costs far fewer tokens — which matters most
for the small local models this defaults to.
"""

from __future__ import annotations

from .catalog import Catalog
from .llm import ToolSpec
from .models import Finding, RunStats

_SEVERITIES = {"critical", "warning", "info"}


def _fmt_entity(entity, indent: str = "") -> str:
    bits = [f"{indent}{entity.entity_type} | {entity.name}", f"{indent}  urn: {entity.urn}"]
    if entity.description:
        bits.append(f"{indent}  desc: {entity.description}")
    if entity.owners:
        bits.append(f"{indent}  owners: {', '.join(o.split(':')[-1] for o in entity.owners)}")
    return "\n".join(bits)


def _fmt_edges(edges) -> str:
    return "\n".join(
        f"degree {e.degree} | {e.entity.entity_type} | {e.entity.name} | {e.entity.urn}"
        + (
            f" | owners: {', '.join(o.split(':')[-1] for o in e.entity.owners)}"
            if e.entity.owners
            else ""
        )
        for e in edges
    )


def build_tools(catalog: Catalog, stats: RunStats) -> list[ToolSpec]:
    """Closures over the catalog, so tests can pass a fake with no globals."""

    def search_catalog(query: str, entity_type: str | None = None) -> str:
        stats.tool_calls += 1
        results = catalog.search(query, [entity_type] if entity_type else None, limit=15)
        stats.entities_inspected += len(results)
        if not results:
            return f"No catalog entities matched {query!r}."
        return "\n".join(_fmt_entity(e) for e in results)

    def get_entity_details(urn: str) -> str:
        stats.tool_calls += 1
        entity = catalog.get_entity(urn)
        if entity is None:
            # Small models invent URNs constantly. Failing with a hint rather
            # than an exception keeps the run recoverable.
            return (
                f"No entity found for urn {urn!r}. Use search_catalog first and "
                "copy the urn exactly as returned."
            )
        stats.entities_inspected += 1
        out = [_fmt_entity(entity)]
        if entity.columns:
            out.append("  columns:")
            for column in entity.columns:
                null = "" if column.nullable else " NOT NULL"
                desc = f" — {column.description}" if column.description else ""
                out.append(f"    - {column.name}: {column.type}{null}{desc}")
        return "\n".join(out)

    def get_downstream(urn: str, max_degree: int = 3) -> str:
        stats.tool_calls += 1
        edges = catalog.lineage(urn, "downstream", max_degree=max_degree)
        stats.entities_inspected += len(edges)
        return _fmt_edges(edges) if edges else f"Nothing downstream of {urn}."

    def get_upstream(urn: str, max_degree: int = 3) -> str:
        stats.tool_calls += 1
        edges = catalog.lineage(urn, "upstream", max_degree=max_degree)
        stats.entities_inspected += len(edges)
        return _fmt_edges(edges) if edges else f"Nothing upstream of {urn}."

    def read_prior_findings(urn: str) -> str:
        stats.tool_calls += 1
        findings = catalog.read_findings(urn)
        if not findings:
            return f"No prior findings recorded on {urn}."
        stats.prior_findings_reused += len(findings)
        return "\n\n".join(
            f"finding_id: {f.finding_id}\nrecorded: {f.created_at}\n"
            f"severity: {f.severity}\ntitle: {f.title}\nbody: {f.body}\n"
            f"evidence: {', '.join(f.evidence_urns) or 'none recorded'}"
            for f in findings
        )

    def record_finding(
        subject_urn: str,
        title: str,
        body: str,
        severity: str = "info",
        kind: str = "finding",
        evidence_urns: list[str] | None = None,
        built_on: list[str] | None = None,
    ) -> str:
        stats.tool_calls += 1
        if catalog.get_entity(subject_urn) is None:
            return (
                f"Refusing to record: {subject_urn!r} is not a real entity. "
                "Look the urn up with search_catalog first."
            )
        slug = "".join(c if c.isalnum() else "-" for c in title.lower())[:48].strip("-")
        finding = Finding(
            finding_id=f"{kind}-{slug}",
            subject_urn=subject_urn,
            kind=kind,
            severity=severity if severity in _SEVERITIES else "info",
            title=title,
            body=body,
            evidence_urns=list(evidence_urns or []),
            built_on=list(built_on or []),
        )
        catalog.write_finding(finding)
        return f"Recorded finding {finding.finding_id} on {subject_urn}."

    _URN = {"type": "string", "description": "Entity URN, exactly as returned by a search."}

    return [
        ToolSpec(
            name="search_catalog",
            description=(
                "Search the data catalog by free text. Use this first to turn a "
                "description like 'the stripe charges table' into a URN."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Free-text search terms."},
                    "entity_type": {
                        "type": "string",
                        "enum": ["DATASET", "DASHBOARD", "MLMODEL", "DATA_JOB"],
                        "description": "Optional type filter.",
                    },
                },
                "required": ["query"],
            },
            fn=search_catalog,
        ),
        ToolSpec(
            name="get_entity_details",
            description="Description, owners and full column list for one entity.",
            parameters={
                "type": "object",
                "properties": {"urn": _URN},
                "required": ["urn"],
            },
            fn=get_entity_details,
        ),
        ToolSpec(
            name="get_downstream",
            description=(
                "Everything that consumes this entity, directly or transitively — "
                "its blast radius. Includes dashboards and ML models."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "urn": _URN,
                    "max_degree": {"type": "integer", "description": "Hops out, 1-3."},
                },
                "required": ["urn"],
            },
            fn=get_downstream,
        ),
        ToolSpec(
            name="get_upstream",
            description="Everything this entity depends on, directly or transitively.",
            parameters={
                "type": "object",
                "properties": {
                    "urn": _URN,
                    "max_degree": {"type": "integer", "description": "Hops out, 1-3."},
                },
                "required": ["urn"],
            },
            fn=get_upstream,
        ),
        ToolSpec(
            name="read_prior_findings",
            description=(
                "Findings an earlier Steward run already recorded here. ALWAYS "
                "call this before investigating an entity, and cite what you find."
            ),
            parameters={
                "type": "object",
                "properties": {"urn": _URN},
                "required": ["urn"],
            },
            fn=read_prior_findings,
        ),
        ToolSpec(
            name="record_finding",
            description=(
                "Write a conclusion back into the catalog permanently, on the "
                "entity a future reader would be looking at."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "subject_urn": _URN,
                    "title": {"type": "string", "description": "One specific line."},
                    "body": {"type": "string", "description": "A short paragraph."},
                    "severity": {"type": "string", "enum": sorted(_SEVERITIES)},
                    "kind": {"type": "string", "description": "e.g. blast_radius"},
                    "evidence_urns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "URNs actually inspected.",
                    },
                    "built_on": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "finding_ids this builds on.",
                    },
                },
                "required": ["subject_urn", "title", "body"],
            },
            fn=record_finding,
        ),
    ]
