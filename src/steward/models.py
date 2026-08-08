"""Plain data structures shared across the codebase.

Nothing here imports DataHub. The adapter converts DataHub's wire types into
these, and every other module works only with these — which is what keeps the
DataHub dependency confined to one file.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

Severity = Literal["critical", "warning", "info"]


@dataclass(frozen=True)
class Column:
    name: str
    type: str
    description: str | None = None
    nullable: bool = True


@dataclass(frozen=True)
class Entity:
    """A node in the catalog: a dataset, dashboard, ML model, job, whatever."""

    urn: str
    entity_type: str
    name: str
    platform: str | None = None
    description: str | None = None
    owners: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    columns: tuple[Column, ...] = ()

    @property
    def is_production(self) -> bool:
        """Cheap heuristic used only to rank output, never to gate an action."""
        haystack = f"{self.urn} {self.name}".lower()
        return "prod" in haystack and "non_prod" not in haystack


@dataclass(frozen=True)
class LineageEdge:
    """One hop away from the entity a traversal started at."""

    entity: Entity
    #: Hops from the origin. 1 is a direct neighbour.
    degree: int
    direction: Literal["upstream", "downstream"]


@dataclass
class Finding:
    """Something an agent concluded, in a form that survives the process.

    A finding is the unit of knowledge this whole project is about: it gets
    written back into DataHub so the next run — or the next engineer — starts
    from it instead of re-deriving it.
    """

    #: Stable identity, so a re-run updates a finding rather than duplicating it.
    finding_id: str
    #: The entity the finding is attached to.
    subject_urn: str
    kind: str
    severity: Severity
    title: str
    body: str
    #: URNs the agent inspected to reach this conclusion. This is what makes a
    #: finding auditable rather than an assertion.
    evidence_urns: list[str] = field(default_factory=list)
    #: Findings from earlier runs that this one built on, by finding_id.
    built_on: list[str] = field(default_factory=list)
    agent: str = "steward"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> Finding:
        return cls(**json.loads(raw))

    def summary_line(self) -> str:
        return f"[{self.severity}] {self.title}"


@dataclass
class RunStats:
    """Instrumentation for the claim that knowledge compounds.

    A demo that asserts the second run is cheaper is worth nothing; one that
    counts tool calls and prior findings reused can be checked.
    """

    tool_calls: int = 0
    entities_inspected: int = 0
    prior_findings_reused: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    wall_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
