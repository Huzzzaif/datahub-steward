"""An in-memory catalog with the same behaviour as the DataHub one.

Built from the same `scenario.py` definition that seeds the live instance, so a
test passing here is a test about the same graph the demo runs on. This is what
lets the suite run in CI with no Docker, no network and no API key.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable

from .catalog import Direction
from .models import Entity, Finding, LineageEdge
from .scenario import ENTITIES, SeedEntity


def _to_entity(seed: SeedEntity) -> Entity:
    return Entity(
        urn=seed.urn,
        entity_type=seed.entity_type,
        name=seed.name,
        platform=seed.platform,
        description=seed.description,
        owners=tuple(seed.owners),
        tags=(),
        columns=tuple(seed.columns),
    )


class FakeCatalog:
    """Satisfies the `Catalog` protocol, entirely in memory."""

    def __init__(self, seeds: Iterable[SeedEntity] | None = None) -> None:
        self._seeds = {s.urn: s for s in (seeds if seeds is not None else ENTITIES)}
        self._findings: dict[str, dict[str, Finding]] = {}
        self._tags: dict[str, set[str]] = {}

        # Adjacency in both directions. An ML model's edge to its training job
        # is modelled exactly as DataHub does it, so traversal depths match.
        self._up: dict[str, list[str]] = {}
        self._down: dict[str, list[str]] = {}
        for seed in self._seeds.values():
            parents = list(seed.upstreams) + list(seed.training_jobs)
            self._up[seed.urn] = parents
            for parent in parents:
                self._down.setdefault(parent, []).append(seed.urn)

    # -- reads ----------------------------------------------------------

    def ping(self) -> bool:
        return True

    def get_entity(self, urn: str) -> Entity | None:
        seed = self._seeds.get(urn)
        if seed is None:
            return None
        entity = _to_entity(seed)
        tags = self._tags.get(urn)
        return entity if not tags else Entity(**{**entity.__dict__, "tags": tuple(sorted(tags))})

    def search(
        self,
        query: str,
        entity_types: Iterable[str] | None = None,
        limit: int = 20,
    ) -> list[Entity]:
        types = set(entity_types) if entity_types else None
        needle = query.strip().lower().strip("*")
        hits = []
        for seed in self._seeds.values():
            if types and seed.entity_type not in types:
                continue
            haystack = f"{seed.name} {seed.description or ''} {seed.urn}".lower()
            if not needle or needle in haystack:
                hits.append(self.get_entity(seed.urn))
        return [h for h in hits if h][:limit]

    def lineage(
        self,
        urn: str,
        direction: Direction = "downstream",
        max_degree: int = 3,
        limit: int = 100,
    ) -> list[LineageEdge]:
        """Breadth-first walk.

        `max_degree` mirrors DataHub's "3+" semantics: asking for 3 returns
        everything at depth 3 *or beyond*, so the fake and the real catalog
        return the same set.
        """
        adjacency = self._down if direction == "downstream" else self._up
        unbounded = max_degree >= 3

        seen = {urn}
        queue = deque([(urn, 0)])
        edges: list[LineageEdge] = []

        while queue:
            current, depth = queue.popleft()
            if not unbounded and depth >= max_degree:
                continue
            for neighbour in adjacency.get(current, []):
                if neighbour in seen:
                    continue
                seen.add(neighbour)
                entity = self.get_entity(neighbour)
                if entity is not None:
                    edges.append(
                        LineageEdge(entity=entity, degree=depth + 1, direction=direction)
                    )
                queue.append((neighbour, depth + 1))

        edges.sort(key=lambda e: (e.degree, e.entity.name))
        return edges[:limit]

    # -- findings -------------------------------------------------------

    def read_findings(self, urn: str) -> list[Finding]:
        return list(self._findings.get(urn, {}).values())

    def write_finding(self, finding: Finding) -> None:
        # Keyed by finding_id so a rewrite updates rather than duplicates,
        # matching the live adapter.
        self._findings.setdefault(finding.subject_urn, {})[finding.finding_id] = finding
        self._tags.setdefault(finding.subject_urn, set()).add("urn:li:tag:steward-reviewed")
