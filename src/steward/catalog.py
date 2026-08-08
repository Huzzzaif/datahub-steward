"""The contract the agents code against.

Both `DataHubAdapter` (live) and `FakeCatalog` (in-memory) satisfy this. The
agents never learn which one they got, which is what lets the whole test suite
run in CI with no Docker and no network.
"""

from __future__ import annotations

from typing import Iterable, Literal, Protocol, runtime_checkable

from .models import Entity, Finding, LineageEdge

Direction = Literal["upstream", "downstream"]


@runtime_checkable
class Catalog(Protocol):
    def ping(self) -> bool: ...

    def get_entity(self, urn: str) -> Entity | None: ...

    def search(
        self,
        query: str,
        entity_types: Iterable[str] | None = None,
        limit: int = 20,
    ) -> list[Entity]: ...

    def lineage(
        self,
        urn: str,
        direction: Direction = "downstream",
        max_degree: int = 3,
        limit: int = 100,
    ) -> list[LineageEdge]: ...

    def read_findings(self, urn: str) -> list[Finding]: ...

    def write_finding(self, finding: Finding) -> None: ...
