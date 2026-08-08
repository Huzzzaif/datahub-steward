"""The only module that knows DataHub's API exists.

Everything above this layer speaks in `models.py` dataclasses. That boundary is
deliberate: DataHub ships a fast-moving surface (the `datahub.sdk` package
currently warns that it is experimental and exempt from backwards-compatibility
guarantees), so this file sticks to the stable trio that every first-party
ingestion source is built on — `DataHubGraph`, `MetadataChangeProposalWrapper`,
and the generated aspect classes — plus GraphQL for graph traversal.

If you later swap in the Agent Context Kit or the MCP Server, this is the file
you rewrite, and nothing else needs to change.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

from datahub.emitter.mcp import MetadataChangeProposalWrapper
from datahub.ingestion.graph.client import DatahubClientConfig, DataHubGraph
from datahub.metadata.schema_classes import (
    AuditStampClass,
    GlobalTagsClass,
    InstitutionalMemoryClass,
    InstitutionalMemoryMetadataClass,
    TagAssociationClass,
    TagPropertiesClass,
)

from .catalog import Direction
from .config import FINDING_PREFIX, STEWARD_TAG, Config
from .models import Column, Entity, Finding, LineageEdge

logger = logging.getLogger(__name__)

_ACTOR = "urn:li:corpuser:steward"

# `properties` resolves to a *different concrete type* per entity (DatasetProperties,
# DashboardProperties, ...), and those types disagree on the nullability of `name`.
# GraphQL therefore refuses to merge them at the same response path — a real
# validation error, not a warning. Aliasing each fragment's fields sidesteps the
# merge entirely, so this must stay aliased even though it reads verbosely.
_ENTITY_FIELDS = """
    ... on Dataset {
      dsName: name
      platform { name }
      dsProps: properties { name description }
    }
    ... on Dashboard { dashProps: properties { name description } }
    ... on Chart { chartProps: properties { name description } }
    ... on DataJob { jobProps: properties { name description } }
    ... on MLModel {
      mlName: name
      mlProps: properties { name description }
    }
    ... on MLModelGroup {
      mlgName: name
      mlgProps: properties { name description }
    }
"""

_DETAIL_FIELDS = """
    ... on Dataset {
      ownership { owners { owner { ... on CorpUser { urn } ... on CorpGroup { urn } } } }
      tags { tags { tag { urn } } }
      schemaMetadata { fields { fieldPath nativeDataType description nullable } }
    }
    ... on Dashboard {
      ownership { owners { owner { ... on CorpUser { urn } ... on CorpGroup { urn } } } }
      tags { tags { tag { urn } } }
    }
    ... on MLModel {
      ownership { owners { owner { ... on CorpUser { urn } ... on CorpGroup { urn } } } }
      tags { tags { tag { urn } } }
    }
    ... on DataJob {
      ownership { owners { owner { ... on CorpUser { urn } ... on CorpGroup { urn } } } }
    }
"""

_LINEAGE_QUERY = """
query stewardLineage($urn: String!, $direction: LineageDirection!, $count: Int!, $degrees: [String!]) {
  searchAcrossLineage(
    input: {
      urn: $urn
      direction: $direction
      query: "*"
      start: 0
      count: $count
      orFilters: [{ and: [{ field: "degree", values: $degrees, condition: EQUAL }] }]
    }
  ) {
    total
    searchResults {
      degree
      entity { urn type __ENTITY_FIELDS__ }
    }
  }
}
""".replace("__ENTITY_FIELDS__", _ENTITY_FIELDS)

_ENTITY_QUERY = """
query stewardEntity($urn: String!) {
  entity(urn: $urn) {
    urn
    type
    __ENTITY_FIELDS__
    __DETAIL_FIELDS__
  }
}
""".replace("__ENTITY_FIELDS__", _ENTITY_FIELDS).replace("__DETAIL_FIELDS__", _DETAIL_FIELDS)

_SEARCH_QUERY = """
query stewardSearch($query: String!, $types: [EntityType!], $count: Int!) {
  searchAcrossEntities(input: { query: $query, types: $types, start: 0, count: $count }) {
    total
    searchResults {
      entity { urn type __ENTITY_FIELDS__ }
    }
  }
}
""".replace("__ENTITY_FIELDS__", _ENTITY_FIELDS)


#: Every alias under which a `properties` object can arrive.
_PROP_ALIASES = ("dsProps", "dashProps", "chartProps", "jobProps", "mlProps", "mlgProps")
_NAME_ALIASES = ("dsName", "mlName", "mlgName")


def _dig(obj: Any, *path: str, default: Any = None) -> Any:
    """Walk a nested dict, tolerating nulls at any level.

    GraphQL returns `null` for whole sub-objects when an entity lacks that
    aspect, which is the common case on a freshly seeded catalog.
    """
    cur = obj
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur if cur is not None else default


def _props(node: dict[str, Any]) -> dict[str, Any]:
    for alias in _PROP_ALIASES:
        value = node.get(alias)
        if isinstance(value, dict):
            return value
    return {}


def _name_from_urn(urn: str) -> str:
    """Last-resort display name when no properties aspect is present."""
    inner = urn.rsplit(",", 2)
    if len(inner) >= 2:
        return inner[-2]
    return urn.rsplit(":", 1)[-1]


class DataHubAdapter:
    """Read and write the catalog, in `models.py` terms."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config.from_env()
        self.graph = DataHubGraph(
            DatahubClientConfig(
                server=self.config.datahub_server,
                token=self.config.datahub_token,
            )
        )

    # -- health ---------------------------------------------------------

    def ping(self) -> bool:
        try:
            self.graph.test_connection()
            return True
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("DataHub unreachable at %s: %s", self.config.datahub_server, exc)
            return False

    # -- reads ----------------------------------------------------------

    def _entity_from_node(self, node: dict[str, Any]) -> Entity:
        urn = node.get("urn", "")
        props = _props(node)

        name = props.get("name")
        if not name:
            for alias in _NAME_ALIASES:
                if node.get(alias):
                    name = node[alias]
                    break
        if not name:
            name = _name_from_urn(urn)

        owners = tuple(
            owner_urn
            for owner in _dig(node, "ownership", "owners", default=[]) or []
            if (owner_urn := _dig(owner, "owner", "urn"))
        )
        tags = tuple(
            tag_urn
            for tag in _dig(node, "tags", "tags", default=[]) or []
            if (tag_urn := _dig(tag, "tag", "urn"))
        )
        columns = tuple(
            Column(
                name=field.get("fieldPath", ""),
                type=field.get("nativeDataType") or "unknown",
                description=field.get("description"),
                nullable=bool(field.get("nullable", True)),
            )
            for field in _dig(node, "schemaMetadata", "fields", default=[]) or []
        )
        return Entity(
            urn=urn,
            entity_type=node.get("type", "UNKNOWN"),
            name=name,
            platform=_dig(node, "platform", "name"),
            description=props.get("description"),
            owners=owners,
            tags=tags,
            columns=columns,
        )

    def get_entity(self, urn: str) -> Entity | None:
        result = self.graph.execute_graphql(_ENTITY_QUERY, variables={"urn": urn})
        node = _dig(result, "entity")
        if not node:
            return None
        return self._entity_from_node(node)

    def search(
        self,
        query: str,
        entity_types: Iterable[str] | None = None,
        limit: int = 20,
    ) -> list[Entity]:
        result = self.graph.execute_graphql(
            _SEARCH_QUERY,
            variables={
                "query": query,
                "types": list(entity_types) if entity_types else None,
                "count": limit,
            },
        )
        results = _dig(result, "searchAcrossEntities", "searchResults", default=[]) or []
        return [self._entity_from_node(r["entity"]) for r in results if r.get("entity")]

    def lineage(
        self,
        urn: str,
        direction: Direction = "downstream",
        max_degree: int = 3,
        limit: int = 100,
    ) -> list[LineageEdge]:
        """Every entity within `max_degree` hops, nearest first.

        DataHub's degree filter is an enum, not an integer range — it accepts
        exactly "1", "2" and "3+", where "3+" means everything three hops or
        further. Passing "3", "4", ... is rejected with a 400.
        """
        degrees = ["1", "2", "3+"][: max(1, min(max_degree, 3))]
        result = self.graph.execute_graphql(
            _LINEAGE_QUERY,
            variables={
                "urn": urn,
                "direction": direction.upper(),
                "count": limit,
                "degrees": degrees,
            },
        )
        results = _dig(result, "searchAcrossLineage", "searchResults", default=[]) or []
        edges = [
            LineageEdge(
                entity=self._entity_from_node(r["entity"]),
                degree=int(r.get("degree") or 1),
                direction=direction,
            )
            for r in results
            if r.get("entity")
        ]
        edges.sort(key=lambda e: e.degree)
        return edges

    # -- findings -------------------------------------------------------

    def read_findings(self, urn: str) -> list[Finding]:
        """Findings previously written to this entity.

        Reading these back is the mechanism by which a later run inherits an
        earlier one's work instead of re-deriving it.
        """
        aspect = self.graph.get_aspect(urn, InstitutionalMemoryClass)
        if not aspect:
            return []

        findings: list[Finding] = []
        for element in aspect.elements or []:
            description = element.description or ""
            if not description.startswith(FINDING_PREFIX):
                continue
            _, _, payload = description.partition(" ")
            try:
                findings.append(Finding.from_json(payload))
            except (ValueError, TypeError) as exc:
                # A hand-edited or malformed link must not break a whole run.
                logger.debug("skipping unparseable finding on %s: %s", urn, exc)
        return findings

    def write_finding(self, finding: Finding) -> None:
        """Persist a finding onto its subject entity, and tag the entity.

        Institutional memory is read-modify-write because emitting the aspect
        replaces it wholesale — a blind write would silently delete every link a
        human had added to that dataset.
        """
        existing = self.graph.get_aspect(finding.subject_urn, InstitutionalMemoryClass)
        elements = list(existing.elements) if existing and existing.elements else []

        marker = f"{FINDING_PREFIX}#{finding.finding_id}"
        elements = [
            element
            for element in elements
            if not (element.description or "").startswith(marker)
        ]

        elements.append(
            InstitutionalMemoryMetadataClass(
                url=f"https://steward.local/findings/{finding.finding_id}",
                description=f"{marker} {finding.to_json()}",
                createStamp=AuditStampClass(time=0, actor=_ACTOR),
            )
        )

        self.graph.emit(
            MetadataChangeProposalWrapper(
                entityUrn=finding.subject_urn,
                aspect=InstitutionalMemoryClass(elements=elements),
            )
        )
        self._ensure_reviewed_tag(finding.subject_urn)

    def _ensure_reviewed_tag(self, urn: str) -> None:
        """Mark the entity as reviewed so the catalog indexes Steward's work."""
        self.graph.emit(
            MetadataChangeProposalWrapper(
                entityUrn=STEWARD_TAG,
                aspect=TagPropertiesClass(
                    name="steward-reviewed",
                    description="An automated Steward agent has recorded a finding here.",
                ),
            )
        )

        existing = self.graph.get_aspect(urn, GlobalTagsClass)
        tags = list(existing.tags) if existing and existing.tags else []
        if any(tag.tag == STEWARD_TAG for tag in tags):
            return
        tags.append(TagAssociationClass(tag=STEWARD_TAG))
        self.graph.emit(
            MetadataChangeProposalWrapper(entityUrn=urn, aspect=GlobalTagsClass(tags=tags))
        )
