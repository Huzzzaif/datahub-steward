"""Load the demo estate from `scenario.py` into a running DataHub.

Idempotent: aspects are emitted by URN, so re-running overwrites rather than
duplicating. Safe to run against a quickstart as many times as you like.
"""

from __future__ import annotations

import logging

from datahub.emitter.mcp import MetadataChangeProposalWrapper
from datahub.metadata.schema_classes import (
    AuditStampClass,
    ChangeAuditStampsClass,
    CorpUserInfoClass,
    DashboardInfoClass,
    DatasetPropertiesClass,
    DataJobInfoClass,
    DataJobInputOutputClass,
    MLModelPropertiesClass,
    OtherSchemaClass,
    OwnerClass,
    OwnershipClass,
    OwnershipTypeClass,
    SchemaFieldClass,
    SchemaFieldDataTypeClass,
    SchemaMetadataClass,
    StringTypeClass,
    NumberTypeClass,
    DateTypeClass,
    TimeTypeClass,
    UpstreamClass,
    UpstreamLineageClass,
    DatasetLineageTypeClass,
)

from .adapter import DataHubAdapter
from .scenario import ENTITIES, SeedEntity

logger = logging.getLogger(__name__)

_ACTOR = "urn:li:corpuser:steward"
_STAMP = AuditStampClass(time=0, actor=_ACTOR)


def _field_type(native: str) -> SchemaFieldDataTypeClass:
    upper = native.upper()
    if "NUMBER" in upper or "INT" in upper or "FLOAT" in upper:
        return SchemaFieldDataTypeClass(type=NumberTypeClass())
    if "TIMESTAMP" in upper:
        return SchemaFieldDataTypeClass(type=TimeTypeClass())
    if "DATE" in upper:
        return SchemaFieldDataTypeClass(type=DateTypeClass())
    return SchemaFieldDataTypeClass(type=StringTypeClass())


def _emit(adapter: DataHubAdapter, urn: str, aspect) -> None:
    adapter.graph.emit(MetadataChangeProposalWrapper(entityUrn=urn, aspect=aspect))


def _seed_owners(adapter: DataHubAdapter, entity: SeedEntity) -> None:
    if not entity.owners:
        return
    for owner_urn in entity.owners:
        # Materialise the user so the UI resolves a name rather than a raw URN.
        _emit(
            adapter,
            owner_urn,
            CorpUserInfoClass(active=True, displayName=owner_urn.split(":")[-1]),
        )
    _emit(
        adapter,
        entity.urn,
        OwnershipClass(
            owners=[
                OwnerClass(owner=o, type=OwnershipTypeClass.TECHNICAL_OWNER)
                for o in entity.owners
            ]
        ),
    )


def _seed_dataset(adapter: DataHubAdapter, entity: SeedEntity) -> None:
    _emit(
        adapter,
        entity.urn,
        DatasetPropertiesClass(name=entity.name, description=entity.description),
    )

    if entity.columns:
        _emit(
            adapter,
            entity.urn,
            SchemaMetadataClass(
                schemaName=entity.name,
                platform=f"urn:li:dataPlatform:{entity.platform}",
                version=0,
                hash="",
                platformSchema=OtherSchemaClass(rawSchema=""),
                fields=[
                    SchemaFieldClass(
                        fieldPath=column.name,
                        type=_field_type(column.type),
                        nativeDataType=column.type,
                        description=column.description,
                        nullable=column.nullable,
                    )
                    for column in entity.columns
                ],
            ),
        )

    if entity.upstreams:
        _emit(
            adapter,
            entity.urn,
            UpstreamLineageClass(
                upstreams=[
                    UpstreamClass(dataset=up, type=DatasetLineageTypeClass.TRANSFORMED)
                    for up in entity.upstreams
                ]
            ),
        )


def _seed_datajob(adapter: DataHubAdapter, entity: SeedEntity) -> None:
    _emit(
        adapter,
        entity.urn,
        DataJobInfoClass(name=entity.name, type="COMMAND", description=entity.description),
    )
    if entity.upstreams:
        _emit(
            adapter,
            entity.urn,
            DataJobInputOutputClass(
                inputDatasets=list(entity.upstreams),
                outputDatasets=[],
            ),
        )


def _seed_dashboard(adapter: DataHubAdapter, entity: SeedEntity) -> None:
    _emit(
        adapter,
        entity.urn,
        DashboardInfoClass(
            title=entity.name,
            description=entity.description or "",
            lastModified=ChangeAuditStampsClass(created=_STAMP, lastModified=_STAMP),
            datasets=list(entity.upstreams),
        ),
    )


def _seed_mlmodel(adapter: DataHubAdapter, entity: SeedEntity) -> None:
    _emit(
        adapter,
        entity.urn,
        MLModelPropertiesClass(
            name=entity.name,
            description=entity.description,
            trainingJobs=list(entity.training_jobs) or None,
        ),
    )


_SEEDERS = {
    "DATASET": _seed_dataset,
    "DATA_JOB": _seed_datajob,
    "DASHBOARD": _seed_dashboard,
    "MLMODEL": _seed_mlmodel,
}


def seed(adapter: DataHubAdapter | None = None) -> int:
    """Emit the whole scenario. Returns the number of entities written."""
    adapter = adapter or DataHubAdapter()
    if not adapter.ping():
        raise RuntimeError(
            "DataHub is not reachable. Start it with `datahub docker quickstart`."
        )

    for entity in ENTITIES:
        seeder = _SEEDERS.get(entity.entity_type)
        if seeder is None:
            logger.warning("no seeder for %s, skipping", entity.entity_type)
            continue
        seeder(adapter, entity)
        _seed_owners(adapter, entity)
        logger.info("seeded %s", entity.urn)

    return len(ENTITIES)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    count = seed()
    print(f"seeded {count} entities")
