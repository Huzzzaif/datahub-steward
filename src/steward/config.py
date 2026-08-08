"""Runtime configuration, all overridable by environment variable."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    #: DataHub GMS endpoint. The quickstart serves this on 8080.
    datahub_server: str = "http://localhost:8080"
    #: Personal access token. Unset is fine on a local quickstart, which
    #: runs with metadata auth disabled by default.
    datahub_token: str | None = None

    #: Claude Opus 5 is the default; the agent reasons over graph structure and
    #: it is the difference between a plausible answer and a correct one.
    model: str = "claude-opus-5"
    #: Long-horizon graph traversal benefits from real thinking depth.
    effort: str = "high"
    max_tokens: int = 16_000

    #: Hard stop on the agent loop so a pathological run cannot spin forever.
    max_iterations: int = 40

    @classmethod
    def from_env(cls) -> Config:
        return cls(
            datahub_server=os.environ.get("DATAHUB_GMS_URL", cls.datahub_server),
            datahub_token=os.environ.get("DATAHUB_GMS_TOKEN") or None,
            model=os.environ.get("STEWARD_MODEL", cls.model),
            effort=os.environ.get("STEWARD_EFFORT", cls.effort),
        )


#: Tag applied to every entity Steward has written a finding about, so the
#: catalog itself becomes the index of what the agents have already looked at.
STEWARD_TAG = "urn:li:tag:steward-reviewed"

#: Findings are written into DataHub as institutional-memory links whose
#: description carries this prefix. Prefixing rather than using a side table is
#: deliberate: the knowledge has to live where the next human will look.
FINDING_PREFIX = "[steward]"
