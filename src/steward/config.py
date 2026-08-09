"""Runtime configuration, all overridable by environment variable."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Provider = Literal["ollama", "groq", "anthropic"]


def _load_dotenv() -> None:
    """Read a local .env if present, without taking a dependency for it.

    Real environment variables always win, so this can never override what a
    deploy platform injects. `.env` is gitignored — API keys belong there, not
    in the repo.
    """
    for candidate in (Path.cwd() / ".env", Path(__file__).resolve().parents[2] / ".env"):
        if not candidate.is_file():
            continue
        for line in candidate.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))
        return


_load_dotenv()


@dataclass(frozen=True)
class Config:
    #: DataHub GMS endpoint. The quickstart serves this on 8080.
    datahub_server: str = "http://localhost:8080"
    #: Personal access token. Unset is fine on a local quickstart, which runs
    #: with metadata auth disabled by default.
    datahub_token: str | None = None

    # -- which brain -----------------------------------------------------
    #
    # Default is Ollama: free, local, no API key, works offline. Everything in
    # this project runs on it.
    #
    # To use Claude instead — better multi-step reasoning, but it is a paid API
    # and billed to your own key:
    #
    #     export STEWARD_PROVIDER=anthropic
    #     export ANTHROPIC_API_KEY=sk-ant-...
    #
    #
    # The hosted demo uses "groq" — free tier, no card, and a free web dyno has
    # neither the GPU nor the RAM to run Ollama:
    #
    #     export STEWARD_PROVIDER=groq
    #     export GROQ_API_KEY=gsk_...        # console.groq.com/keys
    #
    provider: Provider = "ollama"

    #: Ollama settings. llama3.1 is the default because it supports tool calling;
    #: phi3 and most small models do not, and will silently never call a tool.
    ollama_model: str = "llama3.1:8b"
    ollama_host: str = "http://localhost:11434"

    #: Groq settings, used only when provider == "groq".
    groq_model: str = "llama-3.3-70b-versatile"
    groq_base_url: str = "https://api.groq.com/openai/v1"

    #: Anthropic settings, used only when provider == "anthropic".
    model: str = "claude-opus-5"
    effort: str = "high"

    max_tokens: int = 16_000
    #: Hard stop on the agent loop so a pathological run cannot spin forever.
    max_iterations: int = 20

    @classmethod
    def from_env(cls) -> Config:
        provider = os.environ.get("STEWARD_PROVIDER", cls.provider).lower()
        if provider not in ("ollama", "groq", "anthropic"):
            provider = cls.provider
        return cls(
            datahub_server=os.environ.get("DATAHUB_GMS_URL", cls.datahub_server),
            datahub_token=os.environ.get("DATAHUB_GMS_TOKEN") or None,
            provider=provider,  # type: ignore[arg-type]
            ollama_model=os.environ.get("STEWARD_OLLAMA_MODEL", cls.ollama_model),
            ollama_host=os.environ.get("OLLAMA_HOST", cls.ollama_host),
            groq_model=os.environ.get("STEWARD_GROQ_MODEL", cls.groq_model),
            groq_base_url=os.environ.get("GROQ_BASE_URL", cls.groq_base_url),
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
