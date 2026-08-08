"""The single-agent path: one model, all the tools, free-form tool use.

This is the better shape on a capable model — it can decide for itself how deep
to walk and when it has enough. On a small local model it is the weaker option,
which is why `crew.py` exists; see that module's docstring for why.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .catalog import Catalog
from .config import Config
from .llm import LLMProvider, build_provider, run_agent_loop
from .models import RunStats
from .tools import build_tools

_SHARED_RULES = """
You are Steward, an agent that works on a company's data catalog.

Before investigating any entity, call read_prior_findings on it. If an earlier
run already established what you need, cite that finding_id and build on it
instead of re-deriving it, and pass it in `built_on` when recording anything
new. Re-doing settled work is the failure mode this system exists to avoid.

Never type a URN from memory. Get it from search_catalog and copy it exactly.

Ground every claim in a tool result. If the lineage does not support a
conclusion, say what is missing rather than guessing. Lineage proves an edge
exists; it does not prove a column is semantically used. Say which you rely on.

Finish by calling record_finding, then reply with a short plain-prose summary —
no headings, no bullet lists, under 150 words, most consequential fact first.
"""

BLAST_RADIUS_BRIEF = """
Your job: given a proposed change, determine precisely what breaks.

Walk downstream far enough to reach terminal consumers — dashboards and ML
models, not just the next table. Production models and executive dashboards
matter most; say so explicitly. Identify which teams own each affected asset,
because what a human needs is "who do I have to talk to".
"""

ROOT_CAUSE_BRIEF = """
Your job: given a symptom, work upstream to the most likely cause.

Walk upstream from the affected asset and find the nearest upstream entity whose
failure would explain it. Rank candidates rather than asserting one, and say
what evidence would confirm or eliminate each.
"""


@dataclass
class AgentResult:
    answer: str
    stats: RunStats


class StewardAgent:
    def __init__(
        self,
        catalog: Catalog,
        brief: str,
        config: Config | None = None,
        provider: LLMProvider | None = None,
    ) -> None:
        self.catalog = catalog
        self.brief = brief
        self.config = config or Config.from_env()
        self.provider = provider or build_provider(self.config)

    def run(self, question: str) -> AgentResult:
        stats = RunStats()
        started = time.monotonic()

        answer = run_agent_loop(
            provider=self.provider,
            system=_SHARED_RULES + self.brief,
            question=question,
            tools=build_tools(self.catalog, stats),
            stats=stats,
            max_iterations=self.config.max_iterations,
        )

        stats.wall_seconds = round(time.monotonic() - started, 2)
        return AgentResult(answer=answer, stats=stats)


def blast_radius_agent(catalog: Catalog, config: Config | None = None) -> StewardAgent:
    return StewardAgent(catalog, BLAST_RADIUS_BRIEF, config)


def root_cause_agent(catalog: Catalog, config: Config | None = None) -> StewardAgent:
    return StewardAgent(catalog, ROOT_CAUSE_BRIEF, config)
