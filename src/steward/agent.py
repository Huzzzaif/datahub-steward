"""The agents.

Two roles over one catalog and one tool set. They differ only in their brief:
Blast Radius walks forward from a proposed change, Root Cause walks backward
from a symptom. Both are required to check for prior findings before
investigating, and to write what they conclude back into the catalog.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from anthropic import Anthropic

from .catalog import Catalog
from .config import Config
from .models import RunStats
from .tools import build_tools

_SHARED_RULES = """
You are Steward, an agent that works on a company's data catalog.

Before investigating any entity, call read_prior_findings on it. If an earlier
run already established what you need, cite that finding_id and build on it
instead of re-deriving it — and pass it in `built_on` when you record anything
new. Re-doing settled work is the specific failure mode this system exists to
avoid.

Ground every claim in a tool result. Name real entities, real columns and real
owners; never invent a URN. If the lineage does not support a conclusion, say
what is missing rather than guessing.

Distinguish what the graph proves from what you infer. Lineage tells you an
edge exists; it does not tell you a column is semantically used. Say which you
are relying on.

When you have enough to act, act. Do not narrate your plan, restate the
question, or list options you are not taking.

Finish by recording your conclusions with record_finding, then reply to the
user with a short plain-prose summary — no headings, no bullet lists, under 150
words. Lead with the single most consequential fact.
"""

BLAST_RADIUS_BRIEF = """
Your job: given a proposed change to a dataset or column, determine precisely
what breaks.

Walk downstream far enough to reach terminal consumers — dashboards and ML
models, not just the next table. Production ML models and executive dashboards
matter more than intermediate tables; say so explicitly when you find one.
Identify which teams own each affected asset, because the answer a human needs
is "who do I have to talk to".

Record a finding on the entity being changed, so the next person who opens it
sees the blast radius before they repeat the mistake.
"""

ROOT_CAUSE_BRIEF = """
Your job: given a symptom someone has observed, work upstream to the most
likely cause.

Walk upstream from the affected asset and look for the nearest upstream entity
whose failure would explain the symptom. Rank candidates rather than asserting
one. Say what evidence would confirm or eliminate each.

Record a finding on the asset where the symptom appeared, so the next person
who sees it inherits the investigation.
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
        client: Anthropic | None = None,
    ) -> None:
        self.catalog = catalog
        self.brief = brief
        self.config = config or Config.from_env()
        # A bare constructor picks up ANTHROPIC_API_KEY or an `ant auth login`
        # profile, so there is nothing to configure in the common case.
        self.client = client or Anthropic()

    def run(self, question: str) -> AgentResult:
        stats = RunStats()
        started = time.monotonic()

        runner = self.client.beta.messages.tool_runner(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            max_iterations=self.config.max_iterations,
            system=_SHARED_RULES + self.brief,
            output_config={"effort": self.config.effort},
            tools=build_tools(self.catalog, stats),
            messages=[{"role": "user", "content": question}],
        )

        final_text = ""
        for message in runner:
            usage = getattr(message, "usage", None)
            if usage:
                stats.input_tokens += getattr(usage, "input_tokens", 0) or 0
                stats.output_tokens += getattr(usage, "output_tokens", 0) or 0
            text = "\n".join(
                block.text for block in message.content if getattr(block, "type", "") == "text"
            ).strip()
            if text:
                final_text = text

        stats.wall_seconds = round(time.monotonic() - started, 2)
        return AgentResult(answer=final_text, stats=stats)


def blast_radius_agent(catalog: Catalog, config: Config | None = None) -> StewardAgent:
    return StewardAgent(catalog, BLAST_RADIUS_BRIEF, config)


def root_cause_agent(catalog: Catalog, config: Config | None = None) -> StewardAgent:
    return StewardAgent(catalog, ROOT_CAUSE_BRIEF, config)
