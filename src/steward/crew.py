"""A crew of small models, arranged so each one's job is nearly unfailable.

Handing an 8B model a free-form tool loop over a lineage graph fails in a
specific way: it invents URNs, loses the thread across many calls, and
synthesises poorly. Adding more agents doing that same job multiplies the
failure rather than fixing it.

So this pipeline moves the deterministic work out of the model entirely and
gives each model step a narrow, bounded decision:

    resolve   pick the subject from a numbered candidate list   (LLM, choose an index)
    traverse  walk the lineage graph                            (CODE, no model)
    assess    mark which downstream assets are critical         (LLM, choose indices)
    compose   write the finding                                 (LLM, free text, no tools)
    record    persist to the catalog                            (CODE, no model)

Two rules make it robust on small models: the model never emits an identifier
(only an index into a list the code built), and every stage has a deterministic
fallback so a garbled reply degrades instead of crashing.

The single-agent `agent.py` remains the better choice on a frontier model — see
`--single`. This crew exists because it holds up on a free local one.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from .catalog import Catalog
from .config import Config
from .llm import LLMProvider, Turn, build_provider
from .models import Entity, Finding, LineageEdge, RunStats

logger = logging.getLogger(__name__)

_STOPWORDS = {
    "the", "a", "an", "we", "want", "to", "our", "is", "are", "was", "were", "what",
    "breaks", "break", "change", "changing", "rename", "renaming", "and", "or", "of",
    "in", "on", "it", "its", "from", "into", "with", "for", "why", "how", "did",
    "does", "this", "that", "column", "table", "type", "support", "if", "i", "me",
}


@dataclass
class Stage:
    name: str
    detail: str


@dataclass
class CrewResult:
    answer: str
    finding: Finding | None
    stats: RunStats
    trace: list[Stage] = field(default_factory=list)


def _keywords(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_.]+", text.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 2]


def _first_index(reply: str, ceiling: int) -> int | None:
    """Pull the first in-range integer out of a model reply."""
    for token in re.findall(r"\d+", reply):
        value = int(token)
        if 0 <= value < ceiling:
            return value
    return None


def _all_indices(reply: str, ceiling: int) -> list[int]:
    seen: list[int] = []
    for token in re.findall(r"\d+", reply):
        value = int(token)
        if 0 <= value < ceiling and value not in seen:
            seen.append(value)
    return seen


class BlastRadiusCrew:
    def __init__(
        self,
        catalog: Catalog,
        config: Config | None = None,
        provider: LLMProvider | None = None,
    ) -> None:
        self.catalog = catalog
        self.config = config or Config.from_env()
        self.provider = provider or build_provider(self.config)

    # -- one bounded model call -----------------------------------------

    def _ask(self, system: str, prompt: str, stats: RunStats) -> str:
        turn: Turn = self.provider.complete(
            system, [{"role": "user", "content": prompt}], tools=[]
        )
        stats.input_tokens += turn.input_tokens
        stats.output_tokens += turn.output_tokens
        return turn.text.strip()

    # -- stage 1: resolve -----------------------------------------------

    def _resolve(self, question: str, stats: RunStats, trace: list[Stage]) -> Entity | None:
        """Find the entity the question is about.

        Candidates are gathered in code; the model only picks one by number, so
        it can never invent a URN that doesn't exist.
        """
        candidates: dict[str, Entity] = {}
        for word in _keywords(question)[:6]:
            for entity in self.catalog.search(word, limit=8):
                candidates[entity.urn] = entity
        if not candidates:
            for entity in self.catalog.search("", limit=25):
                candidates[entity.urn] = entity

        options = list(candidates.values())
        stats.entities_inspected += len(options)
        if not options:
            trace.append(Stage("resolve", "no candidates found"))
            return None
        if len(options) == 1:
            trace.append(Stage("resolve", f"only candidate: {options[0].name}"))
            return options[0]

        listing = "\n".join(
            f"{i}. {e.name}  ({e.entity_type})" for i, e in enumerate(options)
        )
        reply = self._ask(
            "You match a user's question to exactly one item from a list. "
            "Reply with only the number. No words.",
            f"Question: {question}\n\nItems:\n{listing}\n\n"
            "Which single item is the question about? Reply with the number only.",
            stats,
        )

        index = _first_index(reply, len(options))
        if index is None:
            # Deterministic fallback: most keyword hits in the name wins.
            words = set(_keywords(question))
            index = max(
                range(len(options)),
                key=lambda i: sum(w in options[i].name.lower() for w in words),
            )
            trace.append(Stage("resolve", f"model reply unusable; fell back to {options[index].name}"))
        else:
            trace.append(Stage("resolve", f"model chose {options[index].name}"))
        return options[index]

    # -- stage 2: traverse (no model) ------------------------------------

    def _traverse(
        self, subject: Entity, stats: RunStats, trace: list[Stage]
    ) -> list[LineageEdge]:
        edges = self.catalog.lineage(subject.urn, "downstream", max_degree=3)
        stats.tool_calls += 1
        stats.entities_inspected += len(edges)
        trace.append(Stage("traverse", f"{len(edges)} downstream entities (code, no model)"))
        return edges

    # -- stage 3: assess -------------------------------------------------

    def _assess(
        self, subject: Entity, edges: list[LineageEdge], stats: RunStats, trace: list[Stage]
    ) -> list[LineageEdge]:
        """Which downstream assets make this change dangerous?"""
        if not edges:
            return []

        listing = "\n".join(
            f"{i}. {e.entity.name} — {e.entity.entity_type}, {e.degree} hop(s) away"
            for i, e in enumerate(edges)
        )
        reply = self._ask(
            "You assess data-pipeline risk. Reply with only numbers separated by "
            "commas. No words, no explanation.",
            f"A change is being made to: {subject.name}\n\n"
            f"These assets depend on it:\n{listing}\n\n"
            "Which are the highest-risk ones to break? Production ML models and "
            "executive dashboards are highest risk; intermediate tables are lower. "
            "Reply with just the numbers, most important first.",
            stats,
        )

        picked = [edges[i] for i in _all_indices(reply, len(edges))]
        if not picked:
            # Deterministic fallback: models and dashboards are what people care
            # about, so surface those regardless of what the model said.
            picked = [
                e for e in edges if e.entity.entity_type in {"MLMODEL", "DASHBOARD"}
            ]
            trace.append(Stage("assess", f"model reply unusable; fell back to {len(picked)} terminal consumers"))
        else:
            trace.append(Stage("assess", f"model flagged {len(picked)} high-risk assets"))
        return picked[:6]

    # -- stage 4: compose ------------------------------------------------

    def _compose(
        self,
        subject: Entity,
        critical: list[LineageEdge],
        edges: list[LineageEdge],
        stats: RunStats,
        trace: list[Stage],
    ) -> tuple[str, str]:
        names = ", ".join(e.entity.name for e in critical) or "no terminal consumers"
        owners = sorted(
            {o.split(":")[-1] for e in edges for o in e.entity.owners}
        )

        body = self._ask(
            "You write short, factual notes for data engineers. Plain prose, no "
            "headings, no bullet points, under 90 words. Do not invent details.",
            f"Write a warning note about a proposed change to {subject.name}.\n"
            f"It has {len(edges)} downstream dependents.\n"
            f"The most important are: {names}.\n"
            f"Teams that own affected assets: {', '.join(owners) or 'unknown'}.\n"
            "State what is at risk and who needs to be consulted.",
            stats,
        )
        if not body:
            body = (
                f"{subject.name} has {len(edges)} downstream dependents, including "
                f"{names}. Owners to consult: {', '.join(owners) or 'unknown'}."
            )
            trace.append(Stage("compose", "model returned nothing; used deterministic summary"))
        else:
            trace.append(Stage("compose", f"{len(body.split())} words"))

        top = critical[0].entity if critical else None
        title = (
            f"{subject.name} feeds {top.name} ({top.entity_type}), "
            f"{critical[0].degree} hops downstream"
            if top
            else f"{subject.name} has {len(edges)} downstream dependents"
        )
        return title, body

    # -- orchestration ---------------------------------------------------

    def run(self, question: str) -> CrewResult:
        stats = RunStats()
        trace: list[Stage] = []
        started = time.monotonic()

        subject = self._resolve(question, stats, trace)
        if subject is None:
            stats.wall_seconds = round(time.monotonic() - started, 2)
            return CrewResult(
                answer="I could not find an entity in the catalog matching that question.",
                finding=None,
                stats=stats,
                trace=trace,
            )

        prior = self.catalog.read_findings(subject.urn)
        stats.tool_calls += 1
        if prior:
            stats.prior_findings_reused += len(prior)
            trace.append(
                Stage("prior", f"reusing {len(prior)} existing finding(s) — skipping re-derivation")
            )
            existing = prior[0]
            stats.wall_seconds = round(time.monotonic() - started, 2)
            return CrewResult(
                answer=(
                    f"Already known ({existing.finding_id}): {existing.title}. "
                    f"{existing.body}"
                ),
                finding=existing,
                stats=stats,
                trace=trace,
            )

        edges = self._traverse(subject, stats, trace)
        critical = self._assess(subject, edges, stats, trace)
        title, body = self._compose(subject, critical, edges, stats, trace)

        severity = "critical" if any(
            e.entity.entity_type == "MLMODEL" for e in critical
        ) else "warning" if critical else "info"

        finding = Finding(
            finding_id=f"blast_radius-{re.sub(r'[^a-z0-9]+', '-', subject.name.lower()).strip('-')}",
            subject_urn=subject.urn,
            kind="blast_radius",
            severity=severity,
            title=title,
            body=body,
            evidence_urns=[e.entity.urn for e in edges],
            agent="blast-radius-crew",
        )
        self.catalog.write_finding(finding)
        trace.append(Stage("record", f"wrote {finding.finding_id} to the catalog"))

        stats.wall_seconds = round(time.monotonic() - started, 2)
        return CrewResult(answer=f"{title}. {body}", finding=finding, stats=stats, trace=trace)
