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
from typing import Callable

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


def _named_entity(question: str, options: list[Entity]) -> Entity | None:
    """The entity the question names verbatim, if exactly one does.

    Deterministic on purpose. Substring matching beats any prompt here: when
    someone writes "churn_predictor", no amount of model judgment improves on
    taking them at their word.
    """
    haystack = question.lower()
    hits = [e for e in options if len(e.name) > 3 and e.name.lower() in haystack]
    if not hits:
        return None
    # Longest name wins, so "analytics.features.customer_features" is preferred
    # over a shorter name that happens to be a substring of it.
    hits.sort(key=lambda e: len(e.name), reverse=True)
    if len(hits) > 1 and len(hits[0].name) == len(hits[1].name):
        return None  # genuinely ambiguous — let the model choose
    return hits[0]


def _all_indices(reply: str, ceiling: int) -> list[int]:
    seen: list[int] = []
    for token in re.findall(r"\d+", reply):
        value = int(token)
        if 0 <= value < ceiling and value not in seen:
            seen.append(value)
    return seen


class _CrewBase:
    """Stages both crews share: one bounded model call, and subject resolution."""

    def __init__(
        self,
        catalog: Catalog,
        config: Config | None = None,
        provider: LLMProvider | None = None,
        on_stage: Callable[[Stage], None] | None = None,
    ) -> None:
        self.catalog = catalog
        self.config = config or Config.from_env()
        self.provider = provider or build_provider(self.config)
        #: Called as each stage completes, so a UI can show progress live rather
        #: than revealing a finished list and pretending it streamed.
        self.on_stage = on_stage

    def _emit(self, trace: list[Stage], stage: Stage) -> None:
        trace.append(stage)
        if self.on_stage is not None:
            self.on_stage(stage)

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
            self._emit(trace, Stage("resolve", "no candidates found"))
            return None
        if len(options) == 1:
            self._emit(trace, Stage("resolve", f"only candidate: {options[0].name}"))
            return options[0]

        # If the question names an entity outright, that is not a judgment call
        # and the model doesn't get a vote. Asking anyway is how a run about
        # churn_predictor ends up filed against customer_features.
        named = _named_entity(question, options)
        if named is not None:
            self._emit(trace, Stage("resolve", f"named directly in the question: {named.name}"))
            return named

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
            self._emit(trace, Stage("resolve", f"model reply unusable; fell back to {options[index].name}"))
        else:
            self._emit(trace, Stage("resolve", f"model chose {options[index].name}"))
        return options[index]

    def _prior(self, subject: Entity, stats: RunStats, trace: list[Stage]) -> Finding | None:
        """Whatever an earlier run already concluded here.

        Checked before any traversal — re-deriving settled work is the failure
        mode this whole system exists to avoid.
        """
        prior = self.catalog.read_findings(subject.urn)
        stats.tool_calls += 1
        if not prior:
            return None
        stats.prior_findings_reused += len(prior)
        self._emit(
            trace,
            Stage("prior", f"reusing {len(prior)} existing finding(s) — skipping re-derivation"),
        )
        return prior[0]


class BlastRadiusCrew(_CrewBase):
    """Forward from a proposed change: what breaks, and whose problem is it."""

    # -- stage 2: traverse (no model) ------------------------------------

    def _traverse(
        self, subject: Entity, stats: RunStats, trace: list[Stage]
    ) -> list[LineageEdge]:
        edges = self.catalog.lineage(subject.urn, "downstream", max_degree=3)
        stats.tool_calls += 1
        stats.entities_inspected += len(edges)
        self._emit(trace, Stage("traverse", f"{len(edges)} downstream entities (code, no model)"))
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
            "You assess data-pipeline risk. Reply with at most 5 numbers, "
            "separated by commas, most important first. No words, no explanation.",
            f"A change is being made to: {subject.name}\n\n"
            f"These assets depend on it:\n{listing}\n\n"
            "Which are the highest-risk to break? Production ML models and "
            "executive dashboards are highest risk; intermediate tables and jobs "
            "are lower. Be selective — naming everything is the same as naming "
            "nothing. Reply with at most 5 numbers, most important first.",
            stats,
        )

        picked = [edges[i] for i in _all_indices(reply, len(edges))]

        # A model that flags almost everything has not made a judgment, it has
        # just echoed the list back. Treat that as no answer rather than
        # presenting an undifferentiated wall as "high risk".
        indiscriminate = len(edges) >= 4 and len(picked) >= len(edges) * 0.8
        if indiscriminate:
            self._emit(
                trace,
                Stage("assess", f"model flagged {len(picked)}/{len(edges)} — not selective, ignoring"),
            )
            picked = []

        if not picked:
            # Deterministic fallback: terminal consumers are what people care
            # about, so surface those regardless of what the model said.
            picked = [e for e in edges if e.entity.entity_type in {"MLMODEL", "DASHBOARD"}]
            if not indiscriminate:
                self._emit(
                    trace,
                    Stage("assess", f"model reply unusable; fell back to {len(picked)} terminal consumers"),
                )
        else:
            self._emit(trace, Stage("assess", f"model flagged {len(picked)} high-risk assets"))

        # Models before dashboards before everything else — a production model
        # is the headline, and _compose names picked[0] in the title.
        rank = {"MLMODEL": 0, "DASHBOARD": 1}
        picked.sort(key=lambda e: (rank.get(e.entity.entity_type, 2), e.degree))
        return picked[:5]

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
            self._emit(trace, Stage("compose", "model returned nothing; used deterministic summary"))
        else:
            self._emit(trace, Stage("compose", f"{len(body.split())} words"))

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

        existing = self._prior(subject, stats, trace)
        if existing:
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

        # Nothing downstream means there is no blast radius to warn about, and a
        # finding saying so would be noise on an entity someone has to read.
        # Answer the question, write nothing.
        if not edges:
            self._emit(trace, Stage("record", "nothing downstream — no finding recorded"))
            stats.wall_seconds = round(time.monotonic() - started, 2)
            return CrewResult(
                answer=(
                    f"{subject.name} has nothing downstream of it in the catalog, "
                    "so changing it breaks nothing that DataHub knows about. Note "
                    "that absence of lineage is not proof of no consumers — it can "
                    "also mean lineage was never captured for this asset."
                ),
                finding=None,
                stats=stats,
                trace=trace,
            )

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
        self._emit(trace, Stage("record", f"wrote {finding.finding_id} to the catalog"))

        stats.wall_seconds = round(time.monotonic() - started, 2)
        return CrewResult(answer=f"{title}. {body}", finding=finding, stats=stats, trace=trace)


class RootCauseCrew(_CrewBase):
    """Backward from a symptom: what most likely caused it.

    Same staged shape as the blast-radius crew, and for the same reason — but
    the judgment step is a *ranking* rather than a filter, because the honest
    answer to "why did this break" is usually a shortlist with reasons, not a
    single confident culprit.
    """

    def _traverse(
        self, subject: Entity, stats: RunStats, trace: list[Stage]
    ) -> list[LineageEdge]:
        edges = self.catalog.lineage(subject.urn, "upstream", max_degree=3)
        stats.tool_calls += 1
        stats.entities_inspected += len(edges)
        self._emit(trace, Stage("traverse", f"{len(edges)} upstream entities (code, no model)"))
        return edges

    def _rank(
        self, subject: Entity, edges: list[LineageEdge], stats: RunStats, trace: list[Stage]
    ) -> list[LineageEdge]:
        """Order the upstream candidates by how likely each is to be the cause."""
        if not edges:
            return []

        listing = "\n".join(
            f"{i}. {e.entity.name} — {e.entity.entity_type}, {e.degree} hop(s) upstream"
            for i, e in enumerate(edges)
        )
        reply = self._ask(
            "You rank likely causes of a data incident. Reply with at most 4 "
            "numbers, separated by commas, most likely first. No words.",
            f"Symptom reported on: {subject.name}\n\n"
            f"These feed it, with how many hops away each is:\n{listing}\n\n"
            "Which are the most likely causes of that specific symptom?\n"
            "- Fewer hops away is more likely: a change 1-2 hops up reaches the "
            "symptom directly, one 5 hops up is usually diluted or would have "
            "broken other things first.\n"
            "- Raw source tables change more often than modelled tables.\n"
            "- Prefer sources whose subject matter matches the symptom.\n"
            "Reply with at most 4 numbers, most likely first.",
            stats,
        )

        indices = _all_indices(reply, len(edges))

        # An echo is returning *everything* in the order given — that is the list
        # handed back, not a ranking. A short prefix in order is different: the
        # candidates are sorted nearest-first and the prompt says nearer is more
        # likely, so "the first four" is a legitimate answer, not a non-answer.
        echoed = (
            indices == list(range(len(indices)))
            and len(indices) >= max(3, len(edges) * 0.8)
        )
        if echoed:
            self._emit(
                trace,
                Stage("rank", f"model returned input order for all {len(indices)} — not a ranking, ignoring"),
            )
            indices = []

        ranked = [edges[i] for i in indices]
        if not ranked:
            # Deterministic fallback: raw sources first, then by distance. A
            # garbled model reply degrades to a defensible ordering rather than
            # taking the run down.
            ranked = sorted(
                edges,
                key=lambda e: (0 if e.entity.name.startswith("raw.") else 1, e.degree),
            )
            self._emit(trace, Stage("rank", f"model reply unusable; fell back to {len(ranked)} ordered by source-first"))
        else:
            self._emit(trace, Stage("rank", f"model ranked {len(ranked)} candidate causes"))
        return ranked[:5]

    def _compose(
        self,
        subject: Entity,
        ranked: list[LineageEdge],
        stats: RunStats,
        trace: list[Stage],
    ) -> tuple[str, str]:
        top = ranked[0].entity if ranked else None
        shortlist = ", ".join(e.entity.name for e in ranked[:3]) or "no upstream dependencies"
        owners = sorted({o.split(":")[-1] for e in ranked for o in e.entity.owners})

        body = self._ask(
            "You write short, factual incident notes for data engineers. Plain "
            "prose, no headings, no bullets, under 90 words. Do not invent details.",
            f"A problem was reported on {subject.name}.\n"
            f"Most likely upstream causes, in order: {shortlist}.\n"
            f"Teams owning those: {', '.join(owners) or 'unknown'}.\n"
            "State the leading candidate, say plainly that it is a candidate "
            "rather than confirmed, and say what to check first.",
            stats,
        )
        if not body:
            body = (
                f"Leading candidate is {top.name if top else 'unknown'}. "
                f"Other candidates: {shortlist}. Owners to contact: "
                f"{', '.join(owners) or 'unknown'}. This is a ranking from lineage, "
                "not a confirmed cause — check the leading candidate for recent changes first."
            )
            self._emit(trace, Stage("compose", "model returned nothing; used deterministic summary"))
        else:
            self._emit(trace, Stage("compose", f"{len(body.split())} words"))

        title = (
            f"{subject.name} incident: {top.name} is the leading upstream candidate"
            if top
            else f"{subject.name} has no upstream dependencies recorded"
        )
        return title, body

    def run(self, question: str) -> CrewResult:
        stats = RunStats()
        trace: list[Stage] = []
        started = time.monotonic()

        subject = self._resolve(question, stats, trace)
        if subject is None:
            stats.wall_seconds = round(time.monotonic() - started, 2)
            return CrewResult(
                answer="I could not find an entity in the catalog matching that symptom.",
                finding=None,
                stats=stats,
                trace=trace,
            )

        existing = self._prior(subject, stats, trace)
        if existing:
            stats.wall_seconds = round(time.monotonic() - started, 2)
            return CrewResult(
                answer=f"Already known ({existing.finding_id}): {existing.title}. {existing.body}",
                finding=existing,
                stats=stats,
                trace=trace,
            )

        edges = self._traverse(subject, stats, trace)
        ranked = self._rank(subject, edges, stats, trace)
        title, body = self._compose(subject, ranked, stats, trace)

        finding = Finding(
            finding_id=f"root_cause-{re.sub(r'[^a-z0-9]+', '-', subject.name.lower()).strip('-')}",
            subject_urn=subject.urn,
            kind="root_cause",
            # A ranked shortlist is a lead, not a confirmed defect — severity
            # stays at warning so it never outranks a real blast-radius finding.
            severity="warning" if ranked else "info",
            title=title,
            body=body,
            evidence_urns=[e.entity.urn for e in edges],
            agent="root-cause-crew",
        )
        self.catalog.write_finding(finding)
        self._emit(trace, Stage("record", f"wrote {finding.finding_id} to the catalog"))

        stats.wall_seconds = round(time.monotonic() - started, 2)
        return CrewResult(answer=f"{title}. {body}", finding=finding, stats=stats, trace=trace)
