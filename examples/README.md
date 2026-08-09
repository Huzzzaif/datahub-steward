# Example artifacts

Everything in this folder was **produced by running the agents**, not written by
hand. Regenerate it with:

```bash
uv sync --all-extras
STEWARD_PROVIDER=groq uv run python scripts/build_examples.py
```

If a run is mediocre, the mediocre version is what lands here. That's deliberate
— these exist so the output can be judged without running the code, which only
works if they're honest.

## What Steward produces

The artifact is a **finding**: a conclusion an agent reached, stored back on the
entity inside DataHub so the next person — or the next agent — inherits it
instead of re-deriving it.

## Run transcripts

Full traces: every stage, the answer, and what it cost.

| File | What it shows |
| --- | --- |
| [`runs/01-blast-radius-cold.md`](runs/01-blast-radius-cold.md) | First ask. Walks the graph, judges risk, writes a finding. |
| [`runs/02-blast-radius-warm.md`](runs/02-blast-radius-warm.md) | **The same question again.** No traversal, no model call — it reads what run 1 recorded. |
| [`runs/03-root-cause.md`](runs/03-root-cause.md) | Works backwards from a symptom and ranks candidate causes. |

## The findings themselves

| File | |
| --- | --- |
| [`findings/blast-radius.json`](findings/blast-radius.json) | `critical` — a payments column feeding a production ML model 4 hops away |
| [`findings/root-cause.json`](findings/root-cause.json) | `warning` — a ranked shortlist, never presented as a confirmed cause |

Two fields carry most of the weight:

- **`evidence_urns`** — every entity the agent actually inspected. This is what
  makes a finding auditable rather than an assertion.
- **`built_on`** — earlier findings this one stands on. This is what makes the
  compounding visible rather than merely claimed.

## What lands in DataHub

| File | |
| --- | --- |
| [`datahub/blast-radius-institutional-memory.json`](datahub/blast-radius-institutional-memory.json) | The exact `MetadataChangeProposal` Steward emits |
| [`datahub/root-cause-institutional-memory.json`](datahub/root-cause-institutional-memory.json) | Same, for the root-cause agent |
| [`datahub/lineage.graphql`](datahub/lineage.graphql) | The `searchAcrossLineage` query used for traversal |
| [`datahub/entity.graphql`](datahub/entity.graphql) | The entity hydration query |

Findings are written as `institutionalMemory` links on the subject entity, so
they appear in DataHub's own UI where someone is already looking — not in a side
database only this tool knows about. Writes are read-modify-write (a blind
aspect write would delete human-added links) and keyed by `finding_id`, so a
re-run updates rather than duplicates.

The GraphQL is included because it's where the integration actually lives, and
two details in it were only discoverable by running against a real instance:
fragments must be **aliased per entity type** (`properties.name` has different
nullability on `Dataset` vs `Dashboard`, which makes the obvious query a hard
validation error), and the degree filter is an **enum** — `"1"`, `"2"`, `"3+"`,
not integers.

## The measurement

[`compounding-comparison.json`](compounding-comparison.json) — identical
question, asked twice:

| | Entities inspected | Tokens | Wall time |
| --- | --- | --- | --- |
| First ask | 13 | 566 | 0.88s |
| Second ask | **1** | **0** | **0.0s** |

The second run does no traversal and no model work. It reads the finding the
first one wrote into DataHub and cites it.

That's the whole thesis, and it's a measurement rather than a claim — which is
why `RunStats` is instrumented at all.
