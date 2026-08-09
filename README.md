# Steward

**DataHub agents whose findings compound.**

Built for *Build with DataHub: The Agent Hackathon* — challenges **#1 (Agents
That Do Real Work)** and **#4 (Open / Wildcard)**.

Most catalog agents are read-only advisors: they inspect your metadata and tell
a human something. The interesting clause in challenge #1 is the last one —
*"writes results back so the next person or agent inherits the knowledge."*
Steward is built around making that the measurable centre of the system.

Two agents share one catalog and one tool set:

- **Blast Radius** — given a proposed change, walks *downstream* to terminal
  consumers and reports exactly what breaks and whose sign-off you need.
- **Root Cause** — given a symptom, walks *upstream* and ranks candidate causes
  with the evidence that would confirm or eliminate each.

Every conclusion is written back into DataHub as a structured finding on the
entity itself — not into a side database — and every run reads prior findings
*before* investigating. So the second question about a neighbourhood of the
graph is cheaper than the first, and the knowledge outlives the process.

## The scenario

`src/steward/scenario.py` defines an ordinary analytics estate: Stripe and CRM
raw tables, dbt models, an ML feature table, two Looker dashboards, and two
MLflow models. The interesting property is deliberately non-obvious:

```
raw.stripe.charges.amount_cents
  └─ analytics.core.fct_orders            (revenue_usd derived from it)
      └─ analytics.marts.customer_ltv
          └─ analytics.features.customer_features
              └─ train_churn_predictor    (Airflow)
                  └─ churn_predictor      ← PRODUCTION MODEL, 4 hops away
```

Nobody reading the raw payments table would guess it feeds a production churn
model. That is the whole point: blast radius is invisible precisely when it
matters.

> **Judges / reviewers:** [SUBMISSION.md](SUBMISSION.md) is the short version —
> what it does, what's verified, and the measured before/after.

## Try it in ten seconds

```bash
uv sync
uv run steward serve      # → http://localhost:8000
```

That runs the web UI against the in-memory catalog, so it needs no Docker and no
DataHub. Ask the demo question, watch the crew's stages stream in, then press
**"Ask the same thing again"** and watch the second run finish in about a second
with zero tokens, because it reads the finding the first run wrote.

You'll need one of: Ollama with `llama3.1` (free, local, default), or a free
[Groq key](https://console.groq.com/keys) with `STEWARD_PROVIDER=groq`.

### Deploying it

`Dockerfile` and `render.yaml` are included. On Render, point at the repo and
set `GROQ_API_KEY` in the dashboard — everything else defaults correctly.

## Quickstart against real DataHub

```bash
uv sync

# 1. Bring up DataHub (needs ~10 GB of disk for images)
uv run datahub docker quickstart

# 2. Load the demo estate
uv run steward seed

# 3. Confirm the in-memory fake matches the live instance
uv run steward parity

# 4. Ask something
# Optional: use Claude instead of the free local model
# export STEWARD_PROVIDER=anthropic
# export ANTHROPIC_API_KEY=sk-ant-...
uv run steward blast "We want to rename raw.stripe.charges.amount_cents to \
amount_minor_units and change its type to VARCHAR. What breaks?"

# 5. The compounding demo — two runs, instrumented
uv run steward demo
```

Every command takes `--fake` to run against the in-memory catalog with no
Docker at all.

## Why there's an adapter layer

`src/steward/adapter.py` is the only module that knows DataHub's API exists.
Everything above it speaks in the plain dataclasses in `models.py`, against the
`Catalog` protocol in `catalog.py`.

That boundary earns its keep three ways:

1. **The SDK is moving.** DataHub 1.7's `datahub.sdk` package emits an explicit
   *"experimental — backwards-compatibility guarantees do not apply"* warning on
   import. Steward deliberately builds on the stable trio every first-party
   ingestion source uses instead: `DataHubGraph`, `MetadataChangeProposalWrapper`,
   and the generated aspect classes, plus GraphQL for traversal.
2. **Swapping transports is one file.** Moving to the MCP Server or Agent
   Context Kit means reimplementing `Catalog`; no agent, tool or test changes.
3. **CI runs with no Docker.** `FakeCatalog` implements the same protocol from
   the same scenario definition, so the suite runs in ~0.02s on a clean runner —
   and `steward parity` proves the fake hasn't drifted from the real thing.

## Three things live testing caught

Worth recording, because none would have surfaced from reading docs:

- **GraphQL fragment merging.** `properties.name` has different nullability on
  `Dataset` vs `Dashboard`, so requesting it across inline fragments is a hard
  validation error. Every fragment's fields have to be aliased.
- **Degree filters are an enum.** `searchAcrossLineage` accepts `"1"`, `"2"`,
  `"3+"` — not arbitrary integers. Passing `"3"` returns a 400.
- **ML lineage routes through a training job.** `MLModelProperties` has no
  upstream-dataset field; models connect to data via `trainingJobs`. The
  scenario models it that way because that's how DataHub — and reality — works.

## Findings

A finding is the unit of knowledge:

```python
Finding(
    finding_id="blast_radius-amount-cents-feeds-production-churn",
    subject_urn="urn:li:dataset:(...,raw.stripe.charges,PROD)",
    severity="critical",
    title="amount_cents feeds the production churn model, 4 hops downstream",
    body="...",
    evidence_urns=[...],   # what the agent actually inspected
    built_on=[...],        # prior findings this stands on
)
```

Findings are stored as `institutionalMemory` links on the subject entity, and
the entity is tagged `steward-reviewed` so the catalog itself indexes what the
agents have looked at. Writes are read-modify-write — a blind aspect write would
silently delete human-added links — and keyed by `finding_id`, so a re-run
updates rather than duplicates.

`evidence_urns` is what makes a finding auditable rather than an assertion, and
`built_on` is what makes the compounding visible.

## Development

```bash
uv run pytest          # 13 tests, no Docker or network required
uv run steward parity  # requires a running DataHub
```

## Status

| Piece | State |
| --- | --- |
| DataHub adapter (read + write) | Verified against live DataHub v1.7.0 |
| Lineage traversal, both directions | Verified — 4-hop path to the production model resolves |
| Finding write-back / read-back | Verified — idempotent, tags applied |
| Seeder (15 entities) | Verified |
| Fake ↔ live parity | Verified via `steward parity` |
| Test suite | 23 passing, no Docker or network needed |
| Blast Radius crew on free local llama3.1:8b | Verified — 30 entities / 840 tokens / 9.6s |
| Root Cause crew on free local llama3.1:8b | Verified — ranks candidates, hedges appropriately |
| Knowledge compounding | Verified — second run 18 entities / 281 tokens / 1.1s |
| Web UI with live stage streaming | Verified locally |
| Groq provider | Implemented; deploy path untested end-to-end |
| Anthropic provider | Implemented, not exercised (no key on this machine) |

## Licence

[MIT](LICENSE).
