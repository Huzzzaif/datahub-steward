# Steward — hackathon submission

**Challenges: #1 (Agents That Do Real Work) + #4 (Open / Wildcard)**

**Try it:** *(deployed URL goes here)* · **Code:** this repo · **Local:** `uv sync && uv run steward serve`

---

## The problem

Change a column in a raw table and you find out what depended on it when
something breaks. The knowledge of what connects to what lives in a handful of
senior engineers' heads, and it leaves when they do.

DataHub already holds the map. What it doesn't hold is *conclusions* — the
reasoning someone did once about what that map implies. So every incident
starts from scratch, and the fifth person to ask "what does `amount_cents`
feed?" does the same work as the first.

## What Steward does

Two agent crews read DataHub, reach a conclusion, and **write it back into the
catalog** so nobody has to reach it again.

- **Blast Radius** — walks *downstream* from a proposed change to terminal
  consumers, names the production models and executive dashboards at risk, and
  identifies which teams own them.
- **Root Cause** — walks *upstream* from a symptom and ranks candidate causes,
  stating plainly that a ranking is a lead and not a confirmed defect.

Findings are stored as `institutionalMemory` on the entity itself, with
`evidence_urns` (what was actually inspected) and `built_on` (which earlier
findings it stands on). The entity is tagged `steward-reviewed`, so the catalog
becomes its own index of what has been examined.

## Why this is challenge #1 and not a read-only advisor

The clause that matters in challenge #1 is the last one: *"writes results back
so the next person or agent inherits the knowledge."* Almost any catalog agent
can read metadata and produce prose. The hard part is making knowledge
**accumulate**, and making that claim checkable rather than aspirational.

So Steward instruments it. Same question, twice:

| | Entities inspected | Tokens | Wall time |
|---|---|---|---|
| **Run 1** — cold catalog | 30 | 571 | 0.7s |
| **Run 2** — after run 1 wrote its finding | 18 | 281 | **0.3s** |

Run 2 does no traversal and no assessment. It reads what run 1 left in DataHub
and cites it. The catalog is permanently smarter, and the numbers are on screen
rather than in a claim.

*(Groq `llama-3.3-70b-versatile`. The same runs on local `llama3.1:8b` take 9.6s
and 1.1s — ~10x slower, same conclusions.)*

## Small models, handled deliberately

The crews are built to run on free models, which means designing around the ways
they fail rather than hoping they don't:

- **The model never emits an identifier.** It picks an index into a list the code
  built, so a hallucinated URN is structurally impossible.
- **A named entity skips the model entirely.** If the question says
  `churn_predictor`, that's not a judgment call. This was a real bug: a run about
  the churn model was being filed against a different table.
- **Non-answers are detected and discarded.** A model that flags 12 of 12 assets
  as "high risk", or hands the candidate list back in the order it was given, has
  made no judgment. Both are caught and replaced with a deterministic heuristic.
- **Every stage has a fallback**, so a garbled reply degrades instead of crashing.

## The scenario, and why it's the interesting case

An ordinary analytics estate — Stripe and CRM raw tables, dbt models, an ML
feature table, three Looker dashboards, two MLflow models:

```
raw.stripe.charges.amount_cents
  └─ analytics.core.fct_orders            (revenue_usd derived from it)
      └─ analytics.marts.customer_ltv
          └─ analytics.features.customer_features
              └─ train_churn_predictor    (Airflow)
                  └─ churn_predictor      ← PRODUCTION MODEL, 4 hops away
```

Nothing about the raw payments table tells you it feeds a production churn
model, or the dashboard Finance closes the books with. Blast radius is invisible
exactly when it matters most.

The churn model also draws on `raw.support.tickets` — a second, independent
upstream branch — which is what makes root-cause analysis a genuine ranking
problem rather than a single-path lookup.

## Engineering decisions worth defending

**The agents never emit an identifier.** Free local models invent URNs
constantly. Every model step picks an *index* into a list the code built, so a
hallucinated identifier is structurally impossible. Every stage has a
deterministic fallback, so a garbled reply degrades instead of crashing.

**Traversal is code, not inference.** Walking a lineage graph is deterministic;
an LLM there adds only error and cost. The model does judgment — *"a production
model is scarier than an intermediate table"* — and DataHub supplies truth.

**Multi-agent, but narrow.** Handing one small model a free-form tool loop over
a graph fails specifically: invented URNs, lost thread, weak synthesis. More
agents doing that same job multiplies the failure. The crew instead gives each
step a bounded decision:

```
resolve   pick the subject from a numbered list   (LLM picks an index)
traverse  walk the graph                          (code, no model)
assess    mark high-risk assets                   (LLM picks indices)
compose   write the note                          (LLM, no tools)
record    persist to DataHub                      (code, no model)
```

**One adapter, three brains.** `adapter.py` is the only module that knows
DataHub's API exists; `llm.py` is the only one that knows which model is
running. Swapping to the MCP Server or Agent Context Kit is one file. Runs on
free local Ollama, free hosted Groq, or Claude — same code path.

**It does not write noise.** If an entity has nothing downstream, the crew
answers the question and records *nothing* — and says that absent lineage is not
proof of absent consumers. A catalog filled with vacuous findings is worse than
an empty one.

## Three bugs only live testing found

None of these would have surfaced from reading documentation:

1. **GraphQL fragment merging.** `properties.name` has different nullability on
   `Dataset` vs `Dashboard`; requesting it across inline fragments is a hard
   validation error. Every fragment needs aliasing.
2. **Degree filters are an enum.** `searchAcrossLineage` accepts `"1"`, `"2"`,
   `"3+"` — not integers. Passing `"3"` returns a 400.
3. **ML lineage routes through a training job.** `MLModelProperties` has no
   upstream-dataset field; models reach data via `trainingJobs`. The scenario
   was restructured to match, which made it more realistic too.

## What is verified, and what isn't

| Piece | State |
|---|---|
| DataHub adapter, read + write | Verified against live DataHub **v1.7.0** |
| Lineage traversal, both directions | Verified — 4-hop path to the production model resolves |
| Finding write-back / read-back | Verified — idempotent, tags applied |
| Fake ↔ live parity | Verified via `steward parity` — identical sets |
| Blast Radius crew | Verified on free local llama3.1:8b |
| Root Cause crew | Verified on free local llama3.1:8b |
| Web UI, live streaming | Verified locally |
| Test suite | 23 passing, no Docker or network required |
| Groq provider | Implemented; deploy path untested end-to-end |
| Anthropic provider | Implemented; not exercised (no key on the dev machine) |

## About the hosted demo

The deployed instance runs `FakeCatalog`, because a free web dyno cannot host
DataHub's six-container stack. This is not a way around the integration:

- `FakeCatalog` is built from the **same `scenario.py`** that seeds live DataHub.
- `steward parity` asserts both return identical lineage — 11 downstream and 8
  upstream entities, matching exactly.
- `STEWARD_CATALOG=datahub` runs the **identical code path** against a real
  instance.

To see it against real DataHub:

```bash
uv run datahub docker quickstart   # ~10 GB of images
uv run steward seed
uv run steward parity              # proves the fake matches
uv run steward serve --datahub
```

## Running it

```bash
uv sync

# Free + local: needs Ollama with llama3.1
uv run steward serve

# Free + hosted: needs a Groq key (no credit card)
export STEWARD_PROVIDER=groq GROQ_API_KEY=gsk_...
uv run steward serve

uv run pytest        # 23 tests, no Docker, no network, no API key
```
