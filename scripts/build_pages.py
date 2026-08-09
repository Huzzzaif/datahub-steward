"""Build the static GitHub Pages demo into `docs/`.

GitHub Pages serves files, not processes — so the parts that need a server are
handled differently rather than faked:

* **The map and its lineage are fully live.** Adjacency is exported as JSON and
  traversed in the browser, so clicking an entity and watching its blast radius
  spread is the real graph, computed on the spot.
* **Agent runs are recorded, not simulated.** This script executes the actual
  crews against the actual catalog and captures what they produced — stages,
  wording, token counts, timings. The page replays those. The UI says
  "recorded run" wherever it shows one, and links to the live deployment.

Run it with a provider configured:

    STEWARD_PROVIDER=groq uv run python scripts/build_pages.py
"""

from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from steward import scenario as s  # noqa: E402
from steward.config import Config  # noqa: E402
from steward.crew import BlastRadiusCrew, RootCauseCrew  # noqa: E402
from steward.fake import FakeCatalog  # noqa: E402
from steward.web import _LAYERS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

LIVE_URL = "https://github.com/Huzzzaif/datahub-steward#try-it-in-ten-seconds"


def build_graph() -> dict:
    layers: list[dict] = [
        {"name": t, "plain_name": p, "blurb": tb, "plain_blurb": pb, "entities": []}
        for t, p, tb, pb, _ in _LAYERS
    ]
    for entity in s.ENTITIES:
        for index, (_, _, _, _, matches) in enumerate(_LAYERS):
            if matches(entity):
                plain_name, plain_desc, why = s.plain(entity.urn)
                layers[index]["entities"].append(
                    {
                        "urn": entity.urn,
                        "name": entity.name,
                        "plain_name": plain_name,
                        "type": entity.entity_type,
                        "description": entity.description or "",
                        "plain_description": plain_desc,
                        "why": why,
                        "owners": [o.split(":")[-1] for o in entity.owners],
                        "columns": [c.name for c in entity.columns],
                    }
                )
                break
    return {"layers": layers, "stakes": s.PLAIN_STAKES}


def build_adjacency() -> dict:
    """Both directions, so the browser can walk the graph without a server."""
    down: dict[str, list[str]] = {}
    up: dict[str, list[str]] = {}
    for entity in s.ENTITIES:
        parents = list(entity.upstreams) + list(entity.training_jobs)
        up[entity.urn] = parents
        for parent in parents:
            down.setdefault(parent, []).append(entity.urn)
    return {"down": down, "up": up}


def _bfs(adj: dict[str, list[str]], start: str) -> list[dict]:
    """Mirrors FakeCatalog.lineage, so the static page agrees with the real one."""
    seen = {start}
    queue = deque([(start, 0)])
    out: list[dict] = []
    while queue:
        current, depth = queue.popleft()
        for neighbour in adj.get(current, []):
            if neighbour in seen:
                continue
            seen.add(neighbour)
            out.append({"urn": neighbour, "degree": depth + 1})
            queue.append((neighbour, depth + 1))
    return sorted(out, key=lambda e: e["degree"])


def record_runs() -> dict:
    """Execute the real crews and capture what they actually produced."""
    config = Config.from_env()
    catalog = FakeCatalog()
    runs: dict[str, dict] = {}

    def capture(name: str, crew, question: str) -> None:
        print(f"  recording {name} ...", flush=True)
        result = crew.run(question)
        runs[name] = {
            "question": question,
            "stages": [{"name": st.name, "detail": st.detail} for st in result.trace],
            "answer": result.answer,
            "stats": result.stats.as_dict(),
            "finding": (
                {
                    "finding_id": result.finding.finding_id,
                    "severity": result.finding.severity,
                    "evidence_count": len(result.finding.evidence_urns),
                }
                if result.finding
                else None
            ),
            "provider": config.provider,
        }

    # Order matters: the cold run must come first so the warm one has a finding
    # to reuse. That pair is the whole point of the demo.
    capture("blast_cold", BlastRadiusCrew(catalog, config), s.DEMO_CHANGE)
    capture("blast_warm", BlastRadiusCrew(catalog, config), s.DEMO_CHANGE)
    capture("cause", RootCauseCrew(catalog, config), s.DEMO_SYMPTOM)
    return runs


def main() -> int:
    DOCS.mkdir(exist_ok=True)

    print("building graph ...")
    graph = build_graph()
    adjacency = build_adjacency()

    print("recording real agent runs ...")
    runs = record_runs()

    payload = {
        "graph": graph,
        "adjacency": adjacency,
        "runs": runs,
        "live_url": LIVE_URL,
        "demo_change": s.DEMO_CHANGE,
        "demo_symptom": s.DEMO_SYMPTOM,
    }
    (DOCS / "data.json").write_text(json.dumps(payload, indent=1))

    template = (Path(__file__).parent / "pages_template.html").read_text()
    (DOCS / "index.html").write_text(template)
    (DOCS / ".nojekyll").write_text("")

    print(f"\nwrote {DOCS/'index.html'} and {DOCS/'data.json'}")
    for name, run in runs.items():
        st = run["stats"]
        print(
            f"  {name:<12} {len(run['stages'])} stages | "
            f"{st['entities_inspected']} entities | "
            f"{st['input_tokens'] + st['output_tokens']} tokens | {st['wall_seconds']}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
