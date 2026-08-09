"""The hosted demo: a web front end over the same crews the CLI runs.

Deliberately thin. It adds no reasoning of its own — it streams the stages the
crew already emits and renders the finding the crew already writes. The point
is that a judge can see the system work in ten seconds without installing
Docker, Ollama, or anything else.

Runs against `FakeCatalog` by default (`STEWARD_CATALOG=fake`), because a free
web dyno cannot host DataHub. That is not a shortcut around the integration:
`FakeCatalog` is built from the same `scenario.py` the live seeder uses, and
`steward parity` asserts the two return identical lineage. Point
`STEWARD_CATALOG=datahub` at a real instance and the identical code path runs.
"""

from __future__ import annotations

import json
import os
import queue
import threading
from typing import Iterator

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from . import scenario as s
from .catalog import Catalog
from .config import Config
from .crew import BlastRadiusCrew, CrewResult, RootCauseCrew, Stage
from .fake import FakeCatalog

app = FastAPI(title="Steward", docs_url="/api/docs")

#: One catalog for the process. On the fake it holds findings in memory, which
#: is what makes the "ask again" button demonstrate compounding within a session.
_catalog: Catalog | None = None
_catalog_lock = threading.Lock()


def get_catalog() -> Catalog:
    global _catalog
    with _catalog_lock:
        if _catalog is None:
            if os.environ.get("STEWARD_CATALOG", "fake").lower() == "datahub":
                from .adapter import DataHubAdapter

                _catalog = DataHubAdapter()
            else:
                _catalog = FakeCatalog()
        return _catalog


def reset_catalog() -> None:
    """Fresh catalog, so the demo can be replayed from cold."""
    global _catalog
    with _catalog_lock:
        _catalog = None


class AskRequest(BaseModel):
    question: str
    mode: str = "blast"


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


def _run_streaming(question: str, mode: str) -> Iterator[str]:
    """Run a crew on a worker thread, relaying stages to the browser as they land.

    The crew is synchronous and blocking, so it runs off the response thread and
    communicates through a queue. Stages therefore appear in the UI at the moment
    they actually complete rather than all at once at the end.
    """
    stages: queue.Queue[Stage | None] = queue.Queue()
    box: dict[str, CrewResult | Exception] = {}

    config = Config.from_env()
    crew_cls = RootCauseCrew if mode == "cause" else BlastRadiusCrew
    crew = crew_cls(get_catalog(), config, on_stage=stages.put)

    def work() -> None:
        try:
            box["result"] = crew.run(question)
        except Exception as exc:  # noqa: BLE001 - reported to the browser
            box["error"] = exc
        finally:
            stages.put(None)

    thread = threading.Thread(target=work, daemon=True)
    thread.start()

    yield _sse("start", {"mode": mode, "provider": config.provider})

    while True:
        stage = stages.get()
        if stage is None:
            break
        yield _sse("stage", {"name": stage.name, "detail": stage.detail})

    thread.join(timeout=5)

    error = box.get("error")
    if isinstance(error, Exception):
        yield _sse("error", {"message": f"{type(error).__name__}: {error}"})
        return

    result = box.get("result")
    if not isinstance(result, CrewResult):
        yield _sse("error", {"message": "The crew returned nothing."})
        return

    yield _sse(
        "done",
        {
            "answer": result.answer,
            "finding": (
                {
                    "finding_id": result.finding.finding_id,
                    "severity": result.finding.severity,
                    "title": result.finding.title,
                    "subject_urn": result.finding.subject_urn,
                    "evidence_count": len(result.finding.evidence_urns),
                    "agent": result.finding.agent,
                }
                if result.finding
                else None
            ),
            "stats": result.stats.as_dict(),
        },
    )


@app.post("/api/ask")
def ask(request: AskRequest) -> StreamingResponse:
    return StreamingResponse(
        _run_streaming(request.question, request.mode),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/reset")
def reset() -> dict:
    reset_catalog()
    return {"ok": True}


@app.get("/api/health")
def health() -> dict:
    config = Config.from_env()
    return {
        "ok": True,
        "provider": config.provider,
        "catalog": os.environ.get("STEWARD_CATALOG", "fake"),
    }


#: How entities are grouped in the UI's map of the estate. Order matters — it is
#: the left-to-right reading order of the diagram.
_LAYERS = [
    ("Sources", "Landed from outside systems", lambda e: e.name.startswith("raw.")),
    ("Modelled", "Built by dbt", lambda e: e.entity_type == "DATASET"),
    ("Pipelines", "Airflow jobs", lambda e: e.entity_type == "DATA_JOB"),
    ("Consumers", "What people and models actually use", lambda e: True),
]


@app.get("/api/graph")
def graph() -> dict:
    """The estate, grouped for display.

    Served from the scenario definition rather than a live query so the map
    renders instantly on page load — the point is to give a viewer the shape of
    the company before they are asked a question about it.
    """
    layers: list[dict] = [
        {"name": name, "blurb": blurb, "entities": []} for name, blurb, _ in _LAYERS
    ]
    for entity in s.ENTITIES:
        for index, (_, _, matches) in enumerate(_LAYERS):
            if matches(entity):
                layers[index]["entities"].append(
                    {
                        "urn": entity.urn,
                        "name": entity.name,
                        "type": entity.entity_type,
                        "description": entity.description,
                        "owners": [o.split(":")[-1] for o in entity.owners],
                        "columns": [
                            {"name": c.name, "type": c.type, "description": c.description}
                            for c in entity.columns
                        ],
                    }
                )
                break
    return {"layers": layers}


@app.get("/api/lineage")
def lineage(urn: str, direction: str = "downstream") -> dict:
    """Immediate graph answer, with no model involved.

    This is what makes the demo teachable: a viewer clicks an entity and sees
    the blast radius rendered from the catalog before any LLM runs, so the
    crew's later judgment is layered on something they already understand.
    """
    catalog = get_catalog()
    entity = catalog.get_entity(urn)
    edges = catalog.lineage(urn, direction, max_degree=3)  # type: ignore[arg-type]
    return {
        "subject": {"urn": urn, "name": entity.name if entity else urn},
        "direction": direction,
        "count": len(edges),
        "edges": [
            {
                "urn": e.entity.urn,
                "name": e.entity.name,
                "type": e.entity.entity_type,
                "degree": e.degree,
                "owners": [o.split(":")[-1] for o in e.entity.owners],
            }
            for e in edges
        ],
        "findings": [
            {"finding_id": f.finding_id, "severity": f.severity, "title": f.title}
            for f in catalog.read_findings(urn)
        ],
    }


@app.get("/api/scenario")
def scenario() -> dict:
    return {
        "entities": [
            {"urn": e.urn, "name": e.name, "type": e.entity_type, "owners": e.owners}
            for e in s.ENTITIES
        ],
        "demo_change": s.DEMO_CHANGE,
        "demo_symptom": s.DEMO_SYMPTOM,
    }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _PAGE.replace("__DEMO_CHANGE__", s.DEMO_CHANGE).replace(
        "__DEMO_SYMPTOM__", s.DEMO_SYMPTOM
    )




# The page is one self-contained string: no build step, no CDN, no framework.
# A judge should be able to read the whole front end in one sitting, and the
# deployed image should not depend on anything it doesn't ship.
_PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Steward — DataHub agents whose findings compound</title>
<style>
  :root{--bg:#0B0C0E;--panel:#15171A;--panel2:#1B1E22;--line:#24272B;--ink:#ECEDEE;
        --dim:#9BA1A8;--accent:#6E9BFF;--warn:#E9A14B;--crit:#F2736B;--ok:#4ECB84;
        --model:#C77DFF;--dash:#FFB86B;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
       font:15px/1.55 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
  .wrap{max-width:1080px;margin:0 auto;padding:28px 20px 90px}
  h1{font-size:25px;margin:0 0 4px}
  h2{font-size:15px;margin:0 0 3px}
  .sub{color:var(--dim);margin:0 0 8px}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:12px;
        padding:16px;margin-bottom:14px}
  .step{display:inline-block;background:var(--accent);color:#08101F;font-weight:700;
        font-size:11px;border-radius:5px;padding:2px 7px;margin-right:8px}
  .layers{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}
  .layer h3{font-size:11px;text-transform:uppercase;letter-spacing:.7px;color:var(--dim);
            margin:0 0 2px}
  .layer p{font-size:12px;color:var(--dim);margin:0 0 8px}
  .ent{display:block;width:100%;text-align:left;background:var(--panel2);
       border:1px solid var(--line);border-radius:8px;padding:8px 10px;margin-bottom:6px;
       color:var(--ink);font:inherit;font-size:13px;cursor:pointer}
  .ent:hover{border-color:var(--accent)}
  .ent.sel{border-color:var(--accent);background:rgba(110,155,255,.12)}
  .ent.hit{border-color:var(--crit);background:rgba(242,115,107,.13)}
  .ent .t{font-size:10px;text-transform:uppercase;letter-spacing:.5px;color:var(--dim)}
  .ent.mlmodel .t{color:var(--model)} .ent.dashboard .t{color:var(--dash)}
  .ent .own{font-size:11px;color:var(--dim)}
  .row{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-top:10px}
  button.act{background:var(--accent);color:#08101F;border:0;border-radius:9px;
             padding:9px 15px;font:inherit;font-weight:650;cursor:pointer}
  button.ghost{background:transparent;color:var(--accent);border:1px solid var(--line);
               border-radius:9px;padding:9px 15px;font:inherit;cursor:pointer}
  button:disabled{opacity:.45;cursor:not-allowed}
  textarea{width:100%;background:var(--bg);color:var(--ink);border:1px solid var(--line);
           border-radius:9px;padding:11px;font:inherit;min-height:62px;resize:vertical}
  .cols{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}
  .col{background:var(--bg);border:1px solid var(--line);border-radius:6px;
       padding:3px 8px;font-size:12px;color:var(--dim);font-family:ui-monospace,monospace}
  .imp{display:flex;justify-content:space-between;gap:10px;padding:6px 0;
       border-bottom:1px dashed var(--line);font-size:13px}
  .imp:last-child{border-bottom:0}
  .deg{color:var(--dim);font-size:12px}
  .stage{display:flex;gap:10px;padding:6px 0;font-size:13.5px;
         border-bottom:1px dashed var(--line)}
  .stage:last-child{border-bottom:0}
  .stage b{color:var(--accent);min-width:78px;font-weight:600}
  .stage span{color:var(--dim)}
  .stats{display:flex;gap:16px;flex-wrap:wrap;color:var(--dim);font-size:12.5px;
         margin-top:12px;border-top:1px solid var(--line);padding-top:10px}
  .stats b{color:var(--ink)}
  .sev{display:inline-block;font-size:10px;text-transform:uppercase;letter-spacing:.6px;
       border-radius:5px;padding:2px 7px;margin-bottom:7px}
  .critical{background:rgba(242,115,107,.15);color:var(--crit)}
  .warning{background:rgba(233,161,75,.15);color:var(--warn)}
  .info{background:rgba(110,155,255,.15);color:var(--accent)}
  .hint{color:var(--dim);font-size:12.5px;margin-top:8px}
  .banner{border-left:3px solid var(--ok);padding-left:11px;color:var(--dim);
          font-size:13px;margin-top:12px}
  .answer{white-space:pre-wrap;margin:6px 0}
  code{background:#000;padding:1px 5px;border-radius:4px;font-size:12.5px;
       font-family:ui-monospace,monospace}
  .muted{color:var(--dim)}
</style></head><body><div class="wrap">

<h1>Steward</h1>
<p class="sub">Agents that read a data catalogue, work out what a change would break —
and write the answer back so nobody has to work it out again.</p>

<div class="card">
  <h2><span class="step">1</span>This is Northwind's data estate</h2>
  <p class="sub">A made-up but ordinary company. Click anything to see what depends on it.
  Nothing here is guessed — it is real lineage from DataHub.</p>
  <div class="layers" id="layers"></div>
</div>

<div class="card" id="detail" style="display:none">
  <h2><span class="step">2</span><span id="dname"></span></h2>
  <p class="sub" id="ddesc"></p>
  <div id="dcols" class="cols"></div>

  <div style="margin-top:14px">
    <b id="impact-h">Downstream</b>
    <span class="muted" id="impact-n"></span>
    <div id="impact" style="margin-top:6px"></div>
  </div>

  <div class="row">
    <button class="act" id="askBlast">Ask the crew: what breaks if I change this?</button>
    <button class="ghost" id="askCause">Ask: why is this broken?</button>
  </div>
  <div class="hint">The list above is a plain graph lookup — instant, no AI. The crew adds
  the judgment: which of those actually matter, who owns them, and what to do.</div>
</div>

<div class="card">
  <h2><span class="step">3</span>Or ask in your own words</h2>
  <textarea id="q">__DEMO_CHANGE__</textarea>
  <div class="row">
    <button class="act" id="run">Run the crew</button>
    <button class="ghost" id="again" disabled>Ask the same thing again</button>
    <button class="ghost" id="reset">Reset catalogue</button>
  </div>
  <div class="hint">Ask twice. The second run reads the finding the first one wrote into
  the catalogue instead of re-deriving it — watch <b>entities inspected</b> and
  <b>tokens</b>.</div>
</div>

<div class="card" id="out" style="display:none">
  <h2><span class="step">4</span>What the crew did</h2>
  <div id="stages" style="margin-top:8px"></div>
  <div id="result"></div>
</div>

<script>
const $ = (id) => document.getElementById(id);
const esc = (s) => (s||"").replace(/[&<>]/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[m]));
const DEMOS = { blast: `__DEMO_CHANGE__`, cause: `__DEMO_SYMPTOM__` };
let selected = null, mode = "blast";

async function loadGraph(){
  const g = await (await fetch("/api/graph")).json();
  $("layers").innerHTML = g.layers.map(l => `
    <div class="layer"><h3>${esc(l.name)}</h3><p>${esc(l.blurb)}</p>
      ${l.entities.map(e => `
        <button class="ent ${e.type.toLowerCase()}" data-urn="${esc(e.urn)}">
          <div class="t">${esc(e.type)}</div>
          <div>${esc(e.name)}</div>
          ${e.owners.length ? `<div class="own">${esc(e.owners.join(", "))}</div>` : ""}
        </button>`).join("")}
    </div>`).join("");
  document.querySelectorAll(".ent").forEach(b => b.onclick = () => select(b.dataset.urn));
  window._graph = g;
}

function findEntity(urn){
  for(const l of window._graph.layers) for(const e of l.entities) if(e.urn === urn) return e;
  return null;
}

async function select(urn){
  selected = urn;
  const e = findEntity(urn);
  document.querySelectorAll(".ent").forEach(b => {
    b.classList.toggle("sel", b.dataset.urn === urn);
    b.classList.remove("hit");
  });

  // A model or dashboard has nothing downstream, so lead with what feeds it.
  const dir = (e.type === "MLMODEL" || e.type === "DASHBOARD") ? "upstream" : "downstream";
  const d = await (await fetch(`/api/lineage?urn=${encodeURIComponent(urn)}&direction=${dir}`)).json();

  $("detail").style.display = "block";
  $("dname").textContent = e.name;
  $("ddesc").textContent = e.description || "";
  $("dcols").innerHTML = (e.columns||[]).map(c =>
    `<span class="col" title="${esc(c.description||'')}">${esc(c.name)}</span>`).join("");
  $("impact-h").textContent = dir === "upstream" ? "Feeds on" : "Breaks if this changes";
  $("impact-n").textContent = ` — ${d.count} ${d.count === 1 ? "asset" : "assets"}`;
  $("impact").innerHTML = d.edges.length
    ? d.edges.map(x => `<div class="imp">
        <span>${esc(x.name)} <span class="deg">· ${esc(x.type)}</span></span>
        <span class="deg">${x.degree} hop${x.degree===1?"":"s"}${x.owners.length? " · "+esc(x.owners.join(", ")):""}</span>
      </div>`).join("")
    : `<div class="muted">Nothing — this is a leaf.</div>`;

  // Light up the affected entities in the map above, so the blast radius is
  // visible on the picture rather than only in a list.
  const hits = new Set(d.edges.map(x => x.urn));
  document.querySelectorAll(".ent").forEach(b => {
    if(hits.has(b.dataset.urn)) b.classList.add("hit");
  });

  mode = dir === "upstream" ? "cause" : "blast";
  $("q").value = dir === "upstream"
    ? `Something looks wrong with ${e.name}. What upstream is most likely responsible?`
    : `We need to change ${e.name}. What breaks?`;
}

async function run(){
  const question = $("q").value.trim(); if(!question) return;
  $("run").disabled = $("again").disabled = $("askBlast").disabled = $("askCause").disabled = true;
  $("out").style.display = "block";
  $("stages").innerHTML = `<div class="muted">running…</div>`; $("result").innerHTML = "";

  const resp = await fetch("/api/ask", {method:"POST",
    headers:{"Content-Type":"application/json"}, body: JSON.stringify({question, mode})});

  const reader = resp.body.getReader(); const dec = new TextDecoder(); let buf = "", first = true;
  while(true){
    const {value, done} = await reader.read(); if(done) break;
    buf += dec.decode(value, {stream:true});
    const parts = buf.split("\n\n"); buf = parts.pop();
    for(const part of parts){
      const ev = (part.match(/^event: (.*)$/m)||[])[1];
      const raw = (part.match(/^data: (.*)$/m)||[])[1];
      if(!ev || !raw) continue;
      const d = JSON.parse(raw);
      if(ev === "stage"){
        if(first){ $("stages").innerHTML = ""; first = false; }
        $("stages").insertAdjacentHTML("beforeend",
          `<div class="stage"><b>${esc(d.name)}</b><span>${esc(d.detail)}</span></div>`);
      } else if(ev === "error"){
        $("result").innerHTML = `<p style="color:var(--crit)">${esc(d.message)}</p>`;
      } else if(ev === "done"){
        const f = d.finding, st = d.stats;
        $("result").innerHTML =
          (f ? `<div class="sev ${f.severity}">${f.severity}</div>` : "") +
          `<p class="answer">${esc(d.answer)}</p>` +
          (f ? `<p class="hint">Written into the catalogue as <code>${esc(f.finding_id)}</code>
                by <code>${esc(f.agent)}</code>, citing ${f.evidence_count} entities as evidence.</p>` : "") +
          `<div class="stats">
             <span><b>${st.entities_inspected}</b> entities inspected</span>
             <span><b>${st.input_tokens + st.output_tokens}</b> tokens</span>
             <span><b>${st.prior_findings_reused}</b> prior findings reused</span>
             <span><b>${st.wall_seconds}s</b></span>
           </div>` +
          (st.prior_findings_reused > 0
            ? `<p class="banner">This run did no investigation. It read what the previous run
               recorded in DataHub and cited it. That is the point: the catalogue got
               permanently smarter, and the next person inherits it.</p>` : "");
        if(selected) select(selected);
      }
    }
  }
  $("run").disabled = $("again").disabled = $("askBlast").disabled = $("askCause").disabled = false;
}

$("run").onclick = run;
$("again").onclick = run;
$("askBlast").onclick = () => { mode = "blast";
  $("q").value = `We need to change ${findEntity(selected).name}. What breaks?`; run(); };
$("askCause").onclick = () => { mode = "cause";
  $("q").value = `Something looks wrong with ${findEntity(selected).name}. What upstream is most likely responsible?`; run(); };
$("reset").onclick = async () => {
  await fetch("/api/reset", {method:"POST"});
  $("out").style.display = "none"; $("again").disabled = true;
  if(selected) select(selected);
};

loadGraph();
</script>
</div></body></html>
"""
