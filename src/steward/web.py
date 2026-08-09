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


_PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Steward — DataHub agents whose findings compound</title>
<style>
  :root{--bg:#0B0C0E;--panel:#15171A;--line:#24272B;--ink:#ECEDEE;--dim:#9BA1A8;
        --accent:#6E9BFF;--warn:#E9A14B;--crit:#F2736B;--ok:#4ECB84;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
       font:15px/1.55 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
  .wrap{max-width:860px;margin:0 auto;padding:32px 20px 80px}
  h1{font-size:26px;margin:0 0 6px}
  .sub{color:var(--dim);margin:0 0 26px}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:12px;
        padding:18px;margin-bottom:16px}
  textarea{width:100%;background:var(--bg);color:var(--ink);border:1px solid var(--line);
           border-radius:9px;padding:12px;font:inherit;min-height:76px;resize:vertical}
  .row{display:flex;gap:10px;flex-wrap:wrap;margin-top:12px;align-items:center}
  button{background:var(--accent);color:#08101F;border:0;border-radius:9px;
         padding:10px 16px;font:inherit;font-weight:650;cursor:pointer}
  button.ghost{background:transparent;color:var(--accent);border:1px solid var(--line)}
  button:disabled{opacity:.45;cursor:not-allowed}
  .chips{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}
  .chip{background:transparent;border:1px solid var(--line);color:var(--dim);
        border-radius:999px;padding:6px 12px;font-size:13px;cursor:pointer}
  .chip.on{border-color:var(--accent);color:var(--accent)}
  .stage{display:flex;gap:10px;padding:7px 0;border-bottom:1px dashed var(--line);font-size:14px}
  .stage:last-child{border-bottom:0}
  .stage b{color:var(--accent);min-width:82px;font-weight:600}
  .stage span{color:var(--dim)}
  .answer{white-space:pre-wrap}
  .stats{display:flex;gap:18px;flex-wrap:wrap;color:var(--dim);font-size:13px;
         margin-top:14px;border-top:1px solid var(--line);padding-top:12px}
  .stats b{color:var(--ink);font-weight:650}
  .sev{display:inline-block;font-size:11px;text-transform:uppercase;letter-spacing:.6px;
       border-radius:5px;padding:2px 7px;margin-bottom:8px}
  .critical{background:rgba(242,115,107,.14);color:var(--crit)}
  .warning{background:rgba(233,161,75,.14);color:var(--warn)}
  .info{background:rgba(110,155,255,.14);color:var(--accent)}
  .hint{color:var(--dim);font-size:13px;margin-top:10px}
  .banner{border-left:3px solid var(--ok);padding-left:12px;color:var(--dim);font-size:13.5px}
  code{background:#000;padding:1px 5px;border-radius:4px;font-size:13px}
</style></head><body><div class="wrap">

<h1>Steward</h1>
<p class="sub">DataHub agents whose findings compound. Ask once — the answer is written
back into the catalogue, so asking again is nearly free.</p>

<div class="card">
  <div class="chips">
    <button class="chip on" data-mode="blast">Blast radius — what breaks?</button>
    <button class="chip" data-mode="cause">Root cause — why did it break?</button>
  </div>
  <textarea id="q">__DEMO_CHANGE__</textarea>
  <div class="row">
    <button id="run">Run the crew</button>
    <button id="again" class="ghost" disabled>Ask the same thing again</button>
    <button id="reset" class="ghost">Reset catalogue</button>
  </div>
  <div class="hint">Second run reads the finding the first one recorded, instead of
  re-deriving it. Watch <b>entities inspected</b> and <b>tokens</b>.</div>
</div>

<div class="card" id="out" style="display:none">
  <div id="stages"></div>
  <div id="result"></div>
</div>

<script>
let mode = "blast";
const $ = (id) => document.getElementById(id);
const DEMOS = { blast: `__DEMO_CHANGE__`, cause: `__DEMO_SYMPTOM__` };

document.querySelectorAll(".chip").forEach(c => c.onclick = () => {
  document.querySelectorAll(".chip").forEach(x => x.classList.remove("on"));
  c.classList.add("on");
  mode = c.dataset.mode;
  $("q").value = DEMOS[mode];
});

function esc(s){ return (s||"").replace(/[&<>]/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[m])); }

async function run(){
  const question = $("q").value.trim();
  if(!question) return;
  $("run").disabled = true; $("again").disabled = true;
  $("out").style.display = "block";
  $("stages").innerHTML = ""; $("result").innerHTML = "";

  const resp = await fetch("/api/ask", {
    method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({question, mode})
  });

  // Minimal SSE reader — the payloads are small and strictly ordered.
  const reader = resp.body.getReader(); const dec = new TextDecoder(); let buf = "";
  while(true){
    const {value, done} = await reader.read(); if(done) break;
    buf += dec.decode(value, {stream:true});
    const parts = buf.split("\\n\\n"); buf = parts.pop();
    for(const part of parts){
      const ev = (part.match(/^event: (.*)$/m)||[])[1];
      const raw = (part.match(/^data: (.*)$/m)||[])[1];
      if(!ev || !raw) continue;
      const d = JSON.parse(raw);
      if(ev === "stage"){
        $("stages").insertAdjacentHTML("beforeend",
          `<div class="stage"><b>${esc(d.name)}</b><span>${esc(d.detail)}</span></div>`);
      } else if(ev === "error"){
        $("result").innerHTML = `<p style="color:var(--crit)">${esc(d.message)}</p>`;
      } else if(ev === "done"){
        const f = d.finding, st = d.stats;
        $("result").innerHTML =
          (f ? `<div class="sev ${f.severity}">${f.severity}</div>` : "") +
          `<p class="answer">${esc(d.answer)}</p>` +
          (f ? `<p class="hint">Written to the catalogue as <code>${esc(f.finding_id)}</code>
                by <code>${esc(f.agent)}</code>, citing ${f.evidence_count} entities as evidence.</p>` : "") +
          `<div class="stats">
             <span><b>${st.entities_inspected}</b> entities inspected</span>
             <span><b>${st.input_tokens + st.output_tokens}</b> tokens</span>
             <span><b>${st.prior_findings_reused}</b> prior findings reused</span>
             <span><b>${st.wall_seconds}s</b></span>
           </div>` +
          (st.prior_findings_reused > 0
            ? `<p class="banner" style="margin-top:14px">This run did no new investigation.
               It read what the previous run recorded in DataHub. That is the whole idea:
               the catalogue got permanently smarter.</p>` : "");
      }
    }
  }
  $("run").disabled = false; $("again").disabled = false;
}

$("run").onclick = run;
$("again").onclick = run;
$("reset").onclick = async () => {
  await fetch("/api/reset", {method:"POST"});
  $("out").style.display = "none"; $("again").disabled = true;
};
</script>
</div></body></html>
"""
