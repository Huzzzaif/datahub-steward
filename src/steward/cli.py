"""Command line entry point.

    steward seed              load the demo estate into a running DataHub
    steward parity            check the in-memory fake and live DataHub agree
    steward blast "<change>"  what breaks if I make this change?
    steward cause  "<symptom>" why is this broken?
    steward demo              the compounding-knowledge demo, two runs
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from . import scenario as s
from .adapter import DataHubAdapter
from .agent import blast_radius_agent, root_cause_agent
from .crew import BlastRadiusCrew
from .llm import build_provider
from .catalog import Catalog
from .config import Config
from .fake import FakeCatalog
from .models import RunStats


def _catalog(use_fake: bool) -> Catalog:
    if use_fake:
        return FakeCatalog()
    adapter = DataHubAdapter()
    if not adapter.ping():
        sys.exit(
            "DataHub is not reachable at "
            f"{adapter.config.datahub_server}.\n"
            "Start it with `datahub docker quickstart`, or pass --fake to run "
            "against the in-memory catalog."
        )
    return adapter


def _print_stats(label: str, stats: RunStats) -> None:
    print(
        f"\n  [{label}] {stats.tool_calls} tool calls | "
        f"{stats.entities_inspected} entities inspected | "
        f"{stats.prior_findings_reused} prior findings reused | "
        f"{stats.input_tokens + stats.output_tokens} tokens | "
        f"{stats.wall_seconds}s"
    )


def cmd_seed(args: argparse.Namespace) -> int:
    from .seed import seed

    count = seed()
    print(f"Seeded {count} entities into DataHub.")
    print("Browse them at http://localhost:9002 (datahub / datahub)")
    return 0


def cmd_parity(args: argparse.Namespace) -> int:
    """Assert the fake and the live instance describe the same graph.

    The test suite runs against the fake for speed; this is the check that the
    fake has not drifted from what DataHub actually returns.
    """
    fake = FakeCatalog()
    live = DataHubAdapter()
    if not live.ping():
        sys.exit("DataHub unreachable — start it with `datahub docker quickstart`.")

    failures = 0
    for urn, label in [
        (s.RAW_CHARGES, "downstream of raw.stripe.charges"),
        (s.MODEL_CHURN, "upstream of churn_predictor"),
    ]:
        direction = "downstream" if urn == s.RAW_CHARGES else "upstream"
        fake_set = {e.entity.urn for e in fake.lineage(urn, direction)}
        live_set = {e.entity.urn for e in live.lineage(urn, direction)}

        if fake_set == live_set:
            print(f"  match   {label} ({len(fake_set)} entities)")
        else:
            failures += 1
            print(f"  DIFFER  {label}")
            for missing in sorted(fake_set - live_set):
                print(f"            only in fake: {missing}")
            for extra in sorted(live_set - fake_set):
                print(f"            only in live: {extra}")

    print("\nparity: OK" if not failures else f"\nparity: {failures} mismatch(es)")
    return 1 if failures else 0


def _check_provider(config: Config) -> None:
    """Fail early and helpfully rather than deep inside a tool loop."""
    provider = build_provider(config)
    check = getattr(provider, "available", None)
    if check is None:
        return
    ok, why = check()
    if not ok:
        sys.exit(why)


def _run_agent(kind: str, question: str, use_fake: bool, single: bool) -> int:
    config = Config.from_env()
    _check_provider(config)
    catalog = _catalog(use_fake)

    # The crew is the default because it holds up on a small local model; the
    # single free-form agent is better on a frontier one.
    if kind == "blast" and not single:
        result = BlastRadiusCrew(catalog, config).run(question)
        for stage in result.trace:
            print(f"  · {stage.name}: {stage.detail}")
        print()
        print(result.answer)
        _print_stats(f"{kind} crew ({config.provider})", result.stats)
        return 0

    agent = (blast_radius_agent if kind == "blast" else root_cause_agent)(catalog, config)
    result = agent.run(question)
    print(result.answer)
    _print_stats(f"{kind} ({config.provider})", result.stats)
    return 0


def cmd_blast(args: argparse.Namespace) -> int:
    return _run_agent("blast", args.question, args.fake, getattr(args, "single", False))


def cmd_cause(args: argparse.Namespace) -> int:
    return _run_agent("cause", args.question, args.fake, single=True)


def cmd_serve(args: argparse.Namespace) -> int:
    """Run the web demo.

    Defaults to the in-memory catalog so this works with nothing else running;
    set STEWARD_CATALOG=datahub to point the same UI at a live instance.
    """
    import uvicorn

    # Fake by default so `steward serve` works with nothing else running; opt in
    # to the real thing explicitly.
    os.environ["STEWARD_CATALOG"] = "datahub" if args.datahub else "fake"
    config = Config.from_env()
    _check_provider(config)

    port = int(os.environ.get("PORT", args.port))
    print(f"Steward UI on http://localhost:{port}")
    print(f"  provider: {config.provider}   catalog: {os.environ['STEWARD_CATALOG']}")
    uvicorn.run("steward.web:app", host="0.0.0.0", port=port, log_level="warning")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """The point of the whole project, in two runs.

    Run one investigates cold. Run two asks a related question against a
    catalog that now contains run one's finding — and the instrumentation shows
    it doing less work to get there.
    """
    config = Config.from_env()
    _check_provider(config)
    catalog = _catalog(args.fake)

    print("=" * 72)
    print("RUN 1 — cold catalog, nothing recorded yet")
    print("=" * 72)
    first = BlastRadiusCrew(catalog, config).run(s.DEMO_CHANGE)
    for stage in first.trace:
        print(f"  · {stage.name}: {stage.detail}")
    print()
    print(first.answer)
    _print_stats("run 1", first.stats)

    print("\n" + "=" * 72)
    print("RUN 2 — same question, catalog now carries run 1's finding")
    print("=" * 72)
    second = BlastRadiusCrew(catalog, config).run(s.DEMO_CHANGE)
    for stage in second.trace:
        print(f"  · {stage.name}: {stage.detail}")
    print()
    print(second.answer)
    _print_stats("run 2", second.stats)

    print("\n" + "-" * 72)
    print(
        f"Run 2 reused {second.stats.prior_findings_reused} prior finding(s) "
        f"and inspected {second.stats.entities_inspected} entities "
        f"vs {first.stats.entities_inspected} in run 1."
    )
    print("The knowledge is in DataHub now, not in this process.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="steward", description=__doc__)
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name, handler, help_text, needs_question=False):
        p = sub.add_parser(name, help=help_text)
        if needs_question:
            p.add_argument("question")
        p.add_argument(
            "--fake",
            action="store_true",
            help="use the in-memory catalog instead of a live DataHub",
        )
        p.set_defaults(func=handler)
        return p

    sub.add_parser("seed", help="load the demo estate into DataHub").set_defaults(
        func=cmd_seed, fake=False
    )
    sub.add_parser("parity", help="check the fake matches live DataHub").set_defaults(
        func=cmd_parity, fake=False
    )
    blast = add("blast", cmd_blast, "what breaks if I make this change?", needs_question=True)
    blast.add_argument(
        "--single",
        action="store_true",
        help="use the single free-form agent instead of the crew",
    )
    add("cause", cmd_cause, "why is this broken?", needs_question=True)
    add("demo", cmd_demo, "two-run knowledge-compounding demo")

    serve = sub.add_parser("serve", help="run the web demo")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument(
        "--datahub",
        action="store_true",
        help="use a live DataHub instead of the in-memory catalog",
    )
    serve.set_defaults(func=cmd_serve, fake=False)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
