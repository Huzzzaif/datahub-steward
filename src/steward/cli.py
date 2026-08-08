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
import sys

from . import scenario as s
from .adapter import DataHubAdapter
from .agent import blast_radius_agent, root_cause_agent
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


def _run_agent(kind: str, question: str, use_fake: bool) -> int:
    catalog = _catalog(use_fake)
    agent = (blast_radius_agent if kind == "blast" else root_cause_agent)(catalog)
    result = agent.run(question)
    print(result.answer)
    _print_stats(kind, result.stats)
    return 0


def cmd_blast(args: argparse.Namespace) -> int:
    return _run_agent("blast", args.question, args.fake)


def cmd_cause(args: argparse.Namespace) -> int:
    return _run_agent("cause", args.question, args.fake)


def cmd_demo(args: argparse.Namespace) -> int:
    """The point of the whole project, in two runs.

    Run one investigates cold. Run two asks a related question against a
    catalog that now contains run one's finding — and the instrumentation shows
    it doing less work to get there.
    """
    catalog = _catalog(args.fake)

    print("=" * 72)
    print("RUN 1 — cold catalog, nothing recorded yet")
    print("=" * 72)
    first = blast_radius_agent(catalog).run(s.DEMO_CHANGE)
    print(first.answer)
    _print_stats("run 1", first.stats)

    print("\n" + "=" * 72)
    print("RUN 2 — same catalog, now carrying run 1's finding")
    print("=" * 72)
    second = root_cause_agent(catalog).run(
        "The churn_predictor model's scores shifted sharply last night. "
        "The payments team mentioned they were changing something. What happened?"
    )
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
    add("blast", cmd_blast, "what breaks if I make this change?", needs_question=True)
    add("cause", cmd_cause, "why is this broken?", needs_question=True)
    add("demo", cmd_demo, "two-run knowledge-compounding demo")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
