"""Command line entry point.

  python cli.py download            # 1-minute history for the universe (+ sentiment history)
  python cli.py record              # bid/ask + order book + trades recorder (run continuously)
  python cli.py train               # walk-forward evaluation + final model
  python cli.py backtest            # backtest of record on the held-out test window
  python cli.py experiments         # compare model/label variants (walk-forward only)
  python cli.py experiment-selection  # two-stage selection model vs. EV ranking
  python cli.py fv-research         # Strategy 2: gap stats, lead-lag on recorded bid/ask
  python cli.py fv-backtest         # Strategy 2: event replay with conservative maker fills
  python cli.py fv-shadow           # Strategy 2: paper maker orders on live books (needs `record`)
  python cli.py fv-live --yes       # Strategy 2: REAL maker orders (needs key + `record`)
  python cli.py shadow              # live data, simulated fills
  python cli.py live                # real orders (needs WALLEX_API_KEY)
  python cli.py report RUN_DIR      # rebuild report for a run
  python cli.py verify-api          # read-only checks of private endpoints
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from engine.config import load_config, universe


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


def cmd_download(cfg, args) -> None:
    from engine.data.download import download_all
    from engine.sentiment import fear_greed, funding
    from engine.wallex.client import WallexClient

    client = WallexClient.from_config(cfg)
    days = args.days or cfg.data.history_days
    symbols = args.symbols or universe(cfg)
    manifest = download_all(client, cfg.data.dir, symbols, days)
    print(json.dumps(manifest, indent=2))
    if cfg.sentiment.fear_greed:
        fear_greed.download(cfg.data.dir)
    if cfg.sentiment.funding:
        funding.download(cfg.data.dir, symbols, days)


def cmd_record(cfg, args) -> None:
    from engine.data.recorder import record
    from engine.features.build import usdt_pairs

    tmn = universe(cfg)
    symbols = args.symbols or sorted(set(tmn) | {"USDTTMN"} | set(usdt_pairs(tmn)))
    record(cfg.wallex.base_url, cfg.data.dir, symbols, args.interval)


def cmd_lag_log(cfg, args) -> None:
    from engine.lag.logger import GapLogger

    GapLogger(cfg).run()


def cmd_lag_backtest(cfg, args) -> None:
    from engine.lag.backtest import run

    run(cfg, optimistic=args.optimistic)


def cmd_lag_replay(cfg, args) -> None:
    from engine.lag.replay import run

    run(cfg)


def cmd_train_seq(cfg, args) -> None:
    from engine.model.seq import train_seq

    train_seq(cfg)


def cmd_trend_backtest(cfg, args) -> None:
    from engine.trend.backtest import run

    run(cfg)


def cmd_premium_backtest(cfg, args) -> None:
    from engine.stat.premium import run

    run(cfg)


def cmd_check_net(cfg, args) -> None:
    from engine.netcheck import run

    sys.exit(run())


def cmd_train(cfg, args) -> None:
    from engine.model.train import train_and_evaluate

    train_and_evaluate(cfg)


def cmd_backtest(cfg, args) -> None:
    from engine.backtest.runner import run_backtest

    run_backtest(cfg, start=args.start, end=args.end, model_path=args.model, tag=args.tag,
                 use_agent=args.agent, windows=not args.no_windows)


def cmd_fv_research(cfg, args) -> None:
    from engine.fv.research import run_research

    run_research(cfg)


def cmd_fv_backtest(cfg, args) -> None:
    from engine.fv.sim import run_fv_backtest

    run_fv_backtest(cfg)


def cmd_fv_shadow(cfg, args) -> None:
    from engine.fv.shadow import run_shadow

    run_shadow(cfg)


def cmd_fv_live(cfg, args) -> None:
    from engine.fv.live import run_fv_live

    if not args.yes:
        print("fv-live places REAL orders. Re-run with --yes to confirm.")
        sys.exit(1)
    run_fv_live(cfg)


def cmd_experiments(cfg, args) -> None:
    from engine.model.experiments import run_experiments

    run_experiments(cfg, args.variants)


def cmd_experiment_selection(cfg, args) -> None:
    from engine.model.selection import run_experiment

    run_experiment(cfg)


def cmd_shadow(cfg, args) -> None:
    from engine.live.runner import run_live

    run_live(cfg, mode="shadow", resume=args.resume)


def cmd_live(cfg, args) -> None:
    from engine.live.runner import run_live

    if not args.yes:
        print("Live mode places REAL orders. Re-run with --yes to confirm.")
        sys.exit(1)
    run_live(cfg, mode="live", resume=args.resume)


def cmd_veto_report(cfg, args) -> None:
    from engine.reports.vetoes import report

    report(cfg, args.run_dir)


def cmd_report(cfg, args) -> None:
    from engine.reports.report import build_report

    print(build_report(args.run_dir))


def cmd_smoke_order(cfg, args) -> None:
    from engine.wallex.smoke import smoke

    if not args.yes:
        print("smoke-order places a REAL tiny buy and sell (about 2 x 0.30% fee). Re-run with --yes to confirm.")
        sys.exit(1)
    smoke(cfg, args.symbol, args.tmn)


def cmd_verify_api(cfg, args) -> None:
    from engine.wallex.verify import verify

    verify(cfg)


def parse_overrides(items: list[str]) -> dict:
    import yaml

    out: dict = {}
    for it in items:
        key, _, raw = it.partition("=")
        node = out
        parts = key.strip().split(".")
        for k in parts[:-1]:
            node = node.setdefault(k, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Wallex AI trading engine")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config value, e.g. --set labels.tp_atr=2.5 (repeatable)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("download")
    s.add_argument("--days", type=int)
    s.add_argument("--symbols", nargs="*")
    s.set_defaults(fn=cmd_download)

    s = sub.add_parser("record")
    s.add_argument("--symbols", nargs="*")
    s.add_argument("--interval", type=float, default=2, help="seconds between order book snapshots")
    s.set_defaults(fn=cmd_record)

    sub.add_parser("lag-log", help="Strategy 3: log Wallex vs Binance gaps (no orders)").set_defaults(fn=cmd_lag_log)
    s = sub.add_parser("lag-backtest", help="Strategy 3: historical lag-buy test on 1-minute candles")
    s.add_argument("--optimistic", action="store_true", help="also report fills without needing a Wallex trade")
    s.set_defaults(fn=cmd_lag_backtest)
    sub.add_parser("lag-replay", help="Strategy 3: lag-buy replay on the gap logger's real books").set_defaults(fn=cmd_lag_replay)

    sub.add_parser("train").set_defaults(fn=cmd_train)
    sub.add_parser("check-net", help="test every outside service from this machine (exit 1 if a critical one fails)").set_defaults(fn=cmd_check_net)
    sub.add_parser("premium-backtest", help="Wallex-vs-Binance premium mean reversion, hourly (PLAN.md §17)").set_defaults(fn=cmd_premium_backtest)
    sub.add_parser("trend-backtest", help="daily / 4h trend following on Binance history (PLAN.md §16)").set_defaults(fn=cmd_trend_backtest)
    sub.add_parser("train-seq", help="candidate F: GRU sequence model (PyTorch)").set_defaults(fn=cmd_train_seq)

    s = sub.add_parser("backtest")
    s.add_argument("--start", help="ISO date; default = start of held-out test window")
    s.add_argument("--end", help="ISO date; default = end of data")
    s.add_argument("--model", help="model file; default = models/latest.joblib")
    s.add_argument("--tag", default="", help="suffix for the run directory name")
    s.add_argument("--agent", action="store_true", help="run the GPT overlay on candidates (needs OPENAI_API_KEY; cached)")
    s.add_argument("--no-windows", action="store_true", help="skip rolling 3-day challenge windows")
    s.set_defaults(fn=cmd_backtest)

    sub.add_parser("experiment-selection").set_defaults(fn=cmd_experiment_selection)
    sub.add_parser("fv-research", help="fair-value gap research on recorded bid/ask").set_defaults(fn=cmd_fv_research)
    sub.add_parser("fv-backtest", help="replay recordings with maker fill model").set_defaults(fn=cmd_fv_backtest)
    sub.add_parser("fv-shadow", help="paper maker trading on live recordings (needs `record` running)").set_defaults(fn=cmd_fv_shadow)
    s = sub.add_parser("fv-live", help="REAL maker orders (needs WALLEX_API_KEY and `record` running)")
    s.add_argument("--yes", action="store_true")
    s.set_defaults(fn=cmd_fv_live)

    s = sub.add_parser("experiments")
    s.add_argument("variants", nargs="*", help="subset of variants (default: all)")
    s.set_defaults(fn=cmd_experiments)

    s = sub.add_parser("shadow")
    s.add_argument("--resume", help="run directory to resume (restores open positions)")
    s.set_defaults(fn=cmd_shadow)

    s = sub.add_parser("live")
    s.add_argument("--yes", action="store_true")
    s.add_argument("--resume", help="run directory to resume (restores open positions)")
    s.set_defaults(fn=cmd_live)

    s = sub.add_parser("veto-report", help="what vetoed / gate-skipped entries would have done")
    s.add_argument("run_dir")
    s.set_defaults(fn=cmd_veto_report)

    s = sub.add_parser("report")
    s.add_argument("run_dir")
    s.set_defaults(fn=cmd_report)

    sub.add_parser("verify-api").set_defaults(fn=cmd_verify_api)

    s = sub.add_parser("smoke-order", help="REAL minimum-size buy + sell to verify order handling")
    s.add_argument("--symbol", default="USDTTMN")
    s.add_argument("--tmn", type=float, default=110_000, help="order size; the account minimum is 100,000 TMN")
    s.add_argument("--yes", action="store_true")
    s.set_defaults(fn=cmd_smoke_order)

    args = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows console (cp1252)
    _setup_logging(args.verbose)
    from engine.env import apply_env_overrides, load_dotenv

    loaded = load_dotenv(".env")
    if loaded:
        logging.getLogger("cli").info("loaded from .env: %s", ", ".join(loaded))  # names only, never values
    cfg = load_config(args.config, parse_overrides(args.set))
    apply_env_overrides(cfg)
    args.fn(cfg, args)


if __name__ == "__main__":
    main()
