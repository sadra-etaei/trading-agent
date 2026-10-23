"""Run report: metrics (the judges' list) + charts, from a run directory."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def metrics(trades: pd.DataFrame, equity: pd.DataFrame, initial: float) -> dict:
    final = float(equity["equity"].iloc[-1]) if len(equity) else initial
    m = {
        "initial_tmn": initial,
        "final_tmn": final,
        "net_return_pct": (final / initial - 1) * 100,
        "max_drawdown_pct": float(equity["drawdown"].max() * 100) if len(equity) else 0.0,
        "trades": int(len(trades)),
    }
    if len(trades):
        vol = 2 * trades["notional_tmn"]  # buy + sell legs
        wins = trades[trades["pnl_tmn"] > 0]
        losses = trades[trades["pnl_tmn"] <= 0]
        m.update({
            "trading_volume_tmn": float(vol.sum()),
            "avg_trade_volume_tmn": float(trades["notional_tmn"].mean()),
            "win_rate_pct": len(wins) / len(trades) * 100,
            "avg_trade_ret_pct": float(trades["ret_pct"].mean() * 100),
            "profit_factor": float(wins["pnl_tmn"].sum() / -losses["pnl_tmn"].sum()) if len(losses) and losses["pnl_tmn"].sum() < 0 else float("inf"),
            "fees_tmn": float(trades["fees_tmn"].sum()),
            "avg_hold_hours": float(trades["hold_hours"].mean()),
            "exit_reasons": trades["exit_reason"].value_counts().to_dict(),
            "by_symbol": trades.groupby("symbol")["pnl_tmn"].agg(["count", "sum"]).round(0).to_dict("index"),
        })
    if len(equity) > 24:
        r = equity["equity"].pct_change().dropna()
        m["sharpe_hourly_annualized"] = float(r.mean() / r.std() * np.sqrt(24 * 365)) if r.std() > 0 else 0.0
    return m


def _charts(run_dir: Path, equity: pd.DataFrame) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    if equity.empty:
        return []
    eq = equity.copy()
    eq["ts"] = pd.to_datetime(eq["ts"])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    a1.plot(eq["ts"], eq["equity"] / 1e6, lw=1.2)
    a1.set_ylabel("Equity (M TMN)")
    a1.grid(alpha=0.3)
    a2.fill_between(eq["ts"], -eq["drawdown"] * 100, 0, alpha=0.5, color="tab:red")
    a2.set_ylabel("Drawdown %")
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "equity.png", dpi=110)
    plt.close(fig)
    return ["equity.png"]


def build_report(run_dir: str | Path) -> str:
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text()) if (run_dir / "meta.json").exists() else {}
    tp = run_dir / "trades.csv"
    trades = pd.read_csv(tp) if tp.exists() and tp.stat().st_size > 2 else pd.DataFrame()  # a bare newline = no trades
    equity = pd.read_csv(run_dir / "equity.csv") if (run_dir / "equity.csv").exists() else pd.DataFrame()
    initial = float(meta.get("config", {}).get("risk", {}).get("initial_equity_tmn", 5_000_000))
    m = metrics(trades, equity, initial)
    (run_dir / "metrics.json").write_text(json.dumps(m, indent=2, default=str))
    charts = _charts(run_dir, equity)

    L = [f"# Run report: {meta.get('run_id', run_dir.name)}", ""]
    if meta:
        L += [f"- Mode: **{meta.get('mode')}**", f"- Window: {meta.get('window', ['?', '?'])[0]} .. {meta.get('window', ['?', '?'])[1]}",
              f"- Model: `{meta.get('model_version')}`, config hash `{meta.get('config_hash')}`",
              f"- Pairs: {', '.join(meta.get('symbols', []))}", f"- GPT overlay: {meta.get('agent')}", ""]
    L += ["## Results", "", "| metric | value |", "|---|---|",
          f"| Net return | **{m['net_return_pct']:+.2f}%** |",
          f"| Final value | {m['final_tmn']:,.0f} TMN |",
          f"| Max drawdown | {m['max_drawdown_pct']:.2f}% |",
          f"| Number of trades | {m['trades']} |"]
    if m["trades"]:
        L += [f"| Trading volume | {m['trading_volume_tmn']:,.0f} TMN |",
              f"| Avg trade size | {m['avg_trade_volume_tmn']:,.0f} TMN |",
              f"| Win rate | {m['win_rate_pct']:.1f}% |",
              f"| Avg trade return (net) | {m['avg_trade_ret_pct']:+.3f}% |",
              f"| Profit factor | {m['profit_factor']:.2f} |",
              f"| Fees paid | {m['fees_tmn']:,.0f} TMN |",
              f"| Avg holding time | {m['avg_hold_hours']:.1f} h |"]
    if "sharpe_hourly_annualized" in m:
        L.append(f"| Sharpe (annualized) | {m['sharpe_hourly_annualized']:.2f} |")
    if m["trades"]:
        L += ["", "Exit reasons: " + ", ".join(f"{k} {v}" for k, v in m["exit_reasons"].items()), "",
              "| pair | trades | P&L (TMN) |", "|---|---|---|"]
        for s, v in m["by_symbol"].items():
            L.append(f"| {s} | {int(v['count'])} | {v['sum']:+,.0f} |")
    wp = run_dir / "challenge_windows.csv"
    if wp.exists() and wp.stat().st_size > 1:
        w = pd.read_csv(wp)
        if len(w):
            L += ["", "## Rolling 3-day challenge simulations", "",
                  "Each window: fresh 5,000,000 TMN, challenge rules (ramp-up, no entries near the end, flatten to TMN).", "",
                  "| metric | value |", "|---|---|",
                  f"| Windows | {len(w)} |",
                  f"| Positive windows | {(w['return_pct'] > 0).mean():.0%} |",
                  f"| Median return | {w['return_pct'].median():+.2f}% |",
                  f"| Mean return | {w['return_pct'].mean():+.2f}% |",
                  f"| Worst / best | {w['return_pct'].min():+.2f}% / {w['return_pct'].max():+.2f}% |",
                  f"| Median max drawdown | {w['max_drawdown_pct'].median():.2f}% |",
                  f"| Worst max drawdown | {w['max_drawdown_pct'].max():.2f}% |",
                  f"| Median trades per window | {w['trades'].median():.0f} |",
                  f"| Windows ending fully in TMN | {(w['open_positions_at_end'] == 0).mean():.0%} |"]
    if charts:
        L += ["", *[f"![{c}]({c})" for c in charts]]
    text = "\n".join(L) + "\n"
    (run_dir / "report.md").write_text(text, encoding="utf-8")
    return text
