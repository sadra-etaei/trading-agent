"""Build offline hackathon demo charts and a self-contained evidence page."""
from pathlib import Path
import base64
import hashlib
import html
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "hackathon-demo"
OUT.mkdir(parents=True, exist_ok=True)
BT = ROOT / "runs" / "bt-20261022-154705-test-sz-S1"
TZ = "Asia/Tehran"
BG, PANEL, TEXT, MUTED = "#0b1220", "#111d30", "#edf3fc", "#9eb0c9"
GREEN, BLUE, RED, GOLD = "#43ddb2", "#72b7ff", "#ff8795", "#f3c879"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "text.color": TEXT,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.facecolor": PANEL, "figure.facecolor": BG, "axes.edgecolor": "#283950",
    "grid.color": "#314158", "grid.alpha": .4, "savefig.facecolor": BG})


def local_time(value):
    return pd.Timestamp(value).tz_convert(TZ).strftime("%b %d %H:%M:%S")


def embed(path):
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


meta = json.loads((BT / "meta.json").read_text())
metrics = json.loads((BT / "metrics.json").read_text())
eq = pd.read_csv(BT / "equity.csv")
eq["ts"] = pd.to_datetime(eq.ts, utc=True)
start, end = map(pd.Timestamp, meta["window"])
fx = pd.read_parquet(ROOT / "data/candles_1m/USDTTMN.parquet").sort_values("ts")
# Minute timestamps are opens: a candle strictly before a boundary has closed by that boundary.
def fx_at(ts):
    i = np.searchsorted(fx.ts.to_numpy(), int(ts.timestamp()), side="left") - 1
    assert i >= 0
    row = fx.iloc[i]
    return float(row["close"]), int(row["ts"])

rate0, _ = fx_at(start)
rates = np.array([fx_at(t)[0] for t in eq.ts])
indices = np.array([fx_at(t)[1] for t in eq.ts])
initial = float(metrics["initial_tmn"])
usd_return = (eq.equity.to_numpy() / initial) / (rates / rate0) - 1
hold_usdt = initial * rates / rate0
dates = eq.ts.dt.tz_convert(TZ)
headline = float(metrics["net_return_pct"])
usd_final = float(usd_return[-1] * 100)
fx_change = float((rates[-1] / rate0 - 1) * 100)
fig, axes = plt.subplots(3, 1, figsize=(16, 9), dpi=120, sharex=True,
    gridspec_kw={"height_ratios": [3.1, 1.35, 1], "hspace": .13})
fig.subplots_adjust(left=.075, right=.92, top=.775, bottom=.14)
fig.text(.075, .94, "S1  /  HISTORICAL BACKTEST", fontsize=22, weight="bold")
fig.text(.075, .899, "17 Sep – 22 Oct 2026  ·  16 Toman pairs  ·  Binance features + crash guard", color=MUTED, fontsize=12)
fig.text(.075, .845, f"{headline:+.2f}% Toman", fontsize=19, color=GREEN, weight="bold")
fig.text(.33, .845, f"{usd_final:+.2f}% USDT-equivalent", fontsize=19, color=BLUE, weight="bold")
fig.text(.68, .845, f"42 trades   |   {metrics['max_drawdown_pct']:.2f}% max DD", fontsize=16, color=TEXT)
fig.text(.075, .808, f"Toman balance: {initial:,.0f} → {metrics['final_tmn']:,.0f}   |   Net profit: +{metrics['final_tmn'] - initial:,.0f} Toman", fontsize=12, color=GREEN)
axes[0].plot(dates, (eq.equity / initial - 1) * 100, color=GREEN, lw=2.2, label="S1 · Toman return after modeled costs")
axes[0].plot(dates, (hold_usdt / initial - 1) * 100, color=GOLD, lw=1.4, alpha=.95, label="Hold USDT · Toman return before conversion costs")
axes[0].axhline(0, color=MUTED, lw=1, ls="--", label="Hold Toman · 0%")
axes[0].set_ylabel("Toman return · %")
balance_axis = axes[0].secondary_yaxis("right", functions=(lambda x: initial * (1 + x / 100) / 1e6, lambda x: (x * 1e6 / initial - 1) * 100))
balance_axis.set_ylabel("Balance · million Toman", color=MUTED)
balance_axis.tick_params(colors=MUTED)
axes[0].legend(loc="upper left", fontsize=10, frameon=False)
axes[1].plot(dates, usd_return * 100, color=BLUE, lw=1.6)
axes[1].axhline(0, color=MUTED, ls="--", lw=.9)
axes[1].set_ylabel("USDT-equivalent\nreturn · %")
axes[1].text(.99, .85, "USDT used as a dollar proxy", transform=axes[1].transAxes, ha="right", color=MUTED, fontsize=10)
axes[2].fill_between(dates, -eq.drawdown * 100, 0, color=RED, alpha=.55)
axes[2].set_ylabel("Toman\ndrawdown · %")
for ax in axes:
    ax.grid(True)
    ax.spines[["top", "right"]].set_visible(False)
axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%d %b", tz=dates.dt.tz))
fig.text(.075, .077, "Costs: 0.30% fee + 0.15% slippage per side · 1-minute entry delay · GPT overlay OFF", fontsize=10, color=MUTED)
fig.text(.075, .047, "Historical simulation, not live profit. This period has been examined in earlier experiments.", fontsize=10, color=MUTED)
fig.text(.075, .019, "USDT valuation uses last available quotes across gaps; dollar-equivalent return is before extra conversion costs.", fontsize=9, color=MUTED)
fig.savefig(OUT / "s1-backtest.png", dpi=120)
fig.savefig(OUT / "s1-backtest.svg")
plt.close(fig)

closed, unfinished, run_rows, sources = [], [], [], []
for p in sorted((ROOT / "runs").glob("shadow-*")):
    m = json.loads((p / "meta.json").read_text())
    state = json.loads((p / "state.json").read_text())
    lines = (p / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
    events = [json.loads(x) for x in lines if x.strip()]
    f = p / "trades.csv"
    trades = pd.read_csv(f) if f.exists() and f.stat().st_size > 2 else pd.DataFrame()
    fills = [e for e in events if e.get("kind") == "fill" and e.get("side") == "BUY"]
    for row in trades.to_dict("records"):
        row.update({"run": p.name, "model_version": m.get("model_version"), "mode": "shadow"})
        closed.append(row)
    for symbol, pos in state.get("positions", {}).items():
        unfinished.append({"run": p.name, "symbol": symbol, "decision_id": pos["decision_id"],
            "entry_ts": pos["entry_ts"], "entry_price": pos["entry_price"], "qty": pos["qty"],
            "model_version": m.get("model_version"), "status": "Open at saved run end; not a current account position", "pnl_tmn": None})
    run_rows.append({"run": p.name, "model_version": m.get("model_version"), "quote": m["config"]["universe"]["quote"],
        "agent_enabled": m.get("agent"), "entry_fills": len(fills), "closed": len(trades),
        "unfinished": len(state.get("positions", {})), "closed_pnl_tmn": float(trades.pnl_tmn.sum()) if len(trades) else 0})
    sources.extend([p / "meta.json", p / "state.json", p / "decisions.jsonl", f])
for p in sorted((ROOT / "runs").glob("fv-shadow-*")):
    f = p / "status.md"
    if f.exists():
        sources.append(f)
        rows = [line.split("|") for line in f.read_text().splitlines() if line.startswith("| m")]
        assert all(int(r[2].strip()) == 0 and int(r[-2].strip()) == 0 for r in rows), "New fair-value fills: inspect before reporting"
        run_rows.append({"run": p.name, "model_version": "fair-value variants", "quote": "TMN", "entry_fills": 0, "closed": 0, "unfinished": 0, "closed_pnl_tmn": 0})
ct = pd.DataFrame(closed).sort_values("entry_ts").reset_index(drop=True)
ot = pd.DataFrame(unfinished).sort_values("entry_ts").reset_index(drop=True)
assert len(ct) == 8 and len(ot) == 3
assert ct.decision_id.is_unique and ot.decision_id.is_unique
assert set(ct.decision_id).isdisjoint(set(ot.decision_id))
assert sum(r["entry_fills"] for r in run_rows) == len(ct) + len(ot)
ct["entry_tehran"] = ct.entry_ts.map(local_time)
ct["exit_tehran"] = ct.exit_ts.map(local_time)
ot["entry_tehran"] = ot.entry_ts.map(local_time)
ct.to_csv(OUT / "closed-shadow-trades.csv", index=False)
ot.to_csv(OUT / "unfinished-shadow-entries.csv", index=False)
pd.DataFrame(run_rows).to_csv(OUT / "shadow-run-inventory.csv", index=False)
total_loss = float(ct.pnl_tmn.sum())
fig, ax = plt.subplots(figsize=(16, 9), dpi=120)
fig.subplots_adjust(left=.18, right=.92, top=.80, bottom=.16)
labels = [f"{r.symbol.removesuffix('TMN')}/TMN · {r.entry_tehran}" for r in ct.itertuples()]
bars = ax.barh(labels, ct.pnl_tmn, color=RED, height=.58)
ax.invert_yaxis()
ax.set_xlim(float(ct.pnl_tmn.min()) * 1.28, 1500)
ax.axvline(0, color=MUTED, lw=1)
ax.set_xlabel("Net simulated P&L · Toman · includes recorded modeled fees")
ax.grid(axis="x")
for bar, row in zip(bars, ct.itertuples()):
    ax.text(bar.get_width() - 160, bar.get_y() + bar.get_height()/2, f"{row.pnl_tmn:,.0f}", va="center", ha="right", fontsize=12)
ax.spines[["top", "right", "left"]].set_visible(False)
fig.text(.075, .94, "SHADOW EXECUTION  /  ALL COMPLETED TRADES", fontsize=22, weight="bold")
fig.text(.075, .887, f"8 closed trades · 0 profitable · {total_loss:,.0f} Toman total recorded net P&L", color=RED, fontsize=16)
fig.text(.075, .842, "21–22 Oct 2026 · earlier models/configurations · all timestamps Tehran", color=MUTED, fontsize=12)
fig.text(.075, .075, "Simulated fills on live order books. Independent runs: P&L sum is not a continuous-account return.", color=MUTED, fontsize=10)
fig.text(.075, .045, "3 further entries were unfinished at their saved run ends and are listed separately. No live strategy trades.", color=MUTED, fontsize=10)
fig.savefig(OUT / "shadow-trades.png", dpi=120)
plt.close(fig)


def esc(x):
    return html.escape(str(x))


closed_rows = "".join(f"<tr><td>{i+1}</td><td><b>{esc(r.symbol.removesuffix('TMN'))}/TMN</b><small>{esc(r.run)}</small></td><td>{r.entry_tehran}</td><td>{r.exit_tehran}</td><td>{r.notional_tmn:,.0f}</td><td class='loss'>{r.pnl_tmn:,.0f}</td><td class='loss'>{r.ret_pct*100:+.2f}%</td><td>{esc(r.exit_reason.replace('_',' '))}</td></tr>" for i,r in enumerate(ct.itertuples()))
open_rows = "".join(f"<tr><td><b>{esc(r.symbol.removesuffix('TMN'))}/TMN</b></td><td>{r.entry_tehran}</td><td>{r.qty:g}</td><td>{r.entry_price:,.2f}</td><td>{esc(r.run)}</td><td>Unfinished snapshot</td></tr>" for r in ot.itertuples())
inventory = "".join(f"<tr><td>{esc(r['run'])}</td><td>{esc(r['model_version'])}</td><td>{r['entry_fills']}</td><td>{r['closed']}</td><td>{r['unfinished']}</td></tr>" for r in run_rows)
page = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Wallex | Hackathon evidence</title>
<style>*{box-sizing:border-box}body{margin:0;background:#0b1220;color:#edf3fc;font:17px 'Segoe UI',Arial,sans-serif}header{padding:24px 4vw 18px;border-bottom:1px solid #283950;display:flex;justify-content:space-between;align-items:center}h1{font-size:26px;margin:0}header p{color:#9eb0c9;margin:7px 0 0;font-size:14px}nav{display:flex;gap:8px}button{background:#111d30;color:#9eb0c9;border:1px solid #30435e;padding:11px 17px;border-radius:8px;cursor:pointer;font:inherit}button.active{background:#43ddb2;color:#09241f;border-color:#43ddb2}main{padding:25px 4vw 40px}.panel{display:none}.panel.active{display:block}.banner{padding:12px 17px;border:1px solid #30435e;background:#111d30;border-radius:8px;color:#b7c9e1;font-size:15px;line-height:1.6}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:15px;margin:20px 0}.card{background:#111d30;border:1px solid #283950;border-radius:12px;padding:16px 20px}.card label{display:block;color:#9eb0c9;font-size:14px}.card strong{display:block;font-size:29px;margin-top:8px}.loss{color:#ff8795}.gain{color:#43ddb2}.blue{color:#72b7ff}img{width:100%;height:auto;border:1px solid #283950;border-radius:12px}table{border-collapse:collapse;width:100%;font-size:15px;margin:16px 0 24px}th{text-align:left;color:#9eb0c9;font-weight:500;font-size:13px;letter-spacing:.04em;padding:12px 10px;border-bottom:1px solid #40536e}td{padding:12px 10px;border-bottom:1px solid #24334b}small{display:block;color:#9eb0c9;font-size:11px;margin-top:5px}h2{font-size:22px;margin:22px 0 9px}p{line-height:1.6}.muted{color:#9eb0c9}details{margin:20px 0}summary{cursor:pointer;color:#72b7ff}footer{color:#9eb0c9;font-size:13px;margin-top:25px}a{color:#72b7ff}.scroll{overflow-x:auto}@media(max-width:800px){header{display:block}nav{margin-top:15px;flex-wrap:wrap}.cards{grid-template-columns:repeat(2,1fr)}main{padding:18px}table{min-width:900px}}@media print{nav{display:none}.panel{display:block!important;page-break-after:always}body{print-color-adjust:exact}}</style>
<header><div><h1>WALLEX / AI TRADING ENGINE</h1><p>Starcoach Agentic AI Hackathon · recorded evidence · no live strategy trades</p></div><nav><button class="active" onclick="show('backtest',this)">Backtest</button><button onclick="show('shadow',this)">All shadow trades</button><button onclick="show('method',this)">Method & scope</button></nav></header><main>
<section id="backtest" class="panel active"><div class="banner">HISTORICAL SIMULATION · S1 model __MODEL__ · GPT overlay disabled in this backtest</div><img style="margin-top:20px" src="__BACKTEST__" alt="S1 backtest: Toman equity, USDT-equivalent returns and drawdown"><details><summary>Exact recorded results and interpretation</summary><p>Starting equity: 5,000,000 Toman. Final equity: __FINAL__ Toman. Win rate: __WIN__%. Profit factor: __PF__. Recorded fees: __FEES__ Toman. USDT/TMN increased __FX__% over this window. The USDT benchmark excludes conversion costs; the dollar-equivalent strategy return uses USDT as a proxy and excludes additional final conversion fees.</p><p class="muted">This month has been examined in earlier experiments and is not a fresh prospective holdout. Historical performance does not establish live profitability. Charts preserve the saved equity and drawdown series.</p></details></section>
<section id="shadow" class="panel"><div class="banner">SHADOW MODE · simulated orders using live market data · earlier model/configurations; not current S1 live performance</div><div class="cards"><div class="card"><label>Completed trades</label><strong>8</strong></div><div class="card"><label>Profitable completed trades</label><strong>0 / 8</strong></div><div class="card"><label>Recorded closed P&L · Toman</label><strong class="loss">__LOSS__</strong></div><div class="card"><label>Unfinished saved entries</label><strong>3</strong></div></div><h2>Every completed shadow trade</h2><p class="muted">All dates are October 2026, Tehran time. Seven stop-loss exits and one GPT-requested exit.</p><div class="scroll"><table><thead><tr><th>#</th><th>PAIR / SOURCE RUN</th><th>ENTRY · TEHRAN</th><th>EXIT · TEHRAN</th><th>SIZE · TOMAN</th><th>NET P&L · TOMAN</th><th>TRADE RETURN</th><th>EXIT</th></tr></thead><tbody>__CLOSED__</tbody></table></div><div class="banner">The P&L sum combines independent experiments. It is not a return on one continuous 5,000,000-Toman account. Modeled fees and risk settings differed between earlier runs.</div><details><summary>Show all three unfinished entries</summary><p class="muted">Positions still open in their historical saved states. These are not current account positions and have no recorded realized outcome.</p><div class="scroll"><table><thead><tr><th>PAIR</th><th>ENTRY · TEHRAN</th><th>QUANTITY</th><th>ENTRY PRICE · TOMAN</th><th>SOURCE RUN</th><th>STATUS</th></tr></thead><tbody>__OPEN__</tbody></table></div></details><details><summary>Show chart of all completed outcomes</summary><img src="__SHADOW__" alt="All eight completed shadow trade losses"></details></section>
<section id="method" class="panel"><h2>What the evidence demonstrates</h2><p>ML ranks candidate trades by estimated value after costs. GPT can reduce size, veto or request an exit. Deterministic risk checks control position sizing and exits. The broker is simulated in shadow mode.</p><h2>Recorded UNI example</h2><p>22 October, 13:31 Tehran: ML probability 67.2%, estimated EV +0.216%. GPT reduced proposed size by 25%. Simulated entry notional: 903,357 Toman. At 14:31, GPT requested an early exit on negative headlines. Recorded outcome: −5,329 Toman. This shows a consequential agent decision; it does not establish that the agent improved profit.</p><h2>Complete run inventory</h2><p class="muted">Ten directional shadow runs plus two fair-value runs. Fair-value variants had no fills. Latest S1 shadow run had no trades.</p><div class="scroll"><table><thead><tr><th>RUN</th><th>MODEL</th><th>ENTRY FILLS</th><th>CLOSED</th><th>UNFINISHED</th></tr></thead><tbody>__INVENTORY__</tbody></table></div><h2>Reproduction & source data</h2><p>Build script: research/build_demo_evidence.py. The CSV exports contain decision IDs and source run names. evidence.json contains source hashes. The backtest uses runs/bt-20261022-154705-test-sz-S1. This page embeds its charts and works offline without a backend.</p></section><footer>Read-only evidence package · no API credentials included · production model/configuration unchanged</footer></main><script>function show(id,b){document.querySelectorAll('.panel').forEach(x=>x.classList.remove('active'));document.getElementById(id).classList.add('active');document.querySelectorAll('nav button').forEach(x=>x.classList.remove('active'));b.classList.add('active');window.scrollTo(0,0)}</script></html>"""
replacements = {"__MODEL__": meta["model_version"], "__BACKTEST__": embed(OUT / "s1-backtest.png"), "__SHADOW__": embed(OUT / "shadow-trades.png"),
    "__CLOSED__": closed_rows, "__OPEN__": open_rows, "__INVENTORY__": inventory,
    "__LOSS__": f"{total_loss:,.0f}", "__FINAL__": f"{metrics['final_tmn']:,.0f}",
    "__WIN__": f"{metrics['win_rate_pct']:.1f}", "__PF__": f"{metrics['profit_factor']:.2f}",
    "__FEES__": f"{metrics['fees_tmn']:,.0f}", "__FX__": f"{fx_change:+.2f}"}
for key, value in replacements.items():
    page = page.replace(key, value)
toman_cards = (f'<div class="cards"><div class="card"><label>Toman return</label><strong class="gain">{headline:+.2f}%</strong></div>'
    f'<div class="card"><label>Net profit · Toman</label><strong class="gain">+{metrics["final_tmn"] - initial:,.0f}</strong></div>'
    f'<div class="card"><label>Starting balance · Toman</label><strong>{initial:,.0f}</strong></div>'
    f'<div class="card"><label>Final balance · Toman</label><strong>{metrics["final_tmn"]:,.0f}</strong></div></div>')
page = page.replace('<img style="margin-top:20px"', toman_cards + '<img style="margin-top:20px"', 1)
assert "__CLOSED__" not in page and "__BACKTEST__" not in page
(OUT / "index.html").write_text(page, encoding="utf-8")
sources += [BT / "meta.json", BT / "metrics.json", BT / "equity.csv", BT / "trades.csv", ROOT / "data/candles_1m/USDTTMN.parquet"]
evidence = {"backtest_run": BT.name, "backtest_model": meta["model_version"], "backtest_toman_return_pct": headline,
    "backtest_usdt_equivalent_return_pct": usd_final, "hold_usdt_toman_return_pct": fx_change,
    "fx_start": rate0, "fx_end": float(rates[-1]), "max_fx_quote_age_min": float(np.max((eq.ts.map(lambda t: t.timestamp()).to_numpy() - indices) / 60)),
    "closed_shadow_count": len(ct), "unfinished_shadow_count": len(ot), "closed_shadow_pnl_tmn": total_loss,
    "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources if p.exists()}}
(OUT / "evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
md = ["# Hackathon demo evidence", "", "No live strategy trades. Shadow fills are simulated. Earlier shadow configurations differ from current S1.", "",
    f"Backtest: {headline:+.2f}% Toman, {usd_final:+.2f}% USDT-equivalent, 42 trades, 3.68% maximum Toman drawdown. GPT disabled. Source: {BT.name}.", "",
    "## All completed shadow trades", "", "Times: Tehran, October 2026. P&L includes recorded modeled fees.", "",
    "| Pair | Entry | Exit | Notional TMN | Net P&L TMN | Return | Exit reason |", "|---|---|---|---:|---:|---:|---|"]
for r in ct.itertuples():
    md.append(f"| {r.symbol} | {r.entry_tehran} | {r.exit_tehran} | {r.notional_tmn:,.0f} | {r.pnl_tmn:,.0f} | {r.ret_pct*100:+.2f}% | {r.exit_reason} |")
md += ["", f"Recorded closed P&L sum: {total_loss:,.0f} Toman across independent experiments; not a continuous-account return.", "", "## Unfinished historical entries", "", "| Pair | Entry | Source run |", "|---|---|---|"]
for r in ot.itertuples():
    md.append(f"| {r.symbol} | {r.entry_tehran} | {r.run} |")
md += ["", "Open index.html for the demo view. s1-backtest.png is 1920×1080; s1-backtest.svg is scalable. shadow-trades.png shows all eight outcomes. CSV files retain decision IDs and original UTC timestamps. evidence.json records source hashes.", "", f"USDT-equivalent return was recomputed from the saved equity curve and explicit boundary prices ({rate0:,.0f} to {rates[-1]:,.0f} Toman/USDT), giving {usd_final:+.2f}%; older strategy notes reported +3.38% using a different valuation boundary. Source quotes are carried forward across gaps (maximum age {evidence['max_fx_quote_age_min']:.0f} minutes); no missing prices are interpolated. Additional USDT conversion costs are excluded."]
(OUT / "README.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print(json.dumps({k:v for k,v in evidence.items() if k != "source_sha256"}, indent=2))
print(f"Demo page: {OUT / 'index.html'}")
