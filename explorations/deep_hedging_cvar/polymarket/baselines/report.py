"""Markdown tables from tuned.json, tuned_active.json, eval_val.json and the trial ledger (val only,
frozen env; pre-freeze rows are counted in N but not tabulated).

    python report.py            # prints the tables
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORDER = ["no_quote", "sym_touch", "as_logit", "fair_value", "fv_smart"]
PHASE_FILES = {"primary": "tuned.json", "active": "tuned_active.json"}
ART = ["inspread", "crossed", "stale", "onesided", "last30"]


def fmt(x, d=2):
    if x is None:
        return "-"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return "nan" if x != x else f"{x:+.{d}f}"
    return str(x)


def pfmt(params):
    if not params:
        return "-"
    return ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items())


def main():
    rows = [json.loads(line) for line in (HERE / "trials.jsonl").read_text().splitlines() if line.strip()]
    evals = [r for r in rows if r.get("kind") != "note"]
    pre = [r for r in evals if r.get("pre_freeze")]
    post = [r for r in evals if not r.get("pre_freeze")]
    n = Counter((r.get("phase", "primary"), r["baseline"]) for r in post)
    print(f"Ledger: {len(evals)} val evaluations in total = {len(pre)} pre-freeze (kept, counted) + "
          f"{len(post)} on the frozen env ({', '.join(f'{p}/{b} {c}' for (p, b), c in sorted(n.items()))}).\n")
    ev = json.loads((HERE / "eval_val.json").read_text()) if (HERE / "eval_val.json").exists() else {}

    for phase, fname in PHASE_FILES.items():
        if not (HERE / fname).exists():
            continue
        tuned = json.loads((HERE / fname).read_text())
        names = [b for b in ORDER if b in tuned]
        title = ("primary: unconstrained" if phase == "primary"
                 else "secondary: fills/market >= 5 and two-sided uptime >= 0.25 (best feasible trial)")
        print(f"## Tuned parameters, {title} (val, central, 1 s; objective mean per-market P&L)\n")
        print("| baseline | trials | feasible | untuned P&L/mkt | tuned P&L/mkt | fills | 2-sided uptime | best trial | params |")
        print("|---|---|---|---|---|---|---|---|---|")
        for b in names:
            t = tuned[b]
            print(f"| {b} | {t.get('n_trials', 1)} | {t.get('n_feasible', '-')} | {fmt(t.get('untuned_pnl_mean'), 3)} "
                  f"| {fmt(t.get('val_pnl_mean'), 3)} | {fmt(t.get('best_fills_mean'), 1)} "
                  f"| {fmt(t.get('best_uptime_two_sided_mean'), 2)} | {t.get('best_trial', 0)} | {pfmt(t.get('best_params'))} |")
        print()
        E = ev.get(phase, {})
        if not E:
            continue
        print(f"### {phase}: tuned baselines, central fills, 1 s (USD per market; components sum to P&L)\n")
        print("| baseline | P&L | ex-rebates | spread capture | markout 60 s | inventory | rebates | fills | "
              "2-sided uptime | max abs inv | CVaR95 | 5m | 15m | 4h |")
        print("|---|" + "---|" * 13)
        for b in [x for x in ORDER if x in E]:
            s = E[b]["variants"]["central/L1"]
            print(f"| {b} | {fmt(s['pnl_mean'], 3)} | {fmt(s['pnl_trading_mean'], 3)} | {fmt(s['spread_capture_mean'], 3)} "
                  f"| {fmt(s['markout_60_mean'], 3)} | {fmt(s['inventory_pnl_mean'], 3)} | {fmt(s['rebates_mean'], 3)} "
                  f"| {s['fills_mean']:.1f} | {s['uptime_two_sided_mean']:.2f} | {s['max_abs_inv_max']:.0f} "
                  f"| {fmt(s['cvar95'], 2)} | {fmt(s['pnl_mean_5m'], 2)} | {fmt(s['pnl_mean_15m'], 2)} | {fmt(s['pnl_mean_4h'], 2)} |")
        print(f"\n### {phase}: T5 / T6 / D1 at central, 1 s\n")
        print("T5 P&L share = flag's (spread capture + 60 s markout) / all fills' (NaN when the latter <= 0); "
              "D2 artifact flags judged at 25%. T6 inventory share of total / of positive components (> 50% = "
              "directional).\n")
        print("| baseline | fill P&L $ | " + " | ".join(f"T5 {f}" for f in ART) + " | outside (vol / P&L) | T5 invalid "
              "| T6 inv/total | T6 inv/positive | directional | D1 makes markets |")
        print("|---|---|" + "---|" * len(ART) + "---|---|---|---|---|---|")
        for b in [x for x in ORDER if x in E]:
            s = E[b]["variants"]["central/L1"]
            print(f"| {b} | {fmt(s['fill_pnl_total'], 1)} | " + " | ".join(fmt(s[f'pnl_share_{f}'], 2) for f in ART)
                  + f" | {s['share_vol_outside']:.2f} / {fmt(s['pnl_share_outside'], 2)} | {fmt(s['t5_invalid'])} "
                  f"| {fmt(s['t6_inventory_share_of_total'], 2)} | {fmt(s['t6_inventory_share_of_positive'], 2)} "
                  f"| {fmt(s['t6_directional'])} | {fmt(s['d1_makes_markets'])} |")
        variants = []
        for b in E:
            for k in E[b]["variants"]:
                if k not in variants:
                    variants.append(k)
        for metric, label, d in (("pnl_mean", "mean P&L per market (USD)", 3), ("fills_mean", "fills per market", 1)):
            print(f"\n### {phase}: {label} by fill model / latency / size\n")
            print("| baseline | " + " | ".join(variants) + " |")
            print("|---|" + "---|" * len(variants))
            for b in [x for x in ORDER if x in E]:
                v = E[b]["variants"]
                cells = [(fmt(v[k][metric], d) if metric == "pnl_mean" else f"{v[k][metric]:.1f}") if k in v else "-"
                         for k in variants]
                print(f"| {b} | " + " | ".join(cells) + " |")
        print()


if __name__ == "__main__":
    main()
