"""Markdown tables from tuned.json, eval_val.json and the trial ledger (val only).

    python report.py            # prints the tables
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORDER = ["no_quote", "sym_touch", "as_logit", "fair_value", "fv_smart"]


def fmt(x, d=2):
    return "nan" if x is None or x != x else f"{x:+.{d}f}" if isinstance(x, float) else str(x)


def main():
    tuned = json.loads((HERE / "tuned.json").read_text())
    ev = json.loads((HERE / "eval_val.json").read_text()) if (HERE / "eval_val.json").exists() else {}
    ledger = [json.loads(line) for line in (HERE / "trials.jsonl").read_text().splitlines() if line.strip()]
    notes = [r for r in ledger if r.get("kind") == "note"]
    ledger = [r for r in ledger if r.get("kind") != "note"]
    n = Counter(r["baseline"] for r in ledger)
    names = [b for b in ORDER if b in tuned]

    print("## Tuned parameters (val, central fills, 1 s latency; objective mean per-market P&L)\n")
    print("| baseline | trials | untuned P&L/mkt | tuned P&L/mkt | best trial | params |")
    print("|---|---|---|---|---|---|")
    for b in names:
        t = tuned[b]
        p = ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in t["best_params"].items())
        print(f"| {b} | {t.get('n_trials', 1)} | {fmt(t.get('untuned_pnl_mean'), 3)} | {fmt(t['val_pnl_mean'], 3)} "
              f"| {t.get('best_trial', 0)} | {p or '-'} |")
    for r in notes:
        print(f"\nNote ({r['time']}): {r['text']}")
    print(f"\nLedger: {len(ledger)} val evaluations ({', '.join(f'{k} {v}' for k, v in n.items())}).\n")
    print("## Best logged tuning trial with real activity (>= 5 fills per market), central / 1 s\n")
    print("(read from the ledger, no extra evaluation; shows what each quoter earns when it does quote)\n")
    print("| baseline | trial | P&L/mkt | ex-rebates | spread | markout 60 s | inventory | rebates | fills | CVaR95 | params |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for b in names:
        rows = [r for r in ledger if r["baseline"] == b and r["kind"] == "tune"
                and r["summary"]["fills_mean"] >= 5]
        if not rows:
            continue
        r = max(rows, key=lambda r: r["summary"]["pnl_mean"])
        s = r["summary"]
        p = ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in r["params"].items())
        print(f"| {b} | {r['trial']} | {fmt(s['pnl_mean'], 3)} | {fmt(s['pnl_trading_mean'], 3)} "
              f"| {fmt(s['spread_capture_mean'], 3)} | {fmt(s['markout_60_mean'], 3)} | {fmt(s['inventory_pnl_mean'], 3)} "
              f"| {fmt(s['rebates_mean'], 3)} | {s['fills_mean']:.1f} | {fmt(s['cvar95'], 2)} | {p} |")
    print()
    if not ev:
        return
    print("## Tuned baselines, val, central fills, 1 s (USD per market; attribution sums to P&L)\n")
    print("| baseline | P&L | P&L ex-rebates | spread capture | adverse sel. (markout 60 s) | inventory | "
          "rebates | fills | fill vol (sh) | 2-sided uptime | max abs inv | CVaR95 | median | 5m | 15m | 4h |")
    print("|---|" + "---|" * 16)
    for b in names:
        s = ev.get(b, {}).get("variants", {}).get("central/L1")
        if not s:
            continue
        print(f"| {b} | {fmt(s['pnl_mean'], 3)} | {fmt(s['pnl_trading_mean'], 3)} | {fmt(s['spread_capture_mean'], 3)} "
              f"| {fmt(s['markout_60_mean'], 3)} | {fmt(s['inventory_pnl_mean'], 3)} | {fmt(s['rebates_mean'], 3)} "
              f"| {s['fills_mean']:.1f} | {s['fill_vol_mean']:.0f} | {s['uptime_two_sided_mean']:.2f} "
              f"| {s['max_abs_inv_max']:.0f} | {fmt(s['cvar95'], 2)} | {fmt(s['pnl_median'], 3)} "
              f"| {fmt(s['pnl_mean_5m'], 2)} | {fmt(s['pnl_mean_15m'], 2)} | {fmt(s['pnl_mean_4h'], 2)} |")
    variants = []
    for b in names:
        for k in ev.get(b, {}).get("variants", {}):
            if k not in variants:
                variants.append(k)
    print("\n## Mean P&L per market (USD) by fill model / latency / size, val\n")
    print("| baseline | " + " | ".join(variants) + " |")
    print("|---|" + "---|" * len(variants))
    for b in names:
        v = ev.get(b, {}).get("variants", {})
        print(f"| {b} | " + " | ".join(fmt(v[k]["pnl_mean"], 3) if k in v else "-" for k in variants) + " |")
    print("\n## Fills per market by variant\n")
    print("| baseline | " + " | ".join(variants) + " |")
    print("|---|" + "---|" * len(variants))
    for b in names:
        v = ev.get(b, {}).get("variants", {})
        print(f"| {b} | " + " | ".join(f"{v[k]['fills_mean']:.1f}" if k in v else "-" for k in variants) + " |")
    print("\n## Exploit-flag shares of fill volume (central / 1 s)\n")
    flags = ["crossed", "stale", "onesided", "last30", "last10", "inside2", "extreme", "outside"]
    print("| baseline | " + " | ".join(flags) + " |")
    print("|---|" + "---|" * len(flags))
    for b in names:
        s = ev.get(b, {}).get("variants", {}).get("central/L1")
        if s:
            print(f"| {b} | " + " | ".join(f"{s[f'share_vol_{f}']:.3f}" for f in flags) + " |")


if __name__ == "__main__":
    main()
