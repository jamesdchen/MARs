"""Summarise out/census_markets.parquet: volume by league / sport / market type, fee configs, reward configs."""
import json, pandas as pd
df = pd.read_parquet("out/census_markets.parquet")
c = df[(df.status == "closed")].copy()
c["end_ts"] = pd.to_datetime(c.end, utc=True, errors="coerce")
c = c[(c.end_ts >= "2026-10-01") & (c.end_ts < "2026-10-15")]
tot = c.volume.sum()
SPORT = {"nfl": "american football", "cfb": "american football", "ufl": "american football", "nba": "basketball", "wnba": "basketball",
         "cbb": "basketball", "mlb": "baseball", "kbo": "baseball", "npb": "baseball", "nhl": "hockey", "atp": "tennis", "wta": "tennis", "itf": "tennis",
         "cs2": "esports", "lol": "esports", "dota2": "esports", "val": "esports", "r6siege": "esports", "codmw": "esports", "ow": "esports"}
c["sport"] = c.league.map(SPORT).fillna(c.tags.str.contains("Soccer").map({True: "soccer", False: "other"}))
c.loc[c.esports, "sport"] = "esports"
lines = [f"Sports game events (gamma tag 100639) with endDate in [2026-10-01, 2026-10-15): {c.event_slug.nunique()} events, "
         f"{len(c)} markets, gamma volume ${tot/1e6:.0f}M (gamma 'volume' units; relative shares are what matter)\n"]
g = c.groupby("sport").agg(vol=("volume", "sum"), events=("event_slug", "nunique")).sort_values("vol", ascending=False)
g["share"] = (g.vol / tot).round(3); g["vol_$M"] = (g.vol / 1e6).round(1)
lines += ["## by sport", g[["vol_$M", "share", "events"]].to_string(), ""]
g = c.groupby("league").agg(vol=("volume", "sum"), events=("event_slug", "nunique")).sort_values("vol", ascending=False).head(25)
g["share"] = (g.vol / tot).round(3); g["vol_$M"] = (g.vol / 1e6).round(1); g["$M_per_event"] = (g.vol / g.events / 1e6).round(2)
lines += ["## top 25 leagues", g[["vol_$M", "share", "events", "$M_per_event"]].to_string(), ""]
g = (c.groupby("smt").volume.sum() / tot).sort_values(ascending=False).head(10).round(3)
lines += ["## by sportsMarketType (volume share)", g.to_string(), ""]
lines += ["## fee schedules (markets)", c.groupby(["fee_type", "fee"]).volume.agg(["size", "sum"]).assign(share=lambda d: (d["sum"] / tot).round(3)).drop(columns="sum").to_string(), ""]
lines += ["## reward configs (rewardsMinSize, rewardsMaxSpread)", c.groupby(["rew_min", "rew_spread"]).volume.agg(["size", "sum"]).assign(share=lambda d: (d["sum"] / tot).round(3)).drop(columns="sum").to_string(), ""]
cr = c[c.clob_rewards != "null"]
rates = [r.get("rewardsDailyRate") for x in cr.clob_rewards for r in json.loads(x)]
lines += [f"markets with a clobRewards entry: {len(cr)}; daily rates: {pd.Series(rates).value_counts().head().to_dict()}", ""]
lines += ["## secondsDelay / clearBookOnStart", c.delay.value_counts(dropna=False).to_string(), c.clear_on_start.value_counts(dropna=False).to_string()]
lines += ["", "## delay by sport (market counts)", pd.crosstab(c.sport, c.delay.fillna(-1)).to_string()]
open("out/census_summary.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
