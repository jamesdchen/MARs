# Polymarket Crypto Up/Down Order Books: 1-Second L2 Depth Across Three Contract Lengths

Full bid and ask ladders, captured once per second, for Polymarket's binary
up/down cryptocurrency markets at three contract lengths: **5 minutes, 15 minutes
and 4 hours**. Every market has already settled, so each row carries the winning
outcome.

**897,192 snapshots · 805 resolved markets · 8 coins · 43 MB**

| File | Rows | Markets | Contract length |
| --- | ---: | ---: | --- |
| `updown_5m.parquet` | 167,941 | 471 | 5 minutes |
| `updown_15m.parquet` | 239,310 | 265 | 15 minutes |
| `updown_4h.parquet` | 489,941 | 69 | 4 hours |

Coins: BTC, ETH, SOL, XRP, BNB, DOGE, HYPE, ZEC.
Capture period: 2026-08-21 to 2026-08-24 (UTC).

---

## Why three contract lengths

Published order book datasets for prediction markets, where they exist at all,
cover a single contract length. That makes it impossible to ask whether an
observed effect is a property of prediction markets or a property of *five-minute*
prediction markets.

The three files here share a schema and a capture method, so they are directly
comparable. The difference is already visible in the summary statistics:

| | 5m | 15m | 4h |
| --- | ---: | ---: | ---: |
| One side of the book empty | 16.9% | 9.3% | 0.5% |
| Crossed quotes | 3.24% | 3.08% | 1.50% |
| One side empty, final 60s | 76.2% | 85.4% | 46.5% |
| Median bid levels per snapshot | 39 | 42 | 46 |

Shorter contracts spend far more of their life one-sided. A 4-hour contract holds
a two-sided market for 99.5% of its existence; a 5-minute contract does not. All
three thin out near settlement, but by very different amounts — 85.4% of 15-minute
snapshots in the final minute have an empty side, against 46.5% for 4-hour.

---

## Schema

One row per (token, snapshot). Each market has two tokens, one per outcome.

| Column | Type | Meaning |
| --- | --- | --- |
| `market_slug` | string | Polymarket market identifier, e.g. `btc-updown-5m-1787551200` |
| `coin` | string | `btc`, `eth`, `sol`, `xrp`, `bnb`, `doge`, `hype`, `zec` |
| `contract_length` | string | `5m`, `15m` or `4h` |
| `token_id` | string | Polymarket ERC-1155 token id for this outcome |
| `outcome` | string | Outcome label, `Up` or `Down` |
| `winner` | string | The outcome that settled true, `yes` or `no` |
| `is_winning_outcome` | bool | Whether *this* token is the one that won |
| `market_end_at` | timestamp[us, UTC] | Settlement time |
| `captured_at` | timestamp[us, UTC] | When this snapshot was recorded |
| `seconds_to_close` | double | `market_end_at - captured_at`, in seconds |
| `bid_prices` | list<double> | Bid ladder, **descending**, best first |
| `bid_sizes` | list<double> | Sizes aligned index-for-index with `bid_prices` |
| `ask_prices` | list<double> | Ask ladder, **ascending**, best first |
| `ask_sizes` | list<double> | Sizes aligned index-for-index with `ask_prices` |
| `best_bid` | double | `bid_prices[0]`, or null when the bid side is empty |
| `best_ask` | double | `ask_prices[0]`, or null when the ask side is empty |
| `crossed` | bool | `best_bid >= best_ask` |

Prices are probabilities in [0, 1]. Sizes are share counts.

---

## Loading

```python
import pandas as pd

df = pd.read_parquet("updown_5m.parquet")

# the book five seconds before settlement
final = df[df.seconds_to_close <= 5]

# depth on the winning side, by time to close
won = df[df.is_winning_outcome]
won.groupby(won.seconds_to_close // 30 * 30).bid_sizes.apply(
    lambda s: s.explode().astype(float).sum()
)
```

All three files share a schema, so they concatenate:

```python
import glob
df = pd.concat(pd.read_parquet(f) for f in glob.glob("updown_*.parquet"))
df.groupby("contract_length").crossed.mean()
```

With DuckDB, no download of the whole set required:

```sql
SELECT contract_length,
       avg(crossed)                       AS crossed_rate,
       avg(best_bid IS NULL OR best_ask IS NULL) AS one_sided_rate
FROM read_parquet('updown_*.parquet')
GROUP BY 1;
```

---

## How the data was captured

A websocket collector subscribed to Polymarket's CLOB book feed writes a row
whenever a token's book changes. Snapshots are sparse by design: an unchanged book
is not rewritten, so gaps between consecutive rows mean the book did not move.

Polymarket's feed carries roughly 24,000 events per second across all markets.
Any collector drops messages at that rate, and the delta stream carries no
sequence numbers, so a dropped message cannot be detected from the stream itself.
To bound the resulting drift, the collector re-reads the full book from
Polymarket's REST API **every 60 seconds** and replaces its in-memory state.

That reconcile interval is the main quality control. It is also why data of this
kind cannot be reconstructed after the fact: replaying an archived delta stream
has nothing to re-read against, and the errors accumulate without bound. Measured
against a public Polymarket event archive, replaying its own delta stream between
its own snapshots diverged at 67.8% of 381,093 checkpoints.

---

## Known properties, disclosed rather than cleaned

**One-sided books are real.** 16.9% of 5-minute rows have an empty bid or ask
side, rising to 76.2% in the final minute. As a binary market resolves, nobody
offers the losing side. This is market behaviour, not missing data, but it will
break code that assumes two-sided books. Guard before indexing `bid_prices[0]`.

**Crossed quotes are flagged, not removed.** 1.50% to 3.24% of rows have
`best_bid >= best_ask`, which is not a tradeable state. It means one side of the
in-memory book was stale when the snapshot was taken. Dropping these rows would
hide the capture's true error rate, so they are kept and flagged. Filter with
`df[~df.crossed]` if your analysis needs clean books.

**One collector regime throughout.** On 2026-08-20 the collector began re-reading
every book from Polymarket's REST API on a fixed 60-second interval. This dataset
begins on 2026-08-21, so every row was captured under that regime. A market is
included only when its *entire* lifetime falls inside the window — none are
half-captured across the change.

The difference is measurable. In the final 60 seconds of 5-minute markets, crossed
books ran at 3.54% before the change and 1.92% after. Roughly a million rows
captured under the earlier regime were excluded rather than published alongside.

**The `crossed` column is derived, not copied.** The warehouse column defaults to
false for rows written before it was added, so this dataset computes the flag from
`best_bid >= best_ask` at build time. Verified against the ladders on all 897,192
rows.

**Markets are sampled, not exhaustive.** Markets were drawn at random from those
that resolved in the window rather than taken in sequence, so the sample is not
one contiguous block of market conditions. It is also not the full population:
many more markets resolved in the period than are included here.

**Rows split evenly between winning and losing outcomes** (50.0% in each file),
because both tokens of a market are captured.

---

## Verifying the data yourself

Every claim above is reproducible from the files:

```python
import pandas as pd
df = pd.read_parquet("updown_5m.parquet")

(df.best_bid >= df.best_ask).mean()                    # crossed rate
(df.bid_prices.str.len() == 0).mean()                  # empty bid side
df.groupby(df.seconds_to_close // 60).crossed.mean()   # crossed by minute
(df.bid_prices.apply(lambda r: all(r[i] >= r[i+1] for i in range(len(r)-1)))).all()
```

The last line asserts every bid ladder is correctly ordered. It should return
`True` for all three files.

---

## Citation

```bibtex
@dataset{polyorderbooks_updown_l2,
  title     = {Polymarket Crypto Up/Down Order Books: 1-Second L2 Depth
               Across Three Contract Lengths},
  author    = {PolyOrderbooks},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22084114},
  url       = {https://doi.org/10.5281/zenodo.22084114}
}
```

That is the concept DOI: it always resolves to the newest version. To pin exact
data, cite the version DOI instead — `10.5281/zenodo.22084115` for v1.0.0, or the
DOI shown on the Zenodo page for whichever version you downloaded.

In APA:

> PolyOrderbooks. (2026). *Polymarket Crypto Up/Down Order Books: 1-Second L2
> Depth Across Three Contract Lengths* [Dataset]. Zenodo.
> https://doi.org/10.5281/zenodo.22084114

---

## Licence

Creative Commons Attribution 4.0 International (CC BY 4.0). Use it commercially,
modify it, redistribute it — please cite it.

---

## Related resources

| | |
| --- | --- |
| Live API | [polyorderbooks.com](https://polyorderbooks.com) |
| API documentation | [docs.polyorderbooks.com](https://docs.polyorderbooks.com) |
| OpenAPI specification | [docs.polyorderbooks.com/openapi.json](https://docs.polyorderbooks.com/openapi.json) |
| Python client | [pypi.org/project/polyorderbooks](https://pypi.org/project/polyorderbooks/) · `pip install polyorderbooks` |
| Client source | [github.com/polyorderbooks/polyorderbooks-python](https://github.com/polyorderbooks/polyorderbooks-python) |
| Runnable examples | [github.com/polyorderbooks/python-examples](https://github.com/polyorderbooks/python-examples) |
| Smaller 5-minute dataset | [huggingface.co/datasets/polyorderbooks/polymarket-crypto-5min-l2](https://huggingface.co/datasets/polyorderbooks/polymarket-crypto-5min-l2) |
| Same on Kaggle | [kaggle.com/datasets/polyorderbooks/polymarket-crypto-5min-orderbooks](https://www.kaggle.com/datasets/polyorderbooks/polymarket-crypto-5min-orderbooks) |

The live API covers considerably more than this dataset: all Polymarket crypto
markets including price thresholds and event questions, with history beyond the
capture window here. A free tier queries at the same 1-second resolution.

---

## Contact

Questions about the data, corrections, or requests for a wider capture:

- **Email** — [contact@polyorderbooks.com](mailto:contact@polyorderbooks.com)
- **X** — [@polyorderbooks](https://x.com/polyorderbooks)
- **Issues** — [github.com/polyorderbooks/polyorderbooks-python/issues](https://github.com/polyorderbooks/polyorderbooks-python/issues)

If you publish work using this dataset, I would genuinely like to hear about it.

---

## Disclaimer

Not affiliated with, endorsed by, or connected to Polymarket. Market data is
reproduced for research purposes. Nothing here is financial advice, and past
market behaviour does not predict future outcomes.
