# Prior work: market making on prediction markets, and Polymarket data

Two literature passes run on 2026-10-15. Each source is marked by how much of
it was read: full, sections, or abstract and snippets only.

## Nobody trains a deep market maker on real Polymarket data

No paper, preprint or thesis we found does this. Here is how the closest
works are set up.

| Work | Task | Data | Environment | Fills | Learning | Split | Read |
|---|---|---|---|---|---|---|---|
| Jain 2026, Harvard AB thesis | Market making at the best price, plus a multi-level variant | Kalshi NBA in-game L2, 50 ms snapshots | Calibrated simulator: Hawkes events plus log-normal sizes, with hand-tuned scales | FIFO from the back of the queue, adverse fills, 100 ms delay | PPO, MLP [128,128], discrete post/cancel per side, shaped rewards | None on real data; evaluated in the simulator | full |
| Feil & Nendel, arXiv 2607.17991 | Market making (closed-form HJB) | None (illustrative parameters) | Logit belief diffusion | Poisson, exponential in the spread | Finite-difference HJB; penalty on settlement variance `q^2 p(1-p)` | Monte Carlo | full |
| Dalen, arXiv 2510.15205 | Pricing kernel plus an Avellaneda-Stoikov recipe | "20 Polymarket events"; the reported table is synthetic | Logit jump-diffusion | – | Kalman filter plus EM calibration | Chronological thirds | full |
| PredictionMarketBench, arXiv 2602.00133 | Backtest harness | Kalshi L2 plus trades, 4 episodes | Replay plus queue | Queue behind displayed size, filled by trade prints | Rules and LLMs, no RL | – | full |
| Singh, TUM MSc thesis 2025 | Descriptive LP P&L | Polygon OrderFilled events, Polygonscan, Gamma | – | – | – | – | full |
| OpenMarket, arXiv 2607.26245 | Directional, BTC 15m markets | Self-recorded Polymarket and Binance websockets, 54 days | Walk-forward | Taker with slippage | Logistic regression, which does not beat the mid (Brier 0.165 vs 0.163); Polymarket lags Binance by a median of about 347 ms | Walk-forward | sections |

Taken together:
- The theory papers model fill intensities, but their parameters are made up.
- The one learning paper evaluates only inside the simulator it was trained in.
- Replay harnesses exist, but nobody trains RL in them.
- Nobody models Polymarket's liquidity rewards or maker rebates, although
  Singh shows rewards are a first-order part of LP P&L.

The gaps this project fills:
- training on replayed real L2 data with an explicit queue model;
- a time-based out-of-sample split;
- fees, rebates and rewards in the objective;
- Binance as the source of adverse selection;
- realistic baselines.

## What the descriptive papers use as data

| Paper | Source | Can see | Cannot see |
|---|---|---|---|
| Saguillo et al., arXiv 2508.03474 | Polygon RPC: OrderFilled, split and merge events | Fills and wallets | The book, because matching is off-chain |
| Dubach, arXiv 2604.24366 | A third-party websocket archive (30B events over 52 days) joined to OrderFilled | L2 book changes, fills, maker wallets | Taker identity in the feed; inferring direction from the book is only 59% accurate |
| Qin & Yang, Polymarket-v1, arXiv 2606.04217 | V1 OrderFilled tape, 1.2B rows (Hugging Face `TimeSeventeen/Polymarket-v1`) | Fills, maker/taker roles | Order book snapshots |
| Cheng et al., arXiv 2605.00864 | Polling `GET /book` every 3.6–5.5 s | Top of book | Fills |
| Bürgi, Deng & Whelan, CEPR DP 20631 (Kalshi) | Kalshi trades API | Maker/taker labels | Identities, the book |

The order book is never on-chain. Signed orders are matched off-chain, and
only the matched fills settle on Polygon.

## On-chain reconciliation for one market

Market: `bnb-updown-5m-1787277900`. Scripts are in this folder.

- Contracts: CTFExchangeV2 `0xE111180000d2663C0091e4f400237545B87B996B` and
  NegRiskCtfExchangeV2 `0xe2222d279d744050d28e00520010520000310F59`.
  Deployed at block 84,902,353.
- `tokenId` is not indexed in `OrderFilled`, so an extraction has to pull every
  exchange log and filter locally. That is about 85 OrderFilled logs per block.
- The data-api matches the chain exactly: 152 of 152 fills, 70 taker and 82
  maker. A data-api timestamp is the block time, about 2–4 s after the
  off-chain match.
- The chain adds fees, order hashes and explicit roles. Transaction calldata
  adds each order's original size, limit price and creation time:
  - 67 of the 82 maker fills were partial;
  - a maker order's age at fill has a median of 14.7 s.
- Archive RPCs that work without a key are `rpc-mainnet.matic.quiknode.pro`
  (about 900 blocks/s with 8 concurrent requests),
  `gateway.tenderly.co/public/polygon` and `polygon.drpc.org`. Publicnode
  serves only about the last 40 days.
- Extraction cost:
  - our 805 markets: about 22M logs, 5–35 min;
  - 3 months of crypto up/down markets: about 450M logs, 4–11 h.

## What we use

- **Fills:** the data-api, both `takerOnly` modes, which is equivalent to the
  chain. Pulling directly from the chain is only needed for fees and order
  details.
- **More training data:** extra days of on-chain fills give more flow and
  markout statistics, but no new book states. New L2 episodes need our own
  websocket recorder.
