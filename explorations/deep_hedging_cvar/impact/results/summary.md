| strategy | val CVaR0.95 | mean | std | VaR0.95 | CVaR0.95 | trades | device | train wall time (s) | train sim steps | tuning wall time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| no hedge | 10.119 | 0.007 | 3.470 | 7.460 | 10.134 | 0.000 | – | – | – | – |
| BS delta, daily | 2.504 | 1.388 | 0.470 | 2.218 | 2.519 | 29.714 | – | – | – | – |
| tuned band (h=0.10, edge=0.5) | 2.003 | 0.766 | 0.564 | 1.724 | 2.013 | 9.212 | cpu | 10 | 240M | – |
| pathwise, hard gate (CPU) | 2.075 | 1.258 | 0.577 | 1.921 | 2.077 | 30.000 | cpu | 421 | 983M | – |
| pathwise, ste gate (CPU) | 2.061 | 1.083 | 0.558 | 1.860 | 2.069 | 22.977 | cpu | 404 | 983M | – |
| pathwise, sigmoid gate (CPU) | 15.933 | 4.809 | 4.554 | 13.721 | 16.051 | 5.191 | cpu | 426 | 983M | – |
| PPO, PufferLib 3.0 + Cython env (CPU) | 2.052 | 0.612 | 0.778 | 1.728 | 2.045 | 4.663 | cpu | 608 | 120M | – |
