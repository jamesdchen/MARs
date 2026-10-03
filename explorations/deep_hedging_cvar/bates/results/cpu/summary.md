| strategy | val CVaR0.95 | mean | std | VaR0.95 | CVaR0.95 | stock trades | swap trades | device | train wall time (s) | train sim steps | tuning wall time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| no hedge | 10.625 | 0.058 | 3.890 | 7.168 | 10.707 | 0.0 | 0.0 | – | – | – | – |
| BS delta | 13.164 | 1.828 | 3.206 | 6.975 | 13.221 | 30.0 | 0.0 | – | – | – | – |
| Bates delta | 13.624 | 1.799 | 3.345 | 7.294 | 13.700 | 30.0 | 0.0 | – | – | – | – |
| Bates min-variance delta | 12.540 | 1.771 | 3.019 | 6.444 | 12.561 | 30.0 | 0.0 | – | – | – | – |
| Bates delta-vega | 6.425 | 2.583 | 2.060 | 3.798 | 6.343 | 30.0 | 30.0 | – | – | – | – |
| band, theory point | 5.507 | 0.951 | 2.140 | 4.321 | 5.470 | 3.3 | 1.6 | – | – | – | – |
| tuned no-trade band | 3.445 | 1.006 | 1.943 | 2.972 | 3.447 | 6.6 | 1.0 | cpu | – | 96M | 170 |
| PPO, PufferLib 3.0 + C env, entropy 0.01 | 4.014 | 1.318 | 4.584 | 3.661 | 4.023 | 18.0 | 1.0 | cpu | 1684 | 200M | – |
| pathwise, ste gate | 2.881 | 1.467 | 1.989 | 2.673 | 2.874 | 23.6 | 1.0 | cpu | 582 | 246M | – |
| pathwise, hybrid gate | 2.818 | 1.578 | 1.985 | 2.640 | 2.811 | 27.4 | 1.0 | cpu | 565 | 246M | – |
