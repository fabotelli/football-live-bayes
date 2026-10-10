# Backtest: drive model v1 (2026-10-10)

Holdout states only: the drive tables never saw these seasons. Per state: simulate the rest of the game (1000 sims), compare
P(home wins) with the real result and with a benchmark WP, and check P(final total > closing total) calibration.

**How to read it.** Matching or beating the benchmark says the state model is sound. It does **not** show a trading edge:
Kalshi's live book is the real opponent, and no historical Kalshi prices were available here.

## NFL: train 2016-2023, holdout 2024, 2025, 2026 (to week 5); 29,381 states / 609 games
Benchmark: nflverse `vegas_home_wp`, a strong market-anchored live WP model.

- **Q1–Q2:** level with the benchmark, slightly better (log loss 0.588 vs 0.589 in Q1, 0.512 vs 0.513 in Q2).
- **Second half:** behind the benchmark, worst in the last 5 minutes. That's clock strategy (two-minute drill, timeout use, onside kicks), which the
  drive tables only approximate. The shadow loop blocks the last 5 minutes by policy.
- **Totals:** the model runs ~1–1.5 points low on remaining points. Most of that matches the real market miss in the holdout:
  2024 and 2025 went over the closing total in 54% / 52% of games (+1.5 / +1.2 points vs the line).
- **Bug found and fixed in this pass:** the kneel-out rule assumed the defence had no timeouts on simulated drives, which ended
  live games early (Q4 >5m log loss 0.349 → 0.338).

```
states 29381, sims/state 1000, 381.7s
home-win  phase          n     logloss model / vegas_wp     brier model / vegas_wp
          ALL           29332   0.4545 / 0.4516           0.1509 / 0.1502
          Q1             6655   0.5879 / 0.5889           0.2019 / 0.2022
          Q2             8092   0.5117 / 0.5127           0.1712 / 0.1716
          Q3             6579   0.4320 / 0.4270           0.1409 / 0.1394
          Q4 >5m         4653   0.3383 / 0.3305           0.1054 / 0.1046
          Q4 last 5m     3353   0.2567 / 0.2475           0.0833 / 0.0798

P(final total > closing total) from in-game states, calibration:
  bin        n    mean_p  actual
  0.0-0.1   2924   0.035   0.061
  0.1-0.2   1827   0.150   0.186
  0.2-0.3   1939   0.250   0.258
  0.3-0.4   2608   0.352   0.378
  0.4-0.5   3575   0.451   0.490
  0.5-0.6   2996   0.542   0.587
  0.6-0.7   1893   0.647   0.696
  0.7-0.8   1348   0.749   0.795
  0.8-0.9   1146   0.848   0.824
  0.9-1.0   3259   0.984   0.982
  total-over brier 0.1654 (n=23515)
```

## NCAAF: train 2023-2024 (FBS vs FBS), holdout 2025, 2026 (to date); 10,584 sampled states / 1,031 games
Benchmark: cfbfastR `home_wp_before`. **It is a weak benchmark** (Q1 log loss 0.588 is close to a coin flip). It does not
appear to use the closing spread properly, so the model's large lead here mostly reflects that, not a strong model.

- **Totals:** the opposite of the NFL. The model over-predicts remaining points by ~1–2 (most in Q3 and Q4 with more than 5 minutes left), and the
  holdout went under the line (46% overs, −1.1 points). The likely cause is blowout / garbage-time clock behaviour, which a single λ per team
  doesn't capture. This needs work before NCAAF totals are priced.

```
states 10584, sims/state 1000, 462.9s
home-win  phase          n     logloss model / vegas_wp     brier model / vegas_wp
          ALL           10358   0.3658 / 0.4397           0.1165 / 0.1461
          Q1             2498   0.4704 / 0.5878           0.1567 / 0.2033
          Q2             2870   0.4005 / 0.5026           0.1306 / 0.1692
          Q3             2358   0.3269 / 0.3762           0.1039 / 0.1232
          Q4 >5m         1619   0.2652 / 0.2872           0.0791 / 0.0911
          Q4 last 5m     1013   0.2609 / 0.2874           0.0668 / 0.0808

P(final total > closing total) from in-game states, calibration:
  bin        n    mean_p  actual
  0.0-0.1   1086   0.032   0.020
  0.1-0.2    662   0.149   0.109
  0.2-0.3    775   0.251   0.165
  0.3-0.4   1135   0.352   0.268
  0.4-0.5   1814   0.452   0.418
  0.5-0.6   1358   0.544   0.533
  0.6-0.7    920   0.647   0.616
  0.7-0.8    668   0.747   0.771
  0.8-0.9    566   0.847   0.850
  0.9-1.0   1374   0.983   0.951
  total-over brier 0.1614 (n=10358)
```
