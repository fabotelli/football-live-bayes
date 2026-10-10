# Backtest: drive model v2 (2026-10-10)

Holdout states only: the drive tables never saw these seasons. Per state: simulate the rest of the game (1000 sims), compare
P(home wins) with the real result and with a benchmark WP, and check P(final total > closing total) calibration.
`scripts/diag_remaining.ts` breaks predicted vs actual remaining points down by phase, score gap and spread.

**How to read it.** Matching or beating the benchmark says the state model is sound. It does **not** show a trading edge:
Kalshi's live book is the real opponent, and no historical Kalshi prices were available here.

## Changes since v1
- **NCAAF data bug (the cause of the "totals over-prediction").** Plays were sorted by `id_play`, an 18-digit id that the
  cfbfastR parquet stores as float64, so it had lost precision and the sort scrambled plays within games. That corrupted the
  holdout final scores, pre-play scores and drive end times. Now sorted by period → clock → play number, with scores taken as a
  running max. NCAAF predicted minus actual remaining points by phase was +1.2 to +2.2; it is now −0.4 to +0.1. The
  "market went under in 2025" reading in v1 was the same bug: the 2025 holdout is actually +0.4 vs the closing total.
- **Strength double count (both leagues).** Drive outcomes are now conditioned on the score gap in both halves (7 buckets incl.
  9–16 and 17+). Each cell stores the mean pregame strength of its offenses (college teams leading by 17+ average 1.29× league
  strength), and the engine divides each team's λ by it, so strength isn't counted twice.

## NFL: train 2016-2023, holdout 2024, 2025, 2026 (to week 5); 29,381 states / 609 games
Benchmark: nflverse `vegas_home_wp`, a strong market-anchored live WP model.

- **First half:** ahead of the benchmark (Q1 log loss 0.586 vs 0.589, Q2 0.510 vs 0.513).
- **Second half:** behind it, worst in the last 5 minutes (clock strategy). The shadow loop blocks the last 5 minutes by policy.
- **Totals:** the model runs ~1 point low on remaining points. This matches the real market: 2024 and 2025 went over the closing total in
  54% / 52% of games (+1.5 / +1.2 points).

```
states 29381, sims/state 1000, 388.8s
home-win  phase          n     logloss model / vegas_wp     brier model / vegas_wp
          ALL           29332   0.4529 / 0.4516           0.1503 / 0.1502
          Q1             6655   0.5863 / 0.5889           0.2012 / 0.2022
          Q2             8092   0.5100 / 0.5127           0.1705 / 0.1716
          Q3             6579   0.4290 / 0.4270           0.1402 / 0.1394
          Q4 >5m         4653   0.3387 / 0.3305           0.1052 / 0.1046
          Q4 last 5m     3353   0.2556 / 0.2475           0.0830 / 0.0798

P(final total > closing total) from in-game states, calibration:
  bin        n    mean_p  actual
  0.0-0.1   2848   0.035   0.059
  0.1-0.2   1845   0.150   0.184
  0.2-0.3   1960   0.251   0.256
  0.3-0.4   2636   0.352   0.375
  0.4-0.5   3566   0.452   0.484
  0.5-0.6   3019   0.541   0.593
  0.6-0.7   1941   0.647   0.697
  0.7-0.8   1365   0.748   0.799
  0.8-0.9   1167   0.848   0.834
  0.9-1.0   3168   0.984   0.985
  total-over brier 0.1654 (n=23515)
```

## NCAAF: train 2023-2024 (FBS vs FBS), holdout 2025, 2026 (to date); 10,583 sampled states / 1,031 games
Benchmark: cfbfastR `home_wp_before`. **It is a weak benchmark** (Q1 log loss 0.603, close to a coin flip), so the model's
large lead mostly reflects that.

- **Totals:** no remaining bias in the middle of the range (0.4–0.6 bins on target). In the tails the model is too cautious: it
  says 0.15 where the actual rate is 0.10, and 0.85 where it's 0.91. The remaining-total spread is a bit too wide; the next thing to tune.

```
states 10583, sims/state 1000, 512.3s
home-win  phase          n     logloss model / vegas_wp     brier model / vegas_wp
          ALL           10553   0.3638 / 0.4463           0.1178 / 0.1494
          Q1             2536   0.4715 / 0.6028           0.1571 / 0.2095
          Q2             2919   0.4013 / 0.5113           0.1302 / 0.1724
          Q3             2433   0.3352 / 0.3810           0.1079 / 0.1246
          Q4 >5m         1650   0.2649 / 0.2996           0.0827 / 0.0965
          Q4 last 5m     1015   0.2161 / 0.2633           0.0649 / 0.0779

P(final total > closing total) from in-game states, calibration:
  bin        n    mean_p  actual
  0.0-0.1   1098   0.033   0.027
  0.1-0.2    762   0.150   0.100
  0.2-0.3    825   0.252   0.204
  0.3-0.4   1204   0.352   0.314
  0.4-0.5   1776   0.454   0.455
  0.5-0.6   1320   0.544   0.558
  0.6-0.7    923   0.646   0.674
  0.7-0.8    682   0.747   0.795
  0.8-0.9    545   0.848   0.910
  0.9-1.0   1418   0.984   0.989
  total-over brier 0.1547 (n=10553)
```
