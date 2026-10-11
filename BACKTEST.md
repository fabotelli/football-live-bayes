# Backtest: drive model v2.1 (2026-10-10)

Holdout states only: the drive tables never saw these seasons. Per state: simulate the rest of the game (1000 sims), compare
P(home wins) with the real result and with a benchmark WP, and check P(final total > closing total) calibration.
`scripts/diag_remaining.ts` breaks predicted vs actual remaining points down by phase, score gap and spread.

**How to read it.** Matching or beating the benchmark says the state model is sound. It does **not** show a trading edge:
Kalshi's live book is the real opponent, and no historical Kalshi prices were available here.

## Open: NCAAF in-game under-lean (investigated 2026-10-11, NOT fixed)
Game-weighted holdout test (one random state per game per phase, 1,031 games; per-game standard errors, because states within
a game share one outcome and the play-weighted diagnostic overstates precision): model minus actual remaining points
Q1 −0.44 ± 0.46, **Q2 −0.81 ± 0.37**, Q3 −0.47 ± 0.30, Q4 +0.02 ± 0.20. Pregame there is no lean (207 holdout games:
model mean − line +0.37, actual +0.37).

Where it comes from: **when** college points are scored, not how many. The model front-loads: real share of points by window
(>5:00 / 5:00–2:00 / last 2:00, per half) is 37.6 / 4.8 / 9.0 | 39.2 / 4.8 / 4.6 %, the model's 38.8 / 5.9 / 8.1 | 37.3 / 5.1 / 4.8.
From real first-half states with 2:00–5:00 left (1,127 training games): points before the 2:00 mark real 2.57 vs model 2.96,
from 2:00 to halftime real 4.31 vs model 3.51. In-sample (training seasons) the second half is under-forecast from Q2 states by
1.13 ± 0.30. The NFL's profile matches (35.9 / 5.5 / 9.8 | 37.5 / 5.6 / 5.7 vs real 35.7 / 4.7 / 10.4 | 37.4 / 6.4 / 5.4).
Likely cause: college's end-of-half clock (stops on first downs inside 2:00) packs possessions and points into the last two
minutes, which drive-level tables keyed on the clock at drive start only partly capture.

Ruled out / fixed on the way: game-environment effect (added; fixes pregame width, moves the lean ~0.1); drive-length
narrowing (not the cause); drive lengths by half (added, data-backed, little effect); evaluation weighting (explains part of the
apparent lean in Q1/Q4, not Q2–Q3); a curve-binning bug that filed buzzer scores into the next half (fixed). Per-window scoring
factors fit the shares but need ~0.5 / ~1.7 either side of 2:00, which would misprice red-zone states, so they are not used.

## Changes in v2.1: NCAAF totals distribution tightened
College second halves were simulated with too few drives (10.2 vs 11.5 real) and too much spread in the count (sd 3.0 vs
2.35), so each drive carried too many points and the remaining-total distribution was ~10% too wide before Q4.
- Drive-length tables: "drives" over 10 minutes (feed gaps; college maxima were 1100–1800 s) dropped, quantiles in 2.5% steps
  capped at p99 (the engine interpolates between quantiles, so the old top step was drawn uniformly up to the raw max).
- Fitted `durationShrink` per league (`scripts/fit_dispersion.ts`): each drive length is drawn as mean + k·(draw − mean), with k
  set so the simulated 2H drive-count sd matches real games. NCAAF k = 0.654; NFL k = 1 (its count was already right).
  Check that was not fitted: NCAAF 2H points sd 10.76 vs 10.72 real (was 11.81).
- NCAAF predicted sd vs actual rmse of remaining points: Q2 12.52 vs 11.98 (was 13.61), Q3 9.60 vs 9.69 (was 10.49).
  Q4 with >5 min left is now slightly narrow (6.57 vs 7.35).
- Tail calibration of P(over): 0.15 → 0.131 actual (was 0.100), 0.25 → 0.221 (0.204), 0.75 → 0.777 (0.795). What's left
  is a small lean (the model's mean is ~0.5 points low; bins above 0.5 come in 2–3 points high), not width.

## Changes in v2
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
states 29381, sims/state 1000, 509.8s
home-win  phase          n     logloss model / vegas_wp     brier model / vegas_wp
          ALL           29332   0.4543 / 0.4516           0.1507 / 0.1502
          Q1             6655   0.5871 / 0.5889           0.2015 / 0.2022
          Q2             8092   0.5114 / 0.5127           0.1710 / 0.1716
          Q3             6579   0.4304 / 0.4270           0.1406 / 0.1394
          Q4 >5m         4653   0.3411 / 0.3305           0.1052 / 0.1046
          Q4 last 5m     3353   0.2572 / 0.2475           0.0835 / 0.0798

P(final total > closing total) from in-game states, calibration:
  bin        n    mean_p  actual
  0.0-0.1   2938   0.035   0.061
  0.1-0.2   1869   0.150   0.189
  0.2-0.3   1978   0.251   0.255
  0.3-0.4   2614   0.352   0.381
  0.4-0.5   3539   0.452   0.494
  0.5-0.6   2995   0.542   0.594
  0.6-0.7   1896   0.647   0.696
  0.7-0.8   1358   0.749   0.795
  0.8-0.9   1152   0.848   0.837
  0.9-1.0   3176   0.984   0.984
  total-over brier 0.1656 (n=23515)
```

## NCAAF: train 2023-2024 (FBS vs FBS), holdout 2025, 2026 (to date); 10,583 sampled states / 1,031 games
Benchmark: cfbfastR `home_wp_before`. **It is a weak benchmark** (Q1 log loss 0.603, close to a coin flip), so the model's
large lead mostly reflects that.

- **Totals:** width fixed in v2.1 (see above). What's left is a small under-lean: the mean is ~0.5 points low and the 0.8–0.9
  bin comes in at 0.905.

```
states 10583, sims/state 1000, 751.0s
home-win  phase          n     logloss model / vegas_wp     brier model / vegas_wp
          ALL           10553   0.3633 / 0.4463           0.1177 / 0.1494
          Q1             2536   0.4705 / 0.6028           0.1570 / 0.2095
          Q2             2919   0.3987 / 0.5113           0.1302 / 0.1724
          Q3             2433   0.3346 / 0.3810           0.1078 / 0.1246
          Q4 >5m         1650   0.2642 / 0.2996           0.0823 / 0.0965
          Q4 last 5m     1015   0.2240 / 0.2633           0.0649 / 0.0779

P(final total > closing total) from in-game states, calibration:
  bin        n    mean_p  actual
  0.0-0.1   1295   0.033   0.031
  0.1-0.2    793   0.149   0.131
  0.2-0.3    895   0.252   0.221
  0.3-0.4   1161   0.351   0.359
  0.4-0.5   1631   0.454   0.465
  0.5-0.6   1232   0.544   0.569
  0.6-0.7    858   0.647   0.670
  0.7-0.8    695   0.749   0.777
  0.8-0.9    527   0.849   0.905
  0.9-1.0   1466   0.983   0.988
  total-over brier 0.1542 (n=10553)
```
