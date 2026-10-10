# football-live-bayes: NFL + NCAAF live model for Kalshi (fresh restart)

> **STATUS: SHADOW ONLY. No order code exists in this repo.** The TypeScript side is GET-only (`src/kalshi/gateway.ts`
> `readOnlyFetch`), reads no credentials, and `tests/readonly.test.ts` scans `src/` to keep it that way. Nothing here touches the
> old football bots (`kalshi_ht/ht_bot.py`, `model_live_bridge.py`) or `nhl-live-bayes`; those are parked as-is (see bottom).

## Why a restart, and what's different
The old football work (`kalshi_ht/ht_bot.py`) priced NFL/NCAAF from **fixed lookup tables**: a halftime margin×spread table and
a Q4 lead×minutes table, with "x" cells where n<50. It only looked at two moments of a game, ignored field position/down/distance
except as a coarse "situation" label, and had no total/spread/team-total pricing.

This project applies the NHL live-Bayes approach (`nhl-live-bayes`) to football:
1. **The market sets the prior.** The closing spread + total are inverted into one scoring multiplier per offense (λ), solved so
   the line splits the simulated outcomes 50/50 (lines are medians, not means). We do not try to out-handicap the close; the
   edge has to come from pricing *game state* better or faster than the live book.
2. **A drive-level state model does the live work.** From any snapshot (score, clock, possession, yards to goal, down, to-go,
   timeouts), we simulate the rest of the game drive by drive, using empirical tables fitted on real play-by-play:
   - drive result `P(TD/FG/punt/TO/downs/missed FG/end of half/defensive TD/safety | half-clock bucket, late-game score bucket,
     yards-to-goal bin, down, to-go bin)`, smoothed toward parent cells;
   - how long the rest of the drive takes (separate tables for a drive in progress and for a fresh drive);
   - where the next drive starts (by how the last one ended); kickoff start position per season (the NFL kickoff rules changed
     in 2024 and 2025);
   - TD points (6/7/8), victory-formation kneel-outs, OT (NFL: some ties stand; NCAAF: never tied).
3. **One engine gives every market.** The simulated final-score distribution prices moneyline, spread, full-game total,
   team totals and 1H totals consistently (`src/signals/fair.ts`).
4. **Shadow first, validated before money.** Same discipline as NHL: shadow JSONL → grade against settlements → only then a
   separate, armed, capped bridge (not built yet).

## Layout
| path | what |
|---|---|
| `scripts/fetch_nfl.py`, `scripts/fetch_ncaaf.py` | download nflverse / cfbfastR play-by-play into `data/cache/` (public GitHub release assets) |
| `scripts/build_nfl.py`, `scripts/build_ncaaf.py` | fit drive tables → `data/models/{nfl,ncaaf}_drive_v2.json`; export holdout states for the backtest |
| `src/model/drive.ts` | the engine: market-prior calibration (λ) + seeded Monte Carlo over drives → `SimResult` |
| `scripts/backtest.ts` | holdout backtest vs real outcomes and vs the benchmark WP (nflverse `vegas_home_wp` / cfbfastR `home_wp_before`) |
| `src/data/espn.ts` | ESPN scoreboard → `GameState` (fail closed on dead balls, kickoffs, missing field position, OT) |
| `src/kalshi/gateway.ts` | read-only Kalshi market data, fee formula, series names (CONFIRMED vs to-be-confirmed) |
| `src/signals/fair.ts`, `src/signals/classify.ts` | market → fair P(YES); edge/fee/price-band/liquidity classification (NHL rule shape) |
| `src/live/shadow.ts` | poll loop: ESPN + Kalshi → `runtime/state/signals-<ETdate>.jsonl` + `latest.json`. Never trades. |

## Run
```
npm run build:nfl && npm run build:ncaaf          # fetch + fit (needs python3, pandas, pyarrow for NCAAF)
npm test                                           # node >= 22.18 (runs .ts directly)
node scripts/backtest.ts --league nfl --sims 1000  # or --league ncaaf; --max N samples N states
node scripts/discover_series.ts                    # confirm Kalshi football series + rules text (needs Kalshi API access)
node src/live/shadow.ts [--once] [--poll 15]       # needs ESPN + Kalshi access
```

## Backtest (holdout seasons, never used to fit the tables)
See `BACKTEST.md` for the latest numbers and how to read them.

## Open items before any money
- **Run the shadow loop on the server.** This repo was built in a sandbox without ESPN/Kalshi access: the ESPN parser and
  Kalshi matching are unit-tested on fixtures only.
- **Confirm series + settlement** with `scripts/discover_series.ts`: full-game total/spread/team-total series names are guesses
  (only `KXNFLGAME`, `KXNFL1HTOTAL`, `KXNFL1HTEAMTOTAL`, `KXNCAAFGAME` are confirmed from the old bots), and how `KXNFLGAME` settles a tie.
- **Kalshi price history.** The backtest scores the model against outcomes and a strong benchmark WP, not against Kalshi
  quotes. The real question (does Kalshi's live book lag the state?) needs Kalshi trades/candlesticks for past games, or the
  shadow log.
- **Late game.** The model trails the benchmark in the last 5 minutes (clock strategy); the shadow loop blocks there by policy.
- **NCAAF totals tails.** The v1 over-prediction is fixed (it was a play-ordering data bug, see BACKTEST.md); the remaining-total
  distribution is still a little too wide in college (underconfident at the extremes).
- **NCAAF team codes.** ESPN → Kalshi college codes differ a lot; unmatched games log `NO_MARKET_MATCH` (never mis-priced).
  Needs a mapping table built from `KXNCAAFGAME` tickers (the old `ht_bot.NCAAFTickers` name-normalised fallback is a good model).
- 1H team totals need per-team first-half distributions (not simulated yet); no live Bayesian update of λ from in-game
  performance yet (NHL found in-game pace added nothing beyond the prior; to be tested here with the backtest).

## Parked (not touched by this repo)
- `nhl-live-bayes`: live NHL (real money) keeps running as before.
- `kalshi_ht/ht_bot.py` NFL+NCAAF halftime/Q4 bot: **starts LIVE by default** (real orders unless `--paper`/`--fcs-watch`).
  Parking it means stopping it, or running it with `--paper`, on the server.
- `kalshi_ht/model_live_bridge.py` (NFL 1H totals bridge, real money): stop on the server to park.
