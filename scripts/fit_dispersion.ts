// Fits the two dispersion parameters of a league's drive model against real-game targets (written by the build script from
// training seasons) and writes them into data/models/<league>_drive_v2.json:
//   durationShrink  drive-length dispersion, so the simulated second-half drive-count SD matches real games (targets.h2DrivesSd)
//   (timeFactor     per half x clock window scoring factors: supported by the engine, NOT fitted; the window shares are reported)
//   gameEnv.alpha   game-environment prior strength (g ~ Gamma(alpha, alpha)), so the pregame SD of (total - line) and the
//                   second-half points SD match real games (targets.totalSd, targets.h2PointsSd)
// Reported but NOT fitted, as checks: mean of (total - line) (right skew; lines are medians) and the slope of second-half on
// first-half points (how much in-game scoring tells about the rest; the live update relies on it).
// Everything is simulated at the league's typical closing line with fixed seeds, so the search is deterministic.
//   node scripts/fit_dispersion.ts --league ncaaf
import { readFileSync, writeFileSync } from 'node:fs';
import { existsSync } from 'node:fs';
import { DriveModel, mean, type GameState } from '../src/model/drive.ts';
import { PlayModel } from '../src/model/plays.ts';

const i = process.argv.indexOf('--league'); const LEAGUE = i > 0 ? process.argv[i + 1] : 'ncaaf';
const path = new URL(`../data/models/${LEAGUE}_drive_v2.json`, import.meta.url).pathname;
const json = JSON.parse(readFileSync(path, 'utf8'));
const tg = json.targets; if (!tg?.totalSd) throw new Error('model has no dispersion targets; rebuild it');
// With an end-of-half play table on disk the fit simulates with it, as the shadow loop and backtests do (--no-plays to skip).
const playsPath = new URL(`../data/models/${LEAGUE}_plays_v1.json`, import.meta.url).pathname;
const plays = existsSync(playsPath) && !process.argv.includes('--no-plays') ? new PlayModel(JSON.parse(readFileSync(playsPath, 'utf8'))) : null;
const pr = tg.typicalPrior;
const kickoff = (half: 1 | 2): GameState => ({ half, hsr: 1800, homeScore: 0, awayScore: 0, possHome: true, kickoff: true, y: 75, down: 1, togo: 10, homeTO: 3, awayTO: 3, recv2Home: half === 2 ? true : null });
const sd = (d: Float64Array) => { const m = mean(d); let v = 0; for (let k = 0; k < d.length; k++) v += d[k] * (k - m) ** 2; return Math.sqrt(v); };

let timeFactor: number[][] = [[1, 1, 1], [1, 1, 1]];
function measure(k: number, alpha: number, full = false, withPlays = true) {
  const m = new DriveModel({ ...json, durationShrink: k, gameEnv: { alpha }, timeFactor }, undefined, withPlays ? plays : null);
  // Second half from a 0-0 kickoff with g from its prior: the unconditional second-half distribution at this line.
  const h2 = m.simulate(kickoff(2), pr, { sims: 12000, seed: 777, env: 'prior' });
  const out = { k, alpha, drivesSd: h2.drives.sd, h2Sd: sd(h2.total), totalSd: NaN, meanMinusLine: NaN, slope: NaN };
  if (full) {
    const g = m.simulate(kickoff(1), pr, { sims: 16000, seed: 778, openingCoinFlip: true, env: 'prior', pairs: true });
    out.totalSd = sd(g.total); out.meanMinusLine = mean(g.total) - pr.total;
    const { h1, h2: s2 } = g.pairs!; let m1 = 0, m2 = 0; for (let j = 0; j < h1.length; j++) { m1 += h1[j]; m2 += s2[j]; } m1 /= h1.length; m2 /= h1.length;
    let c = 0, v = 0; for (let j = 0; j < h1.length; j++) { c += (h1[j] - m1) * (s2[j] - m2); v += (h1[j] - m1) ** 2; }
    out.slope = c / v;
  }
  return out;
}

function fitShrink(alpha: number): number {
  // Fitted on the drive-level engine alone: narrowing only applies to drive-level drives, and the play-sampled end of each half
  // has its own (real) possession variance, so fitting with it attached drove k to its bound.
  if (measure(1, alpha, false, false).drivesSd <= tg.h2DrivesSd) return 1;     // only ever narrows
  let lo = 0.2, hi = 1;
  for (let it = 0; it < 12; it++) { const mid = (lo + hi) / 2; if (measure(mid, alpha, false, false).drivesSd > tg.h2DrivesSd) hi = mid; else lo = mid; }
  return Math.round(((lo + hi) / 2) * 1000) / 1000;
}

const fmt = (r: ReturnType<typeof measure>) => `k=${r.k.toFixed(3)} alpha=${String(r.alpha).padStart(4)}  2H drives sd ${r.drivesSd.toFixed(2)}  2H pts sd ${r.h2Sd.toFixed(2)}  total sd ${r.totalSd.toFixed(2)}  mean-line ${r.meanMinusLine >= 0 ? '+' : ''}${r.meanMinusLine.toFixed(2)}  1H->2H slope ${r.slope.toFixed(3)}`;
console.log(`${LEAGUE} targets at ${JSON.stringify(pr)}: 2H drives sd ${tg.h2DrivesSd}  2H pts sd ${tg.h2PointsSd}  total sd ${tg.totalSd}  | checks: mean-line ${tg.meanMinusLine >= 0 ? '+' : ''}${tg.meanMinusLine}  slope ${tg.h1h2Slope}`);
// Drive-length dispersion does not depend on g (g changes which drives score, not how long they take), so fit k once.
const k = fitShrink(0);
// Time profile: multiplicative fixed point on the six window factors. The model's window shares come from its expected-points
// curve at the typical line (lambda recalibrates each round, so only the shape moves). Normalised to a points-weighted mean of 1.
const BOUNDS = [0, 1500, 1680, 1800, 3300, 3480, 3600];   // elapsed seconds: window edges for >5:00, 5:00-2:00, last 2:00 per half
const shares = () => {
  const m = new DriveModel({ ...json, durationShrink: k, gameEnv: { alpha: 0 }, timeFactor }, undefined, plays);
  const E = BOUNDS.map((x) => m.expectedSoFar(pr, x));
  return E.slice(1).map((x, j) => (x - E[j]) / E[6]);
};
if (tg.windowShare) {
  // Reported, not fitted. Fitting scoring factors per window matched the shares but needed factors of ~0.5 and ~1.7 for the windows
  // either side of 2:00 before halftime. That would halve a red-zone touchdown chance at 3:00, i.e. misprice exactly the states
  // the live loop prices, so the factors stay at 1 until the timing mechanism is found (see BACKTEST.md).
  const fmtS = (s: number[]) => s.map((x) => (100 * x).toFixed(1).padStart(5)).join(' ');
  console.log(`  window shares % (>5:00, 5:00-2:00, last 2:00 per half), real: ${fmtS(tg.windowShare)}`);
  console.log(`  model:                                                      ${fmtS(shares())}`);
}
// Then the game-environment strength: alpha = 0 (off) or a grid from weak (CV 5%) to strong (CV 25%); pick the best fit to the two
// points-SD targets, relative errors weighted equally.
let best: ReturnType<typeof measure> | null = null, bestErr = Infinity;
for (const alpha of [0, 400, 200, 120, 80, 60, 45, 35, 25, 16]) {
  const r = measure(k, alpha, true);
  const err = (r.totalSd / tg.totalSd - 1) ** 2 + (r.h2Sd / tg.h2PointsSd - 1) ** 2;
  console.log(`  ${fmt(r)}  err ${err.toFixed(5)}`);
  if (err < bestErr) { bestErr = err; best = r; }
}
json.durationShrink = k;
json.timeFactor = timeFactor;
json.gameEnv = { alpha: best!.alpha };
writeFileSync(path, JSON.stringify(json));
console.log(`chosen: ${fmt(best!)}`);
