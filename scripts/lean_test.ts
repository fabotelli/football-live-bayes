// Game-weighted test of the in-game remaining-points forecast on holdout games: one random state per game per phase, model minus
// actual remaining points, with per-game standard errors. (States within a game share one outcome, so a play-weighted average
// overstates precision; this is the test that showed the NCAAF Q2-Q3 under-lean, BACKTEST.md.)
// Lines are rounded (spread to 1, total to 2) so games share the line calibration; this is a test of the in-game dynamics.
//   node scripts/lean_test.ts [--league ncaaf] [--max 400] [--sims 1500] [--no-plays]
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { loadModel } from '../src/model/load.ts';
import { mean, rng } from '../src/model/drive.ts';

const arg = (k: string, d: string) => { const i = process.argv.indexOf('--' + k); return i > 0 ? process.argv[i + 1] : d; };
const L = arg('league', 'ncaaf'), MAX = Number(arg('max', '400')), SIMS = Number(arg('sims', '1500'));
const plays = !process.argv.includes('--no-plays');
const root = new URL('..', import.meta.url).pathname;
const m = loadModel(L, 2025, plays ? {} : { plays: false });
const csv = gunzipSync(readFileSync(root + `data/cache/${L}/bt_states.csv.gz`)).toString('utf8').trim().split('\n');
const h = csv[0].split(','); const ix = (k: string) => h.indexOf(k);
const byGame = new Map<string, string[][]>();
for (const l of csv.slice(1)) { const r = l.split(','); (byGame.get(r[0]) ?? byGame.set(r[0], []).get(r[0])!).push(r); }
const games = [...byGame.keys()].filter((_, i) => i % Math.max(1, Math.floor(byGame.size / MAX)) === 0).slice(0, MAX);
const rand = rng(7);
const acc: Record<string, { n: number; b: number; ss: number }> = {};
for (const id of games) {
  const rs = byGame.get(id)!;
  for (const ph of ['Q1', 'Q2', 'Q3', 'Q4']) {
    const g = (r: string[], k: string) => Number(r[ix(k)]);
    const inPh = rs.filter((r) => { const gsr = (g(r, 'half') === 1 ? 1800 : 0) + g(r, 'hsr'); return ph === (gsr > 2700 ? 'Q1' : gsr > 1800 ? 'Q2' : gsr > 900 ? 'Q3' : 'Q4'); });
    if (!inPh.length) continue;
    const r = inPh[Math.floor(rand() * inPh.length)];
    const pr = { spread: Math.round(g(r, 'spread')), total: 2 * Math.round(g(r, 'total') / 2) };
    const res = m.simulate({ half: g(r, 'half') as 1 | 2, hsr: g(r, 'hsr'), homeScore: g(r, 'hs'), awayScore: g(r, 'as'), possHome: g(r, 'pos_home') === 1, y: g(r, 'y'),
      down: g(r, 'down') as 1 | 2 | 3 | 4, togo: g(r, 'togo'), homeTO: g(r, 'hto'), awayTO: g(r, 'ato'), recv2Home: g(r, 'recv2_home') === 1 }, pr, { sims: SIMS });
    const d = mean(res.total) - (g(r, 'final_h') + g(r, 'final_a'));
    const a = acc[ph] ??= { n: 0, b: 0, ss: 0 }; a.n++; a.b += d; a.ss += d * d;
  }
}
console.log(`${L} lean test (plays ${plays ? 'on' : 'off'}), ${games.length} holdout games, one random state per game per phase:`);
for (const [ph, a] of Object.entries(acc).sort()) {
  const b = a.b / a.n, se = Math.sqrt((a.ss / a.n - b * b) / a.n);
  console.log(`  ${ph}  model - actual ${b >= 0 ? '+' : ''}${b.toFixed(2)} +/- ${se.toFixed(2)}  (z ${(b / se).toFixed(1)}, n ${a.n})`);
}
