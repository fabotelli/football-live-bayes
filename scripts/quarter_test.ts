// Where does the in-game forecast go wrong? From kickoff and from one random state per game in Q1/Q2/Q3 (holdout games), compare
// the model's expected points in each remaining quarter with the actual points in that quarter (game-weighted, standard errors).
// Needs data/cache/<league>/quarter_scores.csv (real quarter-end totals). Lines rounded like scripts/lean_test.ts.
//   node scripts/quarter_test.ts [--league nfl] [--max 600] [--sims 1500] [--plays]
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { loadModel } from '../src/model/load.ts';
import { rng } from '../src/model/drive.ts';

const arg = (k: string, d: string) => { const i = process.argv.indexOf('--' + k); return i > 0 ? process.argv[i + 1] : d; };
const L = arg('league', 'nfl'), MAX = Number(arg('max', '600')), SIMS = Number(arg('sims', '1500'));
const root = new URL('..', import.meta.url).pathname;
const m = loadModel(L, 2025, process.argv.includes('--plays') ? {} : { plays: false });
const qs = new Map<string, number[]>();
for (const l of readFileSync(root + `data/cache/${L}/quarter_scores.csv`, 'utf8').trim().split('\n').slice(1)) { const r = l.split(','); qs.set(r[0], r.slice(1).map(Number)); }
const csv = gunzipSync(readFileSync(root + `data/cache/${L}/bt_states.csv.gz`)).toString('utf8').trim().split('\n');
const h = csv[0].split(','); const ix = (k: string) => h.indexOf(k);
const byGame = new Map<string, string[][]>();
for (const l of csv.slice(1)) { const r = l.split(','); if (qs.has(r[0])) (byGame.get(r[0]) ?? byGame.set(r[0], []).get(r[0])!).push(r); }
const games = [...byGame.keys()].filter((_, i) => i % Math.max(1, Math.floor(byGame.size / MAX)) === 0).slice(0, MAX);
const rand = rng(11);
const acc: Record<string, { n: number; b: number; ss: number }> = {};
const add = (k: string, d: number) => { const a = acc[k] ??= { n: 0, b: 0, ss: 0 }; a.n++; a.b += d; a.ss += d * d; };
const at = (c: Float64Array, el: number) => { const x = Math.min(el / 10, c.length - 1), i = Math.floor(x), f = x - i; return i + 1 < c.length ? c[i] * (1 - f) + c[i + 1] * f : c[i]; };
for (const id of games) {
  const rs = byGame.get(id)!, q = qs.get(id)!;   // q = cumulative totals at the end of Q1..Q4
  const g = (r: string[], k: string) => Number(r[ix(k)]);
  const pr = { spread: Math.round(g(rs[0], 'spread')), total: 2 * Math.round(g(rs[0], 'total') / 2) };
  const qEnd = [900, 1800, 2700, 3600];
  // pregame
  {
    const c = m.simulate({ half: 1, hsr: 1800, homeScore: 0, awayScore: 0, possHome: true, kickoff: true, y: 75, down: 1, togo: 10, homeTO: 3, awayTO: 3, recv2Home: null }, pr, { sims: SIMS, openingCoinFlip: true, curve: true }).curve!;
    for (let k = 0; k < 4; k++) add(`from kickoff | Q${k + 1}`, (at(c, qEnd[k]) - (k ? at(c, qEnd[k - 1]) : 0)) - (q[k] - (k ? q[k - 1] : 0)));
  }
  for (const [ph, lo, hi] of [['Q1', 2700, 3600], ['Q2', 1800, 2700], ['Q3', 900, 1800]] as const) {
    const inPh = rs.filter((r) => { const gsr = (g(r, 'half') === 1 ? 1800 : 0) + g(r, 'hsr'); return gsr > lo && gsr <= hi; });
    if (!inPh.length) continue;
    const r = inPh[Math.floor(rand() * inPh.length)];
    const el = 3600 - ((g(r, 'half') === 1 ? 1800 : 0) + g(r, 'hsr')), now = g(r, 'hs') + g(r, 'as');
    const c = m.simulate({ half: g(r, 'half') as 1 | 2, hsr: g(r, 'hsr'), homeScore: g(r, 'hs'), awayScore: g(r, 'as'), possHome: g(r, 'pos_home') === 1, y: g(r, 'y'),
      down: g(r, 'down') as 1 | 2 | 3 | 4, togo: g(r, 'togo'), homeTO: g(r, 'hto'), awayTO: g(r, 'ato'), recv2Home: g(r, 'recv2_home') === 1 }, pr, { sims: SIMS, curve: true }).curve!;
    const k0 = qEnd.findIndex((e) => e > el);
    for (let k = k0; k < 4; k++) {
      const modelPts = at(c, qEnd[k]) - (k === k0 ? 0 : at(c, qEnd[k - 1]));
      const actualPts = q[k] - (k === k0 ? now : q[k - 1]);
      add(`from ${ph}       | ${k === k0 ? 'rest of ' : ''}Q${k + 1}`, modelPts - actualPts);
    }
  }
}
console.log(`${L} quarter test, ${games.length} holdout games: model minus actual points in each quarter`);
for (const [k, a] of Object.entries(acc)) { const b = a.b / a.n, se = Math.sqrt((a.ss / a.n - b * b) / a.n); console.log(`  ${k.padEnd(28)} ${b >= 0 ? '+' : ''}${b.toFixed(2)} +/- ${se.toFixed(2)}  (z ${(b / se).toFixed(1)})`); }
