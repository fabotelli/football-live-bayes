// Holdout backtest of the drive model against real NFL outcomes and against nflverse's own market-anchored WP (vegas_home_wp).
// Inputs: data/cache/nfl/bt_states.csv.gz (scripts/build_nfl.py), data/models/nfl_drive_v2.json. Never touches the network.
// Prints log loss / Brier for home-win by game phase, and calibration of P(final total > closing total) from in-game states.
//   node scripts/backtest.ts [--league nfl|ncaaf] [--sims 1000] [--seasons 2024,2025,2026] [--max 0]
// The NCAAF benchmark column is cfbfastR's spread-adjusted home_wp_before (stored in the same vegas_home_wp field).
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { DriveModel, pOver, type GameState } from '../src/model/drive.ts';

const arg = (k: string, d: string) => { const i = process.argv.indexOf('--' + k); return i > 0 ? process.argv[i + 1] : d; };
const SIMS = Number(arg('sims', '1000')), MAX = Number(arg('max', '0'));
const LEAGUE = arg('league', 'nfl');
const SEASONS = new Set(arg('seasons', LEAGUE === 'nfl' ? '2024,2025,2026' : '2025,2026').split(',').map(Number));
const root = new URL('..', import.meta.url).pathname;
const json = JSON.parse(readFileSync(root + `data/models/${LEAGUE}_drive_v2.json`, 'utf8'));
const csv = gunzipSync(readFileSync(root + `data/cache/${LEAGUE}/bt_states.csv.gz`)).toString('utf8').trim().split('\n');
const hdr = csv[0].split(','); const ix = (k: string) => hdr.indexOf(k);
let rows = csv.slice(1).map((l) => l.split(',')).filter((r) => SEASONS.has(Number(r[ix('season')])));
if (MAX > 0) rows = rows.filter((_, i) => i % Math.ceil(rows.length / MAX) === 0);

const models = new Map<number, DriveModel>();
const modelFor = (season: number) => models.get(season) ?? models.set(season, new DriveModel(json, season)).get(season)!;
const ll = (p: number, y: number) => -(y * Math.log(Math.max(p, 1e-6)) + (1 - y) * Math.log(Math.max(1 - p, 1e-6)));
type Acc = { n: number; llM: number; llV: number; brM: number; brV: number };
const phases = new Map<string, Acc>();
const add = (k: string, pm: number, pv: number, y: number) => {
  const a = phases.get(k) ?? phases.set(k, { n: 0, llM: 0, llV: 0, brM: 0, brV: 0 }).get(k)!;
  a.n++; a.llM += ll(pm, y); a.llV += ll(pv, y); a.brM += (pm - y) ** 2; a.brV += (pv - y) ** 2;
};
const calT: { p: number; y: number }[] = [];
const t0 = Date.now();
for (const r of rows) {
  const g = (k: string) => Number(r[ix(k)]);
  const fh = g('final_h'), fa = g('final_a'); if (fh === fa) continue;   // ties are rare; skip for the binary score
  const vwp = g('vegas_home_wp'); if (!Number.isFinite(vwp)) continue;
  const st: GameState = { half: g('half') as 1 | 2, hsr: g('hsr'), homeScore: g('hs'), awayScore: g('as'), possHome: g('pos_home') === 1,
    y: g('y'), down: g('down') as 1 | 2 | 3 | 4, togo: g('togo'), homeTO: g('hto'), awayTO: g('ato'), recv2Home: g('recv2_home') === 1 };
  const prior = { spread: g('spread'), total: g('total') };
  const res = modelFor(g('season')).simulate(st, prior, { sims: SIMS });
  const pm = res.pHome / Math.max(1e-9, res.pHome + res.pAway);
  const y = fh > fa ? 1 : 0;
  const gsr = (g('half') === 1 ? 1800 : 0) + g('hsr');
  const phase = gsr > 2700 ? 'Q1' : gsr > 1800 ? 'Q2' : gsr > 900 ? 'Q3' : gsr > 300 ? 'Q4 >5m' : 'Q4 last 5m';
  add('ALL', pm, vwp, y); add(phase, pm, vwp, y);
  if (Math.abs(g('total') % 1) === 0.5) calT.push({ p: pOver(res.total, g('total')), y: fh + fa > g('total') ? 1 : 0 });
}
console.log(`states ${rows.length}, sims/state ${SIMS}, ${((Date.now() - t0) / 1000).toFixed(1)}s`);
console.log('home-win  phase          n     logloss model / vegas_wp     brier model / vegas_wp');
for (const k of ['ALL', 'Q1', 'Q2', 'Q3', 'Q4 >5m', 'Q4 last 5m']) {
  const a = phases.get(k); if (!a) continue;
  console.log(`          ${k.padEnd(12)} ${String(a.n).padStart(6)}   ${(a.llM / a.n).toFixed(4)} / ${(a.llV / a.n).toFixed(4)}           ${(a.brM / a.n).toFixed(4)} / ${(a.brV / a.n).toFixed(4)}`);
}
console.log('\nP(final total > closing total) from in-game states, calibration:');
console.log('  bin        n    mean_p  actual');
for (let b = 0; b < 10; b++) {
  const xs = calT.filter((c) => Math.min(9, Math.floor(c.p * 10)) === b); if (xs.length < 20) continue;
  const mp = xs.reduce((s, c) => s + c.p, 0) / xs.length, ay = xs.reduce((s, c) => s + c.y, 0) / xs.length;
  console.log(`  ${(b / 10).toFixed(1)}-${((b + 1) / 10).toFixed(1)}  ${String(xs.length).padStart(5)}   ${mp.toFixed(3)}   ${ay.toFixed(3)}`);
}
const brT = calT.reduce((s, c) => s + (c.p - c.y) ** 2, 0) / calT.length;
console.log(`  total-over brier ${brT.toFixed(4)} (n=${calT.length})`);
