// Diagnostic: predicted vs actual REMAINING points from holdout states, split by game phase x current score gap x pregame spread.
// Finds where the total is mis-sized (e.g. NCAAF blowouts). Reads the same files as scripts/backtest.ts; no network.
//   node scripts/diag_remaining.ts [--league ncaaf] [--frac 0.15] [--sims 600]
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { DriveModel, mean } from '../src/model/drive.ts';

const arg = (k: string, d: string) => { const i = process.argv.indexOf('--' + k); return i > 0 ? process.argv[i + 1] : d; };
const LEAGUE = arg('league', 'ncaaf'), FRAC = Number(arg('frac', '0.15')), SIMS = Number(arg('sims', '600'));
const root = new URL('..', import.meta.url).pathname;
const model = new DriveModel(JSON.parse(readFileSync(root + `data/models/${LEAGUE}_drive_v2.json`, 'utf8')), 2025);
const csv = gunzipSync(readFileSync(root + `data/cache/${LEAGUE}/bt_states.csv.gz`)).toString('utf8').trim().split('\n');
const hdr = csv[0].split(','); const ix = (k: string) => hdr.indexOf(k);

type Acc = { n: number; pred: number; act: number };
const acc = new Map<string, Acc>();
const add = (k: string, p: number, a: number) => { const x = acc.get(k) ?? acc.set(k, { n: 0, pred: 0, act: 0 }).get(k)!; x.n++; x.pred += p; x.act += a; };
let i = 0;
for (const line of csv.slice(1)) {
  if (((i++ * 2654435761) >>> 0) / 4294967296 > FRAC) continue;    // deterministic subsample
  const r = line.split(','), g = (k: string) => Number(r[ix(k)]);
  const gsr = (g('half') === 1 ? 1800 : 0) + g('hsr');
  const phase = gsr > 2700 ? 'Q1' : gsr > 1800 ? 'Q2' : gsr > 900 ? 'Q3' : gsr > 300 ? 'Q4>5m' : 'Q4<5m';
  const gap = Math.abs(g('hs') - g('as')), gapB = gap <= 8 ? 'gap0-8' : gap <= 16 ? 'gap9-16' : 'gap17+';
  const sp = Math.abs(g('spread')), spB = sp < 7 ? 'sp<7' : sp < 17 ? 'sp7-16' : 'sp17+';
  const res = model.simulate({ half: g('half') as 1 | 2, hsr: g('hsr'), homeScore: g('hs'), awayScore: g('as'), possHome: g('pos_home') === 1, y: g('y'),
    down: g('down') as 1 | 2 | 3 | 4, togo: g('togo'), homeTO: g('hto'), awayTO: g('ato'), recv2Home: g('recv2_home') === 1 },
    { spread: g('spread'), total: g('total') }, { sims: SIMS });
  const pred = mean(res.total) - g('hs') - g('as'), act = g('final_h') + g('final_a') - g('hs') - g('as');
  add(`${phase} | all`, pred, act); add(`${phase} | ${gapB}`, pred, act); add(`${phase} | ${spB}`, pred, act);
}
console.log(`${LEAGUE}: predicted vs actual remaining points (holdout, frac ${FRAC})`);
console.log('phase | group          n     pred    actual   pred-act');
for (const [k, a] of [...acc].sort()) if (a.n >= 30)
  console.log(`${k.padEnd(20)} ${String(a.n).padStart(5)}  ${(a.pred / a.n).toFixed(2).padStart(6)}  ${(a.act / a.n).toFixed(2).padStart(7)}  ${((a.pred - a.act) / a.n).toFixed(2).padStart(7)}`);
