// Fits `durationShrink` (drive-length dispersion) for a league so the simulated second-half drive count has the same SD as real
// games (`targets.h2DrivesSd`, written by the build script), then writes it into data/models/<league>_drive_v2.json.
// Simulates a second half from a 0-0 kickoff at the league's typical closing line; seeded, so the search is deterministic.
//   node scripts/fit_dispersion.ts --league ncaaf
import { readFileSync, writeFileSync } from 'node:fs';
import { DriveModel } from '../src/model/drive.ts';

const i = process.argv.indexOf('--league'); const LEAGUE = i > 0 ? process.argv[i + 1] : 'ncaaf';
const path = new URL(`../data/models/${LEAGUE}_drive_v2.json`, import.meta.url).pathname;
const json = JSON.parse(readFileSync(path, 'utf8'));
const tg = json.targets; if (!tg) throw new Error('model has no targets; rebuild it');
const half2 = { half: 2 as const, hsr: 1800, homeScore: 0, awayScore: 0, possHome: true, kickoff: true, y: 75, down: 1 as const, togo: 10, homeTO: 3, awayTO: 3, recv2Home: true };
const measure = (k: number) => {
  const m = new DriveModel({ ...json, durationShrink: k });
  const r = m.simulate(half2, tg.typicalPrior, { sims: 30000, seed: 777 });
  let mu = 0, v = 0; for (let x = 0; x < r.total.length; x++) mu += x * r.total[x]; for (let x = 0; x < r.total.length; x++) v += r.total[x] * (x - mu) ** 2;
  return { drivesSd: r.drives.sd, drivesMean: r.drives.mean, ptsSd: Math.sqrt(v) };
};
const raw = measure(1);
let k = 1;
if (raw.drivesSd > tg.h2DrivesSd) {           // only ever narrows; a too-narrow count is left alone
  let lo = 0.2, hi = 1;
  for (let it = 0; it < 14; it++) { const mid = (lo + hi) / 2; if (measure(mid).drivesSd > tg.h2DrivesSd) hi = mid; else lo = mid; }
  k = Math.round(((lo + hi) / 2) * 1000) / 1000;
}
const fit = measure(k);
json.durationShrink = k;
writeFileSync(path, JSON.stringify(json));
console.log(`${LEAGUE}: target 2H drives sd ${tg.h2DrivesSd}, pts sd ${tg.h2PointsSd} (after lines) at prior ${JSON.stringify(tg.typicalPrior)}`);
console.log(`  raw   k=1      drives ${raw.drivesMean.toFixed(2)} sd ${raw.drivesSd.toFixed(2)}, 2H pts sd ${raw.ptsSd.toFixed(2)}`);
console.log(`  fit   k=${k.toFixed(3)}  drives ${fit.drivesMean.toFixed(2)} sd ${fit.drivesSd.toFixed(2)}, 2H pts sd ${fit.ptsSd.toFixed(2)}`);
