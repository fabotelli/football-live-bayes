import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { PlayModel, EV } from '../src/model/plays.ts';
import { loadModel } from '../src/model/load.ts';
import { rng } from '../src/model/drive.ts';

const pm = new PlayModel(JSON.parse(readFileSync(new URL('../data/models/ncaaf_plays_v1.json', import.meta.url), 'utf8')));

test('a 4th down draws 4th-down plays: kicks and punts, not ordinary snaps', () => {
  const r = rng(1); const n = 4000; let kicks = 0, punts = 0;
  for (let i = 0; i < n; i++) { const p = pm.sample(1, 45, 28, 4, 6, 0, 1, 1, r); if (p.ev === EV.FG_GOOD || p.ev === EV.FG_MISS) kicks++; if (p.ev === EV.PUNT) punts++; }
  assert.ok(kicks / n > 0.4, `4th-and-6 at the 28 with 0:45 left: field goal share ${(kicks / n).toFixed(2)}`);
  const r2 = rng(2); let punts2 = 0;
  for (let i = 0; i < n; i++) if (pm.sample(1, 200, 70, 4, 8, 0, 2, 1, r2).ev === EV.PUNT) punts2++;
  assert.ok(punts2 / n > 0.7, `4th-and-8 at own 30 with 3:20 left: punt share ${(punts2 / n).toFixed(2)}`);
});

test('buzzer snaps logged at 0:00 are kept (field goals with seconds left)', () => {
  const r = rng(3); let fg = 0, ends = 0; const n = 4000;
  for (let i = 0; i < n; i++) { const p = pm.sample(1, 4, 20, 2, 10, 0, 0, 1, r); if (p.ev === EV.FG_GOOD || p.ev === EV.FG_MISS) fg++; if (p.clock < 0) ends++; }
  assert.ok(fg / n > 0.3, `ball at the 20 with 0:04 left: field goal share ${(fg / n).toFixed(2)}`);
  assert.ok(ends > 0, 'some samples end the half (logged at 0:00)');
});

test('play mode runs the end of the half snap by snap and consumes the clock', () => {
  const m = loadModel('ncaaf', 2025);
  assert.ok(m.plays, 'play table attached');
  const r = m.simulate({ half: 1, hsr: 90, homeScore: 14, awayScore: 10, possHome: true, y: 60, down: 1, togo: 10, homeTO: 2, awayTO: 2, recv2Home: false },
    { spread: 3.5, total: 55.5 }, { sims: 3000, stats: true, lambda: [1, 1], env: 'off' });
  const snaps = r.stats.snaps[0] / 3000;
  assert.ok(snaps > 4 && snaps < 14, `snaps in the last 1:30 of the half: ${snaps.toFixed(1)}`);
  assert.ok(r.h1Total, 'first-half total recorded');
});
