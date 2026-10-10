import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { DriveModel, mean, pOver, type GameState } from '../src/model/drive.ts';

const json = JSON.parse(readFileSync(new URL('../data/models/nfl_drive_v1.json', import.meta.url), 'utf8'));
const m = new DriveModel(json, 2025);
const kick: GameState = { half: 1, hsr: 1800, homeScore: 0, awayScore: 0, possHome: true, kickoff: true, y: 75, down: 1, togo: 10, homeTO: 3, awayTO: 3, recv2Home: null };

test('outcome table rows are probability vectors', () => {
  for (const [y, d, t] of [[75, 1, 10], [3, 4, 3], [45, 3, 12]]) {
    const p = m.baseProbs(2, 100, -3, y, d, t); const s = p.reduce((a, b) => a + b, 0);
    assert.ok(Math.abs(s - 1) < 1e-3, `sum ${s}`); assert.ok(p.every((x) => x >= 0));
  }
});

test('field position matters: 1st&goal at the 2 scores a TD far more often than 1st&10 at own 25', () => {
  assert.ok(m.baseProbs(1, 1000, 0, 2, 1, 2)[0] > 0.6);
  assert.ok(m.baseProbs(1, 1000, 0, 75, 1, 10)[0] < 0.3);
});

test('pregame calibration reproduces the market line (median margin ~ spread, P(over) ~ 0.5)', () => {
  const prior = { spread: 6.5, total: 45.5 };
  const r = m.simulate(kick, prior, { sims: 20000, openingCoinFlip: true });
  assert.ok(Math.abs(pOver(r.total, 45.5) - 0.5) < 0.03, `P(over) ${pOver(r.total, 45.5)}`);
  let cover = 0; for (const [k, v] of r.margin) if (k > 6.5) cover += v;
  assert.ok(Math.abs(cover - 0.5) < 0.03, `P(cover) ${cover}`);
  assert.ok(r.pHome > 0.65 && r.pHome < 0.76, `pHome ${r.pHome}`);
  assert.ok(Math.abs(mean(r.home) + mean(r.away) - 45.5) < 2.5);
});

test('same state -> same answer (seeded), and a leader with the ball who can kneel wins', () => {
  const st: GameState = { half: 2, hsr: 600, homeScore: 20, awayScore: 17, possHome: false, y: 60, down: 2, togo: 7, homeTO: 2, awayTO: 2, recv2Home: true };
  const pr = { spread: 1.5, total: 44.5 };
  assert.equal(m.simulate(st, pr, { sims: 3000 }).pHome, m.simulate(st, pr, { sims: 3000 }).pHome);
  const kneel = m.simulate({ ...st, hsr: 35, possHome: true, homeTO: 0, awayTO: 0, down: 1 }, pr, { sims: 2000 });
  assert.equal(kneel.pHome, 1);
});

test('score and possession move win probability the right way', () => {
  const pr = { spread: 0, total: 44.5 };
  const base: GameState = { half: 2, hsr: 900, homeScore: 21, awayScore: 21, possHome: true, y: 30, down: 1, togo: 10, homeTO: 3, awayTO: 3, recv2Home: false };
  const pBall = m.simulate(base, pr, { sims: 6000 }).pHome;
  const pOpp = m.simulate({ ...base, possHome: false }, pr, { sims: 6000 }).pHome;
  const pUp7 = m.simulate({ ...base, homeScore: 28 }, pr, { sims: 6000 }).pHome;
  assert.ok(pBall > 0.55 && pOpp < 0.45 && pUp7 > pBall, `${pBall} ${pOpp} ${pUp7}`);
});

test('NCAAF tables load and calibrate a big college spread without hitting the lambda bounds', () => {
  const cfb = new DriveModel(JSON.parse(readFileSync(new URL('../data/models/ncaaf_drive_v1.json', import.meta.url), 'utf8')), 2025);
  assert.equal(cfb.j.ot.pTie, 0);   // college OT never ends tied
  const r = cfb.simulate(kick, { spread: 24.5, total: 58.5 }, { sims: 20000, openingCoinFlip: true });
  let cover = 0; for (const [k, v] of r.margin) if (k > 24.5) cover += v;
  assert.ok(Math.abs(cover - 0.5) < 0.03, `P(cover) ${cover}`);
  assert.ok(r.lambda[0] < 4 && r.lambda[1] > 0.2);
  assert.equal(r.pTie, 0);
});
