import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { parseEvent, parseSpreadDetails, yardsToGoal } from '../src/data/espn.ts';
import { fairYes } from '../src/signals/fair.ts';
import { toMarket } from '../src/kalshi/gateway.ts';
import { evalMarket } from '../src/signals/classify.ts';
import type { SimResult } from '../src/model/drive.ts';

test('spread details -> home-favoured-positive spread', () => {
  assert.equal(parseSpreadDetails('BUF -2.5', 'BUF', 'KC'), 2.5);
  assert.equal(parseSpreadDetails('KC -7', 'BUF', 'KC'), -7);
  assert.equal(parseSpreadDetails('EVEN', 'BUF', 'KC'), 0);
  assert.equal(parseSpreadDetails('NYJ -3', 'BUF', 'KC'), null);   // neither team: refuse rather than guess
});

test('possessionText -> yards to goal from the offense view', () => {
  assert.equal(yardsToGoal('KC 42', 'KC'), 58);    // own 42
  assert.equal(yardsToGoal('BUF 42', 'KC'), 42);   // opponent 42
  assert.equal(yardsToGoal('50', 'KC'), 50);
  assert.equal(yardsToGoal(undefined, 'KC'), null);
});

test('ESPN in-play event -> GameState; dead ball blocks', () => {
  const ev = JSON.parse(readFileSync(new URL('./fixtures/espn_event_in.json', import.meta.url), 'utf8'));
  const g = parseEvent(ev, 'NFL', true);
  assert.equal(g.blocked, null);
  assert.deepEqual(g.prior, { spread: 2.5, total: 47.5 });
  assert.deepEqual(g.game, { half: 2, hsr: 512 + 900, homeScore: 17, awayScore: 13, possHome: false, y: 58, down: 3, togo: 4, homeTO: 3, awayTO: 2, recv2Home: true });
  ev.competitions[0].situation.down = -1;
  assert.equal(parseEvent(ev, 'NFL').blocked, 'no_down_distance');
  ev.competitions[0].status.period = 5;
  assert.equal(parseEvent(ev, 'NFL').blocked, 'overtime_or_bad_period');
});

const fakeSim = (): SimResult => {
  const total = new Float64Array(240); total[40] = 0.5; total[50] = 0.5;
  const home = new Float64Array(120); home[20] = 0.5; home[30] = 0.5;
  const away = new Float64Array(120); away[20] = 1;
  return { n: 1, pHome: 0.5, pAway: 0.45, pTie: 0.05, total, home, away, h1Total: null, margin: new Map([[0, 0.5], [10, 0.5]]), lambda: [1, 1], drives: { mean: 0, sd: 0 }, env: null };
};

test('fair prices per market type', () => {
  const r = fakeSim(), t = { homeK: 'BUF', awayK: 'KC' };
  const mk = (series: string, ticker: string, strike?: number) => toMarket({ ticker, event_ticker: 'E', floor_strike: strike, status: 'active' }, series);
  assert.equal(fairYes(mk('KXNFLGAME', 'KXNFLGAME-26OCT11KCBUF-BUF'), r, t)!.fair, 0.5);
  assert.equal(fairYes(mk('KXNFLGAME', 'KXNFLGAME-26OCT11KCBUF-KC'), r, t)!.fair, 0.45);
  assert.equal(fairYes(mk('KXNFLTOTAL', 'KXNFLTOTAL-26OCT11KCBUF-45', 45.5), r, t)!.fair, 0.5);
  assert.equal(fairYes(mk('KXNFLSPREAD', 'KXNFLSPREAD-26OCT11KCBUF-BUF7', 7.5), r, t)!.fair, 0.5);
  assert.equal(fairYes(mk('KXNFLSPREAD', 'KXNFLSPREAD-26OCT11KCBUF-KC3', 3.5), r, t)!.fair, 0);
  assert.equal(fairYes(mk('KXNFL1HTOTAL', 'KXNFL1HTOTAL-26OCT11KCBUF-21', 21.5), r, t), null);   // no 1H dist once in half 2
  assert.equal(fairYes(mk('KXNFLGAME', 'KXNFLGAME-26OCT11KCBUF-NYJ'), r, t), null);
});

test('both sides qualifying is treated as an inconsistency, not two trades', () => {
  const [y, n] = evalMarket(0.5, 0.30, 100, 0.30, null);
  assert.equal(y.classification, 'MODEL_OR_MARKET_INCONSISTENCY'); assert.equal(n.stake, 0);
  const [y2] = evalMarket(0.70, 0.55, 100, 0.47, null);
  assert.equal(y2.classification, 'TRADE_CANDIDATE');
});
