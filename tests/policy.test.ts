import { test } from 'node:test';
import assert from 'node:assert/strict';
import { marketBlock } from '../src/signals/policy.ts';
import type { GameState } from '../src/model/drive.ts';

const at = (half: 1 | 2, hsr: number): GameState => ({ half, hsr, homeScore: 14, awayScore: 10, possHome: true, y: 60, down: 1, togo: 10, homeTO: 3, awayTO: 3, recv2Home: true });

test('NCAAF totals are blocked in Q2 and Q3 only', () => {
  for (const s of ['KXNCAAFTOTAL', 'KXNCAAF1HTOTAL', 'KXNCAAFTEAMTOTAL']) {
    assert.equal(marketBlock('NCAAF', s, at(1, 1200)), null, `${s} Q1`);
    assert.equal(marketBlock('NCAAF', s, at(1, 600)), 'ncaaf_totals_q2_q3_policy', `${s} Q2`);
    assert.equal(marketBlock('NCAAF', s, at(2, 1200)), 'ncaaf_totals_q2_q3_policy', `${s} Q3`);
    assert.equal(marketBlock('NCAAF', s, at(2, 600)), null, `${s} Q4`);
  }
});

test('other NCAAF markets and all NFL markets are not affected', () => {
  assert.equal(marketBlock('NCAAF', 'KXNCAAFGAME', at(1, 600)), null);
  assert.equal(marketBlock('NCAAF', 'KXNCAAFSPREAD', at(2, 1200)), null);
  assert.equal(marketBlock('NFL', 'KXNFLTOTAL', at(1, 600)), null);
  assert.equal(marketBlock('NFL', 'KXNFL1HTOTAL', at(1, 600)), null);
});
