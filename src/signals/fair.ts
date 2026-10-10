// Market -> fair YES probability from a simulated final-score distribution.
// Inputs: a Kalshi market (series, strike, team suffix) and SimResult. Output: fair P(YES) or null when the market is not one
// the model prices (unknown series, missing strike, team code that matches neither side) -> the caller skips it.
import type { Market } from '../kalshi/gateway.ts';
import { pOver, pMarginOver, type SimResult } from '../model/drive.ts';

export interface Teams { homeK: string; awayK: string }   // Kalshi team codes for this event

export function fairYes(m: Market, r: SimResult, t: Teams): { fair: number; note: string } | null {
  const kind = m.series.replace(/^KX(NFL|NCAAF)/, '');
  const side = m.team === t.homeK ? 'home' : m.team === t.awayK ? 'away' : null;
  switch (kind) {
    case 'GAME':
      if (!side) return null;
      // A tie (NFL regular season only) is neither team winning; settlement for ties must be checked in the market rules text.
      return { fair: side === 'home' ? r.pHome : r.pAway, note: `tie=${r.pTie.toFixed(3)}` };
    case 'TOTAL':
      return m.line == null ? null : { fair: pOver(r.total, m.line), note: '' };
    case '1HTOTAL':
      return m.line == null || !r.h1Total ? null : { fair: pOver(r.h1Total, m.line), note: '' };
    case 'TEAMTOTAL':
      return m.line == null || !side ? null : { fair: pOver(side === 'home' ? r.home : r.away, m.line), note: '' };
    case 'SPREAD':
      // "<team> wins by over X": home margin > X, or away margin > X (i.e. home margin < -X).
      if (m.line == null || !side) return null;
      return { fair: side === 'home' ? pMarginOver(r.margin, m.line) : 1 - pMarginOver(r.margin, -m.line - 1e-9), note: '' };
    default:
      return null;   // 1H team totals etc. need first-half per-team distributions (not simulated yet)
  }
}
