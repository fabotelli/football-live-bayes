// Per-market policy blocks: known model weaknesses where the shadow loop logs BLOCKED_STATE instead of a tradeable signal.
// Each entry names the evidence (BACKTEST.md); remove an entry only with a backtest showing the weakness is fixed.
import type { GameState } from '../model/drive.ts';
import type { League } from '../data/espn.ts';

const TOTAL_KINDS = new Set(['TOTAL', '1HTOTAL', 'TEAMTOTAL', '1HTEAMTOTAL']);

/** Block reason for this market at this state, or null. `series` is the Kalshi series ticker (e.g. KXNCAAFTOTAL). */
export function marketBlock(league: League, series: string, st: GameState): string | null {
  const kind = series.replace(/^KX(NFL|NCAAF)/, '');
  // NCAAF totals in Q2-Q3: the model under-forecasts remaining points by ~0.5-0.8 there (scoring around the end of the first
  // half is mistimed; BACKTEST.md "Open"). Lifted when the end-of-half clock model is validated.
  const quarter = st.half === 1 ? (st.hsr > 900 ? 1 : 2) : (st.hsr > 900 ? 3 : 4);
  if (league === 'NCAAF' && TOTAL_KINDS.has(kind) && (quarter === 2 || quarter === 3)) return 'ncaaf_totals_q2_q3_policy';
  return null;
}
