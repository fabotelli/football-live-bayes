// Per-side edge classification (same rule shape as nhl-live-bayes; the thresholds are NFL starting points, not fitted yet).
// Inputs: fair probability and executable quote per side, plus an optional global block. Outputs: net edge, stake, classification.
// The TypeScript engine only LOGS these; nothing here can place an order.
import { takerFeePerContract } from '../kalshi/gateway.ts';

export const RULES = {
  minNetEdge: 0.08, watchNetEdge: 0.04, askMin: 0.15, askMax: 0.85,
  stakeFloor: 2, stakeCap: 10, edgeFullStake: 0.20,
} as const;

export type Classification = 'TRADE_CANDIDATE' | 'WATCH' | 'NO_TRADE' | 'BLOCKED_PRICE_FLOOR' | 'BLOCKED_PRICE_CEILING' | 'BLOCKED_STATE'
  | 'BLOCKED_LIQUIDITY' | 'MODEL_OR_MARKET_INCONSISTENCY';
export type Side = 'YES' | 'NO';

export function stakeDollars(netEdge: number): number {
  if (!(netEdge >= RULES.minNetEdge)) return 0;
  return Math.min(RULES.stakeCap, RULES.stakeFloor + (RULES.stakeCap - RULES.stakeFloor) * ((netEdge - RULES.minNetEdge) / (RULES.edgeFullStake - RULES.minNetEdge)));
}

export interface SideInput { fair: number; ask: number | null; askSize: number | null }
export interface SideEval { side: Side; fair: number; ask: number | null; fee: number | null; netEdge: number | null; stake: number; classification: Classification; reason: string | null }
export type GlobalBlock = { reason: string } | null;

export function evalSide(side: Side, s: SideInput, block: GlobalBlock): SideEval {
  const base = { side, fair: s.fair, ask: s.ask };
  if (s.ask == null || !(s.ask > 0 && s.ask < 1)) return { ...base, fee: null, netEdge: null, stake: 0, classification: block ? 'BLOCKED_STATE' : 'BLOCKED_LIQUIDITY', reason: block?.reason ?? 'no_executable_ask' };
  const fee = takerFeePerContract(s.ask);
  const netEdge = s.fair - s.ask - fee;
  const stake = stakeDollars(netEdge);
  let cls: Classification, reason: string | null = null;
  if (block) { cls = 'BLOCKED_STATE'; reason = block.reason; }
  else if (netEdge >= RULES.minNetEdge) {
    if (s.ask < RULES.askMin - 1e-9) { cls = 'BLOCKED_PRICE_FLOOR'; reason = `ask_${s.ask.toFixed(2)}_below_${RULES.askMin}`; }
    else if (s.ask > RULES.askMax + 1e-9) { cls = 'BLOCKED_PRICE_CEILING'; reason = `ask_${s.ask.toFixed(2)}_above_${RULES.askMax}`; }
    else {
      const contracts = Math.ceil(stake / s.ask);
      if (s.askSize != null && s.askSize < contracts) { cls = 'BLOCKED_LIQUIDITY'; reason = `ask_size_${s.askSize}_lt_${contracts}`; }
      else cls = 'TRADE_CANDIDATE';
    }
  } else cls = netEdge >= RULES.watchNetEdge ? 'WATCH' : 'NO_TRADE';
  return { ...base, fee, netEdge, stake: cls === 'TRADE_CANDIDATE' ? stake : 0, classification: cls, reason };
}

/** Both sides of one binary market; if both qualify at once the quotes or the model are wrong -> block both. */
export function evalMarket(fairYes: number, yesAsk: number | null, yesAskSize: number | null, noAsk: number | null, block: GlobalBlock): [SideEval, SideEval] {
  const y = evalSide('YES', { fair: fairYes, ask: yesAsk, askSize: yesAskSize }, block);
  const n = evalSide('NO', { fair: 1 - fairYes, ask: noAsk, askSize: null }, block);
  if (y.classification === 'TRADE_CANDIDATE' && n.classification === 'TRADE_CANDIDATE') {
    for (const e of [y, n]) { e.classification = 'MODEL_OR_MARKET_INCONSISTENCY'; e.reason = 'both_sides_qualify'; e.stake = 0; }
  }
  return [y, n];
}
