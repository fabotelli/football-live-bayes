// READ-ONLY Kalshi gateway. GET-only by construction: there is no code path here (or anywhere in src/) that can place an order.
// Public market data needs no credentials, so no key material is read by the TypeScript side at all (tests/readonly.test.ts).
// Inputs: public Kalshi market-data endpoints. Outputs: normalised markets (quotes in dollars, strike, team suffix).

const BASE = 'https://api.elections.kalshi.com/trade-api/v2';

// Series tickers. CONFIRMED = seen on real markets by the earlier kalshi_ht bots (KXNFLGAME, KXNFL1HTOTAL, KXNFL1HTEAMTOTAL,
// KXNCAAFGAME). The rest follow the NHL naming pattern and must be confirmed with `node scripts/discover_series.ts` before use;
// the shadow loop only polls series listed in ENABLED.
export const SERIES = {
  NFL: { GAME: 'KXNFLGAME', H1_TOTAL: 'KXNFL1HTOTAL', H1_TEAM_TOTAL: 'KXNFL1HTEAMTOTAL', TOTAL: 'KXNFLTOTAL', SPREAD: 'KXNFLSPREAD', TEAM_TOTAL: 'KXNFLTEAMTOTAL' },
  NCAAF: { GAME: 'KXNCAAFGAME', TOTAL: 'KXNCAAFTOTAL', SPREAD: 'KXNCAAFSPREAD' },
} as const;
export const CONFIRMED = new Set(['KXNFLGAME', 'KXNFL1HTOTAL', 'KXNFL1HTEAMTOTAL', 'KXNCAAFGAME']);

export interface Market {
  ticker: string; eventTicker: string; series: string; line: number | null; status: string; closeTime: string | null;
  yesBid: number | null; yesAsk: number | null; noBid: number | null; noAsk: number | null;
  yesBidSize: number | null; yesAskSize: number | null; feeMultiplier: number;
  team: string | null;          // ticker suffix team code (moneyline / team totals / spreads), else null
  title: string; rulesPrimary: string;
}

/** Wraps fetch so any non-GET call throws. This is the enforcement point for the "no orders from TypeScript" rule. */
export function readOnlyFetch(f: typeof fetch = fetch): typeof fetch {
  return ((input: any, init?: any) => {
    const method = String(init?.method ?? 'GET').toUpperCase();
    if (method !== 'GET') throw new Error('SHADOW_READ_ONLY: refusing non-GET Kalshi request ' + method);
    return f(input, init);
  }) as typeof fetch;
}

const num = (x: any): number | null => { const n = Number(x); return x != null && x !== '' && Number.isFinite(n) ? n : null; };

export function toMarket(m: any, series: string): Market {
  const suffix = String(m.ticker ?? '').split('-').pop() ?? '';
  // Team-coded suffixes are letters (college codes can include digits); numeric-only suffixes are strikes.
  const team = /^[A-Z][A-Z0-9]{1,5}$/.test(suffix) && !/^\d+$/.test(suffix) ? suffix.replace(/\d+$/, '') : null;
  return {
    ticker: m.ticker, eventTicker: m.event_ticker, series, line: num(m.floor_strike), status: m.status, closeTime: m.close_time ?? null,
    yesBid: num(m.yes_bid_dollars), yesAsk: num(m.yes_ask_dollars), noBid: num(m.no_bid_dollars), noAsk: num(m.no_ask_dollars),
    yesBidSize: num(m.yes_bid_size_fp), yesAskSize: num(m.yes_ask_size_fp), feeMultiplier: Number(m.fee_multiplier_override ?? 1), team,
    title: String(m.title ?? ''), rulesPrimary: String(m.rules_primary ?? ''),
  };
}

export class KalshiReadOnly {
  private f: typeof fetch;
  constructor(f: typeof fetch = fetch) { this.f = readOnlyFetch(f); }
  async get(path: string, attempts = 3): Promise<any> {
    for (let i = 0; i < attempts; i++) {
      const r = await this.f(BASE + path, { headers: { accept: 'application/json' }, signal: AbortSignal.timeout(10000) });
      if (r.status === 429) { await new Promise((res) => setTimeout(res, 500 * 2 ** i)); continue; }
      if (!r.ok) throw new Error(`Kalshi ${path} HTTP ${r.status}`);
      return r.json();
    }
    throw new Error('Kalshi rate limited ' + path);
  }
  /** All markets of a series (default: open), grouped by event ticker. */
  async seriesByEvent(series: string, status = 'open'): Promise<Map<string, Market[]>> {
    const out = new Map<string, Market[]>();
    let cursor: string | undefined;
    do {
      const j = await this.get(`/markets?series_ticker=${series}&status=${status}&limit=1000${cursor ? '&cursor=' + cursor : ''}`);
      for (const raw of j.markets ?? []) { const m = toMarket(raw, series); (out.get(m.eventTicker) ?? out.set(m.eventTicker, []).get(m.eventTicker)!).push(m); }
      cursor = j.cursor || undefined;
    } while (cursor);
    return out;
  }
  async orderbook(ticker: string): Promise<{ yes: [number, number][]; no: [number, number][] }> {
    const j = await this.get(`/markets/${encodeURIComponent(ticker)}/orderbook?depth=25`);
    const ob = j.orderbook_fp ?? j.orderbook ?? {};
    const parse = (a: any[] | undefined): [number, number][] => (a ?? []).map((x) => [Number(x[0]), Number(x[1])] as [number, number]).filter((x) => Number.isFinite(x[0]) && Number.isFinite(x[1])).sort((a, b) => b[0] - a[0]);
    return { yes: parse(ob.yes_dollars ?? ob.yes), no: parse(ob.no_dollars ?? ob.no) };
  }
}

/** Kalshi taker fee per contract: ceil(multiplier * 0.07 * p * (1 - p)) to 1/100 cent (same formula as kalshi_ht / nhl-live-bayes). */
export function takerFeePerContract(price: number, multiplier = 1): number {
  return Math.ceil(multiplier * 0.07 * price * (1 - price) * 10000) / 10000;
}

const MON = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'];
/** Kalshi date prefix for an event ticker, from the game's US/Eastern calendar date: "26OCT11". */
export function kalshiDatePrefix(isoUtc: string): string {
  const et = new Date(new Date(isoUtc).toLocaleString('en-US', { timeZone: 'America/New_York' }));
  return `${String(et.getFullYear()).slice(2)}${MON[et.getMonth()]}${String(et.getDate()).padStart(2, '0')}`;
}
