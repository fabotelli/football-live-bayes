// Lists Kalshi football series (KXNFL*, KXNCAAF*) with one open market each and its rules text, so series names and settlement
// (ties, OT, "wins by over X") are confirmed from Kalshi itself before a series is added to CONFIRMED in src/kalshi/gateway.ts.
// Read-only. Run on a host that can reach api.elections.kalshi.com:  node scripts/discover_series.ts
import { KalshiReadOnly } from '../src/kalshi/gateway.ts';

const k = new KalshiReadOnly();
const j = await k.get('/series?category=Sports');
const series = (j.series ?? []).filter((s: any) => /^KX(NFL|NCAAF)/.test(s.ticker)).sort((a: any, b: any) => a.ticker.localeCompare(b.ticker));
for (const s of series) {
  const mk = (await k.get(`/markets?series_ticker=${s.ticker}&status=open&limit=1`)).markets?.[0];
  console.log(`\n${s.ticker}  ${s.title}`);
  if (mk) console.log(`  e.g. ${mk.ticker} strike=${mk.floor_strike ?? '-'}\n  rules: ${String(mk.rules_primary ?? '').slice(0, 400)}`);
  else console.log('  (no open markets)');
}
