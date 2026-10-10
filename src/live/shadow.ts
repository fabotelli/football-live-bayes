// SHADOW poll loop: ESPN game state -> drive model -> fair prices -> compare with Kalshi quotes -> append-only JSONL. NEVER trades.
// Inputs: ESPN scoreboards (NFL; NCAAF only once data/models/ncaaf_drive_v2.json exists), Kalshi public markets for ENABLED series.
// Outputs: runtime/state/signals-<ETdate>.jsonl (one row per market side per poll) and runtime/state/latest.json.
//   node src/live/shadow.ts [--once] [--poll 15]
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { DriveModel, type Prior } from '../model/drive.ts';
import { fetchScoreboard, parseEvent, type League, type ParsedGame } from '../data/espn.ts';
import { KalshiReadOnly, SERIES, CONFIRMED, kalshiDatePrefix, type Market } from '../kalshi/gateway.ts';
import { fairYes } from '../signals/fair.ts';
import { evalMarket, RULES, type GlobalBlock } from '../signals/classify.ts';

const root = new URL('../..', import.meta.url).pathname;
const arg = (k: string, d: string) => { const i = process.argv.indexOf('--' + k); return i > 0 ? process.argv[i + 1] : d; };
const POLL_S = Number(arg('poll', '15')), ONCE = process.argv.includes('--once');
// Policy, not model: the holdout backtest shows the model trailing nflverse's market-anchored WP in the last 5 minutes, where
// clock strategy dominates. Block there until a late-game model is validated.
const BLOCK_LAST_SECONDS = 300;

// ESPN -> Kalshi team code differences. Unknown mismatches simply fail to match an event (no signal), never mis-price.
const ALIASES: Record<string, string[]> = { WSH: ['WAS'], JAX: ['JAC'], LAR: ['LA'], LV: ['LVR'] };
const codes = (espn: string) => [espn, ...(ALIASES[espn] ?? [])];

const models: Partial<Record<League, DriveModel>> = {};
for (const lg of ['NFL', 'NCAAF'] as League[]) {
  const p = `${root}data/models/${lg.toLowerCase()}_drive_v2.json`;
  if (existsSync(p)) models[lg] = new DriveModel(JSON.parse(readFileSync(p, 'utf8')));
}

interface Memo { prior: Prior | null; openingRecvHome: boolean | null }
const memo = new Map<string, Memo>();
const kalshi = new KalshiReadOnly();

function etDate(): string { return new Date().toLocaleDateString('en-CA', { timeZone: 'America/New_York' }).replaceAll('-', ''); }

/** Find this game's markets across the enabled series. The GAME series fixes the game key ("26OCT11KCBUF"); other series share it. */
function gameMarkets(g: ParsedGame, bySeries: Map<string, Map<string, Market[]>>): { markets: Market[]; homeK: string; awayK: string } | null {
  const S = SERIES[g.league];
  const games = bySeries.get(S.GAME); if (!games) return null;
  const date = kalshiDatePrefix(g.date);
  for (const [ev, ms] of games) {
    if (!ev.includes(date)) continue;
    const teams = new Set(ms.map((m) => m.team));
    const homeK = codes(g.home).find((c) => teams.has(c)), awayK = codes(g.away).find((c) => teams.has(c));
    if (!homeK || !awayK) continue;
    const key = ev.slice(ev.indexOf('-') + 1);
    const markets: Market[] = [];
    for (const [series, evs] of bySeries) for (const [e, list] of evs) if (e === `${series}-${key}`) markets.push(...list);
    return { markets, homeK, awayK };
  }
  return null;
}

async function pollOnce(): Promise<void> {
  const ts = Date.now();
  const rows: any[] = [];
  for (const lg of Object.keys(models) as League[]) {
    const model = models[lg]!;
    const events = await fetchScoreboard(lg);
    const enabled = Object.values(SERIES[lg]).filter((s) => CONFIRMED.has(s));
    const bySeries = new Map<string, Map<string, Market[]>>();
    for (const s of enabled) bySeries.set(s, await kalshi.seriesByEvent(s));
    for (const ev of events) {
      const id = `${lg}:${ev.id}`;
      const m = memo.get(id) ?? memo.set(id, { prior: null, openingRecvHome: null }).get(id)!;
      const first = parseEvent(ev, lg, null);
      // Freeze the prior at kickoff: keep updating it while the game is 'pre', never after (the live line is what we price against).
      if (first.state === 'pre' && first.prior) m.prior = first.prior;
      if (first.state === 'in' && first.period === 1 && m.openingRecvHome == null && first.possessionId && first.clockSec != null && first.clockSec > 800)
        m.openingRecvHome = first.possessionId === first.homeId;
      const g = parseEvent(ev, lg, m.openingRecvHome == null ? null : !m.openingRecvHome);
      if (g.state !== 'in') continue;
      let block: GlobalBlock = g.blocked ? { reason: g.blocked } : null;
      if (!block && !m.prior) block = { reason: 'no_frozen_prior' };   // started watching after kickoff
      if (!block && g.game!.half === 2 && g.game!.hsr <= BLOCK_LAST_SECONDS) block = { reason: 'last_5_minutes_policy' };
      const km = gameMarkets(g, bySeries);
      if (!km) { rows.push({ ts, league: lg, event: g.eventId, game: `${g.away}@${g.home}`, classification: 'NO_MARKET_MATCH' }); continue; }
      const sim = !block ? model.simulate(g.game!, m.prior!, { sims: 20000 }) : null;
      for (const mk of km.markets) {
        const f = sim ? fairYes(mk, sim, { homeK: km.homeK, awayK: km.awayK }) : null;
        if (sim && !f) continue;
        const [yes, no] = evalMarket(f?.fair ?? 0.5, mk.yesAsk, mk.yesAskSize, mk.noAsk, block ?? (f ? null : { reason: 'unpriced' }));
        for (const e of [yes, no]) rows.push({ ts, league: lg, event: g.eventId, game: `${g.away}@${g.home}`, ticker: mk.ticker, series: mk.series,
          line: mk.line, team: mk.team, state: g.game, prior: m.prior, lambda: sim?.lambda, note: f?.note, ...e });
      }
    }
  }
  const dir = `${root}runtime/state`; mkdirSync(dir, { recursive: true });
  if (rows.length) appendFileSync(`${dir}/signals-${etDate()}.jsonl`, rows.map((r) => JSON.stringify(r)).join('\n') + '\n');
  writeFileSync(`${dir}/latest.json`, JSON.stringify({ ts, rules: RULES, rows }, null, 1));
  const cands = rows.filter((r) => r.classification === 'TRADE_CANDIDATE').length;
  console.log(new Date(ts).toISOString(), `rows=${rows.length} candidates=${cands}`);
}

async function main() {
  if (!Object.keys(models).length) throw new Error('no drive model in data/models (run scripts/build_nfl.py)');
  do {
    try { await pollOnce(); } catch (e) { console.error(new Date().toISOString(), 'poll error', (e as Error).message); }
    if (!ONCE) await new Promise((r) => setTimeout(r, POLL_S * 1000));
  } while (!ONCE);
}
if (import.meta.url === `file://${process.argv[1]}`) main();
