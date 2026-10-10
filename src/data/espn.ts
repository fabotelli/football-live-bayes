// ESPN public scoreboard adapter (NFL + college football). Read-only; no credentials.
// Inputs: site.api.espn.com scoreboard JSON. Outputs: ParsedGame rows carrying a model GameState when the state is usable,
// otherwise a `blocked` reason (fail closed: unknown field position / possession / clock -> no fair price, no signal).
//
// Field position comes from `situation.possessionText` ("KC 35" = on KC's 35), not `situation.yardLine`, whose orientation
// is not documented; the text says whose side of the field it is, so yards-to-goal is unambiguous.
import type { GameState, Prior } from '../model/drive.ts';

export type League = 'NFL' | 'NCAAF';
export const SCOREBOARD: Record<League, string> = {
  NFL: 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard',
  // groups=80 -> all FBS games (the default scoreboard only lists ranked/featured games)
  NCAAF: 'https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard?groups=80&limit=300',
};
const QUARTER_SECONDS = 900;

export interface ParsedGame {
  league: League; eventId: string; date: string; home: string; away: string; homeId: string; awayId: string;
  state: 'pre' | 'in' | 'post'; period: number; clockSec: number | null;
  homeScore: number; awayScore: number;
  prior: Prior | null;               // from ESPN's odds block (pregame line); freeze it at kickoff, see src/live/shadow.ts
  game: GameState | null;            // null when blocked
  blocked: string | null;
  possessionId: string | null;
}

/** "KC -3.5" + home/away abbreviations -> spread in nflverse sign (positive = home favoured). "EVEN"/"PK" -> 0. */
export function parseSpreadDetails(details: string | undefined, home: string, away: string): number | null {
  if (!details) return null;
  if (/^(EVEN|PK|PICK)/i.test(details.trim())) return 0;
  const m = details.trim().match(/^([A-Z0-9&]+)\s+(-?\d+(?:\.\d+)?)$/i);
  if (!m) return null;
  const n = Math.abs(Number(m[2]));
  if (m[1].toUpperCase() === home.toUpperCase()) return n;
  if (m[1].toUpperCase() === away.toUpperCase()) return -n;
  return null;
}

/** "KC 35" with the possessing team's abbreviation -> yards to the opponent's goal. Midfield "50" -> 50. */
export function yardsToGoal(possessionText: string | undefined, posAbbr: string): number | null {
  if (!possessionText) return null;
  const t = possessionText.trim();
  if (/^50$/.test(t)) return 50;
  const m = t.match(/^([A-Z0-9&]+)\s+(\d{1,2})$/i);
  if (!m) return null;
  const n = Number(m[2]);
  if (!(n >= 1 && n <= 50)) return null;
  return m[1].toUpperCase() === posAbbr.toUpperCase() ? 100 - n : n;
}

export function parseEvent(ev: any, league: League, recv2Home: boolean | null = null): ParsedGame {
  const comp = ev?.competitions?.[0] ?? {};
  const st = comp.status ?? ev?.status ?? {};
  const cs: any[] = comp.competitors ?? [];
  const H = cs.find((c) => c.homeAway === 'home') ?? {}, A = cs.find((c) => c.homeAway === 'away') ?? {};
  const home = String(H.team?.abbreviation ?? ''), away = String(A.team?.abbreviation ?? '');
  const homeId = String(H.team?.id ?? ''), awayId = String(A.team?.id ?? '');
  const homeScore = Number(H.score ?? 0), awayScore = Number(A.score ?? 0);
  const period = Number(st.period ?? 0);
  const clockSec = Number.isFinite(Number(st.clock)) ? Number(st.clock) : null;
  const state = (st.type?.state ?? 'pre') as ParsedGame['state'];
  const odds = comp.odds?.[0];
  const spread = parseSpreadDetails(odds?.details, home, away);
  const total = Number(odds?.overUnder);
  const prior = spread != null && Number.isFinite(total) && total > 0 ? { spread, total } : null;
  const out: ParsedGame = { league, eventId: String(ev?.id ?? ''), date: String(ev?.date ?? ''), home, away, homeId, awayId, state, period, clockSec,
    homeScore, awayScore, prior, game: null, blocked: null, possessionId: null };
  const block = (r: string) => { out.blocked = r; return out; };
  if (state !== 'in') return block('not_in_play');
  if (st.type?.name === 'STATUS_HALFTIME') return block('halftime');
  if (!(period >= 1 && period <= 4)) return block('overtime_or_bad_period');
  if (clockSec == null) return block('no_clock');
  const s = comp.situation;
  if (!s) return block('no_situation');
  const pos = s.possession != null ? String(s.possession) : null;
  out.possessionId = pos;
  if (pos == null || (pos !== homeId && pos !== awayId)) return block('no_possession');   // dead ball / kickoff / PAT
  const possHome = pos === homeId;
  const down = Number(s.down), togo = Number(s.distance);
  if (!(down >= 1 && down <= 4) || !(togo >= 1)) return block('no_down_distance');      // kickoff, PAT and timeouts report down <= 0
  const y = yardsToGoal(s.possessionText, possHome ? home : away);
  if (y == null) return block('no_field_position');
  out.game = {
    half: period <= 2 ? 1 : 2, hsr: clockSec + (period % 2 === 1 ? QUARTER_SECONDS : 0),
    homeScore, awayScore, possHome, y, down: down as 1 | 2 | 3 | 4, togo: Math.min(togo, y),
    homeTO: Number(s.homeTimeouts ?? 3), awayTO: Number(s.awayTimeouts ?? 3), recv2Home,
  };
  return out;
}

export async function fetchScoreboard(league: League, f: typeof fetch = fetch): Promise<any[]> {
  const r = await f(SCOREBOARD[league], { headers: { accept: 'application/json' }, signal: AbortSignal.timeout(10000) });
  if (!r.ok) throw new Error(`ESPN ${league} HTTP ${r.status}`);
  return (await r.json()).events ?? [];
}
