// Drive-level live model for American football (NFL tables today; NCAAF uses the same engine with its own tables).
// Inputs: a fitted drive model (data/models/<league>_drive_v2.json, built by scripts/build_nfl.py), a market prior (closing
// spread + total), and the live game state (score, clock, possession, field position, down/distance, timeouts).
// Output: the distribution of final scores, from which moneyline / spread / total / team-total / 1H fair prices are read.
//
// How: Monte Carlo over drives with a seeded RNG (same state -> same answer, so fair prices don't jitter between polls).
// Each drive draws its result from P(result | half-clock bucket, half x score-gap bucket, yards to goal, down, to-go), then
// the time it takes, then where the next drive starts. Team strength enters as one multiplier per offense (lambda) on the
// TD and FG probabilities, solved so that pregame expected points match the market's spread and total. The market sets the
// prior; the live edge has to come from processing game state better/faster than the book, not from out-handicapping the close.
//
// Game environment (the Bayesian part): each simulated game also draws a scoring-environment multiplier g ~ Gamma(alpha, alpha)
// (mean 1) applied to both offenses, because real games scatter around the line more than drive-level noise explains (some games
// are shootouts) and that scatter is multiplicative, which is what gives totals their right skew. Live, g is drawn from its
// posterior given the points scored so far vs the points expected so far at this game's line: Gamma(alpha + P/c, alpha + E/c),
// with c = E[x^2]/E[x] of points per drive turning points into effective event counts. alpha is fitted per league
// (scripts/fit_dispersion.ts); alpha = 0 switches the effect off.

import { PlayModel, EV } from './plays.ts';

export const OUTCOMES = ['TD', 'FG', 'PUNT', 'TO', 'DOWNS', 'MISSFG', 'END', 'OPPTD', 'SAFETY'] as const;
export type Outcome = typeof OUTCOMES[number];
const [TD, FG, PUNT, TO, DOWNS, MISSFG, END, OPPTD, SAFETY] = [0, 1, 2, 3, 4, 5, 6, 7, 8];
const K = OUTCOMES.length;

export interface DriveModelJson {
  version: string; league: string; outcomes: string[];
  buckets: { tbEdges: number[]; gapEdges: number[]; togoEdges: number[]; zoneEdges: number[]; yBinWidth: number };
  outcome: number[][][][][][];                        // [tb][half x gap][yb][down-1][tg][outcome]
  strength: number[][][][][];                         // [tb][half x gap][yb][down-1][tg] mean pregame strength of the offenses in the cell
  duration: Record<string, number[] | null>;          // `${outcome}|${half|*}|${tb}|${zone|*}` (drive in progress) and `S:...` (drive start) -> quantiles of seconds (evenly spaced)
  nextStart: Record<string, number[]>;                // `${outcome}|${yb}` -> P(next start yard bin)
  kickoffStart: Record<string, number[]>;             // season -> P(start yard bin) after a kickoff
  tdPoints: Record<string, number>;                   // "6"/"7"/"8" -> prob
  ot: { pTie: number; winnerAdd: Record<string, number>; loserAdd: Record<string, number> };
  halfSeconds?: number;                               // 1800 for NFL and NCAAF
  durationShrink?: number;                            // drive-length dispersion factor, fitted by scripts/fit_dispersion.ts (1 = raw tables)
  targets?: { h2DrivesSd: number; h2PointsSd: number; totalSd: number; meanMinusLine: number; h1h2Slope: number; pointsDisp: number; windowShare?: number[]; typicalPrior: Prior };
  gameEnv?: { alpha: number };                        // game-environment prior strength (0 = off), fitted by scripts/fit_dispersion.ts
  timeFactor?: number[][];                            // [half-1][window] scoring factors (windows >5:00, 5:00-2:00, last 2:00), fitted
}

export interface GameState {
  half: 1 | 2;
  hsr: number;                 // seconds left in this half
  homeScore: number; awayScore: number;
  possHome: boolean;           // team with the ball (or about to receive a kickoff when kickoff=true)
  kickoff?: boolean;           // between drives: possHome is about to receive a kickoff (y/down/togo ignored)
  y: number;                   // yards to the opponent's goal line (1..99)
  down: 1 | 2 | 3 | 4; togo: number;
  homeTO: number; awayTO: number;
  recv2Home: boolean | null;   // who receives the second-half kickoff (null = unknown -> coin flip; only matters in half 1)
}

export interface Prior { spread: number; total: number }   // spread > 0 = home favoured by that many points (nflverse sign)

export interface SimResult {
  n: number; pHome: number; pAway: number; pTie: number;
  total: Float64Array;          // P(final total = k), OT included
  margin: Map<number, number>;  // P(home - away = m)
  home: Float64Array; away: Float64Array;   // P(team final points = k)
  h1Total: Float64Array | null; // P(first-half total = k) when the state is still in half 1, else null
  lambda: [number, number];
  drives: { mean: number; sd: number };   // simulated drives from the state to the end of regulation (diagnostics)
  env: { shape: number; rate: number } | null;   // game-environment distribution used (posterior when live), null when off
  pairs?: { h1: Int16Array; h2: Int16Array };    // per-path first-half / second-half regulation points (opt.pairs, diagnostics)
}

export type EnvMode = 'posterior' | 'prior' | 'off';
export interface SimOptions {
  sims?: number; seed?: number; lambda?: [number, number]; openingCoinFlip?: boolean;
  env?: EnvMode;       // posterior (default): update g from the score so far; prior: ignore the score (used for fitting); off: g = 1
  pairs?: boolean;     // collect per-path 1H/2H points
  curve?: boolean;     // collect the expected cumulative points by elapsed minute (internal, used by lambdas)
  stats?: boolean;     // collect drive-level diagnostics (outcome counts and mean start yards-to-goal, by half)
}

/** Standard normal (Box-Muller) and Gamma(shape >= 1, rate 1) via Marsaglia-Tsang, both from the seeded uniform source. */
function normal(rand: () => number): number { return Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(2 * Math.PI * rand()); }
function gamma(shape: number, rand: () => number): number {
  const d = shape - 1 / 3, c = 1 / Math.sqrt(9 * d);
  for (;;) {
    const x = normal(rand), v = (1 + c * x) ** 3;
    if (v > 0 && Math.log(1 - rand()) < 0.5 * x * x + d - d * v + d * Math.log(v)) return d * v;
  }
}

const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x));
function searchsortedLeft(edges: number[], x: number): number { let i = 0; while (i < edges.length && x > edges[i]) i++; return i; }

/** mulberry32: small, fast, seedable. */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
export function hashState(s: object): number {
  const str = JSON.stringify(s); let h = 2166136261;
  for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 16777619); }
  return h >>> 0;
}
const normCdf = (z: number) => 0.5 * (1 + erf(z / Math.SQRT2));
function erf(x: number): number { // Abramowitz-Stegun 7.1.26
  const s = Math.sign(x); x = Math.abs(x); const t = 1 / (1 + 0.3275911 * x);
  return s * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x));
}

function sampleDiscrete(p: ArrayLike<number>, u: number): number { let c = 0; for (let i = 0; i < p.length; i++) { c += p[i]; if (u < c) return i; } return p.length - 1; }
function toDist(rec: Record<string, number>): { v: number[]; p: number[] } {
  const v = Object.keys(rec).map(Number), p = v.map((k) => rec[String(k)]); const s = p.reduce((a, b) => a + b, 0);
  return { v, p: p.map((x) => x / s) };
}

export class DriveModel {
  readonly j: DriveModelJson;
  private probs: Float64Array;
  private strength: Float64Array;
  private nGap: number;
  private dur = new Map<string, Float64Array>();
  private durMean = new Map<string, number>();
  /** Drive lengths are drawn as mean + shrink * (draw - mean). Public so scripts/fit_dispersion.ts can search it. */
  durShrink: number;
  private next = new Map<string, Float64Array>();
  private kick: Float64Array;
  private tdPts: { v: number[]; p: number[] };
  private otWin: { v: number[]; p: number[] };
  private otLose: { v: number[]; p: number[] };
  private lamCache = new Map<string, [number, number]>();
  private curveCache = new Map<string, Float64Array>();   // expected cumulative total points by elapsed minute (61 values)
  /** Game-environment prior strength (0 = off) and points dispersion. Public so scripts/fit_dispersion.ts can search them. */
  envAlpha: number;
  pointsDisp: number;
  /** Scoring factor by [half-1][window]: >5:00 left, 5:00-2:00, last 2:00. Fitted (scripts/fit_dispersion.ts) so the share of points
   *  scored in each window matches real games; without it the model front-loaded college scoring (~0.7 points too many in the
   *  first half, too few at its end and in the second half), which showed up live as a ~0.5-point under-forecast of what is left. */
  timeFactor: number[][];
  readonly halfSeconds: number;

  /** End-of-half play sampler; when set, the last `plays.start` seconds of each half are simulated play by play. */
  plays: PlayModel | null;

  constructor(j: DriveModelJson, season?: number, plays?: PlayModel | null) {
    this.plays = plays ?? null;
    this.j = j;
    if (j.outcomes.join() !== OUTCOMES.join()) throw new Error('drive model outcome order mismatch');
    this.nGap = j.buckets.gapEdges.length + 1;
    const cells = 4 * 2 * this.nGap * 10 * 4 * 5;
    this.probs = new Float64Array(j.outcome.flat(5) as number[]);
    this.strength = new Float64Array(j.strength.flat(4) as number[]);
    if (this.probs.length !== cells * K || this.strength.length !== cells) throw new Error('drive model tables have the wrong shape');
    for (const [k, q] of Object.entries(j.duration)) if (q) { this.dur.set(k, new Float64Array(q)); this.durMean.set(k, q.reduce((s, x) => s + x, 0) / q.length); }
    this.durShrink = j.durationShrink ?? 1;
    this.envAlpha = j.gameEnv?.alpha ?? 0;
    this.pointsDisp = j.targets?.pointsDisp ?? 6.3;
    this.timeFactor = j.timeFactor ?? [[1, 1, 1], [1, 1, 1]];
    for (const [k, p] of Object.entries(j.nextStart)) this.next.set(k, new Float64Array(p));
    const seasons = Object.keys(j.kickoffStart).map(Number).sort((a, b) => a - b);
    // Kickoff rules changed in 2024 and 2025: use the requested season's start distribution, else the latest season on file.
    const ks = season != null && j.kickoffStart[String(season)] ? season : seasons[seasons.length - 1];
    this.kick = new Float64Array(j.kickoffStart[String(ks)]);
    this.tdPts = toDist(j.tdPoints);
    this.otWin = toDist(j.ot.winnerAdd); this.otLose = toDist(j.ot.loserAdd);
    this.halfSeconds = j.halfSeconds ?? 1800;
  }

  private cell(half: number, hsr: number, diff: number, y: number, down: number, togo: number): number {
    const b = this.j.buckets;
    const tb = searchsortedLeft(b.tbEdges, hsr);
    const hg = (half === 2 ? this.nGap : 0) + searchsortedLeft(b.gapEdges, diff);
    const yb = clamp(Math.floor((y - 1) / b.yBinWidth), 0, 9);
    const tg = searchsortedLeft(b.togoEdges, togo);
    return (((tb * 2 * this.nGap + hg) * 10 + yb) * 4 + (clamp(down, 1, 4) - 1)) * 5 + tg;
  }
  /** P(outcome) for the state, before the strength multiplier. */
  baseProbs(half: number, hsr: number, diff: number, y: number, down: number, togo: number): Float64Array {
    const i = this.cell(half, hsr, diff, y, down, togo) * K;
    return this.probs.subarray(i, i + K);
  }
  /** Mean pregame strength of the offenses that produced this cell's frequencies (1 = average team). */
  cellStrength(half: number, hsr: number, diff: number, y: number, down: number, togo: number): number {
    return this.strength[this.cell(half, hsr, diff, y, down, togo)];
  }

  /** Seconds the rest of the drive takes. `fresh` = the drive is just starting (simulated future drive), else in progress. */
  private duration(o: number, half: number, hsr: number, y: number, fresh: boolean, u: number): number {
    const b = this.j.buckets, tb = searchsortedLeft(b.tbEdges, hsr), zone = searchsortedLeft(b.zoneEdges, y), pre = fresh ? 'S:' : '', O = OUTCOMES[o];
    const key = [`${pre}${O}|${half}|${tb}|${zone}`, `${pre}${O}|${half}|${tb}|*`, `${pre}${O}|*|${tb}|*`, `${O}|${half}|${tb}|*`, `${O}|*|${tb}|*`].find((k) => this.dur.has(k));
    if (!key) return 150;
    const q = this.dur.get(key)!;
    const x = u * (q.length - 1), i = Math.floor(x), f = x - i;
    const d = i + 1 < q.length ? q[i] * (1 - f) + q[i + 1] * f : q[i];
    // Real drives in one game are more regular than independent draws from a table pooled over all teams and games, so the
    // simulated drive COUNT came out too variable in college (2H sd 2.6 vs 2.35 real) and the totals distribution too wide.
    // Shrinking each draw toward its table mean keeps the mean drive length and fixes the count dispersion.
    return this.durShrink === 1 ? d : this.durMean.get(key)! + this.durShrink * (d - this.durMean.get(key)!);
  }

  private startFrom(dist: Float64Array, u1: number, u2: number): number {
    const bin = sampleDiscrete(dist, u1);
    return clamp(bin * 10 + 1 + Math.floor(u2 * 10), 1, 99);
  }
  /** Where the other team starts after a non-scoring end. `from`: '' = drive in progress (keyed by its current spot),
   *  'S:' = simulated drive drawn at its start, 'E:' = the ball's spot when possession changes (play sampler). */
  private nextStart(o: number, y: number, u1: number, u2: number, from: '' | 'S:' | 'E:' = ''): number {
    const yb = clamp(Math.floor((y - 1) / 10), 0, 9), O = OUTCOMES[o];
    const get = (b: number) => this.next.get(`${from}${O}|${b}`) ?? this.next.get(`${O}|${b}`);
    let d = get(yb);
    for (let k = 1; !d && k < 10; k++) d = get(clamp(yb - k, 0, 9)) ?? get(clamp(yb + k, 0, 9));
    // Fallback (should not happen with the shipped tables): the receiving team starts at the spot of the change of possession.
    return d ? this.startFrom(d, u1, u2) : clamp(100 - y, 1, 99);
  }

  /** Lambda pair (home, away) such that, pregame, the market's spread and total split the outcomes 50/50: P(margin > spread) =
   *  P(margin < spread) and P(total > total) = P(total < total). Lines are medians, and football scores are right-skewed, so
   *  matching means instead leaves the model short of the over (the first holdout run showed ~5 points of under-bias).
   *  Parametrised as lambda = exp(a +/- b): `a` sets overall scoring (total), `b` the home/away tilt (spread). Each is found by
   *  a secant search on its own balance with common random numbers (fixed seed), so the target function is deterministic. */
  lambdas(prior: Prior, sims = 8000): [number, number] {
    const key = `${prior.spread}|${prior.total}`;
    const hit = this.lamCache.get(key); if (hit) return hit;
    const start: GameState = { half: 1, hsr: this.halfSeconds, homeScore: 0, awayScore: 0, possHome: true, kickoff: true, y: 75, down: 1, togo: 10, homeTO: 3, awayTO: 3, recv2Home: null };
    const bal = (d: Iterable<[number, number]>, line: number) => { let gt = 0, lt = 0; for (const [k, v] of d) { if (k > line) gt += v; else if (k < line) lt += v; } return gt - lt; };
    const evalAB = (a: number, b: number) => {
      const r = this.simulate(start, prior, { sims, seed: 12345, lambda: [Math.exp(a + b), Math.exp(a - b)], openingCoinFlip: true, env: 'prior' });
      return { t: bal(r.total.entries(), prior.total), m: bal(r.margin.entries(), prior.spread), r };
    };
    // Start from the mean-matching guess: one simulation at lambda = 1 gives points per unit lambda for this ruleset.
    const r0 = this.simulate(start, prior, { sims, seed: 12345, lambda: [1, 1], openingCoinFlip: true, env: 'prior' });
    const base = (mean(r0.home) + mean(r0.away)) / 2;
    let a = Math.log(Math.max(prior.total / 2, 1) / base), b = 0.5 * Math.log(Math.max(prior.total + prior.spread, 2) / Math.max(prior.total - prior.spread, 2));
    // Newton steps with fixed slopes: d(balance)/d(a) and d(balance)/d(b) are both ~2.7 for NFL-like scoring (measured). Secant
    // slopes were tried and diverge: with common random numbers the balance is a step function of lambda, so small steps give
    // garbage slopes. Keep the best iterate seen.
    const SLOPE = 2.7;
    let cur = evalAB(a, b), best = { a, b, err: Math.abs(cur.t) + Math.abs(cur.m) };
    for (let it = 0; it < 10 && best.err > 0.008; it++) {
      a += clamp(-cur.t / SLOPE, -0.3, 0.3); b += clamp(-cur.m / SLOPE, -0.3, 0.3);
      cur = evalAB(a, b);
      const err = Math.abs(cur.t) + Math.abs(cur.m);
      if (err < best.err) best = { a, b, err };
    }
    a = best.a; b = best.b;
    const lam: [number, number] = [clamp(Math.exp(a + b), 0.2, 4), clamp(Math.exp(a - b), 0.2, 4)];
    this.lamCache.set(key, lam);
    // Expected points so far, by elapsed minute, at this line: the denominator of the live game-environment update.
    const curve = this.simulate(start, prior, { sims, seed: 4242, lambda: lam, openingCoinFlip: true, env: 'prior', curve: true }).curve!;
    this.curveCache.set(key, curve);
    return lam;
  }

  /** Expected total points scored by `elapsed` game seconds (regulation) at this prior's calibrated strengths. */
  expectedSoFar(prior: Prior, elapsed: number): number {
    this.lambdas(prior);
    const c = this.curveCache.get(`${prior.spread}|${prior.total}`)!;
    const x = clamp(elapsed / 60, 0, c.length - 1), i = Math.floor(x), f = x - i;
    return i + 1 < c.length ? c[i] * (1 - f) + c[i + 1] * f : c[i];
  }

  /** P(home wins OT | tied after regulation, no OT tie): mild lean toward the market favourite (heuristic, unfitted). */
  private pHomeOT(prior: Prior): number { return 0.5 + 0.7 * (normCdf(prior.spread / 13.5) - 0.5); }

  simulate(st: GameState, prior: Prior, opt: SimOptions & { curve?: boolean } = {}): SimResult & { curve?: Float64Array } {
    const sims = opt.sims ?? 20000;
    const lam = opt.lambda ?? this.lambdas(prior);
    const rand = rng(opt.seed ?? hashState({ st, prior }));
    // Game-environment distribution for this call: the prior, or (live) the posterior given points so far vs expected so far.
    const mode: EnvMode = this.envAlpha > 0 ? (opt.env ?? 'posterior') : 'off';
    let env: { shape: number; rate: number } | null = null;
    if (mode !== 'off') {
      env = { shape: this.envAlpha, rate: this.envAlpha };
      if (mode === 'posterior') {
        const elapsed = (st.half - 1) * this.halfSeconds + (this.halfSeconds - st.hsr);
        env.shape += (st.homeScore + st.awayScore) / this.pointsDisp;
        env.rate += this.expectedSoFar(prior, elapsed) / this.pointsDisp;
      }
    }
    const curveBins = opt.curve ? new Float64Array(61) : null;
    const dstats = opt.stats ? { n: [0, 0], y: [0, 0], o: [new Float64Array(K), new Float64Array(K)], nextY: new Float64Array(K), nextN: new Float64Array(K), win: Array.from({ length: 6 }, () => ({ n: 0, td: 0, fg: 0, y: 0, gap: 0 })) } : null;
    const pairsH1 = opt.pairs ? new Int16Array(sims) : null, pairsH2 = opt.pairs ? new Int16Array(sims) : null;
    const MAXP = 120;
    const total = new Float64Array(2 * MAXP), home = new Float64Array(MAXP), away = new Float64Array(MAXP);
    const h1 = st.half === 1 ? new Float64Array(2 * MAXP) : null;
    const margin = new Map<number, number>();
    let pH = 0, pA = 0, pT = 0, dSum = 0, dSq = 0;
    const p = new Float64Array(K);
    const pOTH = this.pHomeOT(prior);

    for (let s = 0; s < sims; s++) {
      let half = st.half, t = st.hsr, h = st.homeScore, a = st.awayScore;
      const g = env ? gamma(env.shape, rand) / env.rate : 1;
      let h1Pts = st.half === 1 ? -1 : -2;   // first-half points for opt.pairs (-2 = unknown, state already in half 2)
      let possHome = opt.openingCoinFlip ? rand() < 0.5 : st.possHome;
      let y = st.y, down = st.down, togo = st.togo;
      const recv2 = st.recv2Home ?? (opt.openingCoinFlip ? !possHome : rand() < 0.5);
      let first = true;
      // Timeouts: the state's for the current half, 3 each after halftime. The drive-level part only uses them for the kneel-out
      // rule; the end-of-half play sampler spends them.
      let hTO = st.homeTO, aTO = st.awayTO;
      if (st.kickoff) { y = this.startFrom(this.kick, rand(), rand()); down = 1; togo = 10; }
      let h1Done = st.half === 2, nd = 0;
      for (let guard = 0; guard < 200; guard++) {
        if (t <= 0) {
          if (half === 1) {
            if (h1 && !h1Done) { h1[Math.min(h + a, 2 * MAXP - 1)]++; h1Done = true; }
            h1Pts = h + a;
            half = 2; t = this.halfSeconds; possHome = recv2; hTO = 3; aTO = 3;
            y = this.startFrom(this.kick, rand(), rand()); down = 1; togo = 10; first = false; continue;
          }
          break;
        }
        const diff = possHome ? h - a : a - h;
        // End of half: hand off to the play sampler (real end-of-half snaps carry the clock rules, timeouts, hurry-up and kneels).
        if (this.plays && t <= this.plays.start) {
          for (let snaps = 0; t > 0 && snaps < 400; snaps++) {
            const df = possHome ? h - a : a - h;
            const tilt = g * this.timeFactor[half - 1][t > 300 ? 0 : t > 120 ? 1 : 2] * (possHome ? lam[0] : lam[1]) / this.strength[this.cell(half, t, df, y, down, togo)];
            const pl = this.plays.sample(half, t, y, down, togo, df, possHome ? hTO : aTO, tilt, rand);
            if (possHome) { hTO = Math.max(0, hTO - pl.offTO); aTO = Math.max(0, aTO - pl.defTO); } else { aTO = Math.max(0, aTO - pl.offTO); hTO = Math.max(0, hTO - pl.defTO); }
            t -= pl.clock;
            const before = h + a;
            let ev: number = pl.ev, flip = false;
            if (ev === EV.PLAY) {
              y -= pl.gain;
              if (y <= 0) ev = EV.TD;
              else if (y >= 100) ev = EV.SAFETY;
              else if (pl.gain >= togo) { down = 1; togo = Math.min(10, y); }
              else if (down >= 4) { flip = true; y = clamp(100 - y, 1, 99); nd++; }          // turnover on downs at the spot
              else { down++; togo -= pl.gain; }
            }
            switch (ev) {
              case EV.TD: { const pts = this.tdPts.v[sampleDiscrete(this.tdPts.p, rand())]; if (possHome) h += pts; else a += pts; flip = true; y = this.startFrom(this.kick, rand(), rand()); nd++; break; }
              case EV.FG_GOOD: { if (possHome) h += 3; else a += 3; flip = true; y = this.startFrom(this.kick, rand(), rand()); nd++; break; }
              case EV.FG_MISS: { flip = true; y = this.nextStart(MISSFG, y, rand(), rand(), 'E:'); nd++; break; }
              case EV.PUNT: { flip = true; y = this.nextStart(PUNT, y, rand(), rand(), 'E:'); nd++; break; }
              case EV.TO: { flip = true; y = this.nextStart(TO, y, rand(), rand(), 'E:'); nd++; break; }
              case EV.DEF_TD: { const pts = this.tdPts.v[sampleDiscrete(this.tdPts.p, rand())]; if (possHome) a += pts; else h += pts; y = this.startFrom(this.kick, rand(), rand()); nd++; break; }
              case EV.SAFETY: { if (possHome) a += 2; else h += 2; flip = true; y = clamp(Math.round(55 + rand() * 15), 1, 99); nd++; break; }
            }
            if (flip) possHome = !possHome;
            if (ev !== EV.PLAY || flip) { down = 1; togo = Math.min(10, y); }
            if (curveBins && h + a > before) curveBins[clamp(Math.floor(((half - 1) * this.halfSeconds + this.halfSeconds - Math.max(t, 1)) / 60), 0, 59)] += h + a - before;
          }
          t = 0; first = false;
          continue;
        }
        // Victory formation: the leader has the ball late and can kneel the clock out (timeouts only known for the current drive).
        if (half === 2 && diff > 0) {
          const snaps = 5 - down, defTO = possHome ? aTO : hTO;
          if (t <= 40 * Math.max(0, snaps - defTO)) break;
        }
        const c = this.cell(half, t, diff, y, down, togo);
        const base = this.probs.subarray(c * K, c * K + K);
        // Strength relative to the offenses this cell was fitted on: a 17-point leader's cell already reflects mostly strong teams,
        // so multiplying by the full lambda would count team strength twice (v1 over-predicted late NCAAF blowout scoring).
        const l = g * this.timeFactor[half - 1][t > 300 ? 0 : t > 120 ? 1 : 2] * (possHome ? lam[0] : lam[1]) / this.strength[c];
        const score = base[TD] + base[FG], scaledScore = Math.min(0.98, l * score);
        const rest = (1 - scaledScore) / Math.max(1e-9, 1 - score);
        for (let i = 0; i < K; i++) p[i] = base[i] * rest;
        p[TD] = score > 0 ? base[TD] * scaledScore / score : 0; p[FG] = score > 0 ? base[FG] * scaledScore / score : 0;
        const o = sampleDiscrete(p, rand()); nd++;
        if (dstats && down === 1) { dstats.n[half - 1]++; dstats.y[half - 1] += y; dstats.o[half - 1][o]++; }
        if (dstats && down === 1) { const w = dstats.win[(half - 1) * 3 + (t > 300 ? 0 : t > 120 ? 1 : 2)]; w.n++; w.y += y; w.gap += Math.abs(diff); if (o === TD) w.td++; if (o === FG) w.fg++; }
        const fresh = !first || !!st.kickoff;   // a drive starting now (vs the live state's drive in progress)
        t -= this.duration(o, half, t, y, fresh, rand());
        const before = h + a;
        first = false;
        switch (o) {
          case TD: { const pts = this.tdPts.v[sampleDiscrete(this.tdPts.p, rand())]; if (possHome) h += pts; else a += pts; possHome = !possHome; y = this.startFrom(this.kick, rand(), rand()); break; }
          case FG: { if (possHome) h += 3; else a += 3; possHome = !possHome; y = this.startFrom(this.kick, rand(), rand()); break; }
          case OPPTD: { const pts = this.tdPts.v[sampleDiscrete(this.tdPts.p, rand())]; if (possHome) a += pts; else h += pts; y = this.startFrom(this.kick, rand(), rand()); break; } // scorer kicks off back to the same offense
          case SAFETY: { if (possHome) a += 2; else h += 2; possHome = !possHome; y = clamp(Math.round(55 + rand() * 15), 1, 99); break; } // free kick to the team that scored
          case END: { t = 0; break; }
          default: { const ny = this.nextStart(o, y, rand(), rand(), fresh ? 'S:' : ''); possHome = !possHome; y = ny; }
        }
        if (dstats && t > 0 && o !== END) { dstats.nextY[o] += y; dstats.nextN[o]++; }   // where the next drive starts (diagnostics)
        down = 1; togo = Math.min(10, y);
        // Bin the points at the drive's end time; a drive whose clock ran past 0:00 scored at the end of ITS half (t >= 1 keeps it
        // in that half's last minute; clamping at 0 filed buzzer scores into the next half's first minute).
        if (curveBins && h + a > before) curveBins[clamp(Math.floor(((half - 1) * this.halfSeconds + this.halfSeconds - Math.max(t, 1)) / 60), 0, 59)] += h + a - before;
      }
      if (pairsH1) { if (h1Pts === -1) h1Pts = h + a; pairsH1[s] = h1Pts; pairsH2![s] = h + a - h1Pts; }
      if (h1 && !h1Done) h1[Math.min(h + a, 2 * MAXP - 1)]++;
      dSum += nd; dSq += nd * nd;
      // Overtime (regular-season rules): some ties stand; otherwise the winner adds its OT points, the loser its own.
      if (h === a) {
        if (rand() < this.j.ot.pTie) { /* stays tied */ }
        else {
          const homeWins = rand() < pOTH;
          const w = this.otWin.v[sampleDiscrete(this.otWin.p, rand())];
          let lo = this.otLose.v[sampleDiscrete(this.otLose.p, rand())]; if (lo >= w) lo = w - 1 >= 0 ? Math.min(lo, w - 1) : 0;
          const addW = Math.max(w, lo + 1);
          if (homeWins) { h += addW; a += lo; } else { a += addW; h += lo; }
        }
      }
      if (h > a) pH++; else if (a > h) pA++; else pT++;
      total[Math.min(h + a, 2 * MAXP - 1)]++; home[Math.min(h, MAXP - 1)]++; away[Math.min(a, MAXP - 1)]++;
      margin.set(h - a, (margin.get(h - a) ?? 0) + 1);
    }
    const norm = (x: Float64Array | null) => { if (x) for (let i = 0; i < x.length; i++) x[i] /= sims; return x; };
    for (const [k, v] of margin) margin.set(k, v / sims);
    return { n: sims, pHome: pH / sims, pAway: pA / sims, pTie: pT / sims, total: norm(total)!, home: norm(home)!, away: norm(away)!, h1Total: norm(h1), margin, lambda: lam,
      drives: { mean: dSum / sims, sd: Math.sqrt(Math.max(0, dSq / sims - (dSum / sims) ** 2)) }, env,
      ...(pairsH1 ? { pairs: { h1: pairsH1, h2: pairsH2! } } : {}),
      ...(dstats ? { stats: dstats } : {}),
      ...(curveBins ? { curve: (() => { const c = new Float64Array(61); let acc = 0; for (let i = 0; i < 61; i++) { c[i] = acc; acc += curveBins[i] / sims; } return c; })() } : {}) };
  }
}

/** Continuous (interpolated) median of an integer distribution: treats value k as spread over [k-0.5, k+0.5]. */
export function median(dist: ArrayLike<number>, offset = 0): number {
  let c = 0;
  for (let k = 0; k < dist.length; k++) { if (c + dist[k] >= 0.5 && dist[k] > 0) return k + offset - 0.5 + (0.5 - c) / dist[k]; c += dist[k]; }
  return dist.length - 1 + offset;
}
export function medianMargin(m: Map<number, number>): number {
  const keys = [...m.keys()]; const lo = Math.min(...keys), hi = Math.max(...keys);
  const arr = new Float64Array(hi - lo + 1); for (const [k, v] of m) arr[k - lo] = v;
  return median(arr, lo);
}
export function mean(dist: Float64Array): number { let m = 0; for (let i = 0; i < dist.length; i++) m += i * dist[i]; return m; }
/** P(X > line) for a points distribution; lines are usually x.5 so there is no push. */
export function pOver(dist: Float64Array, line: number): number { let p = 0; for (let i = Math.floor(line) + 1; i < dist.length; i++) p += dist[i]; return p; }
/** P(home - away > x), e.g. home "wins by over 3.5". */
export function pMarginOver(m: Map<number, number>, x: number): number { let p = 0; for (const [k, v] of m) if (k > x) p += v; return p; }
