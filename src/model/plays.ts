// End-of-half play sampler. The drive engine (drive.ts) hands off to this when a half has <= startSeconds left: from there the
// half is simulated snap by snap by drawing real plays (data/models/<league>_plays_v1.json, scripts/build_plays.py) from the
// most detailed group of similar situations with enough samples (half, clock, yards to goal, down, to-go, score diff, offence
// timeouts). A sample carries its own clock use and timeout use, so the league's end-of-half clock rules come with it.
// Team strength enters by tilting the share of scoring plays (TD, FG) in the drawn group, the same way the drive engine scales
// TD/FG probabilities.

export const EV = { PLAY: 0, TD: 1, FG_GOOD: 2, FG_MISS: 3, PUNT: 4, TO: 5, DEF_TD: 6, SAFETY: 7 } as const;
const MIN_GROUP = 25;

export interface PlaysJson {
  version: string; league: string; startSeconds: number;
  buckets: { tbEdges: number[]; ybEdges: number[]; tgEdges: number[]; dfEdges: number[] };
  events: string[]; nSamples: number; samples: number[];   // flat [event, gain, clock, offTimeoutUsed, defTimeoutUsed] * n
  levels: { fields: string[]; groups: Record<string, number[]> }[];
}
export interface Play { ev: number; gain: number; clock: number; offTO: number; defTO: number }

const bucket = (edges: number[], x: number) => { let i = 0; while (i < edges.length && x > edges[i]) i++; return i; };

export class PlayModel {
  readonly start: number;
  private s: Int32Array;
  private levels: { fields: string[]; groups: Map<string, { score: Int32Array; other: Int32Array; pScore: number }> }[];
  private b: PlaysJson['buckets'];

  constructor(j: PlaysJson) {
    this.start = j.startSeconds; this.b = j.buckets;
    this.s = Int32Array.from(j.samples);
    this.levels = j.levels.map((lv) => {
      const groups = new Map<string, { score: Int32Array; other: Int32Array; pScore: number }>();
      for (const [k, ids] of Object.entries(lv.groups)) {
        if (ids.length < MIN_GROUP) continue;
        const score = ids.filter((i) => this.s[i * 5] === EV.TD || this.s[i * 5] === EV.FG_GOOD), other = ids.filter((i) => !(this.s[i * 5] === EV.TD || this.s[i * 5] === EV.FG_GOOD));
        groups.set(k, { score: Int32Array.from(score), other: Int32Array.from(other), pScore: score.length / ids.length });
      }
      return { fields: lv.fields, groups };
    });
  }

  /** Draw one snap for the offence. `tilt` multiplies the group's share of scoring plays (team strength x game environment). */
  sample(half: number, t: number, y: number, down: number, togo: number, diff: number, offTimeouts: number, tilt: number, rand: () => number): Play {
    const v: Record<string, number> = {
      h: half, tb: bucket(this.b.tbEdges, t), yb: bucket(this.b.ybEdges, y), dn: Math.max(1, Math.min(4, down)),
      tg: bucket(this.b.tgEdges, Math.max(1, togo)), df: bucket(this.b.dfEdges, diff), to: Math.max(0, Math.min(2, offTimeouts)),
    };
    for (const lv of this.levels) {
      const g = lv.groups.get(lv.fields.map((f) => v[f]).join('|'));
      if (!g) continue;
      const p = Math.min(0.95, tilt * g.pScore);
      const pool = (rand() < p && g.score.length) || !g.other.length ? g.score : g.other;
      const i = pool[Math.floor(rand() * pool.length)] * 5;
      return { ev: this.s[i], gain: this.s[i + 1], clock: this.s[i + 2], offTO: this.s[i + 3], defTO: this.s[i + 4] };
    }
    return { ev: EV.PLAY, gain: 0, clock: Math.min(t, 30), offTO: 0, defTO: 0 };   // no group at all (cannot happen with shipped tables)
  }
}
