#!/usr/bin/env python3
"""Build the NFL drive-model tables from nflverse play-by-play (data/cache/nfl/pbp_<season>.csv.gz, games.csv).

Outputs
  data/models/nfl_drive_v2.json   tables the TypeScript engine reads (src/model/drive.ts)
  data/cache/nfl/bt_states.csv.gz sampled in-game states from the holdout seasons, for scripts/backtest.ts

Model pieces (all empirical, fitted on TRAIN seasons only; holdout seasons are never used for tables except kickoff start position,
which is per-season because the kickoff rules changed in 2024 and 2025):
  outcome  P(drive result | half-clock bucket, half x score-gap bucket, yards-to-goal bin, down, to-go bin), hierarchically smoothed
  strength mean pregame strength r of the offenses behind each outcome cell (r = market-implied points / league average). Cells
           conditioned on the score gap are dominated by strong teams when leading big and weak ones when trailing big; the engine
           divides each team's strength by the cell's r so team strength is not counted twice (the cause of the NCAAF late-game
           over-prediction in drive model v1)
  duration quantiles (2.5% steps, capped at p99, feed gaps > 10 min dropped) of the seconds the rest of the drive takes, by
           (result, half, time bucket, field zone)
  nextStart where the other team starts after a non-scoring drive end, by (result, yards-to-goal bin of the state)
  kickoff  start-position distribution after a score, per season
  tdPoints distribution of points on a touchdown drive (6/7/8)
  ot       P(tie | OT) and the OT points distribution (regular season, 2017+ 10-minute OT)

  python3 scripts/build_nfl.py [--train 2016-2023] [--holdout 2024-2026]
"""
import argparse, gzip, json, os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, 'data', 'cache', 'nfl')

OUTCOMES = ['TD', 'FG', 'PUNT', 'TO', 'DOWNS', 'MISSFG', 'END', 'OPPTD', 'SAFETY']
RESULT_MAP = {'Touchdown': 'TD', 'Field goal': 'FG', 'Punt': 'PUNT', 'Turnover': 'TO', 'Turnover on downs': 'DOWNS',
              'Missed field goal': 'MISSFG', 'End of half': 'END', 'Opp touchdown': 'OPPTD', 'Safety': 'SAFETY'}
SMOOTH_M = 20.0  # pseudo-count pulling each cell toward its parent cell
STRENGTH_M = 50.0  # pseudo-count for the per-cell mean offense strength

# Bucket definitions. src/model/drive.ts duplicates these; tests/drive.test.ts checks the two agree on the JSON's `buckets` block.
TB_EDGES = [30, 120, 300]          # half seconds remaining: <=30 / <=120 / <=300 / >300
GAP_EDGES = [-17, -9, -1, 0, 8, 16]  # posteam score diff: <=-17 / -16..-9 / -8..-1 / 0 / 1..8 / 9..16 / >=17 (both halves)
N_GAP = len(GAP_EDGES) + 1
N_HG = 2 * N_GAP                   # half x gap
TOGO_EDGES = [1, 3, 6, 10]         # to-go: 1 / 2-3 / 4-6 / 7-10 / 11+
ZONE_EDGES = [20, 60]              # yards to goal: <=20 / 21-60 / 61+


def tb_of(hsr): return np.searchsorted(TB_EDGES, hsr, side='left')            # 0..3
def ybin_of(y): return np.clip((np.asarray(y) - 1) // 10, 0, 9).astype(int)   # 1-10 -> 0 ... 91-99 -> 9
def togo_of(t): return np.searchsorted(TOGO_EDGES, t, side='left')            # 0..4
def zone_of(y): return np.searchsorted(ZONE_EDGES, y, side='left')            # 0..2
def hg_of(half2, diff):
    # half x score-gap bucket: 0..6 in half 1, 7..13 in half 2.
    return np.asarray(half2, dtype=int) * N_GAP + np.searchsorted(GAP_EDGES, diff, side='left')


def load(seasons):
    cols = ['game_id', 'season', 'season_type', 'week', 'home_team', 'away_team', 'posteam', 'defteam', 'play_type', 'down', 'ydstogo',
            'yardline_100', 'half_seconds_remaining', 'game_seconds_remaining', 'game_half', 'qtr', 'fixed_drive', 'fixed_drive_result',
            'total_home_score', 'total_away_score', 'posteam_score', 'defteam_score', 'posteam_timeouts_remaining',
            'defteam_timeouts_remaining', 'spread_line', 'total_line', 'vegas_home_wp', 'home_score', 'away_score', 'play_id',
            # play-level fields for scripts/build_plays.py
            'yards_gained', 'touchdown', 'td_team', 'field_goal_result', 'punt_attempt', 'interception', 'fumble_lost', 'safety']
    out = []
    for s in seasons:
        p = os.path.join(CACHE, f'pbp_{s}.csv.gz')
        if not os.path.exists(p): print('missing', p, file=sys.stderr); continue
        out.append(pd.read_csv(p, usecols=cols, low_memory=False))
    d = pd.concat(out, ignore_index=True)
    return d.sort_values(['game_id', 'play_id'], kind='stable').reset_index(drop=True)


def drive_frame(d):
    """One row per (game, drive): posteam, half, result, end-of-drive half-seconds, points for/against."""
    d = d[d.game_half.isin(['Half1', 'Half2']) & d.fixed_drive.notna()]
    g = d.groupby(['game_id', 'fixed_drive'], sort=False)
    dr = g.agg(posteam=('posteam', lambda s: s.dropna().iloc[0] if s.notna().any() else None), half=('game_half', 'first'),
               result=('fixed_drive_result', 'first'), home=('home_team', 'first'),
               end_home=('total_home_score', 'last'), end_away=('total_away_score', 'last')).reset_index()
    dr['result'] = dr.result.map(RESULT_MAP)
    # Drive end time = start time of the next drive in the same half (0 if this was the half's last drive).
    first = g.half_seconds_remaining.max().reset_index(name='start_hsr')
    dr = dr.merge(first, on=['game_id', 'fixed_drive'])
    nxt = dr.groupby('game_id').shift(-1)
    dr['end_hsr'] = np.where(nxt.half == dr.half, nxt.start_hsr, 0.0)
    # Points on the drive = score change from the previous drive's end (scores are post-play).
    prev = dr.groupby('game_id')[['end_home', 'end_away']].shift(1).fillna(0)
    dh, da = dr.end_home - prev.end_home, dr.end_away - prev.end_away
    is_home = dr.posteam == dr.home
    dr['pts_for'] = np.where(is_home, dh, da)
    dr['pts_against'] = np.where(is_home, da, dh)
    # The next drive's starting state (for nextStart): first scrimmage yardline of the following drive.
    # Kickoff / PAT rows are not snaps: college (CFBD) kickoff rows carry down 1 and the kickoff spot (65 yards to go), so letting
    # them count gave every post-kickoff drive a start ~10 yards too far forward (drive model v2 college bug: simulated drives
    # started at 66.7 yards to goal on average vs 71.3 real, over-producing points that the line calibration then took back from
    # every team, starving end-of-half scoring).
    scr = d[d.down.notna() & d.yardline_100.notna() & ~d.play_type.isin(['kickoff', 'extra_point'])]
    st = scr.groupby(['game_id', 'fixed_drive']).yardline_100.first().reset_index(name='start_y')
    dr = dr.merge(st, on=['game_id', 'fixed_drive'], how='left')
    dr['next_start_y'] = np.where(dr.groupby('game_id').half.shift(-1) == dr.half, dr.groupby('game_id').start_y.shift(-1), np.nan)
    dr['next_posteam'] = dr.groupby('game_id').posteam.shift(-1)
    return dr


def states(d, dr):
    """Scrimmage states joined to their drive's result and end time."""
    s = d[d.game_half.isin(['Half1', 'Half2']) & d.down.notna() & d.yardline_100.notna() & d.posteam.notna()
          & ~d.play_type.isin(['kickoff', 'extra_point'])].copy()
    s = s.merge(dr[['game_id', 'fixed_drive', 'result', 'end_hsr']], on=['game_id', 'fixed_drive'], how='inner')
    s = s[s.result.notna()]
    s['y'] = s.yardline_100.astype(int)
    s['tb'] = tb_of(s.half_seconds_remaining.values)
    s['hg'] = hg_of((s.game_half == 'Half2').values, (s.posteam_score - s.defteam_score).values)
    # Offense strength: market-implied points for the team with the ball, relative to the average team (NaN without a line).
    sgn = np.where(s.posteam == s.home_team, 1.0, -1.0)
    implied = (s.total_line + sgn * s.spread_line) / 2
    s['r'] = implied / (s.groupby('game_id').total_line.first().mean() / 2)
    s['yb'] = ybin_of(s.y.values)
    s['dn'] = s.down.astype(int) - 1
    s['tg'] = togo_of(s.ydstogo.values)
    s['zone'] = zone_of(s.y.values)
    s['rem'] = (s.half_seconds_remaining - s.end_hsr).clip(lower=0)
    return s


LEVELS = [(['tb', 'hg', 'yb'], (4, N_HG, 10)), (['tb', 'hg', 'yb', 'dn'], (4, N_HG, 10, 4)), (['tb', 'hg', 'yb', 'dn', 'tg'], (4, N_HG, 10, 4, 5))]


def smoothed_strength(s):
    """Mean offense strength r per outcome cell, each level pulled toward its parent with pseudo-count STRENGTH_M."""
    x = s[s.r.notna()]
    def sums(keys, shape):
        n, v = np.zeros(shape), np.zeros(shape)
        idx = tuple(x[k].values for k in keys)
        np.add.at(n, idx, 1); np.add.at(v, idx, x.r.values)
        return n, v
    n0, v0 = sums(['yb'], (10,)); p = (v0 + STRENGTH_M) / (n0 + STRENGTH_M)
    for i, (keys, shape) in enumerate(LEVELS):
        n, v = sums(keys, shape)
        p = (v + STRENGTH_M * p[(None, None) if i == 0 else (..., None)]) / (n + STRENGTH_M)
    return np.round(p, 4)


def smoothed_table(s):
    """P(outcome | tb, hg, yb, dn, tg): each level is pulled toward its parent with pseudo-count SMOOTH_M."""
    K = len(OUTCOMES)
    oi = s.result.map({o: i for i, o in enumerate(OUTCOMES)}).values
    def counts(keys, shape):
        c = np.zeros(shape + (K,))
        np.add.at(c, tuple(s[k].values for k in keys) + (oi,), 1)
        return c
    def blend(c, parent):
        n = c.sum(-1, keepdims=True)
        return (c + SMOOTH_M * parent) / (n + SMOOTH_M)
    c0 = counts(['yb'], (10,));                         p0 = (c0 + 1) / (c0.sum(-1, keepdims=True) + K)
    (k1, s1), (k2, s2), (k3, s3) = LEVELS
    p1 = blend(counts(k1, s1), p0[None, None])
    p2 = blend(counts(k2, s2), p1[..., None, :])
    p3 = blend(counts(k3, s3), p2[..., None, :])
    return np.round(p3, 5), int(len(s))


MAX_DRIVE_SECONDS = 600   # no real drive lasts 10 minutes; longer "drives" are feed gaps (skipped drives), mostly in college data


def quantiles(x, q=41):
    """q evenly spaced quantiles (2.5% steps), the top one capped at the 99th percentile. The engine interpolates linearly between
    neighbouring quantiles, so an outlier at q100 would be drawn from uniformly across the whole top step; drive model v2 used 5%
    steps up to the raw max (college maxima of 1100-1800s), which made simulated drives too long and their count too variable,
    and the NCAAF totals distribution too wide."""
    x = np.asarray(x, dtype=float)
    if not len(x): return None
    qs = np.quantile(x, np.linspace(0, 1, q))
    qs[-1] = min(qs[-1], np.quantile(x, 0.99))
    return [round(float(v), 1) for v in qs]


def duration_table(s, prefix=''):
    """Quantiles of the seconds left in the drive. Called twice: on every state (for the drive in progress, `prefix=''`) and on
    drive-start states only (`prefix='S:'`, for simulated future drives). Pooling mid-drive states into future drives would make
    them too short and fit too many drives into the game. States whose drive "lasts" over MAX_DRIVE_SECONDS are feed gaps and
    are left out."""
    s = s[s.rem <= MAX_DRIVE_SECONDS]
    s = s.assign(h=np.where(s.game_half == 'Half2', 2, 1))
    out = {}
    # Keyed by half too: first-half drives are quicker (offences race to halftime; college drives starting inside 2:00 take 53 s
    # in the first half vs 66 s in the second, where leaders burn clock). Pooling the halves moved simulated scoring earlier.
    for (o, h, tb, z), g in s.groupby(['result', 'h', 'tb', 'zone']):
        if len(g) >= 30: out[f'{prefix}{o}|{h}|{tb}|{z}'] = quantiles(g.rem)
    for (o, h, tb), g in s.groupby(['result', 'h', 'tb']):   # fallback without field zone
        out[f'{prefix}{o}|{h}|{tb}|*'] = quantiles(g.rem)
    for (o, tb), g in s.groupby(['result', 'tb']):           # fallback without half
        out[f'{prefix}{o}|*|{tb}|*'] = quantiles(g.rem)
    return out


def next_start_table(s, dr):
    """Where the OTHER team starts after a non-scoring end, keyed by (result, yards-to-goal bin). Three variants by which state the
    key's yards-to-goal comes from, matching how the engine asks:
      ''   any state of the drive          (the drive in progress, drawn from its current state)
      'S:' the drive's first snap          (a simulated future drive, drawn at its start)
      'E:' the drive's last snap           (the end-of-half play sampler: the ball is where possession changes)
    Pooling every state for fresh drives handed the opponent ~2 yards too good a start after punts (~4 after turnovers on downs),
    because later states of a drive sit further downfield than its start."""
    x = s.merge(dr[['game_id', 'fixed_drive', 'next_start_y', 'next_posteam']], on=['game_id', 'fixed_drive'])
    x = x[x.result.isin(['PUNT', 'TO', 'DOWNS', 'MISSFG']) & x.next_start_y.notna() & (x.next_posteam != x.posteam)]
    g = x.groupby(['game_id', 'fixed_drive'])
    out = {}
    for prefix, sub in [('', x), ('S:', x[g.cumcount() == 0]), ('E:', x[g.cumcount(ascending=False) == 0])]:
        for (o, yb), grp in sub.groupby(['result', 'yb']):
            h = np.bincount(ybin_of(grp.next_start_y.values), minlength=10).astype(float)
            out[f'{prefix}{o}|{yb}'] = [round(v, 5) for v in ((h + 0.5) / (h.sum() + 5)).tolist()]
    return out


def end_y_table(s):
    """Mean yards-to-goal of a drive's LAST snap, by (result, yards-to-goal bin of its first snap). Used when the engine interrupts a
    drive-level drive at the hand-off to the end-of-half play sampler: the ball's spot at that moment is interpolated between the
    drive's start and this typical end spot."""
    g = s.groupby(['game_id', 'fixed_drive'])
    d = pd.DataFrame({'result': g.result.first(), 'y0': g.y.first(), 'y1': g.y.last()})
    d['yb'] = ybin_of(d.y0.values)
    return {f'{o}|{yb}': round(float(x.y1.mean()), 1) for (o, yb), x in d.groupby(['result', 'yb']) if len(x) >= 20}


def kickoff_table(dr):
    """Start-position distribution (10-yard bins) after a TD/FG kickoff, per season."""
    x = dr.copy()
    x['season'] = x.game_id.str.slice(0, 4).astype(int)
    x = x[x.result.isin(['TD', 'FG']) & x.next_start_y.notna() & (x.next_posteam != x.posteam)]
    return {str(s): [round(v, 5) for v in (np.bincount(ybin_of(g.next_start_y.values), minlength=10) / len(g)).tolist()]
            for s, g in x.groupby('season')}


def td_points(dr):
    v = dr[dr.result == 'TD'].pts_for.round().astype(int)
    v = v[v.between(6, 8)]
    h = v.value_counts(normalize=True)
    return {str(k): round(float(h.get(k, 0)), 4) for k in (6, 7, 8)}


def ot_table(d, games):
    """Regular-season OT (2017+): share ending tied, and points added by the OT winner / loser."""
    g = games[(games.game_type == 'REG') & (games.season >= 2017) & (games.overtime == 1)]
    reg = d[d.qtr == 4].groupby('game_id')[['total_home_score', 'total_away_score']].max()
    x = g.merge(reg, left_on='game_id', right_index=True)
    x = x[(x.total_home_score == x.total_away_score)]
    add_h, add_a = x.home_score - x.total_home_score, x.away_score - x.total_away_score
    tie = (x.home_score == x.away_score)
    win_add = np.maximum(add_h, add_a)[~tie].astype(int)
    lose_add = np.minimum(add_h, add_a)[~tie].astype(int)
    return {'n': int(len(x)), 'pTie': round(float(tie.mean()), 4),
            'winnerAdd': {str(k): round(float(v), 4) for k, v in win_add.value_counts(normalize=True).sort_index().items()},
            'loserAdd': {str(k): round(float(v), 4) for k, v in lose_add.value_counts(normalize=True).sort_index().items()}}


WINDOWS = [(1800, 300), (300, 120), (120, 0)]   # half-seconds-remaining windows for the time profile: >5:00, 5:00-2:00, last 2:00


def window_share(d, games):
    """Share of regulation points scored in each half x window (6 values, half 1 first), over the given games. Scores are
    post-play running totals, so points in a window = score at its end - score at its start."""
    d = d[d.game_id.isin(set(games)) & d.game_half.isin(['Half1', 'Half2'])]
    tot = d.total_home_score + d.total_away_score
    d = d.assign(tot=tot)
    out, prev = [], pd.Series(0.0, index=pd.Index(sorted(set(games))))
    for half in ['Half1', 'Half2']:
        h = d[d.game_half == half]
        for _, end in WINDOWS:
            x = h if end == 0 else h[h.half_seconds_remaining > end]
            cur = x.groupby('game_id').tot.max().reindex(prev.index).fillna(prev).combine(prev, max)
            out.append(float((cur - prev).mean())); prev = cur
    s = sum(out)
    return [round(v / s, 5) for v in out]


def dispersion_targets(d, dr):
    """Real-data targets for scripts/fit_dispersion.ts (all from training seasons, regular season, games with a closing line):
      h2DrivesSd      SD of second-half drive counts per game
      h2PointsSd      SD of second-half points after regressing on the closing total and |spread|
      totalSd         SD of (final total - closing total)
      meanMinusLine   mean of (final total - closing total); lines are medians, so this measures right skew (validation only)
      h1h2Slope       slope of second-half on first-half points, both after regressing on the lines (validation only)
      pointsDisp      E[x^2]/E[x] of points per drive: converts points into effective event counts for the live update
      typicalPrior    median lines, where the fit simulates"""
    d = d[d.season_type == 'REG']
    c2 = dr[(dr.half == 'Half2') & dr.game_id.isin(set(d.game_id))].groupby('game_id').size()
    g = d.groupby('game_id').first()
    h1 = d[d.game_half == 'Half1'].groupby('game_id')[['total_home_score', 'total_away_score']].max().sum(axis=1)
    reg = d[d.game_half != 'Overtime'].groupby('game_id')[['total_home_score', 'total_away_score']].max().sum(axis=1)
    x = pd.DataFrame({'c2': c2, 'h1': h1, 'h2': reg - h1, 'fin': g.home_score + g.away_score, 'tl': g.total_line, 'sp': g.spread_line.abs()}).dropna()
    X = np.column_stack([np.ones(len(x)), x.tl, x.sp])
    resid = lambda y: y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    e1, e2 = resid(x.h1), resid(x.h2)
    pts = dr.pts_for.clip(lower=0); pts = pts[pts <= 8]
    return {'h2DrivesSd': round(float(x.c2.std()), 3), 'h2PointsSd': round(float(e2.std()), 3),
            'totalSd': round(float((x.fin - x.tl).std()), 3), 'meanMinusLine': round(float((x.fin - x.tl).mean()), 3),
            'h1h2Slope': round(float(np.cov(e1, e2)[0, 1] / e1.var()), 4), 'nGames': int(len(x)),
            'pointsDisp': round(float((pts ** 2).mean() / pts.mean()), 3),
            'windowShare': window_share(d, x.index),
            'typicalPrior': {'spread': round(float(x.sp.median()) * 2) / 2, 'total': round(float(x.tl.median()) * 2) / 2}}


def export_bt(d, dr, seasons, path, every=3):
    x = d[d.season.isin(seasons) & (d.season_type == 'REG')]
    s = x[x.game_half.isin(['Half1', 'Half2']) & x.down.notna() & x.yardline_100.notna() & x.posteam.notna()
          & ~x.play_type.isin(['kickoff', 'extra_point'])].copy()
    s = s[s.groupby('game_id').cumcount() % every == 0]
    # Who receives the second-half kickoff: the team that did NOT receive the opening kickoff.
    kick = x[(x.play_type == 'kickoff') & (x.game_half == 'Half1') & x.posteam.notna()].groupby('game_id').posteam.first()
    s['open_recv'] = s.game_id.map(kick)
    home = s.posteam == s.home_team
    out = pd.DataFrame({
        'game_id': s.game_id, 'season': s.season, 'week': s.week, 'half': np.where(s.game_half == 'Half2', 2, 1),
        'hsr': s.half_seconds_remaining.astype(int), 'pos_home': home.astype(int), 'y': s.yardline_100.astype(int),
        'down': s.down.astype(int), 'togo': s.ydstogo.astype(int),
        'hs': np.where(home, s.posteam_score, s.defteam_score).astype(int), 'as': np.where(home, s.defteam_score, s.posteam_score).astype(int),
        'hto': np.where(home, s.posteam_timeouts_remaining, s.defteam_timeouts_remaining),
        'ato': np.where(home, s.defteam_timeouts_remaining, s.posteam_timeouts_remaining),
        'recv2_home': (s.open_recv != s.home_team).astype(int),
        'spread': s.spread_line, 'total': s.total_line, 'final_h': s.home_score, 'final_a': s.away_score, 'vegas_home_wp': s.vegas_home_wp,
    })
    out = out.dropna(subset=['spread', 'total', 'hto', 'ato'])
    out.to_csv(path, index=False, compression='gzip')
    return len(out), out.game_id.nunique()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default='2016-2023'); ap.add_argument('--holdout', default='2024-2026')
    a = ap.parse_args()
    rng = lambda r: list(range(int(r.split('-')[0]), int(r.split('-')[-1]) + 1))
    train, hold = rng(a.train), rng(a.holdout)
    d = load(sorted(set(train + hold)))
    games = pd.read_csv(os.path.join(CACHE, 'games.csv'), low_memory=False)
    dr_all = drive_frame(d)
    dtr = d[d.season.isin(train)]
    dr = dr_all[dr_all.game_id.str.slice(0, 4).astype(int).isin(train)]
    s = states(dtr, dr)
    table, n_states = smoothed_table(s)
    model = {
        'version': 'nfl_drive_v2', 'league': 'NFL', 'trainSeasons': train, 'nStates': n_states, 'nDrives': int(len(dr)),
        'outcomes': OUTCOMES,
        'buckets': {'tbEdges': TB_EDGES, 'gapEdges': GAP_EDGES, 'togoEdges': TOGO_EDGES, 'zoneEdges': ZONE_EDGES, 'yBinWidth': 10},
        'outcome': table.tolist(),                       # [tb][half x gap][yb][down-1][tg][outcome]
        'strength': smoothed_strength(s).tolist(),       # [tb][half x gap][yb][down-1][tg]
        'duration': {**duration_table(s), **duration_table(s[s.groupby(['game_id', 'fixed_drive']).cumcount() == 0], 'S:')},
        'nextStart': next_start_table(s, dr),
        'kickoffStart': kickoff_table(dr_all),
        'endY': end_y_table(s),
        'tdPoints': td_points(dr),
        'ot': ot_table(d[d.season.isin(train)], games),
        'meanPointsPerGame': round(float(games[(games.game_type == 'REG') & games.season.isin(train)].total.mean()), 3),
        'targets': dispersion_targets(dtr, dr),
        'durationShrink': 1.0,   # set by scripts/fit_dispersion.ts
        'gameEnv': {'alpha': 0},  # set by scripts/fit_dispersion.ts (0 = no game-environment effect)
    }
    os.makedirs(os.path.join(ROOT, 'data', 'models'), exist_ok=True)
    with open(os.path.join(ROOT, 'data', 'models', 'nfl_drive_v2.json'), 'w') as f: json.dump(model, f, separators=(',', ':'))
    n, ng = export_bt(d, dr_all, hold, os.path.join(CACHE, 'bt_states.csv.gz'))
    print(f'train {train[0]}-{train[-1]}: {len(dr)} drives, {n_states} states; holdout {hold}: {n} states / {ng} games')
    print('tdPoints', model['tdPoints'], 'ot', {k: model['ot'][k] for k in ('n', 'pTie')})
    print('kickoff start (most common bin by season)', {k: int(np.argmax(v)) for k, v in model['kickoffStart'].items()})


if __name__ == '__main__':
    main()
