#!/usr/bin/env python3
"""Build the NFL drive-model tables from nflverse play-by-play (data/cache/nfl/pbp_<season>.csv.gz, games.csv).

Outputs
  data/models/nfl_drive_v1.json   tables the TypeScript engine reads (src/model/drive.ts)
  data/cache/nfl/bt_states.csv.gz sampled in-game states from the holdout seasons, for scripts/backtest.ts

Model pieces (all empirical, fitted on TRAIN seasons only; holdout seasons are never used for tables except kickoff start position,
which is per-season because the kickoff rules changed in 2024 and 2025):
  outcome  P(drive result | half-time bucket, late-game score bucket, yards-to-goal bin, down, to-go bin), hierarchically smoothed
  duration quantiles of the seconds the rest of the drive takes, by (result, time bucket, field zone)
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

# Bucket definitions. src/model/drive.ts duplicates these; tests/drive.test.ts checks the two agree on the JSON's `buckets` block.
TB_EDGES = [30, 120, 300]          # half seconds remaining: <=30 / <=120 / <=300 / >300
LATE_EDGES = [-9, -1, 0, 8]        # posteam score diff when late in half 2: <=-9 / -8..-1 / 0 / 1..8 / >=9
TOGO_EDGES = [1, 3, 6, 10]         # to-go: 1 / 2-3 / 4-6 / 7-10 / 11+
ZONE_EDGES = [20, 60]              # yards to goal: <=20 / 21-60 / 61+


def tb_of(hsr): return np.searchsorted(TB_EDGES, hsr, side='left')            # 0..3
def ybin_of(y): return np.clip((np.asarray(y) - 1) // 10, 0, 9).astype(int)   # 1-10 -> 0 ... 91-99 -> 9
def togo_of(t): return np.searchsorted(TOGO_EDGES, t, side='left')            # 0..4
def zone_of(y): return np.searchsorted(ZONE_EDGES, y, side='left')            # 0..2
def late_of(half2, tb, diff):
    # 0 = not late; 1..5 = score-diff bucket in the last 5 minutes of the second half.
    d = np.searchsorted(LATE_EDGES, diff, side='left') + 1
    return np.where(half2 & (tb <= 2), d, 0)


def load(seasons):
    cols = ['game_id', 'season', 'season_type', 'week', 'home_team', 'away_team', 'posteam', 'defteam', 'play_type', 'down', 'ydstogo',
            'yardline_100', 'half_seconds_remaining', 'game_seconds_remaining', 'game_half', 'qtr', 'fixed_drive', 'fixed_drive_result',
            'total_home_score', 'total_away_score', 'posteam_score', 'defteam_score', 'posteam_timeouts_remaining',
            'defteam_timeouts_remaining', 'spread_line', 'total_line', 'vegas_home_wp', 'home_score', 'away_score', 'play_id']
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
    scr = d[d.down.notna() & d.yardline_100.notna()]
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
    s['late'] = late_of((s.game_half == 'Half2').values, s.tb.values, (s.posteam_score - s.defteam_score).values)
    s['yb'] = ybin_of(s.y.values)
    s['dn'] = s.down.astype(int) - 1
    s['tg'] = togo_of(s.ydstogo.values)
    s['zone'] = zone_of(s.y.values)
    s['rem'] = (s.half_seconds_remaining - s.end_hsr).clip(lower=0)
    return s


def smoothed_table(s):
    """P(outcome | tb, late, yb, dn, tg): each level is pulled toward its parent with pseudo-count SMOOTH_M."""
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
    c1 = counts(['tb', 'late', 'yb'], (4, 6, 10));      p1 = blend(c1, p0[None, None])
    c2 = counts(['tb', 'late', 'yb', 'dn'], (4, 6, 10, 4)); p2 = blend(c2, p1[..., None, :])
    c3 = counts(['tb', 'late', 'yb', 'dn', 'tg'], (4, 6, 10, 4, 5)); p3 = blend(c3, p2[..., None, :])
    return np.round(p3, 5), int(len(s))


def quantiles(x, q=21):
    x = np.asarray(x, dtype=float)
    return [round(float(v), 1) for v in np.quantile(x, np.linspace(0, 1, q))] if len(x) else None


def duration_table(s, prefix=''):
    """Quantiles of the seconds left in the drive. Called twice: on every state (for the drive in progress, `prefix=''`) and on
    drive-start states only (`prefix='S:'`, for simulated future drives). Pooling mid-drive states into future drives would make
    them too short and fit too many drives into the game."""
    out = {}
    for (o, tb, z), g in s.groupby(['result', 'tb', 'zone']):
        if len(g) >= 30: out[f'{prefix}{o}|{tb}|{z}'] = quantiles(g.rem)
    for (o, tb), g in s.groupby(['result', 'tb']):          # fallback without field zone
        out[f'{prefix}{o}|{tb}|*'] = quantiles(g.rem)
    return out


def next_start_table(s, dr):
    """Where the OTHER team starts after a non-scoring end, keyed by (result, yards-to-goal bin of the state)."""
    x = s.merge(dr[['game_id', 'fixed_drive', 'next_start_y', 'next_posteam']], on=['game_id', 'fixed_drive'])
    x = x[x.result.isin(['PUNT', 'TO', 'DOWNS', 'MISSFG']) & x.next_start_y.notna() & (x.next_posteam != x.posteam)]
    out = {}
    for (o, yb), g in x.groupby(['result', 'yb']):
        h = np.bincount(ybin_of(g.next_start_y.values), minlength=10).astype(float)
        out[f'{o}|{yb}'] = [round(v, 5) for v in ((h + 0.5) / (h.sum() + 5)).tolist()]
    return out


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
        'version': 'nfl_drive_v1', 'league': 'NFL', 'trainSeasons': train, 'nStates': n_states, 'nDrives': int(len(dr)),
        'outcomes': OUTCOMES,
        'buckets': {'tbEdges': TB_EDGES, 'lateEdges': LATE_EDGES, 'togoEdges': TOGO_EDGES, 'zoneEdges': ZONE_EDGES, 'yBinWidth': 10},
        'outcome': table.tolist(),                       # [tb][late][yb][down-1][tg][outcome]
        'duration': {**duration_table(s), **duration_table(s[s.groupby(['game_id', 'fixed_drive']).cumcount() == 0], 'S:')},
        'nextStart': next_start_table(s, dr),
        'kickoffStart': kickoff_table(dr_all),
        'tdPoints': td_points(dr),
        'ot': ot_table(d[d.season.isin(train)], games),
        'meanPointsPerGame': round(float(games[(games.game_type == 'REG') & games.season.isin(train)].total.mean()), 3),
    }
    os.makedirs(os.path.join(ROOT, 'data', 'models'), exist_ok=True)
    with open(os.path.join(ROOT, 'data', 'models', 'nfl_drive_v1.json'), 'w') as f: json.dump(model, f, separators=(',', ':'))
    n, ng = export_bt(d, dr_all, hold, os.path.join(CACHE, 'bt_states.csv.gz'))
    print(f'train {train[0]}-{train[-1]}: {len(dr)} drives, {n_states} states; holdout {hold}: {n} states / {ng} games')
    print('tdPoints', model['tdPoints'], 'ot', {k: model['ot'][k] for k in ('n', 'pTie')})
    print('kickoff start (most common bin by season)', {k: int(np.argmax(v)) for k, v in model['kickoffStart'].items()})


if __name__ == '__main__':
    main()
