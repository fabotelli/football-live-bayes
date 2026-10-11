#!/usr/bin/env python3
"""Build the NCAAF drive-model tables from cfbfastR (CFBD-based) play-by-play: data/cache/ncaaf/cfbd_<season>.parquet.

Same model pieces and JSON shape as scripts/build_nfl.py (whose table functions are reused): this script only normalises the
college data into nflverse column names. FBS-vs-FBS games only. Default train 2023-2024 (the 2023 clock rule, clock keeps running
after first downs, changed drive lengths, so older seasons would teach the wrong pace); holdout 2025-2026.

Outputs data/models/ncaaf_drive_v2.json and data/cache/ncaaf/bt_states.csv.gz.
Source: github.com/sportsdataverse/sportsdataverse-data releases, tag cfbfastR_cfb_pbp (fetch with scripts/fetch_ncaaf.py).
  python3 scripts/build_ncaaf.py [--train 2023-2024] [--holdout 2025-2026]
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_nfl as B  # noqa: E402  (table functions; NFL-only parts are not called)

ROOT = B.ROOT
CACHE = os.path.join(ROOT, 'data', 'cache', 'ncaaf')

# CFBD drive_result_detailed -> nflverse fixed_drive_result strings (B.RESULT_MAP then maps those to model outcomes).
# Return touchdowns by the defence are "Opp touchdown"; a kickoff-return TD belongs to the receiving team's drive -> "Touchdown".
RESULT = {
    'Rushing Touchdown': 'Touchdown', 'Passing Touchdown': 'Touchdown', 'Kickoff Return Touchdown': 'Touchdown', 'Penalty Touchdown': 'Touchdown',
    'Field Goal Good': 'Field goal',
    'Punt': 'Punt', 'Blocked Punt': 'Punt', 'Punt Team Fumble Recovery': 'Punt',
    'Interception Return': 'Turnover', 'Fumble': 'Turnover', 'Fumble Recovery (Opponent)': 'Turnover', 'Kickoff Team Fumble Recovery': 'Turnover',
    'Downs Turnover': 'Turnover on downs',
    'Field Goal Missed': 'Missed field goal', 'Blocked Field Goal': 'Missed field goal', 'Missed Field Goal Return': 'Missed field goal',
    'End Half': 'End of half', 'End Game': 'End of half', 'End of Half': 'End of half', 'End of Game': 'End of half', 'End of Regulation': 'End of half',
    'Interception Return Touchdown': 'Opp touchdown', 'Fumble Return Touchdown': 'Opp touchdown', 'Blocked Punt Touchdown': 'Opp touchdown',
    'Fumble Recovery (Opponent) Touchdown': 'Opp touchdown', 'Blocked Field Goal Touchdown': 'Opp touchdown', 'Punt Return Touchdown': 'Opp touchdown',
    'Safety': 'Safety', 'Blocked Punt (Safety)': 'Safety', 'Punt (Safety)': 'Safety',
}
COLS = ['game_id', 'id_play', 'game_play_number', 'season', 'week', 'season_type', 'home', 'away', 'home_team_division', 'away_team_division', 'pos_team', 'def_pos_team',
        'period', 'clock_minutes', 'clock_seconds', 'down', 'distance', 'yards_to_goal', 'play_type', 'drive_number', 'drive_result_detailed',
        'pos_team_score', 'def_pos_team_score', 'pos_team_timeouts_rem_before', 'def_pos_team_timeouts_rem_before', 'spread', 'over_under', 'home_wp_before',
        'yards_gained']


def load(seasons):
    out = []
    for s in seasons:
        p = os.path.join(CACHE, f'cfbd_{s}.parquet')
        if not os.path.exists(p): print('missing', p, file=sys.stderr); continue
        x = pd.read_parquet(p, columns=COLS)
        x = x[(x.home_team_division == 'fbs') & (x.away_team_division == 'fbs')]
        out.append(x)
    x = pd.concat(out, ignore_index=True)
    # Play order: period, then game clock (descending), then the source's play number for plays in the same second.
    # NOT id_play: it is an 18-digit ESPN id stored as float64 in the parquet, so it has lost precision and sorting by it
    # scrambles plays within a game (drive model v1 did that, which corrupted final scores and drive end times).
    x['clock_left'] = x.clock_minutes.astype(float) * 60 + x.clock_seconds.astype(float)
    x['gpn'] = x.game_play_number.fillna(x.groupby('game_id').cumcount())
    x = x.sort_values(['game_id', 'period', 'clock_left', 'gpn'], ascending=[True, True, False, True], kind='stable').reset_index(drop=True)
    home = x.pos_team == x.home
    # Scores in this source are post-play. A few rows carry a stale score (timeouts, penalties logged in the same second as a
    # touchdown), so take the running maximum: a team's score never goes down. Pre-play = previous row's post-play score.
    x['h_raw'] = np.where(home, x.pos_team_score, x.def_pos_team_score).astype(float)
    x['a_raw'] = np.where(home, x.def_pos_team_score, x.pos_team_score).astype(float)
    post_h = x.groupby('game_id').h_raw.cummax().values
    post_a = x.groupby('game_id').a_raw.cummax().values
    x['total_home_score'], x['total_away_score'] = post_h, post_a
    pre_h = x.groupby('game_id').total_home_score.shift(1).fillna(0).values
    pre_a = x.groupby('game_id').total_away_score.shift(1).fillna(0).values
    per = x.period.astype(int)
    clock = x.clock_minutes.astype(float) * 60 + x.clock_seconds.astype(float)
    pt = x.play_type.fillna('')
    d = pd.DataFrame({
        'game_id': x.game_id.astype(str), 'season': x.season.astype(int), 'week': x.week,
        'season_type': np.where(x.season_type == 'regular', 'REG', 'POST'),
        'home_team': x.home, 'away_team': x.away, 'posteam': x.pos_team, 'defteam': x.def_pos_team,
        'play_type': np.where(pt.str.contains('Kickoff') & ~pt.str.contains('Touchdown'), 'kickoff',
                     np.where(pt.isin(['Timeout', 'End Period', 'End of Half', 'End of Game', 'End of Regulation', 'placeholder', 'Uncategorized'])
                              | pt.str.contains('Two Point|2pt'), 'extra_point', 'scrimmage')),
        'down': x.down.where(x.down.between(1, 4)), 'ydstogo': x.distance, 'yardline_100': x.yards_to_goal.where(x.yards_to_goal.between(1, 99)),
        'half_seconds_remaining': clock + np.where(per % 2 == 1, 900, 0), 'qtr': per,
        'game_half': np.where(per <= 2, 'Half1', np.where(per <= 4, 'Half2', 'Overtime')),
        'fixed_drive': x.drive_number, 'fixed_drive_result': x.drive_result_detailed.map(RESULT),
        'total_home_score': post_h, 'total_away_score': post_a,
        'posteam_score': np.where(home, pre_h, pre_a), 'defteam_score': np.where(home, pre_a, pre_h),
        'posteam_timeouts_remaining': x.pos_team_timeouts_rem_before, 'defteam_timeouts_remaining': x.def_pos_team_timeouts_rem_before,
        'spread_line': -x.spread,          # CFBD: negative = home favoured; model/nflverse: positive = home favoured
        'total_line': x.over_under, 'vegas_home_wp': x.home_wp_before, 'play_id': np.arange(len(x)),
        'yards_gained': x.yards_gained, 'play_type_raw': pt,
    })
    fin = d.groupby('game_id')[['total_home_score', 'total_away_score']].transform('last')
    d['home_score'], d['away_score'] = fin.total_home_score, fin.total_away_score
    d['game_seconds_remaining'] = np.nan
    return d


def ot_table(d):
    """College OT: no ties. Points added by the OT winner / loser (alternating possessions from the 25)."""
    reg = d[d.game_half != 'Overtime'].groupby('game_id')[['total_home_score', 'total_away_score']].last()
    fin = d.groupby('game_id')[['home_score', 'away_score']].first()
    x = reg.join(fin)
    x = x[(x.total_home_score == x.total_away_score) & (x.home_score != x.away_score)]
    ah, aa = x.home_score - x.total_home_score, x.away_score - x.total_away_score
    win, lose = np.maximum(ah, aa).astype(int), np.minimum(ah, aa).astype(int)
    return {'n': int(len(x)), 'pTie': 0.0,
            'winnerAdd': {str(k): round(float(v), 4) for k, v in win.value_counts(normalize=True).sort_index().items()},
            'loserAdd': {str(k): round(float(v), 4) for k, v in lose.value_counts(normalize=True).sort_index().items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default='2023-2024'); ap.add_argument('--holdout', default='2025-2026')
    a = ap.parse_args()
    rng = lambda r: list(range(int(r.split('-')[0]), int(r.split('-')[-1]) + 1))
    train, hold = rng(a.train), rng(a.holdout)
    d = load(sorted(set(train + hold)))
    dr_all = B.drive_frame(d)
    dtr = d[d.season.isin(train)]
    dr = dr_all[dr_all.game_id.isin(set(dtr.game_id))]
    s = B.states(dtr, dr)
    table, n_states = B.smoothed_table(s)
    seasons = d.groupby('game_id').season.first()
    ko = {}
    x = dr_all.assign(season=dr_all.game_id.map(seasons))
    x = x[x.result.isin(['TD', 'FG']) & x.next_start_y.notna() & (x.next_posteam != x.posteam)]
    for season, g in x.groupby('season'):
        ko[str(season)] = [round(v, 5) for v in (np.bincount(B.ybin_of(g.next_start_y.values), minlength=10) / len(g)).tolist()]
    g = d[d.season.isin(train) & (d.season_type == 'REG')].groupby('game_id').first()
    model = {
        'version': 'ncaaf_drive_v2', 'league': 'NCAAF', 'trainSeasons': train, 'nStates': n_states, 'nDrives': int(len(dr)),
        'outcomes': B.OUTCOMES,
        'buckets': {'tbEdges': B.TB_EDGES, 'gapEdges': B.GAP_EDGES, 'togoEdges': B.TOGO_EDGES, 'zoneEdges': B.ZONE_EDGES, 'yBinWidth': 10},
        'outcome': table.tolist(),
        'strength': B.smoothed_strength(s).tolist(),
        'duration': {**B.duration_table(s), **B.duration_table(s[s.groupby(['game_id', 'fixed_drive']).cumcount() == 0], 'S:')},
        'nextStart': B.next_start_table(s, dr),
        'kickoffStart': ko,
        'endY': B.end_y_table(s),
        'tdPoints': B.td_points(dr),
        'ot': ot_table(d[d.season.isin(train)]),
        'meanPointsPerGame': round(float((g.home_score + g.away_score).mean()), 3),
        'targets': B.dispersion_targets(dtr, dr),
        'durationShrink': 1.0,   # set by scripts/fit_dispersion.ts
        'gameEnv': {'alpha': 0},  # set by scripts/fit_dispersion.ts (0 = no game-environment effect)
    }
    with open(os.path.join(ROOT, 'data', 'models', 'ncaaf_drive_v2.json'), 'w') as f: json.dump(model, f, separators=(',', ':'))
    n, ng = B.export_bt(d, dr_all, hold, os.path.join(CACHE, 'bt_states.csv.gz'))
    print(f'train {train}: {len(dr)} drives, {n_states} states, {g.shape[0]} games, mean total {model["meanPointsPerGame"]}; holdout {hold}: {n} states / {ng} games')
    print('tdPoints', model['tdPoints'], 'ot', {k: model['ot'][k] for k in ('n', 'pTie')}, 'kickoff bins', {k: int(np.argmax(v)) for k, v in ko.items()})


if __name__ == '__main__':
    main()
