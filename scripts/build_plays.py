#!/usr/bin/env python3
"""Build the end-of-half PLAY table for a league: every scrimmage play with <= 5:00 left in a half (training seasons), as a sample
the engine draws from when it simulates the end of a half play by play (src/model/drive.ts, PlayModel).

Why: drive-level tables keyed on the clock at drive start mistime end-of-half scoring. College stops the clock on first downs
inside 2:00, offences hurry and use timeouts, leaders kneel, and the result is that the drive model scored too early and too little
around halftime (BACKTEST.md "Open"). Sampling real plays carries the clock rules with them: each sample is what happened on one
snap: its event, the yards it moved the ball, the game-clock seconds until the next snap (incl. PAT and kickoff after a score), and
whether either team used a timeout before the next snap.

Each sample: [event, gain, clock, offTimeoutUsed, defTimeoutUsed, automaticFirstDown, downDelta, turnoverOnDowns]
  event  0 PLAY (incl. penalties, sacks, kneels, spikes), 1 TD (offence), 2 FG_GOOD, 3 FG_MISS (incl. blocked), 4 PUNT,
         5 TO (interception / lost fumble), 6 DEF_TD (defence or return touchdown), 7 SAFETY
  clock  game-clock seconds until the next snap (incl. PAT and kickoff after a score); -1 = logged at 0:00, the half ends after it
  gain   yards toward the goal: ball spot change when the same team snaps next, else the play's yards gained (PLAY only)
  firstDown / downDelta / turnoverOnDowns   what happened to the series (PLAY only), taken from the next snap rather than inferred
         from yards: penalties replay the down or award a first down, and inferring "gain >= to-go else down + 1" made simulated
         drives stall too early (v1 of this table)
Samples are grouped at several levels of detail (backoff); the engine uses the most detailed group with >= 25 samples.

Window: the last 10:00 of each half (START).
Outputs data/models/<league>_plays_v1.json.   python3 scripts/build_plays.py --league ncaaf [--train 2023-2024]
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_nfl as B  # noqa: E402

# Hand-off point: the last 10:00 of each half is simulated play by play. At 5:00 the drive-level hand-off placed the ball ~11 yards
# too far from goal (55 vs 44 real) and always at 1st and 10; starting at 10:00 lets the play dynamics produce realistic 5:00 and
# 2:00 situations.
START = 600
TB_EDGES = [10, 20, 30, 60, 120, 180, 300, 450]   # seconds left: fine at the very end (spikes, timeouts, buzzer field goals), then to 10:00
YB_EDGES = [5, 10, 15, 20, 30, 40, 60, 80]   # yards to goal: fine near the goal line (a 20-yard-wide bucket let plays from the 35 be applied at the 22)
TG_EDGES = [3, 7]                    # to go: 1-3 / 4-7 / 8+
DF_EDGES = [-9, -1, 0, 8]            # offence score diff: <=-9 / -8..-1 / 0 / 1..8 / >=9
TC_EDGES = [30, 120, 300]            # coarse clock: <=30 / <=120 / <=300 / <=600
YC_EDGES = [20, 40, 60]              # coarse yards to goal: <=20 / <=40 / <=60 / 61+
# Backoff order: the down is never dropped (a 4th down drawn from all-downs groups gets ordinary snaps instead of kicks and
# punts; an earlier version did that and under-produced end-of-half field goals). Clock and field position get coarser first.
LEVELS = [['h', 'tb', 'yb', 'dn', 'tg', 'df', 'to'], ['h', 'tb', 'yb', 'dn', 'tg', 'df'], ['h', 'tb', 'yb', 'dn', 'tg'],
          ['h', 'tb', 'yb', 'dn'], ['h', 'tc', 'yb', 'dn', 'tg'], ['h', 'tc', 'yb', 'dn'], ['h', 'tc', 'yc', 'dn'], ['h', 'dn']]
EVENTS = ['PLAY', 'TD', 'FG_GOOD', 'FG_MISS', 'PUNT', 'TO', 'DEF_TD', 'SAFETY']

NCAAF_EVENT = {
    'Rushing Touchdown': 1, 'Passing Touchdown': 1, 'Penalty Touchdown': 1,
    'Field Goal Good': 2, 'Field Goal Missed': 3, 'Blocked Field Goal': 3, 'Missed Field Goal Return': 3,
    'Punt': 4, 'Blocked Punt': 4, 'Punt Team Fumble Recovery': 4,
    'Interception Return': 5, 'Fumble Recovery (Opponent)': 5,
    'Interception Return Touchdown': 6, 'Fumble Return Touchdown': 6, 'Fumble Recovery (Opponent) Touchdown': 6,
    'Blocked Punt Touchdown': 6, 'Blocked Field Goal Touchdown': 6, 'Punt Return Touchdown': 6,
    'Safety': 7, 'Blocked Punt (Safety)': 7, 'Punt (Safety)': 7,
}


def events_nfl(s):
    td_off = (s.touchdown == 1) & (s.td_team == s.posteam)
    td_def = (s.touchdown == 1) & (s.td_team != s.posteam)
    fg = s.field_goal_result.fillna('')
    ev = np.select([td_off, td_def, s.safety == 1, fg == 'made', fg.isin(['missed', 'blocked']), s.punt_attempt == 1,
                    (s.interception == 1) | (s.fumble_lost == 1)], [1, 6, 7, 2, 3, 4, 5], 0)
    return ev


def load(league, seasons):
    if league == 'nfl':
        d = B.load(seasons)
    else:
        import build_ncaaf as N
        d = N.load(seasons)
    d = d[(d.season_type == 'REG') & d.game_half.isin(['Half1', 'Half2'])]
    s = d[d.down.notna() & d.yardline_100.notna() & d.posteam.notna() & ~d.play_type.isin(['kickoff', 'extra_point'])].copy()
    s['ev'] = events_nfl(s) if league == 'nfl' else s.play_type_raw.map(NCAAF_EVENT).fillna(0).astype(int).values
    return s


def samples(s):
    s = s.copy()
    s['h'] = np.where(s.game_half == 'Half2', 2, 1)
    s['y'] = s.yardline_100.astype(int)
    g = s.groupby(['game_id', 'h'])
    for c in ['half_seconds_remaining', 'posteam', 'y', 'down', 'posteam_timeouts_remaining', 'defteam_timeouts_remaining']:
        s['n_' + c] = g[c].shift(-1)
    last = s.n_half_seconds_remaining.isna()
    s['clock'] = np.where(last, s.half_seconds_remaining, s.half_seconds_remaining - s.n_half_seconds_remaining).clip(0, 300)
    same = s.n_posteam == s.posteam
    s['gain'] = np.where((s.ev == 0) & same, s.y - s.n_y, np.where(s.ev == 0, s.yards_gained.fillna(0), 0))
    play = (s.ev == 0)
    # Automatic first down: the next snap is 1st down although the play did not gain the distance (defensive penalties). Normal
    # first downs are re-derived in the engine from gain >= to-go, so a sample from 4th-and-4 cannot convert 4th-and-7.
    s['fd'] = (play & same & (s.n_down == 1) & (s.gain < s.ydstogo)).astype(int)
    # Down change when the series continues: +1 normally, 0 when the down is replayed (offsetting / pre-snap penalties).
    s['dd'] = np.where(play & same & (s.n_down != 1), (s.n_down - s.down).fillna(1), 1).clip(-3, 3).astype(int)
    s['tod'] = (play & ~same & s.n_posteam.notna() & (s.down == 4)).astype(int)
    # Timeouts used before the next snap, per team (the next row may have the other team on offence).
    off_next = np.where(same, s.n_posteam_timeouts_remaining, s.n_defteam_timeouts_remaining)
    def_next = np.where(same, s.n_defteam_timeouts_remaining, s.n_posteam_timeouts_remaining)
    s['off_used'] = np.clip(np.nan_to_num(s.posteam_timeouts_remaining - off_next, nan=0), 0, 1).astype(int)
    s['def_used'] = np.clip(np.nan_to_num(s.defteam_timeouts_remaining - def_next, nan=0), 0, 1).astype(int)
    # Snaps logged at 0:00 are real: the source records many buzzer plays (38% of college end-of-first-half field goals) with the
    # clock already at zero. Keep them, with clock -1 = "the half ends after this play" (dropping them lost a third of those kicks).
    # Only the half's real last snap ends it; an earlier 0:00 snap (an untimed down after a defensive penalty, a scoring play
    # with the PAT/kickoff still to come) takes 0 seconds and play goes on.
    s.loc[(s.half_seconds_remaining == 0) & last, 'clock'] = -1
    s.loc[(s.half_seconds_remaining == 0) & ~last, 'clock'] = 0
    s = s[(s.half_seconds_remaining <= START) & (s.half_seconds_remaining >= 0)]
    s['tb'] = np.searchsorted(TB_EDGES, s.half_seconds_remaining, side='left')
    s['tc'] = np.searchsorted(TC_EDGES, s.half_seconds_remaining, side='left')
    s['yc'] = np.searchsorted(YC_EDGES, s.y, side='left')
    s['yb'] = np.searchsorted(YB_EDGES, s.y, side='left')
    s['dn'] = s.down.astype(int)
    s['tg'] = np.searchsorted(TG_EDGES, s.ydstogo.clip(lower=1), side='left')
    s['df'] = np.searchsorted(DF_EDGES, s.posteam_score - s.defteam_score, side='left')
    s['to'] = s.posteam_timeouts_remaining.fillna(3).clip(0, 2).astype(int)
    s = s[s.gain.between(-40, 99)].reset_index(drop=True)
    return s


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--league', default='ncaaf'); ap.add_argument('--train', default=None)
    a = ap.parse_args()
    train = a.train or ('2023-2024' if a.league == 'ncaaf' else '2016-2023')
    seasons = list(range(int(train.split('-')[0]), int(train.split('-')[-1]) + 1))
    s = samples(load(a.league, seasons))
    flat = np.column_stack([s.ev, s.gain.round().astype(int), s.clock.round().astype(int), s.off_used, s.def_used, s.fd, s.dd, s.tod]).ravel().tolist()
    levels = []
    for fields in LEVELS:
        groups = {}
        for key, idx in s.groupby(fields).groups.items():
            key = key if isinstance(key, tuple) else (key,)
            groups['|'.join(str(int(k)) for k in key)] = [int(i) for i in idx]
        levels.append({'fields': fields, 'groups': groups})
    out = {'version': f'{a.league}_plays_v1', 'fields': 8, 'league': a.league.upper(), 'trainSeasons': seasons, 'startSeconds': START,
           'buckets': {'tbEdges': TB_EDGES, 'ybEdges': YB_EDGES, 'tgEdges': TG_EDGES, 'dfEdges': DF_EDGES, 'tcEdges': TC_EDGES, 'ycEdges': YC_EDGES},
           'events': EVENTS, 'nSamples': int(len(s)), 'samples': flat, 'levels': levels}
    path = os.path.join(B.ROOT, 'data', 'models', f'{a.league}_plays_v1.json')
    with open(path, 'w') as f: json.dump(out, f, separators=(',', ':'))
    ev = s.ev.value_counts(normalize=True).sort_index()
    print(f'{a.league} plays (<= {START // 60}:00 left, {seasons[0]}-{seasons[-1]}): {len(s)} samples, {os.path.getsize(path) / 1e6:.1f} MB;',
          'events', {EVENTS[i]: round(float(v), 3) for i, v in ev.items()}, '| clock/play mean', round(float(s.clock.mean()), 1))


if __name__ == '__main__':
    main()
