#!/usr/bin/env python3
"""Download nflverse play-by-play (per season) and the games file (closing spread/total, finals) into data/cache/nfl/.
Public GitHub release assets; no credentials.   python3 scripts/fetch_nfl.py [--from 2016] [--to 2026]"""
import argparse, os, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'data', 'cache', 'nfl')
PBP = 'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{}.csv.gz'
GAMES = 'https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv'

ap = argparse.ArgumentParser(); ap.add_argument('--from', dest='a', type=int, default=2016); ap.add_argument('--to', dest='b', type=int, default=2026)
args = ap.parse_args()
os.makedirs(OUT, exist_ok=True)
for y in range(args.a, args.b + 1):
    urllib.request.urlretrieve(PBP.format(y), os.path.join(OUT, f'pbp_{y}.csv.gz')); print('pbp', y)
urllib.request.urlretrieve(GAMES, os.path.join(OUT, 'games.csv')); print('games.csv')
