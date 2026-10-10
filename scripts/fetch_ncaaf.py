#!/usr/bin/env python3
"""Download cfbfastR (CFBD-based) college play-by-play parquet files into data/cache/ncaaf/ (about 110-130 MB per season).
Public GitHub release assets (sportsdataverse/sportsdataverse-data, tag cfbfastR_cfb_pbp); no credentials. Reading them needs pyarrow.
  python3 scripts/fetch_ncaaf.py [--from 2023] [--to 2026]"""
import argparse, os, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'data', 'cache', 'ncaaf')
URL = 'https://github.com/sportsdataverse/sportsdataverse-data/releases/download/cfbfastR_cfb_pbp/play_by_play_{}.parquet'

ap = argparse.ArgumentParser(); ap.add_argument('--from', dest='a', type=int, default=2023); ap.add_argument('--to', dest='b', type=int, default=2026)
args = ap.parse_args()
os.makedirs(OUT, exist_ok=True)
for y in range(args.a, args.b + 1):
    urllib.request.urlretrieve(URL.format(y), os.path.join(OUT, f'cfbd_{y}.parquet')); print('pbp', y)
