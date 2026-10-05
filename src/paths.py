"""
Where things live. Every script takes its Statcast files from here.

The season files are pitch-level Statcast data, one parquet per season, fetched with
00_fetch_statcast.py. They are not part of the repository (MLB Advanced Media's terms do
not allow redistributing them); by default they are expected in data/ next to src/, and
the environment variable PITCH_VECTORS_DATA can point somewhere else.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("PITCH_VECTORS_DATA", os.path.join(ROOT, "data"))
OUTPUT = os.path.join(ROOT, "output")
WEB = os.path.join(ROOT, "web")
SEASONS = (2024, 2025, 2026)


def statcast_path(season):
    return os.path.join(DATA, "statcast_%d.parquet" % season)


SNAP = {s: statcast_path(s) for s in SEASONS}
