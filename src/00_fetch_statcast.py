"""
Fetch one season of pitch-level Statcast data from Baseball Savant (via pybaseball).

One request per day, each in try/except: a single request for a whole season dies on
the first broken response, and that loses everything. Days that fail are listed at the
end and can be fetched again by rerunning with --start/--end.

If the season file already exists, the fetch resumes from the last date in it (that day
is fetched again, since it may have been partial) and the new rows are merged in on the
pitch key, so the same command both builds a season and brings it up to date.

The data belong to MLB Advanced Media. They are for your own non-commercial use and are
not part of this repository; see README.

Usage:  python src/00_fetch_statcast.py 2026
        python src/00_fetch_statcast.py 2026 --start 2026-09-02 --end 2026-09-28
"""
import sys, os, argparse, datetime as dt
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
from pybaseball import statcast, cache
from paths import statcast_path

COLS = ["game_date", "game_pk", "game_type", "home_team", "away_team", "inning", "inning_topbot",
        "pitcher", "batter", "player_name", "p_throws", "stand", "pitch_type", "pitch_name",
        "release_speed", "release_spin_rate", "release_extension", "release_pos_x", "release_pos_z",
        "spin_axis", "pfx_x", "pfx_z", "vx0", "vy0", "vz0", "ax", "ay", "az", "plate_x", "plate_z", "zone",
        "sz_top", "sz_bot", "balls", "strikes", "outs_when_up", "on_1b", "on_2b", "on_3b",
        "at_bat_number", "pitch_number", "description", "events", "type", "bb_type",
        "launch_speed", "launch_angle", "hit_distance_sc", "hc_x", "hc_y",
        "estimated_woba_using_speedangle", "estimated_ba_using_speedangle", "estimated_slg_using_speedangle",
        "woba_value", "woba_denom", "babip_value", "iso_value", "delta_run_exp"]
KEY = ["game_pk", "at_bat_number", "pitch_number", "pitcher", "batter"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("season", type=int)
    ap.add_argument("--start", help="first day (default: resume from the file, or March 1)")
    ap.add_argument("--end", help="last day (default: November 30, or yesterday if earlier)")
    a = ap.parse_args()
    path = statcast_path(a.season)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    old = pd.read_parquet(path) if os.path.exists(path) else None

    if a.start:
        start = dt.date.fromisoformat(a.start)
    elif old is not None:
        start = pd.Timestamp(old.game_date.max()).date()
    else:
        start = dt.date(a.season, 3, 1)
    end = dt.date.fromisoformat(a.end) if a.end else min(dt.date(a.season, 11, 30), dt.date.today() - dt.timedelta(days=1))
    print("%d: %s -> %s  (%s)" % (a.season, start, end, "update" if old is not None else "new file"), flush=True)

    cache.enable()
    parts, failed = [], []
    d = start
    while d <= end:
        s = d.isoformat()
        try:
            x = statcast(start_dt=s, end_dt=s, verbose=False)
            n = 0 if x is None else len(x)
            if n:
                parts.append(x[[c for c in COLS if c in x.columns]])
            print(s, n, flush=True)
        except Exception as e:
            failed.append(s)
            print("fail", s, repr(e)[:200], flush=True)
        d += dt.timedelta(days=1)

    if not parts:
        print("no new data"); print("failed days:", failed); return
    new = pd.concat(parts, ignore_index=True)
    both = new if old is None else pd.concat([old, new], ignore_index=True)
    both["game_date"] = pd.to_datetime(both.game_date)
    both = both.drop_duplicates(subset=KEY, keep="last")
    both.to_parquet(path, index=False)
    print("wrote %s: %d rows (%s -> %s)" % (path, len(both), both.game_date.min().date(), both.game_date.max().date()))
    print("game_type:", both.game_type.value_counts().to_dict())
    print("failed days:", failed)


if __name__ == "__main__":
    main()
