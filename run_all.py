"""
Run every analysis script in dependency order and log what each prints.

The numbered scripts mostly run in number order, with these exceptions:
  08 (the Pitch Plane) runs first: 04, 09, 10, 13, 16 and later read output/plane_2026.npz.
  15, 17, 19, 27 are run with the arguments the published figures used.
00 (fetching data) is not run here; see README.

Stops at the first script that fails. Output goes to output/run_all.log.

Usage:  python run_all.py            # everything
        python run_all.py 21 22 26   # only these, in the order given
"""
import os, sys, subprocess, time

ROOT = os.path.dirname(os.path.abspath(__file__))
STEPS = [
    ["08_the_plane.py"],
    ["01_vector_space.py"], ["02_origin_and_metric.py"], ["03_sphere.py"], ["04_viz_data.py"],
    ["05_plane.py"], ["06_flat_arsenal.py"], ["07_what_the_plane_is.py"],
    ["09_what_savant_loses.py"], ["10_pitcher_cards_data.py"], ["11_vs_savant.py"],
    ["12_spin_rate_route.py"], ["13_pp_vs_outcomes.py"], ["14_resolution.py"],
    ["15_spectra_figure.py", "Glasnow, Tyler", "Sale, Chris", "Misiorowski, Jacob"],
    ["16_mcr.py"], ["17_mcr_components.py", "abs", "8"], ["17_mcr_components.py", "ref", "10"],
    ["18_peak_table.py"], ["19_peak_alignment.py"], ["19_peak_alignment.py", "0.9"],
    ["20_glove_fan.py"], ["21_ff_reference.py"], ["22_two_fans.py"], ["23_four_numbers.py"],
    ["24_four_map_data.py"], ["25_projection.py"], ["26_three_axes_data.py"],
    ["27_separation_check.py"], ["28_logratio.py"], ["29_map_vs_stuff.py"], ["30_threats.py"],
    ["31_projection_vs_raw.py"], ["32_note_figures.py"], ["33_offplane_check.py"],
]


def main():
    only = sys.argv[1:]
    steps = [s for s in STEPS if not only or s[0][:2] in only]
    os.makedirs(os.path.join(ROOT, "output"), exist_ok=True)
    log = open(os.path.join(ROOT, "output", "run_all.log"), "w", encoding="utf-8")
    env = dict(os.environ, PYTHONIOENCODING="utf-8", LOKY_MAX_CPU_COUNT=os.environ.get("LOKY_MAX_CPU_COUNT", "4"))
    t_all = time.time()
    for step in steps:
        t = time.time()
        head = "===== %s" % " ".join(step)
        print(head, flush=True); log.write(head + "\n"); log.flush()
        r = subprocess.run([sys.executable, os.path.join("src", step[0])] + step[1:], cwd=ROOT, env=env,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        log.write(r.stdout)
        if r.returncode:
            log.write(r.stderr)
        msg = "      exit %d  %.0f s" % (r.returncode, time.time() - t)
        print(msg, flush=True); log.write(msg + "\n"); log.flush()
        if r.returncode:
            print(r.stderr[-2000:]); sys.exit(r.returncode)
    print("all done in %.0f s" % (time.time() - t_all))


if __name__ == "__main__":
    main()
