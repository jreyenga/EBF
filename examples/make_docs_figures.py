# -*- coding: utf-8 -*-
"""
Regenerate every figure used by the documentation.

The docs embed PNGs that are produced by the example scripts rather than
checked in by hand.  Whenever a figure-producing example changes — a new
colormap, a retuned fit, a different basis function in the registry —
the committed PNG goes stale until someone reruns the script.  This runs
all of them in one command:

    python examples/make_docs_figures.py

Each script is launched in its own process with a non-interactive
matplotlib backend and ``--save-only``, so nothing opens a window.  The
fits are real, so this is slow (several minutes) — ``loss_comparison``
alone trains six models.

Options
-------
--list          show the scripts and their output figures, then exit
--only NAME     run just one script (repeatable), e.g. --only RBF_vs_EBF
"""
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
ASSETS = HERE.parent / "docs" / "assets"

# script stem -> figure it writes into docs/assets/
FIGURES = {
    "RBF_vs_EBF":            "rbf_vs_ebf.png",
    "node_ellipsoids":       "node_ellipsoids.png",
    "comp_map_ebf":          "compressor_map_summary.png",
    "loss_comparison":       "loss_comparison.png",
    "loss_function_gallery": "loss_functions.png",
    "basis_function_gallery": "basis_functions.png",
}


def run_one(stem):
    """Run one example with --save-only.  Returns True on success."""
    script = HERE / f"{stem}.py"
    target = ASSETS / FIGURES[stem]
    before = target.stat().st_mtime if target.exists() else 0.0

    env = dict(os.environ, MPLBACKEND="Agg")
    print(f"  running {stem}.py ...", end="", flush=True)
    start = time.time()
    proc = subprocess.run([sys.executable, str(script), "--save-only"],
                          cwd=str(HERE), env=env,
                          capture_output=True, text=True)
    elapsed = time.time() - start

    if proc.returncode != 0:
        print(f" FAILED ({elapsed:.0f}s)")
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-12:]
        for line in tail:
            print(f"      | {line}")
        return False

    # A zero exit code is not proof the figure was written — verify.
    if not target.exists() or target.stat().st_mtime <= before:
        print(f" ran but {FIGURES[stem]} was not updated ({elapsed:.0f}s)")
        return False

    size_kb = target.stat().st_size / 1024
    print(f" ok ({elapsed:.0f}s) -> {FIGURES[stem]} [{size_kb:.0f} KB]")
    return True


def main(argv):
    if "--list" in argv:
        print("Documentation figures:\n")
        for stem, fig in FIGURES.items():
            mark = "ok     " if (ASSETS / fig).exists() else "MISSING"
            print(f"  [{mark}] examples/{stem}.py  ->  docs/assets/{fig}")
        return 0

    only = [argv[i + 1] for i, a in enumerate(argv) if a == "--only"
            and i + 1 < len(argv)]
    unknown = [name for name in only if name not in FIGURES]
    if unknown:
        print(f"Unknown script(s): {', '.join(unknown)}")
        print(f"Choose from: {', '.join(FIGURES)}")
        return 2

    targets = only or list(FIGURES)
    print(f"Regenerating {len(targets)} documentation figure(s) into "
          f"{ASSETS}\nThis retrains every model and takes a few minutes.\n")

    start = time.time()
    failed = [stem for stem in targets if not run_one(stem)]

    print(f"\n{len(targets) - len(failed)}/{len(targets)} succeeded "
          f"in {time.time() - start:.0f}s")
    if failed:
        print(f"Failed: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
