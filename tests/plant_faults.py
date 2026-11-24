"""Can the feature tests fail? Break features.py four ways, one at a time, and run the feature tests on each broken copy.

    python tests/plant_faults.py

A test that still passes on a broken copy is not testing that thing. Nothing in the project folder is changed:
each broken copy lives in a temporary folder.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAULTS = {
    "the 3-day window peeks one day ahead":
        ("        win = order[start[max(0, d - WINDOW + 1)]:start[d + 1]]         # the window, today included",
         "        win = order[start[max(0, d - WINDOW + 1)]:start[min(d + 2, n_days)]]"),
    "the 7-day history includes today":
        ("        week = order[start[max(0, d - WEEK)]:start[d]]                  # the 7 days before today",
         "        week = order[start[max(0, d - WEEK)]:start[d + 1]]"),
    "transfers to self are counted as activity":
        ("    real = s != r                                                       # transfers between two different accounts",
         "    real = np.ones(n, dtype=bool)"),
    "a loop needs only a one-way connection":
        ('    _, label = connected_components(graph, directed=True, connection="strong")',
         '    _, label = connected_components(graph, directed=True, connection="weak")'),
}


def main():
    source = (ROOT / "features.py").read_text(encoding="utf-8")
    missed = 0
    for name, (old, new) in FAULTS.items():
        if source.count(old) != 1:
            sys.exit(f"features.py no longer holds the line this fault replaces: {name}")
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            shutil.copy(ROOT / "common.py", folder / "common.py")
            (folder / "features.py").write_text(source.replace(old, new), encoding="utf-8")
            (folder / "tests").mkdir()
            shutil.copy(ROOT / "tests" / "test_features.py", folder / "tests" / "test_features.py")
            run = subprocess.run([sys.executable, "tests/test_features.py"], cwd=folder, capture_output=True, text=True)
        failing = [line for line in run.stdout.splitlines() if line.startswith("FAIL")]
        caught = run.returncode != 0
        missed += not caught
        print(f"{'CAUGHT' if caught else 'MISSED'}  {name}: {len(failing)} checks failed" + (f", first: {failing[0][6:90]}" if failing else ""))
    print("\nEvery planted fault was caught." if not missed else f"\n{missed} planted fault(s) went through.")
    sys.exit(1 if missed else 0)


if __name__ == "__main__":
    main()
