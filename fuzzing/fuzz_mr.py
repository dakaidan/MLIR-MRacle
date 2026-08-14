#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = Path(os.environ.get("MLIR_MR_BUILD_DIR", PROJECT_ROOT / "build"))
BINARY_NAME = "mlir_mr_opt"

CANDIDATE_PATHS = [
    BUILD_DIR / "mlir-mr" / "src" / "app" / BINARY_NAME,
    BUILD_DIR / "bin" / BINARY_NAME,
    BUILD_DIR / BINARY_NAME,
    BUILD_DIR / "mlir-mr" / BINARY_NAME,
]


def find_binary():
    override = os.environ.get("MLIR_MR_OPT")
    if override:
        return Path(override)
    for path in CANDIDATE_PATHS:
        if path.is_file() and os.access(path, os.X_OK):
            return path
    return None


def build_binary():
    cmake = shutil.which("cmake")
    if not cmake:
        sys.exit("cmake not found in PATH")
    if not (BUILD_DIR / "CMakeCache.txt").exists():
        subprocess.run(
            [cmake, "-G", "Ninja", "-S", str(PROJECT_ROOT), "-B", str(BUILD_DIR)],
            check=True,
        )
    subprocess.run(
        [cmake, "--build", str(BUILD_DIR), "--target", BINARY_NAME, "-j"],
        check=True,
    )
    binary = find_binary()
    if not binary:
        sys.exit(f"build succeeded but {BINARY_NAME} not found")
    return binary


def main():
    parser = argparse.ArgumentParser(
        description="Run the mlir-mr-opt metamorphic pass pipeline."
    )
    parser.add_argument(
        "file", nargs="?", help="path to a single .mlir file"
    )
    parser.add_argument("--seed", type=int, help="fixed seed for all runs")
    parser.add_argument(
        "--run", action="store_true",
        help="execution mode: run each file as-is, record joint outcome sets"
    )
    parser.add_argument(
        "--reps", type=int, default=10000,
        help="executions per program per thread count (default 10000)"
    )
    parser.add_argument(
        "--iter", type=int, default=1,
        help="number of pipeline repetitions (default 1)"
    )
    parser.add_argument(
        "--transform", action="append", default=[],
        help="transformation(s) to apply; repeat the flag or comma-separate names"
    )
    parser.add_argument(
        "--multi", metavar="FOLDER",
        help="folder of .mlir files; random one selected per run"
    )
    parser.add_argument(
        "--apply", type=int, default=1,
        help="maximum number of transformations to apply per run"
    )
    parser.add_argument(
        "--tsan", type=int, default=100,
        help="percentage of compilations instrumented with TSan (0-100, default 100)"
    )
    parser.add_argument(
        "--campaign-dir", metavar="PATH",
        help="resume/continue a campaign in an existing log folder"
    )
    parser.add_argument(
        "--threshold", type=int, default=5,
        help="transformed-only outcomes below this percent of transformed runs "
             "are WARN, above are FAIL (default 5)"
    )
    parser.add_argument(
        "--reruns", type=int, default=5000,
        help="extra source runs when the transformed side finds a new outcome "
             "(default 5000)"
    )
    parser.add_argument(
        "--max-runs", type=int, default=25000,
        help="hard cap for source runs per baseline (default 25000)"
    )
    parser.add_argument("--binary", metavar="PATH", help="override path to mlir_mr_opt")
    args = parser.parse_args()

    if args.file and args.multi:
        sys.exit("Cannot specify both FILE and --multi.")
    if not args.file and not args.multi:
        sys.exit("Must provide either FILE or --multi FOLDER.")
    if args.apply < 0:
        sys.exit("--apply must be >= 0.")
    if args.reps <= 0:
        sys.exit("--reps must be > 0.")

    binary = Path(args.binary) if args.binary else find_binary()
    if not binary:
        print(f"{BINARY_NAME} not found; building...", file=sys.stderr)
        binary = build_binary()

    cmd = [str(binary)]
    if args.seed is not None:
        cmd.append(f"--seed={args.seed}")
    if args.run:
        cmd.append("--run")
    cmd.append(f"--reps={args.reps}")
    cmd.append(f"--iter={args.iter}")
    transforms = []
    for value in args.transform:
        transforms.extend(
            name.strip() for name in value.split(",") if name.strip()
        )
    if transforms:
        cmd.append(f"--transform={','.join(transforms)}")
    if args.multi:
        cmd.append(f"--multi={args.multi}")
    cmd.append(f"--apply={args.apply}")
    cmd.append(f"--tsan={args.tsan}")
    cmd.append(f"--threshold={args.threshold}")
    cmd.append(f"--reruns={args.reruns}")
    cmd.append(f"--max-runs={args.max_runs}")
    if args.campaign_dir:
        cmd.append(f"--campaign-dir={args.campaign_dir}")
    if args.file:
        cmd.append(args.file)

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        sys.exit(result.returncode)

    campaign_dir = result.stdout.strip()
    if not campaign_dir:
        sys.stderr.write(result.stderr)
        sys.exit("no campaign dir on stdout")

    result_path = Path(campaign_dir) / "result.json"
    try:
        runs = json.loads(result_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        sys.stderr.write(result.stderr)
        sys.exit(f"failed to read {result_path}: {exc}")

    if args.run:
        ok = fail = 0
        for run in runs:
            if run.get("error"):
                fail += 1
            else:
                ok += 1
        print("=== Campaign completed ===")
        print(f"OK: {ok}  WARN: 0  FAIL: {fail}")
        print(f"Campaign directory: {campaign_dir}")
        return 0

    ok = warn = fail = 0
    for run in runs:
        status = run.get("status", "OK")
        if status == "ERROR":
            fail += 1
        elif status == "WARN":
            warn += 1
        else:
            ok += 1
    print("=== Campaign completed ===")
    print(f"OK: {ok}  WARN: {warn}  FAIL: {fail}")
    print(f"Campaign directory: {campaign_dir}")


if __name__ == "__main__":
    sys.exit(main())
