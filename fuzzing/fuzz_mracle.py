#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BINARY_NAME = "mlir_mracle_opt"
CACHE_DIR = PROJECT_ROOT / "cache"
# stderr markers that indicate a TSan report on the tool or the JIT'd code
SANITIZER_PATTERNS = [
    r"WARNING: ThreadSanitizer",
    r"SUMMARY: ThreadSanitizer",
]

# returns the root of the mlir_wheel installation, or exits if not found
def mlir_wheel_root():
    try:
        return subprocess.check_output(
            [sys.executable, "-m", "mlir_wheel", "--root-dir"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        sys.exit("mlir_wheel not installed; create the uv venv and run "
                 "`uv pip install -r requirements.txt` first")

# returns the build directory, or exits if not found
def build_dir_for():
    env = os.environ.get("MLIR_MRACLE_BUILD_DIR")
    if env:
        return Path(env)
    return PROJECT_ROOT / "build"


# canonical location of the built tool; the runner builds it on first use
def binary_path():
    return build_dir_for() / "mlir-mracle" / "src" / "app" / BINARY_NAME


def find_binary():
    path = binary_path()
    if path.is_file() and os.access(path, os.X_OK):
        return path
    return None


def build_binary():
    cmake = shutil.which("cmake")
    if not cmake:
        sys.exit("cmake not found in PATH")
    build_dir = build_dir_for()
    config = [cmake, "-S", str(PROJECT_ROOT), "-B", str(build_dir),
              f"-DCMAKE_PREFIX_PATH={mlir_wheel_root()}"]
    if not (build_dir / "CMakeCache.txt").exists():
        config[1:1] = ["-G", "Ninja"]
    subprocess.run(config, check=True)
    subprocess.run(
        [cmake, "--build", str(build_dir), "--target", BINARY_NAME, "-j"],
        check=True,
    )
    binary = find_binary()
    if not binary:
        sys.exit(f"build succeeded but {BINARY_NAME} not found")
    return binary


# returns the stderr lines that look like sanitizer reports
def scan_sanitizer_output(stderr_text):
    return [
        line for line in stderr_text.splitlines()
        if any(re.search(p, line) for p in SANITIZER_PATTERNS)
    ]


def save_sanitizer_log(campaign_dir, stderr_text):
    if not campaign_dir:
        return
    hits = scan_sanitizer_output(stderr_text)
    if not hits:
        return
    log = Path(campaign_dir) / "sanitizer.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("\n".join(hits) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run the mlir_mracle-opt metamorphic pass pipeline."
    )
    parser.add_argument(
        "file", nargs="?", help="path to a single .mlir file"
    )
    parser.add_argument("--seed", type=int, help="fixed seed for all runs")
    parser.add_argument(
        "--mode", default="multi",
        help="pipeline mode(s) to run, comma-separated: emit, execution, "
             "legacy, multi (default: multi)"
    )
    parser.add_argument(
        "--reps", type=int, default=5000,
        help="with --mode=emit: number of times transformations are picked "
             "and applied; otherwise: executions per program per thread "
             "count (default 5000)"
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
        "--model", default="",
        help="memory model gating the applicable transforms: empty = generic "
             "transforms only; 'armv8' adds the ARMv8-specific ones "
             "(default: generic only)"
    )
    parser.add_argument(
        "--campaign-dir", metavar="PATH",
        help="resume/continue a campaign, or set the output directory for --mode=emit"
    )
    parser.add_argument(
        "--threshold", type=int, default=5,
        help="fail threshold: an outcome rare in the source (below this %% of "
             "its runs) appearing at/above this %% of transformed runs FAILs; "
             "other rate deviations beyond the Poisson bound WARN (default 5)"
    )
    parser.add_argument(
        "--reruns", type=int, default=5000,
        help="extra source runs when the transformed side finds a new outcome "
             "(default 5000)"
    )
    parser.add_argument(
        "--max-runs", type=int, default=100000,
        help="hard cap for source runs per baseline (default 100000)"
    )
    args = parser.parse_args()

    if args.file and args.multi:
        sys.exit("Cannot specify both FILE and --multi.")
    if not args.file and not args.multi:
        sys.exit("Must provide either FILE or --multi FOLDER.")
    if args.apply < 0:
        sys.exit("--apply must be >= 0.")
    if args.reps <= 0:
        sys.exit("--reps must be > 0.")

    modes = []
    for name in args.mode.split(","):
        name = name.strip()
        if not name:
            continue
        if name not in ("emit", "execution", "legacy", "multi"):
            sys.exit(f"unknown mode '{name}' (expected emit, execution, "
                     "legacy, or multi)")
        if name not in modes:
            modes.append(name)
    if not modes:
        modes.append("multi")
    if "emit" in modes and not args.transform:
        sys.exit("--mode=emit requires --transform.")

    binary = find_binary()
    if not binary:
        print(f"{BINARY_NAME} not found; building...", file=sys.stderr)
        binary = build_binary()

    cmd = [str(binary)]
    if args.seed is not None:
        cmd.append(f"--seed={args.seed}")
    cmd.append(f"--mode={','.join(modes)}")
    cmd.append(f"--reps={args.reps}")
    cmd.append(f"--iter={args.iter}")
    cmd.append(f"--apply={args.apply}")
    cmd.append(f"--model={args.model}")
    cmd.append(f"--threshold={args.threshold}")
    cmd.append(f"--reruns={args.reruns}")
    cmd.append(f"--max-runs={args.max_runs}")
    transforms = []
    for value in args.transform:
        transforms.extend(
            name.strip() for name in value.split(",") if name.strip()
        )
    if transforms:
        cmd.append(f"--transform={','.join(transforms)}")
    if args.multi:
        cmd.append(f"--multi={args.multi}")
    if args.campaign_dir:
        cmd.append(f"--campaign-dir={args.campaign_dir}")
    if args.file:
        cmd.append(args.file)

    # pin the cache root so every invocation shares the project baseline
    # cache regardless of the working directory
    env = dict(os.environ)
    env["MLIR_MRACLE_CACHE_DIR"] = str(CACHE_DIR / "v2")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    sanitizer_hits = scan_sanitizer_output(result.stderr)

    # a crashing tool may not print the campaign dir; best-effort log under
    # the user-provided campaign dir when there is one
    if result.returncode != 0:
        if sanitizer_hits and args.campaign_dir:
            save_sanitizer_log(args.campaign_dir, result.stderr)
        sys.stderr.write(result.stderr)
        sys.exit(result.returncode)

    campaign_dirs = [ln for ln in result.stdout.splitlines() if ln.strip()]
    if len(campaign_dirs) != len(modes):
        sys.stderr.write(result.stderr)
        sys.exit(f"expected {len(modes)} campaign dir(s) on stdout, "
                 f"got {len(campaign_dirs)}")

    for campaign_dir in campaign_dirs:
        save_sanitizer_log(campaign_dir, result.stderr)

    ok = warn = fail = 0
    emitted = emit_errors = 0
    for mode, campaign_dir in zip(modes, campaign_dirs):
        if mode == "emit":
            out_dir = Path(campaign_dir)
            emitted += len(list(out_dir.glob("run*_seed*.mlir")))
            emit_errors += len(list(out_dir.glob("run*_seed*.error.txt")))
            continue
        result_path = Path(campaign_dir) / "result.json"
        try:
            runs = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            sys.stderr.write(result.stderr)
            sys.exit(f"failed to read {result_path}: {exc}")
        for run in runs:
            if mode == "execution":
                if run.get("error"):
                    fail += 1
                else:
                    ok += 1
            else:
                status = run.get("status", "OK")
                if status == "ERROR":
                    fail += 1
                elif status == "WARN":
                    warn += 1
                else:
                    ok += 1

    print("=== Campaign completed ===")
    if emitted or emit_errors:
        print(f"Emitted MLIR: {emitted}  Errors: {emit_errors}")
    print(f"OK: {ok}  WARN: {warn}  FAIL: {fail}")
    if sanitizer_hits:
        for line in sanitizer_hits[:20]:
            print(line, file=sys.stderr)
        print(f"Sanitizer warnings detected; campaign classified as ERROR "
              f"({len(sanitizer_hits)} hit lines, see sanitizer.log)")
    print(f"Campaign directory: {', '.join(campaign_dirs)}")
    return 1 if sanitizer_hits or emit_errors else 0


if __name__ == "__main__":
    sys.exit(main())
