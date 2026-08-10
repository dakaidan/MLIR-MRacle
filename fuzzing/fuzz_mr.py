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
    parser.add_argument(
        "--multi", metavar="FOLDER",
        help="folder of .mlir files; random one selected per run"
    )
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--print-mlir", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--binary", help="override path to mlir_mr_opt")
    parser.add_argument("--seed", type=int, help="fixed seed for all runs")
    parser.add_argument(
        "--transform", action="append", default=[],
        help="transformation(s) to apply; repeat the flag or comma-separate names"
    )
    parser.add_argument(
        "--apply", type=int, default=1,
        help="maximum number of transformations to apply per run"
    )
    parser.add_argument(
        "--log", action="store_true",
        help="log the transformed MLIR, LLVM IR and executable to the logs folder"
    )

    args = parser.parse_args()

    if args.file and args.multi:
        sys.exit("Cannot specify both FILE and --multi.")
    if not args.file and not args.multi:
        sys.exit("Must provide either FILE or --multi FOLDER.")
    if args.print_mlir and args.runs > 1:
        sys.exit("--print-mlir option only available for a singular run.")
    if args.apply < 1:
        sys.exit("--apply must be at least 1.")

    binary = Path(args.binary) if args.binary else find_binary()
    if not binary:
        print(f"{BINARY_NAME} not found; building...", file=sys.stderr)
        binary = build_binary()

    cmd = [str(binary)]
    if args.print_mlir:
        cmd.append("--print-mlir")
    if args.log:
        cmd.append("--log")
    if args.verbose:
        cmd.append("--verbose")
    if args.seed is not None:
        cmd.append(f"--seed={args.seed}")
    transforms = []
    for value in args.transform:
        transforms.extend(
            name.strip() for name in value.split(",") if name.strip()
        )
    if transforms:
        cmd.append(f"--transform={','.join(transforms)}")
    if args.apply:
        cmd.append(f"--apply={args.apply}")
    cmd.append(f"--runs={args.runs}")
    if args.multi:
        cmd.append(f"--multi={args.multi}")
    if args.file:
        cmd.append(args.file)

    result = subprocess.run(cmd, capture_output=True, text=True)

    runs = []
    raw = result.stdout
    start = raw.find("[")
    end = raw.rfind("]")

    # parse the JSON array of run info from stdout
    if start != -1 and end != -1 and end > start:
        try:
            runs = json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            sys.stderr.write(result.stderr)
            sys.stderr.write("failed to parse JSON from stdout\n")
            sys.stderr.write(raw)
            sys.exit(1)
    elif raw.strip():
        sys.stderr.write(result.stderr)
        sys.stderr.write("no JSON array found in stdout\n")
        sys.stderr.write(raw)
        sys.exit(1)

    # collect run lines; error and warn runs are deferred so they print
    # together at the end, regardless of verbosity
    summary = []
    errors = []
    warns = []
    for run in runs:
        is_error = run.get("error", "") != ""
        is_warn = run.get("warn", "") != ""
        if not (args.verbose or is_error or is_warn):
            continue
        if is_error:
            status = "[FAIL]"
        elif is_warn:
            status = "[WARN]"
        else:
            status = "[OK]"
        applied = run.get("applied_transforms", [])
        xform = ", ".join(
            f"{t['name']}@{t.get('target_function', '?')}"
            for t in applied
        ) if applied else "none"
        line = f"{status} run {run['run']}, seed {run['seed']}, " \
               f"file {run['file']}, transform(s): {xform}"
        if is_error:
            errors.append(f"{line}\n       error: {run['error']}")
        elif is_warn:
            # warn detail is raw C++ analysis text; only surface it in
            # verbose mode so the default view stays python-level only
            if args.verbose:
                warns.append(f"{line}\n{run['warn']}")
            else:
                warns.append(line)
        else:
            summary.append(line)

    for line in summary:
        print(line)

    # if singular run and --print-mlir, print the MLIR output to stdout
    if args.print_mlir and runs:
        mlir = runs[0].get("mlir_output", "")
        if mlir:
            sys.stdout.write(mlir)

    # collect raw binary stderr and per-run failures/warnings into one block
    # at the end; warns are shown as plain status lines, under all run results
    if errors or warns or result.stderr.strip():
        print("-" * 60)
        if result.stderr.strip():
            sys.stdout.write(result.stderr)
            if not result.stderr.endswith("\n"):
                sys.stdout.write("\n")
        for line in warns:
            print(line)
        for line in errors:
            print(line)


if __name__ == "__main__":
    sys.exit(main())
