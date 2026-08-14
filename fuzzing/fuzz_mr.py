#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BINARY_NAME = "mlir_mr_opt"
CACHE_DIR = PROJECT_ROOT / "cache"
EXECUTABLE_CACHE_DIR = CACHE_DIR / "executables"

# sanitizer stderr markers that indicate a bug in the tool or the JIT'd code
SANITIZER_PATTERNS = [
    r"WARNING: ThreadSanitizer",
    r"WARNING: AddressSanitizer",
    r"ERROR: AddressSanitizer",
    r"ERROR: LeakSanitizer",
    r"UndefinedBehaviorSanitizer",
    r"runtime error:",
    r"SUMMARY: (Thread|Address|UndefinedBehavior|Leak)Sanitizer",
]


def build_dir_for(sanitizers):
    env = os.environ.get("MLIR_MR_BUILD_DIR")
    if env:
        return Path(env)
    if sanitizers == "thread":
        return PROJECT_ROOT / "build"
    return PROJECT_ROOT / f"build-{sanitizer_build_suffix(sanitizers)}"


def sanitizer_build_suffix(sanitizers):
    if sanitizers == "thread":
        return "tsan"
    if "address" in sanitizers or "undefined" in sanitizers:
        return "asan-ubsan"
    return "none"


def candidate_paths_for(sanitizers):
    build_dir = build_dir_for(sanitizers)
    return [
        build_dir / "mlir-mr" / "src" / "app" / BINARY_NAME,
        build_dir / "bin" / BINARY_NAME,
        build_dir / BINARY_NAME,
        build_dir / "mlir-mr" / BINARY_NAME,
    ]


def find_binary(sanitizers="thread"):
    override = os.environ.get("MLIR_MR_OPT")
    if override:
        return Path(override)
    for path in candidate_paths_for(sanitizers):
        if path.is_file() and os.access(path, os.X_OK):
            return path
    return None


# fingerprints the mlir-mr source tree so cached executables are invalidated
# when the tool itself changes
def source_fingerprint():
    hasher = hashlib.sha256()
    for path in sorted((PROJECT_ROOT / "mlir-mr").rglob("*")):
        if not path.is_file():
            continue
        if "build" in path.parts or path.suffix in (".o", ".a", ".d"):
            continue
        hasher.update(path.relative_to(PROJECT_ROOT).as_posix().encode())
        try:
            hasher.update(path.read_bytes())
        except OSError:
            pass
    return hasher.hexdigest()[:16]


def cached_binary_path(opt_level, sanitizers):
    h = hashlib.sha256()
    h.update(str(opt_level).encode())
    h.update(sanitizers.encode())
    h.update(source_fingerprint().encode())
    digest = h.hexdigest()[:16]
    label = sanitizers.replace(",", "-") if sanitizers else "none"
    return EXECUTABLE_CACHE_DIR / f"opt{opt_level}_{label}_{digest}" / BINARY_NAME


def build_binary(opt_level=1, sanitizers="thread"):
    cmake = shutil.which("cmake")
    if not cmake:
        sys.exit("cmake not found in PATH")
    cached = cached_binary_path(opt_level, sanitizers)
    if cached.is_file() and os.access(cached, os.X_OK):
        print(f"using cached {BINARY_NAME} (opt=-O{opt_level}, "
              f"sanitizers={sanitizers or 'none'})", file=sys.stderr)
        return cached
    build_dir = build_dir_for(sanitizers)
    config = [cmake, "-S", str(PROJECT_ROOT), "-B", str(build_dir),
              f"-DMLIR_MR_OPT_LEVEL={opt_level}",
              f"-DMLIR_MR_SANITIZERS={sanitizers}"]
    if not (build_dir / "CMakeCache.txt").exists():
        config[1:1] = ["-G", "Ninja"]
    subprocess.run(config, check=True)
    subprocess.run(
        [cmake, "--build", str(build_dir), "--target", BINARY_NAME, "-j"],
        check=True,
    )
    binary = find_binary(sanitizers)
    if not binary:
        sys.exit(f"build succeeded but {BINARY_NAME} not found")
    cached.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(binary, cached)
    return cached


# returns the stderr lines that look like sanitizer reports
def scan_sanitizer_output(stderr_text):
    return [
        line for line in stderr_text.splitlines()
        if any(re.search(p, line) for p in SANITIZER_PATTERNS)
    ]


def save_sanitizer_log(campaign_dir, stderr_text):
    if not campaign_dir:
        return
    log = Path(campaign_dir) / "sanitizer.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(stderr_text)


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
        "--emit-mlir", action="store_true",
        help="generator mode: emit transformed MLIR files without executing"
    )
    parser.add_argument(
        "--reps", type=int, default=5000,
        help="with --emit-mlir: number of times transformations are picked "
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
        "--tsan", type=int, default=None,
        help="percentage of compilations instrumented with TSan (0-100, "
             "default 100; forced to 0 for non-TSan builds)"
    )
    parser.add_argument(
        "--sanitizers", default="thread",
        help="sanitizer build to use: 'thread' (default), 'address,undefined', "
             "or 'none' (separate build dir per config; ASan and TSan are "
             "never combined)"
    )
    parser.add_argument(
        "--campaign-dir", metavar="PATH",
        help="resume/continue a campaign, or output directory with --emit-mlir"
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
        "--max-runs", type=int, default=100000,
        help="hard cap for source runs per baseline (default 100000)"
    )
    parser.add_argument("--binary", metavar="PATH", help="override path to mlir_mr_opt")
    parser.add_argument(
        "--opt-level", type=int, default=1,
        help="optimisation level for mlir-mr targets (default 1)"
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
    if args.run and args.emit_mlir:
        sys.exit("Cannot combine --run and --emit-mlir.")
    if args.emit_mlir and not args.transform:
        sys.exit("--emit-mlir requires --transform.")

    sanitizers = ",".join(sorted(
        s.strip() for s in args.sanitizers.lower().split(",") if s.strip()
    ))
    if sanitizers not in ("thread", "address,undefined", "none"):
        sys.exit("--sanitizers must be one of: thread, address,undefined, none")
    if "address" in sanitizers and "thread" in sanitizers:
        sys.exit("ASan and TSan must not be combined in one build; "
                 "run separate campaigns per sanitizer")
    if args.tsan is None:
        args.tsan = 100 if sanitizers == "thread" else 0

    binary = Path(args.binary) if args.binary else find_binary(sanitizers)
    if not binary:
        print(f"{BINARY_NAME} not found; building...", file=sys.stderr)
        binary = build_binary(args.opt_level, sanitizers)

    cmd = [str(binary)]
    if args.seed is not None:
        cmd.append(f"--seed={args.seed}")
    if args.emit_mlir:
        cmd.append("--emit-mlir")
    elif args.run:
        cmd.append("--run")
    if args.emit_mlir:
        cmd.append(f"--reps={args.reps}")
        cmd.append(f"--apply={args.apply}")
    else:
        cmd.append(f"--reps={args.reps}")
        cmd.append(f"--iter={args.iter}")
        cmd.append(f"--apply={args.apply}")
        cmd.append(f"--tsan={args.tsan}")
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

    result = subprocess.run(cmd, capture_output=True, text=True)
    sanitizer_hits = scan_sanitizer_output(result.stderr)

    # a crashing tool may not print the campaign dir; best-effort log under
    # the user-provided campaign dir when there is one
    if result.returncode != 0:
        if sanitizer_hits and args.campaign_dir:
            save_sanitizer_log(args.campaign_dir, result.stderr)
        sys.stderr.write(result.stderr)
        sys.exit(result.returncode)

    campaign_dir = result.stdout.strip()
    if not campaign_dir:
        sys.stderr.write(result.stderr)
        sys.exit("no campaign dir on stdout")

    if sanitizer_hits:
        save_sanitizer_log(campaign_dir, result.stderr)

    if args.emit_mlir:
        out_dir = Path(campaign_dir)
        generated = sorted(out_dir.glob("run*_seed*.mlir"))
        errors = sorted(out_dir.glob("run*_seed*.error.txt"))
        print("=== Emit completed ===")
        print(f"Generated: {len(generated)}  Errors: {len(errors)}")
        if sanitizer_hits:
            for line in sanitizer_hits[:20]:
                print(line, file=sys.stderr)
            print(f"Sanitizer warnings detected; campaign classified as ERROR "
                  f"({len(sanitizer_hits)} hit lines, see sanitizer.log)")
        print(f"Output directory: {campaign_dir}")
        return 1 if sanitizer_hits or errors else 0

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
        if sanitizer_hits:
            for line in sanitizer_hits[:20]:
                print(line, file=sys.stderr)
            print(f"Sanitizer warnings detected; campaign classified as ERROR "
                  f"({len(sanitizer_hits)} hit lines, see sanitizer.log)")
        print(f"Campaign directory: {campaign_dir}")
        return 1 if sanitizer_hits else 0

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
    if sanitizer_hits:
        for line in sanitizer_hits[:20]:
            print(line, file=sys.stderr)
        print(f"Sanitizer warnings detected; campaign classified as ERROR "
              f"({len(sanitizer_hits)} hit lines, see sanitizer.log)")
    print(f"Campaign directory: {campaign_dir}")
    return 1 if sanitizer_hits else 0


if __name__ == "__main__":
    sys.exit(main())
