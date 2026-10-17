#!/usr/bin/env python3
"""
sandbox.py - Verify a generated reproduction test + patch against a real checkout.

A patch is only "verified" when, on a throwaway copy of the repository:
  1. the reproduction test FAILS on the unpatched code (the bug is real),
  2. the diff applies cleanly,
  3. the same test PASSES afterwards, and
  4. the repo's existing test suite did not regress (when it was green before).

Isolation here is a temp-dir copy, a wall-clock timeout and a scrubbed
environment (no tokens/API keys). It is NOT a security boundary: only point
it at repositories you trust. A Docker backend (no network, CPU/memory caps)
is the planned replacement before running untrusted fork code.

Usage:
    python scripts/sandbox.py <repo_dir> <test_file.py> <fix.patch>
    python scripts/sandbox.py --clone https://github.com/o/r.git <test_file.py> <fix.patch>
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

TEST_TIMEOUT = 120
CLONE_TIMEOUT = 180
REPRO_TEST_NAME = "test_patchpilot_repro.py"
COPY_IGNORE = shutil.ignore_patterns(".git", "__pycache__", "*.pyc", ".pytest_cache", ".venv", "node_modules")

# Verdicts
VERIFIED = "verified"
NOT_REPRODUCED = "not_reproduced"        # test passed on the unpatched code
INVALID_TEST = "invalid_test"            # test errored/timed out instead of failing
EMPTY_PATCH = "empty_patch"
PATCH_DID_NOT_APPLY = "patch_did_not_apply"
STILL_FAILING = "still_failing"          # patch applied but the test still fails
REGRESSION = "regression"                # test passes but the existing suite broke


@dataclass
class VerifyResult:
    status: str
    detail: str = ""
    log: str = ""
    steps: dict = field(default_factory=dict)

    @property
    def ok(self):
        return self.status == VERIFIED


def _clean_env():
    """Environment for child processes: never leak credentials into tested code."""
    secret_suffixes = ("_TOKEN", "_API_KEY", "_SECRET", "_PASSWORD")
    env = {k: v for k, v in os.environ.items() if not k.upper().endswith(secret_suffixes)}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _run(cmd, cwd, timeout):
    """Run a command; returns (returncode, combined output). Timeout -> (124, ...)."""
    try:
        proc = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout, env=_clean_env(),
        )
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, f"Timed out after {timeout}s"


def _pytest(cwd, *args, timeout=TEST_TIMEOUT):
    return _run([sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider", *args],
                cwd, timeout)


def clone_repo(clone_url, dest, branch=None):
    """Shallow-clone a public repo into dest."""
    cmd = ["git", "clone", "--depth", "1", "--quiet"]
    if branch:
        cmd += ["--branch", branch]
    code, out = _run(cmd + [clone_url, dest], cwd=None, timeout=CLONE_TIMEOUT)
    if code != 0:
        raise RuntimeError(f"git clone failed: {out.strip()}")


def _tail(text, lines=40):
    return "\n".join(text.strip().splitlines()[-lines:])


def verify_patch(source_dir, test_code, diff, run_suite=True):
    """Check test_code + diff against a throwaway copy of source_dir."""
    steps = {}
    if not (diff or "").strip() or diff.lstrip().startswith("# Unified diff placeholder"):
        return VerifyResult(EMPTY_PATCH, "The model produced no usable diff.")
    if not (test_code or "").strip():
        return VerifyResult(INVALID_TEST, "The model produced no reproduction test.")

    with tempfile.TemporaryDirectory(prefix="patchpilot_", ignore_cleanup_errors=True) as tmp:
        ws = os.path.join(tmp, "ws")
        shutil.copytree(source_dir, ws, ignore=COPY_IGNORE)

        # Baseline: is the existing suite green before we touch anything?
        baseline_green = False
        if run_suite:
            code, _ = _pytest(ws)
            baseline_green = code == 0
            steps["baseline_suite"] = "green" if baseline_green else f"not green (exit {code}), regression gate skipped"

        test_path = os.path.join(ws, REPRO_TEST_NAME)
        with open(test_path, "w", encoding="utf-8") as f:
            f.write(test_code.rstrip("\n") + "\n")

        # 1. The test must fail on unpatched code.
        code, out = _pytest(ws, REPRO_TEST_NAME)
        if code == 0:
            steps["test_before_patch"] = "passed"
            return VerifyResult(NOT_REPRODUCED,
                                "The reproduction test passes without the patch, so it does not demonstrate the bug.",
                                _tail(out), steps)
        if code != 1:
            steps["test_before_patch"] = f"error (exit {code})"
            return VerifyResult(INVALID_TEST,
                                "The reproduction test errored (syntax/import/timeout) instead of failing on an assertion.",
                                _tail(out), steps)
        steps["test_before_patch"] = "failed (as required)"

        # 2. Apply the patch. --recount tolerates the wrong hunk line counts LLMs often emit.
        patch_path = os.path.join(tmp, "fix.patch")
        with open(patch_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(diff.strip("\n") + "\n")
        code, out = _run(["git", "apply", "--recount", "--whitespace=nowarn", patch_path], ws, 30)
        if code != 0:
            steps["apply_patch"] = "failed"
            return VerifyResult(PATCH_DID_NOT_APPLY, "git apply rejected the diff.", _tail(out), steps)
        steps["apply_patch"] = "ok"

        # 3. The same test must now pass.
        code, out = _pytest(ws, REPRO_TEST_NAME)
        if code != 0:
            steps["test_after_patch"] = f"failed (exit {code})"
            return VerifyResult(STILL_FAILING, "The patch applied but the reproduction test still fails.",
                                _tail(out), steps)
        steps["test_after_patch"] = "passed"

        # 4. No regressions (only meaningful if the suite was green to begin with).
        if run_suite and baseline_green:
            code, out = _pytest(ws)
            if code != 0:
                steps["suite_after_patch"] = f"failed (exit {code})"
                return VerifyResult(REGRESSION, "The reproduction test passes but the existing suite now fails.",
                                    _tail(out), steps)
            steps["suite_after_patch"] = "green"

    return VerifyResult(VERIFIED, "Test fails before the patch, passes after, no regressions.", steps=steps)


def main(argv=None):
    p = argparse.ArgumentParser(description="Verify a reproduction test + patch against a repo.")
    p.add_argument("repo", help="local repo directory (or a git URL with --clone)")
    p.add_argument("test_file")
    p.add_argument("patch_file")
    p.add_argument("--clone", action="store_true", help="treat <repo> as a git URL and shallow-clone it")
    p.add_argument("--branch")
    p.add_argument("--no-suite", action="store_true", help="skip the full-suite regression check")
    args = p.parse_args(argv)

    with open(args.test_file, encoding="utf-8") as f:
        test_code = f.read()
    with open(args.patch_file, encoding="utf-8") as f:
        diff = f.read()

    with tempfile.TemporaryDirectory(prefix="patchpilot_src_", ignore_cleanup_errors=True) as tmp:
        source = args.repo
        if args.clone:
            source = os.path.join(tmp, "src")
            clone_repo(args.repo, source, args.branch)
        result = verify_patch(source, test_code, diff, run_suite=not args.no_suite)

    print(f"{result.status.upper()}: {result.detail}")
    for step, outcome in result.steps.items():
        print(f"  - {step}: {outcome}")
    if result.log:
        print("\n" + result.log)
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
