#!/usr/bin/env python3
"""
patch_pilot.py - End-to-end CLI orchestrator for Patch-Pilot.

Executes the autonomous, strictly-scoped open-source maintainer pipeline:
1. Triage: Ingests GitHub issue + repo tree.
2. Target: LLM selects the 1-3 culprit files based on SKILL.md.
3. Map: Progressively discloses architecture via AST mapping (saving ~80%+ tokens).
4. Patch: Generates failing reproduction test (TDD) + surgical unified git diff.
5. Verify: Validates syntax and outputs maintainer-ready artifacts.

Usage:
    python scripts/patch_pilot.py https://github.com/owner/repo/issues/123
    python scripts/patch_pilot.py https://github.com/owner/repo/issues/123 --provider ollama --model qwen2.5-coder:7b
    python scripts/patch_pilot.py --mock  # Run offline demo
"""

import argparse
import ast
import json
import os
import re
import sys
import tempfile
import time

# Auto-load environment variables from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Support importing local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_issue import parse_issue_url, GitHubClient, fetch_issue, fetch_file_paths, fetch_file_content
from analyze_ast import analyze_code
from llm_client import LLMClient
from sandbox import clone_repo, verify_patch

# Visual analysis modules (Gemma 4 Vision upgrade)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from ai.gemma_vision import analyze_screenshot, VisionAnalysisResult
    from utils.image_annotator import draw_bounding_box, create_comparison
    from fetch_issue import extract_image_urls, download_image
    from github.pr_builder import VisualPRContext, build_pr_body
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False

try:
    from runner.playwright_capture import capture_screenshot, capture_after_patch
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

DEMO_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples", "demo_repo")
MAX_ATTEMPTS = 3


# Windows consoles default to cp1252; force UTF-8 support
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

BANNER = r"""
======================================================================
  ____       _       _          ____  _ _       _   
 |  _ \ __ _| |_ ___| |__      |  _ \(_) | ___ | |_ 
 | |_) / _` | __/ __| '_ \ ____| |_) | | |/ _ \| __|
 |  __/ (_| | || (__| | | |____|  __/| | | (_) | |_ 
 |_|   \__,_|\__\___|_| |_|    |_|   |_|_|\___/ \__|
                                                    
  Open-Source Maintainer Agent & TDD Model Harness
  Adheres to the Agent Skill Open Standard
======================================================================
"""



def load_skill_directive():
    """Load system instructions from SKILL.md."""
    skill_path = os.path.join(os.path.dirname(__file__), "..", "SKILL.md")
    if os.path.exists(skill_path):
        with open(skill_path, "r", encoding="utf-8") as f:
            return f.read()
    return "You are Patch-Pilot, an open-source maintainer agent strictly scoped to fixing bugs via TDD."


def extract_json_array(text):
    """Extract a raw JSON list from model response."""
    text = text.strip()
    # Try finding markdown code block
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    else:
        # Match outermost square brackets
        m = re.search(r"(\[.*?\])", text, re.DOTALL)
        if m:
            text = m.group(1).strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def extract_markdown_block(text, block_name):
    """Extract a named section or code block from model response."""
    pattern = rf"(?i)###?\s*{block_name}.*?\n```(?:\w+)?\n(.*?)```"
    m = re.search(pattern, text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


def validate_python_syntax(code_str):
    """Check if Python code string compiles without syntax error."""
    try:
        ast.parse(code_str)
        return True, "Valid syntax"
    except SyntaxError as e:
        return False, str(e)


def list_repo_files(root):
    """Flat, sorted list of file paths under root (forward slashes), skipping VCS/cache dirs."""
    skip = {".git", "__pycache__", ".pytest_cache"}
    paths = []
    for base, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d not in skip]
        for name in names:
            paths.append(os.path.relpath(os.path.join(base, name), root).replace(os.sep, "/"))
    return sorted(paths)


def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def retry_feedback(result):
    """Prompt suffix telling the model exactly why its previous attempt was rejected."""
    return (
        f"\n\nYour previous attempt was REJECTED by automated verification: {result.status} - {result.detail}\n"
        f"Verifier output:\n{result.log}\n"
        "Produce a corrected Reproduction Test and Surgical Patch in the same format."
    )


def run_visual_analysis(image_urls: list, gh_token: str | None, code_context: str) -> dict | None:
    """Download and analyze the first screenshot from a GitHub issue using Gemma 4 Vision.

    Returns a dict with 'result', 'annotated_path', and 'original_path' keys, or None.
    """
    if not VISION_AVAILABLE or not image_urls:
        return None

    print("\n[*] 👁️ Visual Bug Analysis: Gemma 4 Vision detected screenshot attachments.")
    # Download or read the first image
    img_data = download_image(image_urls[0], token=gh_token)
    if not img_data:
        print("    [!] Failed to download screenshot, falling back to text-only mode.")
        return None

    if isinstance(img_data, bytes):
        if os.path.exists(image_urls[0]):
            local_path = os.path.abspath(image_urls[0])
        else:
            temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png", prefix="patchpilot_img_")
            temp_img.write(img_data)
            temp_img.close()
            local_path = temp_img.name
    else:
        local_path = img_data

    print(f"    -> Downloaded/loaded screenshot: {local_path}")

    try:
        vision_result = analyze_screenshot(local_path, code_context)
    except Exception as e:
        print(f"    [!] Gemma 4 Vision analysis failed: {e}")
        return None

    if not vision_result.has_visual_bug:
        print("    -> No visual bug detected by Gemma 4, proceeding with text-only analysis.")
        return None

    print(f"    -> 🎯 Visual defect detected: {vision_result.label}")
    print(f"    -> Bounding box (normalized): {vision_result.box_2d}")
    print(f"    -> Explanation: {vision_result.explanation}")

    # Annotate the screenshot with the bounding box
    annotated_path = draw_bounding_box(
        local_path, vision_result.box_2d, vision_result.label
    )
    print(f"    -> Annotated screenshot saved: {annotated_path}")

    return {
        "result": vision_result,
        "annotated_path": annotated_path,
        "original_path": local_path,
    }


def generate_verified_patch(llm, system_prompt, patch_prompt, source_dir, max_attempts):
    """Ask the model for a test + diff; if source_dir is given, verify and retry on failure.

    Returns (response, test_code, diff, result) where result is None when verification is off.
    """
    feedback = ""
    for attempt in range(1, max_attempts + 1):
        response = llm.complete(patch_prompt + feedback, system_prompt=system_prompt)
        test_code = extract_markdown_block(response, "Reproduction Test")
        diff = extract_markdown_block(response, "Surgical Patch")
        if not source_dir:
            return response, test_code, diff, None

        print(f"    -> Attempt {attempt}/{max_attempts}: verifying in sandbox...")
        result = verify_patch(source_dir, test_code, diff)
        print(f"       {result.status.upper()}: {result.detail}")
        if result.ok:
            return response, test_code, diff, result
        feedback = retry_feedback(result)
    return response, test_code, diff, result


def run_pipeline(issue_url=None, provider=None, model=None, branch=None, output_dir="output", is_mock=False,
                 verify=False, repo_dir=None, max_attempts=MAX_ATTEMPTS, image_path=None):
    print(BANNER)
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()

    # 1. Initialize LLM Client
    llm = LLMClient(
        provider="mock" if is_mock else provider,
        model=model
    )
    print(f"[*] Model Provider: [{llm.provider.upper()}]  Model: [{llm.model}]")

    # 2. Ingest Issue & File Tree
    if is_mock or not issue_url:
        print("[*] Running in Mock/Demo Mode with bundled reference issue...")
        issue_data = {
            "title": "Fix JSONDecodeError on empty string input",
            "body": "When parse_input('') is called with an empty string, json.loads raises JSONDecodeError instead of returning an empty dict.",
            "number": 42,
            "url": "https://github.com/example/demo/issues/42",
            "image_urls": []
        }
        repo_info = {"owner": "example", "name": "demo", "branch": "main"}
        files = list_repo_files(DEMO_REPO)
        sample_code = {f: read_text(os.path.join(DEMO_REPO, f)) for f in files if f.endswith(".py")}
    else:
        print(f"[*] Phase 1: Fetching issue and repository file tree from GitHub...")
        owner, repo, number = parse_issue_url(issue_url)
        gh = GitHubClient(os.environ.get("GITHUB_TOKEN"))
        repo_info = gh.get(f"/repos/{owner}/{repo}")
        target_branch = branch or repo_info.get("default_branch") or "main"
        issue_data = fetch_issue(gh, owner, repo, number)
        files, truncated = fetch_file_paths(gh, owner, repo, target_branch)
        print(f"    - Target: {owner}/{repo} (Branch: {target_branch})")
        print(f"    - Issue #{issue_data['number']}: {issue_data['title']}")
        print(f"    - Total Files in Repo: {len(files)}")

    # Check for screenshots / images
    image_urls = list(issue_data.get("image_urls", []))
    if image_path:
        image_urls.insert(0, image_path)
    if image_urls and VISION_AVAILABLE:
        print(f"    - 📸 Found {len(image_urls)} screenshot attachment(s). Prioritizing Gemma 4 Vision.")

    # Source tree for verification: bundled demo, a local checkout, or a shallow clone (opt-in).
    source_dir = None
    tmp_clone = None
    if is_mock or not issue_url:
        source_dir = DEMO_REPO
    elif repo_dir:
        source_dir = repo_dir
    elif verify:
        print("[!] --verify runs the repository's own tests on this machine. Only use it on code you trust.")
        tmp_clone = tempfile.TemporaryDirectory(prefix="patchpilot_clone_", ignore_cleanup_errors=True)
        source_dir = os.path.join(tmp_clone.name, "repo")
        print("[*] Cloning repository for sandbox verification...")
        clone_repo(repo_info.get("clone_url") or f"https://github.com/{owner}/{repo}.git", source_dir, target_branch)

    # 3. Target / Triage (Agent Skill Step 1)
    print("\n[*] Phase 2: Agent Skill Triage (Pinpointing 1-3 culprit files)...")
    triage_payload = {
        "issue_title": issue_data["title"],
        "issue_body": issue_data["body"],
        "files": files[:800]  # Scoped window of paths
    }
    system_prompt = load_skill_directive()
    triage_prompt = (
        "Identify the 1 to 3 files most likely to contain the code that needs to be fixed.\n"
        "Input Data:\n" + json.dumps(triage_payload, indent=2) + "\n\n"
        "Output ONLY a raw, valid JSON array containing the file paths. No markdown, no commentary."
    )

    triage_response = llm.complete(triage_prompt, system_prompt=system_prompt)
    target_files = extract_json_array(triage_response)
    if not target_files:
        # Fallback to smart heuristic if response wasn't clean JSON
        target_files = [f for f in files if "parser" in f or "test" in f][:2] or files[:2]

    print(f"    -> Scoped Target Files: {target_files}")

    # 4. Map Architecture via AST (Progressive Disclosure)
    print("\n[*] Phase 3: Progressive Disclosure & AST Mapping...")
    ast_maps = []
    total_raw_bytes = 0
    total_ast_bytes = 0
    file_contents = {}

    for path in target_files:
        if is_mock or not issue_url:
            code = sample_code.get(path, "# File placeholder\n")
        else:
            code = fetch_file_content(gh, owner, repo, path, target_branch)
            if code is None:
                print(f"    [!] Could not fetch {path}, skipping...")
                continue
        file_contents[path] = code
        total_raw_bytes += len(code.encode("utf-8"))

        ast_summary = analyze_code(code, path)
        ast_maps.append(ast_summary)
        total_ast_bytes += len(ast_summary.encode("utf-8"))
        print(f"    -> Mapped AST for `{path}`")

    # Compute Token Reduction
    token_savings = 0.0
    if total_raw_bytes > 0:
        token_savings = max(0.0, ((total_raw_bytes - total_ast_bytes) / total_raw_bytes) * 100)
    print(f"    -> Raw Source Size: {total_raw_bytes} bytes (~{total_raw_bytes // 4} tokens)")
    print(f"    -> AST Architecture Size: {total_ast_bytes} bytes (~{total_ast_bytes // 4} tokens)")
    print(f"    -> 🚀 Token Reduction Efficiency: {token_savings:.1f}% context tokens saved!")

    # 4.5. Visual Bug Analysis with Gemma 4 Vision (if screenshots present)
    visual_context = None
    if image_urls and VISION_AVAILABLE:
        code_ctx_str = "\n---\n".join(f"# {p}\n{c}" for p, c in file_contents.items())
        gh_token = os.environ.get("GITHUB_TOKEN")
        visual_context = run_visual_analysis(image_urls, gh_token, code_ctx_str)

    # 5. Generate Scoped TDD Patch & Test
    print("\n[*] Phase 4: Generating TDD Reproduction Test & Surgical Unified Diff...")
    visual_supplement = ""
    if visual_context:
        vr = visual_context["result"]
        visual_supplement = f"""

Visual Bug Analysis (Gemma 4 Vision):
- Defect Label: {vr.label}
- Bounding Box (normalized [y1, x1, y2, x2]): {vr.box_2d}
- Visual Root Cause: {vr.explanation}
- Suggested CSS/React Fix: {vr.suggested_patch}

Prioritize the visual analysis findings when generating the patch.
"""

    patch_prompt = f"""
Issue Title: {issue_data['title']}
Issue Body: {issue_data['body']}
{visual_supplement}
Target Files & Architecture Map:
{chr(10).join(ast_maps)}

Source Code Excerpts:
{json.dumps(file_contents, indent=2)}

Generate:
1. ### Reproduction Test (TDD): A minimal pytest test proving the bug exists. It is run as a standalone file from the
   repository root, so it must import the code under test itself and must FAIL on an assertion before the fix.
2. ### Surgical Patch: A clean unified git diff (```diff) modifying strictly the minimal lines necessary.
3. ### Maintainer Summary: A concise root-cause explanation and safety assurance.
"""
    patch_response, reproduce_test, patch_diff, verdict = generate_verified_patch(
        llm, system_prompt, patch_prompt, source_dir, max_attempts)
    if tmp_clone:
        tmp_clone.cleanup()

    # 6. Validate & save artifacts
    test_valid = False
    test_msg = "No test found"
    if reproduce_test:
        test_valid, test_msg = validate_python_syntax(reproduce_test)

    # Save artifacts
    test_path = os.path.join(output_dir, "test_reproduce.py")
    patch_path = os.path.join(output_dir, "fix.patch")
    report_path = os.path.join(output_dir, "MAINTAINER_REPORT.md")
    if verdict is None:
        verification = "NOT RUN (pass --verify or --repo-dir to check the patch in the sandbox)"
    elif verdict.ok:
        verification = f"VERIFIED - {verdict.detail}"
    else:
        verification = f"FAILED ({verdict.status}) - {verdict.detail}"

    with open(test_path, "w", encoding="utf-8") as f:
        f.write(reproduce_test or "# Reproduction test placeholder\n")

    with open(patch_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(patch_diff.strip("\n") + "\n" if patch_diff else "# Unified diff placeholder\n")

    visual_report = ""
    annotated_output_path = None
    pr_desc_path = None
    if visual_context:
        vr = visual_context["result"]
        # Copy or save annotated image to output_dir
        annotated_output_path = os.path.join(output_dir, "before_annotated.png")
        if os.path.abspath(visual_context["annotated_path"]) != os.path.abspath(annotated_output_path):
            import shutil
            shutil.copy2(visual_context["annotated_path"], annotated_output_path)

        visual_report = (
            f"- **Visual Defect Identified:** `{vr.label}`\n"
            f"- **Spatial Bounding Box [y1, x1, y2, x2]:** `{vr.box_2d}`\n"
            f"- **Visual Root Cause:** {vr.explanation}\n"
            f"- **Annotated Screenshot:** `{annotated_output_path}`\n"
        )

        # Generate PR description markdown
        if VISION_AVAILABLE:
            v_ctx = VisualPRContext(
                before_image_path=annotated_output_path,
                after_image_path=os.path.join(output_dir, "after_fixed.png") if os.path.exists(os.path.join(output_dir, "after_fixed.png")) else None,
                explanation=vr.explanation,
                suggested_patch=vr.suggested_patch or patch_diff,
                label=vr.label,
                box_2d=vr.box_2d,
            )
            pr_body_md = build_pr_body(
                visual_ctx=v_ctx,
                before_url="<URL_BEFORE>",
                after_url="<URL_AFTER>",
                patch_diff=patch_diff,
                maintainer_summary=extract_markdown_block(patch_response, "Maintainer Summary") or "Fix identified via Gemma 4 visual bug investigation.",
            )
            pr_desc_path = os.path.join(output_dir, "PR_DESCRIPTION.md")
            with open(pr_desc_path, "w", encoding="utf-8") as f:
                f.write(pr_body_md)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Patch-Pilot Maintainer Report\n\n"
                f"- **Issue:** {issue_data.get('url', 'local')} - {issue_data['title']}\n"
                f"- **Target Files:** `{target_files}`\n"
                f"- **Token Savings:** {token_savings:.1f}%\n"
                f"- **TDD Test Syntax Valid:** {test_valid} ({test_msg})\n"
                f"- **Sandbox Verification:** {verification}\n"
                f"{visual_report}\n"
                f"## Full Agent Output\n\n{patch_response}\n")

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("  ✅ PIPELINE RUN COMPLETED" if verdict is None or verdict.ok
          else "  ⚠️  PIPELINE RUN COMPLETED - PATCH NOT VERIFIED")
    print(f"  Verification: {verification}")
    print(f"  Execution Time: {elapsed:.2f}s")
    print(f"  Artifacts Generated:")
    print(f"    - Reproduction Test:   {test_path} [Syntax Check: {'PASSED' if test_valid else 'WARNING'}]")
    print(f"    - Surgical Patch:      {patch_path}")
    print(f"    - Maintainer Report:   {report_path}")
    if annotated_output_path:
        print(f"    - Annotated Defect:    {annotated_output_path}")
    if pr_desc_path:
        print(f"    - PR Description:      {pr_desc_path}")
    print("=" * 70)
    print("\nGenerated Maintainer Report Preview:\n")
    print(patch_response)


def main():
    parser = argparse.ArgumentParser(
        description="Patch-Pilot: Autonomous Open-Source Maintainer Agent & TDD Harness"
    )
    parser.add_argument("issue_url", nargs="?", help="GitHub issue URL")
    parser.add_argument("--provider", choices=["ollama", "openai", "gemini", "mock"], help="LLM provider")
    parser.add_argument("--model", help="Model name (e.g. qwen2.5-coder:7b, llama3.2)")
    parser.add_argument("--branch", help="Repository branch")
    parser.add_argument("--output-dir", default="output", help="Output directory for generated patches")
    parser.add_argument("--mock", action="store_true", help="Run deterministic offline mock demo")
    parser.add_argument("--verify", action="store_true",
                        help="Clone the repo and verify the test+patch by running its tests (trusted repos only)")
    parser.add_argument("--repo-dir", help="Local checkout of the repo to verify against (skips cloning)")
    parser.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS,
                        help="Generate/verify retries, feeding failures back to the model")
    parser.add_argument("--image", help="Path or URL to a screenshot image for visual bug analysis")
    args = parser.parse_args()

    if not args.issue_url and not args.mock:
        # No issue given: use the bundled demo issue. Only fake the model if none was requested.
        if args.provider and args.provider != "mock":
            print(f"No issue URL provided. Running the bundled demo issue with the real '{args.provider}' model...")
        else:
            print("No issue URL provided. Running mock demo (--mock)...")
            args.mock = True

    run_pipeline(
        issue_url=args.issue_url,
        provider=args.provider,
        model=args.model,
        branch=args.branch,
        output_dir=args.output_dir,
        is_mock=args.mock,
        verify=args.verify,
        repo_dir=args.repo_dir,
        max_attempts=args.max_attempts,
        image_path=args.image,
    )


if __name__ == "__main__":
    main()
