#!/usr/bin/env python3
"""
eval_harness.py - Model Evaluation & Benchmark Harness for Patch-Pilot.

Measures:
1. Triage Accuracy: Ground-truth retrieval of buggy files across repository trees.
2. Scope Adherence: Strict enforcement of <= 3 files (anti-hallucination metric).
3. Token Reduction Efficiency: AST progressive disclosure token compression ratio.
4. TDD Test Validity: AST syntax parsing of generated reproduction tests.

Usage:
    python scripts/eval_harness.py
    python scripts/eval_harness.py --provider ollama --model qwen2.5-coder:7b
    python scripts/eval_harness.py --mock
"""

import argparse
import ast
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_ast import analyze_code
from llm_client import LLMClient
from patch_pilot import load_skill_directive, extract_json_array, extract_markdown_block, validate_python_syntax

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def load_benchmarks():
    bm_path = os.path.join(os.path.dirname(__file__), "..", "references", "benchmarks.json")
    if not os.path.exists(bm_path):
        raise FileNotFoundError(f"Benchmark file not found at {bm_path}")
    with open(bm_path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_benchmark(item, llm, system_prompt):
    result = {
        "id": item["id"],
        "name": item["name"],
        "repo": item["repo"],
        "total_repo_files": len(item["files"]),
        "triage_passed": False,
        "scope_compliant": False,
        "selected_files": [],
        "token_reduction_pct": 0.0,
        "test_valid": False,
        "latency_sec": 0.0,
    }

    t0 = time.time()
    
    # 1. Triage Directive
    triage_prompt = (
        f"Issue Title: {item['issue_title']}\n"
        f"Issue Body: {item['issue_body']}\n\n"
        f"Repository Files ({len(item['files'])}):\n"
        + json.dumps(item["files"], indent=2) + "\n\n"
        "Identify the 1 to 3 files most likely to contain the code that needs to be fixed.\n"
        "Output ONLY a raw, valid JSON array containing the file paths."
    )
    
    if llm.provider == "mock":
        # Mock ground truth for deterministic test
        selected_files = item["ground_truth_culprits"]
    else:
        resp = llm.complete(triage_prompt, system_prompt=system_prompt)
        selected_files = extract_json_array(resp)

    result["selected_files"] = selected_files
    result["scope_compliant"] = (1 <= len(selected_files) <= 3)

    # Check triage accuracy against ground truth
    ground_truth = set(item["ground_truth_culprits"])
    hits = set(selected_files).intersection(ground_truth)
    result["triage_passed"] = len(hits) >= 1  # At least 1 true culprit caught

    # 2. Token Reduction via AST
    total_raw_bytes = 0
    total_ast_bytes = 0
    ast_summaries = []

    for path, code in item.get("sample_source", {}).items():
        total_raw_bytes += len(code.encode("utf-8"))
        ast_map = analyze_code(code, path)
        ast_summaries.append(ast_map)
        total_ast_bytes += len(ast_map.encode("utf-8"))

    if total_raw_bytes > 0:
        result["token_reduction_pct"] = round(
            max(0.0, ((total_raw_bytes - total_ast_bytes) / total_raw_bytes) * 100), 1
        )

    # 3. Test Syntax Validation
    patch_prompt = (
        f"Issue: {item['issue_title']}\n"
        f"AST Architecture:\n{chr(10).join(ast_summaries)}\n\n"
        "Generate a reproduction unit test:\n"
        "### Reproduction Test (TDD)\n```python\n# test here\n```"
    )
    patch_resp = llm.complete(patch_prompt, system_prompt=system_prompt)
    reproduce_test = extract_markdown_block(patch_resp, "Reproduction Test")
    
    if reproduce_test:
        valid, _ = validate_python_syntax(reproduce_test)
        result["test_valid"] = valid
    else:
        result["test_valid"] = False

    result["latency_sec"] = round(time.time() - t0, 2)
    return result


def main():
    parser = argparse.ArgumentParser(description="Patch-Pilot Open-Source Model Evaluation Harness")
    parser.add_argument("--provider", choices=["ollama", "openai", "gemini", "mock"], default="mock")
    parser.add_argument("--model", default="qwen2.5-coder:7b")
    parser.add_argument("--output", default="eval_results.json")
    args = parser.parse_args()

    benchmarks = load_benchmarks()
    llm = LLMClient(provider=args.provider, model=args.model)
    system_prompt = load_skill_directive()

    print("=" * 70)
    print("  PATCH-PILOT: OPEN-SOURCE MODEL EVALUATION HARNESS")
    print(f"  Provider: [{llm.provider.upper()}]  Model: [{llm.model}]")
    print(f"  Benchmark Count: {len(benchmarks)} issues")
    print("=" * 70)

    results = []
    for bm in benchmarks:
        print(f"[*] Running {bm['id']} ({bm['name']})...", end="", flush=True)
        res = evaluate_benchmark(bm, llm, system_prompt)
        results.append(res)
        status = "PASSED" if (res["triage_passed"] and res["scope_compliant"]) else "FAILED"
        print(f" [{status}] (Triage: {'YES' if res['triage_passed'] else 'NO'}, Scope: {'OK' if res['scope_compliant'] else 'OVER'}, Tokens Saved: {res['token_reduction_pct']}%)")

    # Scorecard
    total = len(results)
    triage_acc = (sum(1 for r in results if r["triage_passed"]) / total) * 100
    scope_acc = (sum(1 for r in results if r["scope_compliant"]) / total) * 100
    test_acc = (sum(1 for r in results if r["test_valid"]) / total) * 100
    avg_tokens_saved = sum(r["token_reduction_pct"] for r in results) / total

    print("\n" + "=" * 70)
    print("  EVALUATION SCORECARD")
    print("=" * 70)
    print(f"  - Triage Accuracy:            {triage_acc:.1f}% ({sum(1 for r in results if r['triage_passed'])}/{total})")
    print(f"  - Strict Scope Compliance:    {scope_acc:.1f}% ({sum(1 for r in results if r['scope_compliant'])}/{total})")
    print(f"  - TDD Test Syntax Validity:   {test_acc:.1f}% ({sum(1 for r in results if r['test_valid'])}/{total})")
    print(f"  - Average Context Reduction:  {avg_tokens_saved:.1f}% fewer tokens via AST Progressive Disclosure")
    print("=" * 70)

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), "..", args.output)
    report = {
        "provider": llm.provider,
        "model": llm.model,
        "summary": {
            "total_benchmarks": total,
            "triage_accuracy_pct": triage_acc,
            "scope_compliance_pct": scope_acc,
            "test_syntax_validity_pct": test_acc,
            "avg_token_reduction_pct": avg_tokens_saved
        },
        "details": results
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[+] Full benchmark results saved to: {output_path}")


if __name__ == "__main__":
    main()
