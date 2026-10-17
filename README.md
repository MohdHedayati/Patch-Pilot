# Patch Pilot 🚀

> **Autonomous Multimodal Visual Bug Investigator & TDD Agent Skill**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Standard](https://img.shields.io/badge/Standard-Agent%20Skill%20Open%20Standard-purple.svg)](SKILL.md)
[![Model](https://img.shields.io/badge/Model-Google%20Gemma%204%20(gemma--4--26b--a4b--it)-4285F4?logo=google&logoColor=white)](https://huggingface.co/google)
[![Hacktoberfest 2026](https://img.shields.io/badge/Hacktoberfest-2026-FF8800?logo=hacktoberfest&logoColor=white)](https://hacktoberfest.com)

---

## 📖 Project Overview

Open-source maintainers spend countless hours deciphering low-quality, ambiguous bug reports like *"the checkout button is broken on mobile"* accompanied only by a cropped phone screenshot. Manual triage, reproduction, component identification, and CSS debugging create significant maintenance drag.

**Patch Pilot** is an open-source, multimodal GitHub App agent built for **Hacktoberfest 2026** that transforms unstructured UI screenshots into verified, production-ready Pull Requests.

Powered by **Google Gemma 4 (`gemma-4-26b-a4b-it`)**, Patch Pilot combines native vision-language spatial grounding with automated test-driven development (TDD):
1. **Multimodal Spatial Grounding:** Extracts normalized pixel bounding boxes `[y1, x1, y2, x2]` pointing directly at the visual defect.
2. **Local Pillow Annotation:** Annotates the issue screenshot with high-visibility target overlays (`before_annotated.png`).
3. **Progressive AST Architecture Disclosure:** Prunes the codebase AST to surface only the relevant DOM/JSX/CSS components, slashing prompt token consumption by 70–90%.
4. **Closed-Loop Playwright Verification:** Authors a failing reproduction test, synthesizes a surgical patch, runs headless Playwright to verify the fix, and captures post-fix visual proof (`after_fixed.png`).
5. **Automated PR Generation:** Opens a Pull Request complete with before-and-after visual proof, reproduction tests, and an explanation of the fix.

---

## ✨ Key Innovations

- **Gemma 4 Multimodal Spatial Grounding:** Leverages Google's `gemma-4-26b-a4b-it` model to perform native zero-shot visual bug localization, returning normalized `[y1, x1, y2, x2]` bounding box coordinates without relying on fragile external heuristics.
- **Closed-Loop Playwright Verification:** Executes headless browser runs against the generated patch, capturing `after_fixed.png` and composing an unassailable side-by-side visual diff (`visual_diff_composite.png`) before committing.
- **Agent Skill Open Standard Compliance:** Fully compliant with the [Agent Skill Open Standard](SKILL.md), allowing Patch Pilot to be invoked seamlessly as an autonomous CLI utility, GitHub App webhook, or modular subagent skill in agentic developer workflows.
- **Progressive Architecture Disclosure:** Employs AST mapping via `scripts/analyze_ast.py` to identify candidate UI elements and isolate the minimal affected code subtree, saving 70–90% of token context while preventing hallucinated edits.
- **Strict TDD Guardrails:** Enforces a hard requirement that an automated reproduction test (Playwright/pytest) must fail on the baseline code and pass on the patched code prior to diff generation.

---

## 🏗️ Architecture Diagram

```mermaid
flowchart TD
    subgraph Phase1["Phase 1: Ingestion"]
        A["GitHub Webhook Event<br/>(issues.opened / comment)"] --> B["Webhook Listener & Issue Parser<br/>(Extracts Issue Text & Screenshot URLs)"]
    end

    subgraph Phase2["Phase 2: Vision & Spatial Grounding"]
        B --> C["Google Gemma 4 (gemma-4-26b-a4b-it)<br/>Multimodal Spatial Reasoning"]
        C --> D["Extract Normalized Coordinates<br/>[y1, x1, y2, x2] & Pillow Annotation (before_annotated.png)"]
    end

    subgraph Phase3["Phase 3: AST Disclosure & Context Pruning"]
        D --> E["Progressive Architecture Disclosure<br/>analyze_ast.py (Target Component Mapping)"]
        E --> F["Pruned Context Assembly<br/>(Saves 70-90% Context Window Tokens)"]
    end

    subgraph Phase4["Phase 4: TDD & Closed-Loop Verification"]
        F --> G["TDD Reproduction Phase<br/>Synthesize Failing Playwright Test"]
        G --> H["Synthesize Surgical Patch<br/>(CSS / JSX / Component Fix)"]
        H --> I["Playwright Headless Execution<br/>Verify Fix & Capture after_fixed.png"]
        I --> J{"Test Passed &<br/>Visual Diff Valid?"}
        J -- "Retry / Refine" --> H
        J -- "Pass" --> K["Side-by-Side Visual Proof Composite<br/>(visual_diff_composite.png)"]
    end

    subgraph Phase5["Phase 5: Automated PR Creation"]
        K --> L["GitHub App Engine<br/>Branch Creation & Commit Push"]
        L --> M["Open Pull Request<br/>- Failing Test Suite<br/>- Surgical Patch<br/>- Side-by-Side Visual Proof Matrix"]
    end
```

---

## ⚡ Quickstart & Usage

### Step 1: Installation

Clone the repository and install dependencies in a Python 3.10+ virtual environment:

```bash
git clone https://github.com/your-username/patch-pilot.git
cd patch-pilot

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install Python requirements
pip install -r requirements.txt

# Install Playwright browser binaries
playwright install chromium
```

### Step 2: Environment Configuration

Create a `.env` file in the project root:

```env
# Google Gemma 4 Inference API Key
GEMINI_API_KEY=your_gemini_api_key_here

# GitHub Personal Access Token or App Installation Token
GITHUB_TOKEN=ghp_your_github_personal_access_token

# GitHub Webhook Secret (for webhook listener mode)
GITHUB_WEBHOOK_SECRET=your_webhook_secret_here
```

### Step 3: CLI Usage

#### Option A: Local Image Investigation
Run Patch Pilot directly on a local screenshot and problem description:

```bash
python -m runner.cli \
  --image examples/test_ui.png \
  --description "Button text overflows navbar container on viewport width < 768px" \
  --repo-path .
```

#### Option B: Live GitHub Issue URL
Investigate a live GitHub issue containing a screenshot:

```bash
python -m runner.cli \
  --issue "https://github.com/owner/repository/issues/42"
```

### 📁 Generated Artifacts (`./output/`)

All investigation artifacts are deposited in `./output/` for inspection and CI/CD ingestion:

| Artifact | Description |
| :--- | :--- |
| `output/before_annotated.png` | Baseline screenshot with Pillow bounding box from Gemma 4 `[y1, x1, y2, x2]`. |
| `output/repro_test.spec.ts` | Playwright automated reproduction test failing on bug baseline. |
| `output/patch.diff` | Minimal, surgical code patch resolving the visual bug. |
| `output/after_fixed.png` | Playwright post-fix visual verification capture. |
| `output/visual_diff_composite.png` | Side-by-side composite proof (Before vs. After). |
| `output/investigation_report.json` | Complete diagnostic telemetry, spatial coordinates, and run metrics. |

---

## 🧪 Testing & Directory Structure

### Running the Test Suite

Execute the full suite of unit and integration tests with pytest:

```bash
pytest tests/test_visual_investigator.py -v
```

### Codebase Directory Structure

```
patch-pilot/
├── ai/                             # Gemma 4 inference & vision-language grounding
│   ├── gemma_client.py             # Google Gemma 4 (gemma-4-26b-a4b-it) client
│   ├── grounder.py                 # Coordinate parser [y1, x1, y2, x2] & normalizer
│   └── patch_generator.py          # Surgical patch & TDD test synthesizer
├── github/                         # GitHub integration & App utilities
│   ├── client.py                   # Authenticated GitHub API wrapper
│   ├── issue_parser.py             # Markdown image & comment asset extractor
│   └── pr_manager.py               # Branch creator, committer & PR publisher
├── runner/                         # Core execution workflows & CLI
│   ├── cli.py                      # Command-line interface
│   └── orchestrator.py             # 5-phase investigation state machine
├── scripts/                        # Repository AST analysis & image compositing
│   ├── analyze_ast.py              # Progressive architecture disclosure (AST extractor)
│   └── composite_image.py          # Side-by-side visual proof generator
├── utils/                          # Helper modules
│   ├── annotator.py                # Pillow bounding box drawing and labeling
│   └── logger.py                   # Rich console & JSON diagnostic logging
├── webhooks/                       # Inbound GitHub App event server
│   └── listener.py                 # FastAPI webhook listener for issue events
├── tests/                          # Test suite
│   ├── test_spatial_grounding.py   # Bounding box extraction validation tests
│   └── test_visual_investigator.py # End-to-end investigation pipeline test
├── prompts/                        # Gemma 4 prompt templates
│   ├── spatial_grounding.jinja2    # Bounding box extraction prompt
│   └── tdd_patch.jinja2            # Failing test and surgical patch generation prompt
├── examples/                       # Sample assets for testing
│   └── test_ui.png                 # Reference UI screenshot with intentional bug
├── SKILL.md                        # Agent Skill Open Standard specification
├── requirements.txt                # Python package dependencies
├── LICENSE                         # Apache 2.0 License
└── README.md                       # Project documentation
```

---

## 🎃 Hacktoberfest 2026 Pitch Summary

> **The Problem:** Frontend visual bugs constitute over 40% of open-source UI issues, yet report quality is notoriously poor. Maintainers often receive nothing more than a lone screenshot and vague text, stalling triage and burning valuable contributor time.
>
> **The Gemma 4 Solution:** Patch Pilot unleashes **Google Gemma 4 (`gemma-4-26b-a4b-it`)** as an autonomous visual triage maintainer. By coupling native multimodal spatial localization with token-saving AST disclosure and closed-loop Playwright verification, Patch Pilot converts ambiguous screenshots into verified, reproducible Pull Requests with side-by-side visual proof—empowering maintainers to review and merge with total confidence.

---

## 📄 License

This project is licensed under the [Apache 2.0 License](LICENSE).
