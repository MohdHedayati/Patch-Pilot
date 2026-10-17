"""GitHub Pull Request builder with visual bug evidence support.

Creates GitHub Pull Requests that include before/after screenshots
annotated by Gemma 4 Vision spatial grounding, providing maintainers
with clear visual context for bug resolutions.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


@dataclass
class VisualPRContext:
    """Context for a visually-grounded pull request.

    Holds paths to before/after screenshots, the defect explanation,
    a suggested patch, the defect label, and an optional bounding box.
    """

    before_image_path: str | None = None
    after_image_path: str | None = None
    explanation: str = ""
    suggested_patch: str = ""
    label: str = ""
    box_2d: list[int] | None = field(default=None)


def _github_headers(token: str) -> dict[str, str]:
    """Return standard GitHub API request headers.

    Args:
        token: GitHub personal access token.

    Returns:
        Dictionary of HTTP headers for GitHub API calls.
    """
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "patch-pilot-pr-builder",
    }


def upload_image_to_repo(
    token: str,
    owner: str,
    repo: str,
    branch: str,
    local_path: str,
    repo_path: str,
    commit_message: str = "Add visual evidence",
) -> str:
    """Upload an image file to a GitHub repository via the Contents API.

    If the file already exists at *repo_path*, its SHA is fetched first so
    the API call updates the existing blob rather than failing.

    Args:
        token: GitHub personal access token.
        owner: Repository owner (user or organisation).
        repo: Repository name.
        branch: Target branch for the commit.
        local_path: Absolute or relative path to the local image file.
        repo_path: Destination path inside the repository (e.g.
            ``".patch-pilot/evidence/before.png"``).
        commit_message: Commit message for the upload.

    Returns:
        The raw download URL for the uploaded file.

    Raises:
        FileNotFoundError: If *local_path* does not exist.
        requests.HTTPError: If the GitHub API returns an error status.
    """
    image_path = Path(local_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {local_path}")

    content_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")

    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{repo_path}"
    headers = _github_headers(token)

    # Check whether the file already exists so we can supply its SHA.
    sha: str | None = None
    existing_resp = requests.get(
        api_url,
        headers=headers,
        params={"ref": branch},
        timeout=30,
    )
    if existing_resp.status_code == 200:
        sha = existing_resp.json().get("sha")
        logger.info("File already exists at %s – will update (sha=%s)", repo_path, sha)

    payload: dict[str, str] = {
        "message": commit_message,
        "content": content_b64,
        "branch": branch,
    }
    if sha is not None:
        payload["sha"] = sha

    resp = requests.put(api_url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()

    download_url: str = resp.json()["content"]["download_url"]
    logger.info("Uploaded %s → %s", local_path, download_url)
    return download_url


def build_pr_body(
    visual_ctx: VisualPRContext,
    before_url: str | None = None,
    after_url: str | None = None,
    patch_diff: str = "",
    maintainer_summary: str = "",
) -> str:
    """Construct a markdown PR description with optional visual evidence.

    When *before_url* and *after_url* are both provided the body includes a
    side-by-side comparison table.  In text-only mode (no images) the visual
    section is omitted and only the patch diff and maintainer summary are
    shown.

    Args:
        visual_ctx: Visual context dataclass with defect metadata.
        before_url: Public URL to the "before" screenshot (or ``None``).
        after_url: Public URL to the "after" screenshot (or ``None``).
        patch_diff: Unified diff of the applied patch.
        maintainer_summary: Human-readable summary for maintainers.

    Returns:
        Formatted markdown string ready to use as a PR body.
    """
    sections: list[str] = ["## 👁️ Patch Pilot: Visual Bug Resolution\n"]

    # --- Visual comparison section (only when images are available) --------
    has_images = before_url is not None or after_url is not None
    if has_images:
        before_cell = f"![Before]({before_url})" if before_url else "*not available*"
        after_cell = f"![After]({after_url})" if after_url else "*not available*"

        box_display = (
            f"`{visual_ctx.box_2d}`" if visual_ctx.box_2d else "*not provided*"
        )

        sections.append(
            "### 🎯 Spatial Bug Localization\n"
            "Gemma 4 identified a visual defect in the rendered UI:\n\n"
            "| ❌ Before (Annotated Defect) | ✅ After (Verified Patch) |\n"
            "| :---: | :---: |\n"
            f"| {before_cell} | {after_cell} |\n\n"
            f"**Bounding Box (normalized):** {box_display}\n"
            f"**Defect Label:** {visual_ctx.label or '*not provided*'}\n"
            f"**Visual Root Cause:** {visual_ctx.explanation or '*not provided*'}\n\n"
            "---"
        )

    # --- Patch diff section ------------------------------------------------
    if patch_diff:
        sections.append(
            "### 🛠️ Applied Patch\n"
            "```diff\n"
            f"{patch_diff}\n"
            "```\n\n"
            "---"
        )

    # --- Maintainer summary ------------------------------------------------
    if maintainer_summary:
        sections.append(
            "### 📋 Maintainer Summary\n"
            f"{maintainer_summary}\n\n"
            "---"
        )

    # --- Footer ------------------------------------------------------------
    sections.append(
        "*Generated by [Patch Pilot](https://github.com/Patch-Pilot) "
        "with Gemma 4 Vision spatial grounding*"
    )

    return "\n".join(sections)


def create_visual_pr(
    token: str,
    owner: str,
    repo: str,
    base_branch: str,
    head_branch: str,
    title: str,
    visual_ctx: VisualPRContext,
    patch_diff: str = "",
    maintainer_summary: str = "",
) -> dict:
    """Create a GitHub Pull Request with optional visual bug evidence.

    If the *visual_ctx* contains image paths the images are first uploaded
    to the repository so they can be embedded in the PR body.

    Args:
        token: GitHub personal access token.
        owner: Repository owner (user or organisation).
        repo: Repository name.
        base_branch: Branch the PR should be merged into.
        head_branch: Branch that contains the changes.
        title: Pull request title.
        visual_ctx: Visual context with screenshots and defect metadata.
        patch_diff: Unified diff of the applied patch.
        maintainer_summary: Human-readable summary for maintainers.

    Returns:
        The parsed JSON response from the GitHub "create pull request" API
        endpoint.

    Raises:
        requests.HTTPError: If any GitHub API call fails.
    """
    before_url: str | None = None
    after_url: str | None = None

    if visual_ctx.before_image_path:
        logger.info("Uploading before-image: %s", visual_ctx.before_image_path)
        before_url = upload_image_to_repo(
            token=token,
            owner=owner,
            repo=repo,
            branch=head_branch,
            local_path=visual_ctx.before_image_path,
            repo_path=".patch-pilot/evidence/before.png",
            commit_message="Add before-screenshot visual evidence",
        )

    if visual_ctx.after_image_path:
        logger.info("Uploading after-image: %s", visual_ctx.after_image_path)
        after_url = upload_image_to_repo(
            token=token,
            owner=owner,
            repo=repo,
            branch=head_branch,
            local_path=visual_ctx.after_image_path,
            repo_path=".patch-pilot/evidence/after.png",
            commit_message="Add after-screenshot visual evidence",
        )

    body = build_pr_body(
        visual_ctx=visual_ctx,
        before_url=before_url,
        after_url=after_url,
        patch_diff=patch_diff,
        maintainer_summary=maintainer_summary,
    )

    api_url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
    headers = _github_headers(token)

    payload: dict[str, str] = {
        "title": title,
        "head": head_branch,
        "base": base_branch,
        "body": body,
    }

    logger.info("Creating PR '%s' (%s → %s)", title, head_branch, base_branch)
    resp = requests.post(api_url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()

    pr_data: dict = resp.json()
    logger.info("PR created: %s", pr_data.get("html_url", pr_data.get("url")))
    return pr_data
