"""GitHub App Webhook Handler for Patch Pilot (Visual Bug Investigator).

Processes incoming GitHub App webhooks ('issues' and 'issue_comment' events),
extracts screenshot attachments, invokes Gemma 4 Vision spatial grounding,
and dispatches to the patch pipeline with visual evidence.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any

# Ensure project root is in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.fetch_issue import extract_image_urls, download_image
from ai.gemma_vision import analyze_screenshot, VisionAnalysisResult
from utils.image_annotator import draw_bounding_box
from github.pr_builder import VisualPRContext, build_pr_body

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class WebhookIssueContext:
    """Parsed issue context from a GitHub App webhook payload."""

    owner: str
    repo: str
    issue_number: int
    title: str
    body: str
    author: str
    action: str
    image_urls: list[str] = field(default_factory=list)
    downloaded_images: list[str] = field(default_factory=list)
    has_images: bool = False


def parse_issue_webhook_payload(
    payload: dict[str, Any],
    download_attachments: bool = True,
    token: str | None = None,
    dest_dir: str | None = None,
) -> WebhookIssueContext:
    """Parse a GitHub webhook payload for issue/comment events and extract images.

    Scans `issue.body` and `comment.body` (if present) for markdown `![...](url)`
    and HTML `<img src="...">` image references. If found, optionally downloads
    the payloads locally to a temporary directory.

    Args:
        payload: The raw JSON dictionary from the GitHub webhook event.
        download_attachments: Whether to download detected images to local disk.
        token: Optional GitHub token for downloading private attachments.
        dest_dir: Optional custom directory to save downloaded images.

    Returns:
        A :class:`WebhookIssueContext` object containing the parsed issue metadata,
        extracted image URLs, and paths to any locally downloaded image files.
    """
    repository = payload.get("repository", {})
    owner = repository.get("owner", {}).get("login", "")
    repo = repository.get("name", "")

    issue = payload.get("issue", {})
    issue_number = issue.get("number", 0)
    title = issue.get("title", "")
    body = issue.get("body") or ""
    action = payload.get("action", "")
    author = issue.get("user", {}).get("login", "")

    # Scan issue body for image URLs
    all_image_urls = extract_image_urls(body)

    # If this is an issue_comment event or comment is embedded, scan comment body too
    comment = payload.get("comment")
    if isinstance(comment, dict):
        comment_body = comment.get("body") or ""
        comment_images = extract_image_urls(comment_body)
        for img in comment_images:
            if img not in all_image_urls:
                all_image_urls.append(img)

    downloaded: list[str] = []
    if download_attachments and all_image_urls:
        target_dir = dest_dir or tempfile.mkdtemp(prefix="patchpilot_webhook_img_")
        auth_token = token or os.environ.get("GITHUB_TOKEN")
        for url in all_image_urls:
            local_img = download_image(url, dest_dir=target_dir, token=auth_token)
            if local_img:
                if isinstance(local_img, bytes):
                    if os.path.exists(url):
                        downloaded.append(os.path.abspath(url))
                    else:
                        temp_file_path = os.path.join(target_dir, f"attachment_{len(downloaded)}.png")
                        with open(temp_file_path, "wb") as f:
                            f.write(local_img)
                        downloaded.append(temp_file_path)
                else:
                    downloaded.append(local_img)

    return WebhookIssueContext(
        owner=owner,
        repo=repo,
        issue_number=issue_number,
        title=title,
        body=body,
        author=author,
        action=action,
        image_urls=all_image_urls,
        downloaded_images=downloaded,
        has_images=len(downloaded) > 0 or len(all_image_urls) > 0,
    )


def handle_issue_event(
    payload: dict[str, Any],
    code_context: str = "",
    token: str | None = None,
    gemini_key: str | None = None,
) -> dict[str, Any]:
    """Main dispatch handler for incoming GitHub issue webhooks.

    If an image is detected in the issue:
      1. Downloads image payload locally.
      2. Invokes Gemma 4 Multimodal Vision to extract [y1, x1, y2, x2] spatial bounding box.
      3. Overlays bounding box onto screenshot via image_annotator.
      4. Returns visual bug analysis result with annotated screenshot path.

    Fallback:
      If no image is provided, returns text-only pipeline instruction.

    Args:
        payload: GitHub webhook event payload.
        code_context: Optional frontend code context for the visual defect.
        token: Optional GitHub token.
        gemini_key: Optional Gemini API key.

    Returns:
        Dict containing investigation status, mode ('visual' or 'text_fallback'),
        and metadata.
    """
    ctx = parse_issue_webhook_payload(payload, download_attachments=True, token=token)

    logger.info(
        "Received webhook for %s/%s #%d [%s]. Images detected: %d",
        ctx.owner,
        ctx.repo,
        ctx.issue_number,
        ctx.action,
        len(ctx.downloaded_images),
    )

    if not ctx.has_images or not ctx.downloaded_images:
        logger.info(
            "No images detected in issue #%d. Falling back to text-based reproduction loop.",
            ctx.issue_number,
        )
        return {
            "mode": "text_fallback",
            "context": ctx,
            "message": "Fallback: No image attachments found; proceeding with text-only reproduction loop.",
            "visual_result": None,
        }

    # Vision-guided investigation
    primary_image = ctx.downloaded_images[0]
    logger.info("Executing Gemma 4 Vision analysis on %s", primary_image)

    vision_result = analyze_screenshot(
        image_path=primary_image,
        code_context=code_context,
        api_key=gemini_key,
    )

    annotated_path: str | None = None
    if vision_result.has_visual_bug and vision_result.box_2d:
        try:
            annotated_path = draw_bounding_box(
                image_path=primary_image,
                box_2d=vision_result.box_2d,
                label=vision_result.label or "UI Defect",
            )
            logger.info("Annotated visual defect image saved to %s", annotated_path)
        except Exception as e:
            logger.warning("Could not draw bounding box: %s", e)

    return {
        "mode": "visual",
        "context": ctx,
        "visual_result": vision_result,
        "annotated_image": annotated_path,
        "original_image": primary_image,
        "message": (
            f"Visual Defect Localized: {vision_result.label}"
            if vision_result.has_visual_bug
            else "Gemma 4 Vision found no visual defects in the screenshot."
        ),
    }

