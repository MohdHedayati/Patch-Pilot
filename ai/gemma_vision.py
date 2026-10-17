"""Gemma 4 multimodal vision integration for UI bug detection using google-genai SDK.

Uses the Google GenAI SDK (google.genai) to send screenshot images alongside
code context to Gemma 4, returning spatial bounding boxes and suggested patches
for visual defects.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types

logger: logging.Logger = logging.getLogger(__name__)

_DEFAULT_MODEL: str = "gemma-4-26b-a4b-it"

_MIME_TYPE_MAP: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}

_PROMPT_TEMPLATE: str = (
    "You are an expert frontend QA engineer and UI bug analyzer. "
    "Examine the attached screenshot alongside the code snippet.\n"
    "1. Identify any visual bugs (misalignments, text overlaps, "
    "incorrect spacing, broken components).\n"
    "2. Output the exact spatial bounding box coordinates of the primary "
    "visual defect in normalized [y1, x1, y2, x2] format (0 to 1000 scale).\n"
    "3. Explain the visual defect and output the exact CSS/React "
    "modification required to fix it.\n"
    "\n"
    "Code context:\n"
    "{code_context}\n"
    "\n"
    "Return strictly JSON in this format:\n"
    "{{\n"
    '  "has_visual_bug": true,\n'
    '  "box_2d": [y1, x1, y2, x2],\n'
    '  "label": "short_bug_description",\n'
    '  "explanation": "detailed reason",\n'
    '  "suggested_patch": "CSS or React code diff snippet"\n'
    "}}"
)


@dataclass
class VisionAnalysisResult:
    """Result of a multimodal vision analysis on a UI screenshot."""

    has_visual_bug: bool
    box_2d: list[int] = field(default_factory=list)
    label: str = ""
    explanation: str = ""
    suggested_patch: str = ""


def _get_mime_type(image_path: str) -> str:
    """Determine the MIME type from the file extension.

    Args:
        image_path: Filesystem path to the image.

    Returns:
        The corresponding MIME type string.

    Raises:
        ValueError: If the extension is not recognised.
    """
    ext: str = Path(image_path).suffix.lower()
    mime_type: str | None = _MIME_TYPE_MAP.get(ext)
    if mime_type is None:
        raise ValueError(
            f"Unsupported image extension '{ext}'. "
            f"Supported: {', '.join(sorted(_MIME_TYPE_MAP))}"
        )
    return mime_type


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences (e.g. ```json … ```) from *text*.

    Args:
        text: Raw text that may be wrapped in code fences.

    Returns:
        The inner content with fences removed.
    """
    stripped: str = text.strip()
    pattern: re.Pattern[str] = re.compile(
        r"^```(?:\w+)?\s*\n?(.*?)\n?\s*```$", re.DOTALL
    )
    match: re.Match[str] | None = pattern.match(stripped)
    if match:
        return match.group(1).strip()
    return stripped


def _build_default_result() -> VisionAnalysisResult:
    """Return a default *no-bug-found* result."""
    return VisionAnalysisResult(
        has_visual_bug=False,
        box_2d=[0, 0, 0, 0],
        label="",
        explanation="",
        suggested_patch="",
    )


def analyze_screenshot(
    image_path: str,
    code_context: str,
    api_key: str | None = None,
    model: str = _DEFAULT_MODEL,
) -> VisionAnalysisResult:
    """Analyze a UI screenshot for visual bugs using Gemma 4 multimodal vision via google-genai.

    Sends the screenshot together with surrounding code context to Gemma 4
    using the google-genai Client and returns structured information about
    any detected visual defect.

    Args:
        image_path: Absolute or relative path to the screenshot image file.
        code_context: Source-code snippet surrounding the UI component shown
            in the screenshot.
        api_key: Optional Google Gemini API key. If not passed, Client
            automatically resolves GEMINI_API_KEY from environment.
        model: Model identifier (defaults to 'gemma-4-26b-a4b-it').

    Returns:
        A VisionAnalysisResult describing the detected bug (if any).
        When no bug is found or an error occurs, returns a result with
        has_visual_bug=False.
    """
    # ── Validate image file ──────────────────────────────────────────────
    if not os.path.isfile(image_path):
        logger.error("Image file not found: %s", image_path)
        return _build_default_result()

    try:
        mime_type: str = _get_mime_type(image_path)
        with open(image_path, "rb") as fh:
            image_bytes: bytes = fh.read()
    except (ValueError, OSError) as exc:
        logger.error("Failed to read image '%s': %s", image_path, exc)
        return _build_default_result()

    # ── Initialize genai.Client ──────────────────────────────────────────
    try:
        if api_key:
            client = genai.Client(api_key=api_key)
        else:
            client = genai.Client()
    except Exception as exc:
        logger.error("Failed to initialize genai.Client: %s", exc)
        return _build_default_result()

    # ── Build prompt & execute model call ────────────────────────────────
    prompt_text: str = _PROMPT_TEMPLATE.format(code_context=code_context)

    try:
        logger.info("Sending vision analysis request to %s via google.genai …", model)
        response = client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                prompt_text,
            ],
        )
    except Exception as exc:
        logger.error("google-genai model generation failed: %s", exc)
        return _build_default_result()

    # ── Parse response ───────────────────────────────────────────────────
    try:
        raw_text: str = response.text or ""
        logger.debug("Raw model response text: %s", raw_text)

        cleaned_text: str = _strip_code_fences(raw_text)
        result_data: dict[str, object] = json.loads(cleaned_text)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError, AttributeError) as exc:
        logger.error("Failed to parse model response: %s", exc)
        return _build_default_result()

    # ── Build result ─────────────────────────────────────────────────────
    has_bug: bool = bool(result_data.get("has_visual_bug", False))
    if not has_bug:
        logger.info("No visual bug detected.")
        return _build_default_result()

    result: VisionAnalysisResult = VisionAnalysisResult(
        has_visual_bug=True,
        box_2d=[int(c) for c in result_data.get("box_2d", [0, 0, 0, 0])],  # type: ignore[union-attr]
        label=str(result_data.get("label", "")),
        explanation=str(result_data.get("explanation", "")),
        suggested_patch=str(result_data.get("suggested_patch", "")),
    )
    logger.info("Visual bug detected: %s", result.label)
    return result
