"""Tests for Gemma 4 Vision & Spatial Grounding Integration in Patch Pilot."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from ai.gemma_vision import (
    VisionAnalysisResult,
    _get_mime_type,
    _strip_code_fences,
    analyze_screenshot,
)
from github.pr_builder import (
    VisualPRContext,
    build_pr_body,
    upload_image_to_repo,
)
from scripts.fetch_issue import download_image, extract_image_urls
from utils.image_annotator import create_comparison, draw_bounding_box
from webhooks.issue_handler import handle_issue_event, parse_issue_webhook_payload


# ─── Fetch Issue & Image Extraction Tests ───────────────────────────────────

def test_extract_image_urls():
    text = """
    ## Bug Report
    Look at this misalignment:
    ![screenshot](https://user-images.githubusercontent.com/123/456.png)
    
    Here is another view:
    <img src="https://example.com/assets/bug_view.jpg?v=2" width="400" />

    And non-image links should be ignored:
    [documentation](https://example.com/docs)
    """
    urls = extract_image_urls(text)
    assert len(urls) == 2
    assert "https://user-images.githubusercontent.com/123/456.png" in urls
    assert "https://example.com/assets/bug_view.jpg?v=2" in urls


def test_extract_image_urls_empty():
    assert extract_image_urls("Plain text with no images or links.") == []


def test_download_image_local_file(tmp_path):
    local_img = tmp_path / "local_sample.png"
    sample_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDRtest_data"
    local_img.write_bytes(sample_bytes)

    with patch("requests.get") as mock_get:
        data = download_image(str(local_img))
        assert isinstance(data, bytes)
        assert data == sample_bytes
        # Ensure requests.get was NOT called for local file
        mock_get.assert_not_called()


def test_download_image_http_url(tmp_path):
    dest_dir = tmp_path / "downloads"
    dest_dir.mkdir()
    fake_content = b"\x89PNGfake_remote_bytes"

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.iter_content.return_value = [fake_content]

    with patch("requests.get", return_value=mock_resp) as mock_get:
        result_path = download_image("https://example.com/bug.png", dest_dir=str(dest_dir))
        assert isinstance(result_path, str)
        assert os.path.exists(result_path)
        with open(result_path, "rb") as f:
            assert f.read() == fake_content
        mock_get.assert_called_once()


def test_download_image_invalid():
    assert download_image("non_existent_file.png") is None


# ─── Gemma 4 Vision Integration Tests ───────────────────────────────────────

def test_get_mime_type():
    assert _get_mime_type("test.png") == "image/png"
    assert _get_mime_type("photo.jpeg") == "image/jpeg"
    assert _get_mime_type("graphic.webp") == "image/webp"
    with pytest.raises(ValueError):
        _get_mime_type("document.pdf")


def test_strip_code_fences():
    fenced_json = "```json\n{\"has_visual_bug\": true}\n```"
    assert _strip_code_fences(fenced_json) == "{\"has_visual_bug\": true}"

    plain = "{\"has_visual_bug\": false}"
    assert _strip_code_fences(plain) == plain


def test_analyze_screenshot_success(tmp_path):
    img_file = tmp_path / "test_screenshot.png"
    img = Image.new("RGBA", (200, 100), color="blue")
    img.save(img_file)

    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "has_visual_bug": True,
        "box_2d": [100, 200, 400, 600],
        "label": "Button text clipping",
        "explanation": "The submit button label overflows its container.",
        "suggested_patch": "padding: 8px 16px; overflow: visible;",
    })

    with patch("ai.gemma_vision.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_client_cls.return_value = mock_client

        result = analyze_screenshot(
            image_path=str(img_file),
            code_context="<button className='btn'>Submit Now</button>",
            api_key="fake-key-for-test",
        )

        assert result.has_visual_bug is True
        assert result.box_2d == [100, 200, 400, 600]
        assert result.label == "Button text clipping"
        assert "overflow" in result.suggested_patch
        mock_client.models.generate_content.assert_called_once()
        call_kwargs = mock_client.models.generate_content.call_args.kwargs
        assert call_kwargs["model"] == "gemma-4-26b-a4b-it"


def test_analyze_screenshot_no_bug(tmp_path):
    img_file = tmp_path / "normal_screenshot.png"
    img = Image.new("RGBA", (100, 100), color="white")
    img.save(img_file)

    mock_response = MagicMock()
    mock_response.text = json.dumps({"has_visual_bug": False})

    with patch("ai.gemma_vision.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_client_cls.return_value = mock_client

        result = analyze_screenshot(
            image_path=str(img_file),
            code_context="<div>Normal</div>",
            api_key="fake-key",
        )

        assert result.has_visual_bug is False


# ─── Bounding Box Rendering Tests ──────────────────────────────────────────

def test_draw_bounding_box(tmp_path):
    img_file = tmp_path / "defect.png"
    original = Image.new("RGBA", (1000, 1000), color="white")
    original.save(img_file)

    output_file = tmp_path / "before_annotated.png"
    # Box: y1=200, x1=100, y2=500, x2=800 in 0-1000 normalized coordinates
    annotated_path = draw_bounding_box(
        image_path=str(img_file),
        box_2d=[200, 100, 500, 800],
        label="Header Overlap Defect",
        output_path=str(output_file),
    )

    assert os.path.exists(annotated_path)
    annotated_img = Image.open(annotated_path)
    assert annotated_img.size == (1000, 1000)


def test_create_comparison(tmp_path):
    before_path = tmp_path / "before.png"
    after_path = tmp_path / "after.png"
    Image.new("RGBA", (400, 300), color="red").save(before_path)
    Image.new("RGBA", (400, 300), color="green").save(after_path)

    comparison_path = tmp_path / "comparison.png"
    res = create_comparison(
        before_path=str(before_path),
        after_path=str(after_path),
        output_path=str(comparison_path),
    )

    assert os.path.exists(res)
    comp_img = Image.open(res)
    # Total width is 400 + 400 = 800, height is 300 + 40 (header) = 340
    assert comp_img.size == (800, 340)


# ─── Webhook Ingestion & Fallback Tests ─────────────────────────────────────

def test_parse_issue_webhook_payload_with_images():
    payload = {
        "action": "opened",
        "repository": {"owner": {"login": "octocat"}, "name": "hello-world"},
        "issue": {
            "number": 101,
            "title": "Navigation bar items overlap on mobile",
            "body": "See attached screenshot:\n![navbar bug](https://user-images.githubusercontent.com/navbar.png)",
            "user": {"login": "mona"},
        },
    }

    ctx = parse_issue_webhook_payload(payload, download_attachments=False)
    assert ctx.owner == "octocat"
    assert ctx.repo == "hello-world"
    assert ctx.issue_number == 101
    assert ctx.has_images is True
    assert "https://user-images.githubusercontent.com/navbar.png" in ctx.image_urls


def test_handle_issue_event_fallback_when_no_images():
    payload = {
        "action": "opened",
        "repository": {"owner": {"login": "octocat"}, "name": "hello-world"},
        "issue": {
            "number": 102,
            "title": "ZeroDivisionError in calculator.py",
            "body": "When input is 0, code throws unhandled ZeroDivisionError.",
            "user": {"login": "mona"},
        },
    }

    result = handle_issue_event(payload)
    assert result["mode"] == "text_fallback"
    assert "Fallback: No image attachments found" in result["message"]
    assert result["visual_result"] is None


# ─── PR Creator & Markdown Template Tests ──────────────────────────────────

def test_build_pr_body_visual_template():
    ctx = VisualPRContext(
        before_image_path="before_annotated.png",
        after_image_path="after_fixed.png",
        explanation="CSS flex-wrap was missing on navbar-container.",
        suggested_patch="+  flex-wrap: wrap;",
        label="Navbar Item Collision",
        box_2d=[120, 50, 240, 950],
    )

    body = build_pr_body(
        visual_ctx=ctx,
        before_url="https://github.com/o/r/raw/main/before.png",
        after_url="https://github.com/o/r/raw/main/after.png",
        patch_diff="--- a/nav.css\n+++ b/nav.css\n@@ -2,1 +2,2 @@\n+  flex-wrap: wrap;",
        maintainer_summary="Added missing flex-wrap rule to prevent horizontal component collision.",
    )

    assert "## 👁️ Patch Pilot: Visual Bug Resolution" in body
    assert "### 🎯 Spatial Bug Localization" in body
    assert "| ❌ Before (Annotated Defect) | ✅ After (Verified Patch) |" in body
    assert "![Before](https://github.com/o/r/raw/main/before.png)" in body
    assert "![After](https://github.com/o/r/raw/main/after.png)" in body
    assert "**Bounding Box (normalized):** `[120, 50, 240, 950]`" in body
    assert "**Defect Label:** Navbar Item Collision" in body
    assert "**Visual Root Cause:** CSS flex-wrap was missing" in body
    assert "### 🛠️ Applied Patch" in body
    assert "flex-wrap: wrap;" in body
    assert "### 📋 Maintainer Summary" in body


def test_build_pr_body_text_fallback():
    ctx = VisualPRContext()
    body = build_pr_body(
        visual_ctx=ctx,
        before_url=None,
        after_url=None,
        patch_diff="--- a/calc.py\n+++ b/calc.py\n+ if y == 0: return 0",
        maintainer_summary="Handled zero denominator edge case.",
    )

    # Should omit visual comparison table
    assert "### 🎯 Spatial Bug Localization" not in body
    assert "| ❌ Before (Annotated Defect) |" not in body
    assert "### 🛠️ Applied Patch" in body
    assert "### 📋 Maintainer Summary" in body


# ─── End-to-End Pipeline Integration Test ──────────────────────────────────

def test_pipeline_with_visual_defect(tmp_path):
    from scripts.patch_pilot import run_pipeline

    test_img = tmp_path / "screenshot.png"
    Image.new("RGBA", (500, 400), color="white").save(test_img)

    out_dir = tmp_path / "output"

    mock_vision_result = VisionAnalysisResult(
        has_visual_bug=True,
        box_2d=[150, 100, 350, 450],
        label="Dropdown cut off",
        explanation="Dropdown element overflows parent container with hidden overflow.",
        suggested_patch=".dropdown { z-index: 100; overflow: visible; }",
    )

    with patch("scripts.patch_pilot.analyze_screenshot", return_value=mock_vision_result):
        run_pipeline(
            issue_url=None,
            output_dir=str(out_dir),
            is_mock=True,
            image_path=str(test_img),
        )

    # Check generated visual artifacts
    assert os.path.exists(out_dir / "before_annotated.png")
    assert os.path.exists(out_dir / "PR_DESCRIPTION.md")
    assert os.path.exists(out_dir / "MAINTAINER_REPORT.md")

    report_content = (out_dir / "MAINTAINER_REPORT.md").read_text(encoding="utf-8")
    assert "Visual Defect Identified" in report_content
    assert "Dropdown cut off" in report_content
    assert "[150, 100, 350, 450]" in report_content

    pr_desc = (out_dir / "PR_DESCRIPTION.md").read_text(encoding="utf-8")
    assert "## 👁️ Patch Pilot: Visual Bug Resolution" in pr_desc
    assert "Dropdown cut off" in pr_desc

