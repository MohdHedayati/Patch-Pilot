"""Image annotation utilities for Patch-Pilot.

Draws spatial bounding boxes returned by Gemma 4 vision analysis onto
screenshot images and creates before/after comparison composites.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

logger: logging.Logger = logging.getLogger(__name__)


def _get_font(size: int = 14) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Return a TrueType font at *size*, falling back to the PIL default."""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except (OSError, IOError):
        logger.debug("TrueType font not available, falling back to default font.")
        return ImageFont.load_default()


def draw_bounding_box(
    image_path: str,
    box_2d: list[int],
    label: str,
    output_path: str | None = None,
) -> str:
    """Draw a glowing red bounding box with a label onto an image.

    Parameters
    ----------
    image_path:
        Filesystem path to the source image.
    box_2d:
        Normalised coordinates ``[y1, x1, y2, x2]`` on a 0–1000 scale.
    label:
        Text label rendered above the bounding box.
    output_path:
        Destination path for the annotated image.  When *None*, the file is
        saved next to the original with an ``_annotated`` suffix.

    Returns
    -------
    str
        Absolute path of the saved annotated image.
    """
    image: Image.Image = Image.open(image_path).convert("RGBA")
    img_w: int
    img_h: int
    img_w, img_h = image.size

    # --- Scale normalised coords (0-1000) to pixel coords ----------------
    y1_px: int = int(box_2d[0] / 1000 * img_h)
    x1_px: int = int(box_2d[1] / 1000 * img_w)
    y2_px: int = int(box_2d[2] / 1000 * img_h)
    x2_px: int = int(box_2d[3] / 1000 * img_w)

    logger.info(
        "Bounding box pixels: (%d, %d) -> (%d, %d) for label '%s'",
        x1_px,
        y1_px,
        x2_px,
        y2_px,
        label,
    )

    # --- Glow effect (semi-transparent red overlay) -----------------------
    glow_expand: int = 4
    overlay: Image.Image = Image.new("RGBA", image.size, (0, 0, 0, 0))
    glow_draw: ImageDraw.ImageDraw = ImageDraw.Draw(overlay)
    glow_draw.rectangle(
        [
            x1_px - glow_expand,
            y1_px - glow_expand,
            x2_px + glow_expand,
            y2_px + glow_expand,
        ],
        outline=(255, 0, 0, 80),
        width=6,
    )
    image = Image.alpha_composite(image, overlay)

    # --- Main bounding box ------------------------------------------------
    draw: ImageDraw.ImageDraw = ImageDraw.Draw(image)
    draw.rectangle(
        [x1_px, y1_px, x2_px, y2_px],
        outline="#FF0000",
        width=3,
    )

    # --- Label background + text ------------------------------------------
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont = _get_font()
    bbox: tuple[int, int, int, int] = font.getbbox(label)
    text_w: int = bbox[2] - bbox[0]
    text_h: int = bbox[3] - bbox[1]
    padding: int = 4
    label_bg: list[int] = [
        x1_px,
        y1_px - text_h - 2 * padding,
        x1_px + text_w + 2 * padding,
        y1_px,
    ]
    # Ensure the label stays within image bounds
    if label_bg[1] < 0:
        label_bg[1] = y1_px
        label_bg[3] = y1_px + text_h + 2 * padding

    draw.rectangle(label_bg, fill="#FF0000")
    draw.text(
        (label_bg[0] + padding, label_bg[1] + padding),
        label,
        fill="white",
        font=font,
    )

    # --- Resolve output path ----------------------------------------------
    if output_path is None:
        src: Path = Path(image_path)
        output_path = str(src.with_name(f"{src.stem}_annotated{src.suffix}"))

    image.save(output_path)
    logger.info("Annotated image saved to %s", output_path)
    return output_path


def create_comparison(
    before_path: str,
    after_path: str,
    output_path: str | None = None,
) -> str:
    """Create a side-by-side comparison image with headers.

    Parameters
    ----------
    before_path:
        Path to the "before" screenshot.
    after_path:
        Path to the "after" screenshot.
    output_path:
        Destination path for the comparison image.  When *None*, the file is
        saved as ``comparison.png`` in the same directory as *before_path*.

    Returns
    -------
    str
        Absolute path of the saved comparison image.
    """
    before: Image.Image = Image.open(before_path).convert("RGBA")
    after: Image.Image = Image.open(after_path).convert("RGBA")

    header_h: int = 40
    total_w: int = before.width + after.width
    total_h: int = max(before.height, after.height) + header_h

    canvas: Image.Image = Image.new("RGBA", (total_w, total_h), (26, 26, 46, 255))
    draw: ImageDraw.ImageDraw = ImageDraw.Draw(canvas)
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont = _get_font(size=18)

    # --- Headers ----------------------------------------------------------
    before_label: str = "\u274c Before (Defect)"
    after_label: str = "\u2705 After (Fixed)"

    before_bbox: tuple[int, int, int, int] = font.getbbox(before_label)
    after_bbox: tuple[int, int, int, int] = font.getbbox(after_label)

    before_text_w: int = before_bbox[2] - before_bbox[0]
    after_text_w: int = after_bbox[2] - after_bbox[0]

    draw.text(
        ((before.width - before_text_w) // 2, (header_h - (before_bbox[3] - before_bbox[1])) // 2),
        before_label,
        fill="white",
        font=font,
    )
    draw.text(
        (before.width + (after.width - after_text_w) // 2, (header_h - (after_bbox[3] - after_bbox[1])) // 2),
        after_label,
        fill="white",
        font=font,
    )

    # --- Paste images below header ----------------------------------------
    canvas.paste(before, (0, header_h))
    canvas.paste(after, (before.width, header_h))

    # --- Save -------------------------------------------------------------
    if output_path is None:
        output_path = str(Path(before_path).parent / "comparison.png")

    canvas.save(output_path)
    logger.info("Comparison image saved to %s", output_path)
    return output_path
