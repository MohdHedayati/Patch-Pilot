"""Playwright-based screenshot capture for visual patch verification.

This module provides utilities to capture 'after' screenshots using a headless
Chromium browser, enabling automated visual regression checks after applying
patches.
"""

from __future__ import annotations

import logging
import subprocess
import time
from typing import Optional

logger: logging.Logger = logging.getLogger(__name__)


def _ensure_playwright_available() -> None:
    """Verify that Playwright is installed and importable.

    Raises:
        RuntimeError: If ``playwright`` is not installed, with instructions
            on how to install it.
    """
    try:
        import playwright.sync_api  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Playwright is not installed. "
            "Install it with:\n"
            "  pip install playwright\n"
            "  python -m playwright install chromium"
        ) from exc


def capture_screenshot(
    url: str,
    output_path: str = "after_fixed.png",
    viewport_width: int = 1280,
    viewport_height: int = 720,
    wait_for_selector: Optional[str] = None,
    wait_timeout: int = 10000,
    full_page: bool = False,
) -> str:
    """Capture a screenshot of a web page using headless Chromium.

    Args:
        url: The URL to navigate to and capture.
        output_path: File path where the screenshot PNG will be saved.
        viewport_width: Browser viewport width in pixels.
        viewport_height: Browser viewport height in pixels.
        wait_for_selector: Optional CSS selector to wait for before capturing.
            When provided, the function waits until this element appears in the
            DOM.  When ``None``, the function waits for the ``networkidle``
            load state instead.
        wait_timeout: Maximum time in milliseconds to wait for the selector
            or load state.
        full_page: If ``True``, capture the full scrollable page rather than
            just the viewport.

    Returns:
        The *output_path* where the screenshot was saved.

    Raises:
        RuntimeError: If Playwright is not installed.
    """
    _ensure_playwright_available()

    from playwright.sync_api import sync_playwright

    logger.info("Launching headless Chromium for screenshot capture.")

    browser = None
    playwright_ctx = sync_playwright()
    playwright_instance = playwright_ctx.start()

    try:
        browser = playwright_instance.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": viewport_width, "height": viewport_height},
        )
        page = context.new_page()

        logger.info("Navigating to %s", url)
        page.goto(url)

        if wait_for_selector is not None:
            logger.info(
                "Waiting for selector '%s' (timeout=%dms)",
                wait_for_selector,
                wait_timeout,
            )
            page.wait_for_selector(wait_for_selector, timeout=wait_timeout)
        else:
            logger.info(
                "Waiting for networkidle load state (timeout=%dms)",
                wait_timeout,
            )
            page.wait_for_load_state("networkidle", timeout=wait_timeout)

        page.screenshot(path=output_path, full_page=full_page)
        logger.info("Screenshot saved to %s", output_path)

    finally:
        if browser is not None:
            browser.close()
            logger.debug("Browser closed.")
        playwright_instance.stop()
        logger.debug("Playwright instance stopped.")

    return output_path


def capture_after_patch(
    server_command: str,
    url: str,
    output_path: str = "after_fixed.png",
    startup_wait: float = 3.0,
    viewport_width: int = 1280,
    viewport_height: int = 720,
    wait_for_selector: Optional[str] = None,
) -> str:
    """Start a dev server, capture a screenshot, then tear the server down.

    This is a convenience wrapper around :func:`capture_screenshot` that
    manages the lifecycle of a local development server so you can capture
    the state of the application *after* a patch has been applied.

    Args:
        server_command: Shell command to start the server (e.g.
            ``"npm start"``).
        url: The URL to capture once the server is ready.
        output_path: File path where the screenshot PNG will be saved.
        startup_wait: Seconds to wait after launching the server before
            attempting to capture the screenshot.
        viewport_width: Browser viewport width in pixels.
        viewport_height: Browser viewport height in pixels.
        wait_for_selector: Optional CSS selector to wait for before
            capturing.  Passed through to :func:`capture_screenshot`.

    Returns:
        The *output_path* where the screenshot was saved.

    Raises:
        RuntimeError: If Playwright is not installed.
    """
    _ensure_playwright_available()

    logger.info("Starting server with command: %s", server_command)

    server_process: subprocess.Popen[bytes] = subprocess.Popen(
        server_command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        logger.info(
            "Waiting %.1f seconds for the server to start…", startup_wait
        )
        time.sleep(startup_wait)

        result: str = capture_screenshot(
            url=url,
            output_path=output_path,
            viewport_width=viewport_width,
            viewport_height=viewport_height,
            wait_for_selector=wait_for_selector,
        )
    finally:
        logger.info("Terminating server process (pid=%d).", server_process.pid)
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning(
                "Server did not terminate gracefully; killing pid=%d.",
                server_process.pid,
            )
            server_process.kill()

    return result
