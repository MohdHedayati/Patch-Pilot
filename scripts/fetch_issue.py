#!/usr/bin/env python3
"""
fetch_issue.py - Patch-Pilot step 1: data ingestion from GitHub.

Given a public GitHub issue URL, fetches the issue's title/body and a flat
list of every file path in the repository's default branch, then prints a
single JSON object to stdout.

Usage:
    python fetch_issue.py https://github.com/owner/repo/issues/123
    python fetch_issue.py https://github.com/owner/repo/issues/123 --branch dev -o issue.json

Set GITHUB_TOKEN to raise the rate limit from 60 to 5000 requests/hour.
Without a token it still works for public repos (a run costs 3 requests,
more only for very large repos whose tree is truncated).

Exit codes: 0 ok, 1 GitHub/network error, 2 bad arguments.
"""

import argparse
import json
import os
import re
import sys
import tempfile
import time

import requests

API = "https://api.github.com"
TIMEOUT = 30          # seconds per HTTP request
MAX_RETRIES = 4       # for 5xx / network errors / secondary rate limits
MAX_RATE_WAIT = 90    # longest we'll sleep waiting for a rate-limit reset

ISSUE_URL_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?github\.com/"
    r"(?P<owner>[A-Za-z0-9-_.]+)/(?P<repo>[A-Za-z0-9-_.]+)/"
    r"(?:issues|pull)/(?P<number>\d+)"
    r"(?:[/?#].*)?$"
)


class GitHubError(Exception):
    pass


def log(msg):
    print(f"[fetch_issue] {msg}", file=sys.stderr)


def parse_issue_url(url):
    m = ISSUE_URL_RE.match(url.strip())
    if not m:
        raise ValueError(
            f"Not a GitHub issue URL: {url!r} "
            "(expected https://github.com/<owner>/<repo>/issues/<number>)"
        )
    repo = m["repo"]
    if repo.endswith(".git"):
        repo = repo[:-4]
    return m["owner"], repo, int(m["number"])


class GitHubClient:
    def __init__(self, token=None):
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "patch-pilot-fetch-issue",
        })
        self.token = token
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"

    def _rate_limit_wait(self, resp):
        """Seconds to wait if resp is a rate-limit rejection, else None."""
        if resp.status_code not in (403, 429):
            return None
        retry_after = resp.headers.get("Retry-After")
        if retry_after and retry_after.isdigit():
            return int(retry_after)
        if resp.headers.get("X-RateLimit-Remaining") == "0":
            reset = int(resp.headers.get("X-RateLimit-Reset", "0"))
            return max(reset - int(time.time()), 0) + 1
        if "rate limit" in resp.text.lower():  # secondary limit, no headers
            return 60
        return None

    def get(self, path, params=None):
        url = f"{API}{path}"
        attempt = 0
        while True:
            attempt += 1
            try:
                resp = self.session.get(url, params=params, timeout=TIMEOUT)
            except (requests.ConnectionError, requests.Timeout) as e:
                if attempt > MAX_RETRIES:
                    raise GitHubError(f"Network error calling {url}: {e}") from e
                delay = 2 ** attempt
                log(f"Network error ({e.__class__.__name__}), retrying in {delay}s...")
                time.sleep(delay)
                continue

            if resp.status_code == 200:
                return resp.json()

            # Bad/expired token: fall back to anonymous access for public repos.
            if resp.status_code == 401 and self.token:
                log("GITHUB_TOKEN was rejected (401); retrying without it.")
                self.session.headers.pop("Authorization", None)
                self.token = None
                attempt -= 1
                continue

            wait = self._rate_limit_wait(resp)
            if wait is not None:
                hint = "" if self.token else " Set GITHUB_TOKEN to raise the limit."
                if wait > MAX_RATE_WAIT or attempt > MAX_RETRIES:
                    raise GitHubError(
                        f"GitHub rate limit exceeded; resets in ~{wait}s.{hint}"
                    )
                log(f"Rate limited, waiting {wait}s...{hint}")
                time.sleep(wait)
                continue

            if resp.status_code >= 500 and attempt <= MAX_RETRIES:
                delay = 2 ** attempt
                log(f"GitHub returned {resp.status_code}, retrying in {delay}s...")
                time.sleep(delay)
                continue

            try:
                detail = resp.json().get("message", resp.text)
            except ValueError:
                detail = resp.text
            if resp.status_code == 404:
                raise GitHubError(
                    f"Not found: {path} (repo/issue doesn't exist, or it's private "
                    "and GITHUB_TOKEN is missing/lacks access)"
                )
            raise GitHubError(f"GitHub API {resp.status_code} on {path}: {detail}")


def extract_image_urls(text: str) -> list[str]:
    """Extract image URLs from markdown and HTML image tags.

    Parses ``![alt](url)``, ``<img src="url">``, direct markdown links, and
    modern GitHub attachment URLs (including extensionless
    ``github.com/user-attachments/assets/`` UUIDs). Results are deduplicated
    while preserving order.
    """
    _IMAGE_EXTENSIONS = re.compile(
        r"\.(png|jpe?g|gif|webp|svg|bmp|tiff?)(\?.*)?$", re.IGNORECASE
    )
    _GITHUB_ATTACHMENT_DOMAINS = re.compile(
        r"https?://(?:user-images\.githubusercontent\.com|"
        r"private-user-images\.githubusercontent\.com|"
        r"github\.com/(?:[^/\s]+/){1,2}assets/|"
        r"github\.com/user-attachments/assets/)",
        re.IGNORECASE,
    )

    # Markdown image syntax: ![alt](url) or ![alt](<url> "title")
    md_pattern = re.compile(r"!\[[^\]]*\]\((<?[^\s>)]+>?)(?:\s+[\"'][^\"']*[\"'])?\)")
    # HTML <img> tag: <img ... src="url" ...>
    html_pattern = re.compile(r"<img\s[^>]*?src=[\"']([^\"']+)[\"']", re.IGNORECASE)
    # Markdown link syntax: [alt](url)
    link_pattern = re.compile(r"\[[^\]]*\]\((<?[^\s>)]+>?)(?:\s+[\"'][^\"']*[\"'])?\)")

    candidates: list[str] = []
    for match in md_pattern.finditer(text):
        url = match.group(1).strip("<>").strip()
        candidates.append(url)
    for match in html_pattern.finditer(text):
        url = match.group(1).strip("<>").strip()
        candidates.append(url)
    for match in link_pattern.finditer(text):
        url = match.group(1).strip("<>").strip()
        candidates.append(url)

    filtered: list[str] = []
    for url in candidates:
        if _IMAGE_EXTENSIONS.search(url) or _GITHUB_ATTACHMENT_DOMAINS.search(url):
            if url not in filtered:
                filtered.append(url)
    return filtered


def download_image(
    url_or_path: str,
    dest_dir: str | None = None,
    token: str | None = None,
) -> bytes | str | None:
    """Download an image from a URL or read bytes from a local file.

    Parameters
    ----------
    url_or_path:
        The remote image URL to download or local file path to read.
    dest_dir:
        Directory to save the file in when downloading from a URL.
        A temporary directory is created when *None*.
    token:
        Optional GitHub personal-access token used for authenticated
        requests (useful for private-repo attachments).

    Returns
    -------
    bytes | str | None
        Raw bytes if *url_or_path* is an existing local file, absolute filepath
        string if downloaded from a URL, or ``None`` on failure.
    """
    import logging

    logger = logging.getLogger(__name__)

    # If it is a local file path, read raw bytes directly from disk
    if os.path.exists(url_or_path):
        try:
            with open(url_or_path, "rb") as fh:
                return fh.read()
        except Exception:
            logger.warning("Failed to read local image file: %s", url_or_path, exc_info=True)
            return None

    # If it starts with http:// or https://, proceed with existing requests.get() logic
    if url_or_path.startswith(("http://", "https://")):
        if dest_dir is None:
            dest_dir = tempfile.mkdtemp(prefix="patchpilot_img_")

        headers: dict[str, str] = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        try:
            resp = requests.get(url_or_path, headers=headers, timeout=TIMEOUT, stream=True)
            resp.raise_for_status()

            # Derive a filename from the URL, ensuring an image extension for Pillow compatibility
            filename = os.path.basename(url_or_path.split("?")[0]) or "image"
            if not os.path.splitext(filename)[1]:
                filename = f"{filename}.png"
            filepath = os.path.join(dest_dir, filename)

            with open(filepath, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=8192):
                    fh.write(chunk)

            logger.debug("Downloaded image %s -> %s", url_or_path, filepath)
            return filepath
        except Exception:
            logger.warning("Failed to download image from %s", url_or_path, exc_info=True)
            return None

    logger.warning("Invalid image URL or path: %s", url_or_path)
    return None


def fetch_issue(gh, owner, repo, number):
    data = gh.get(f"/repos/{owner}/{repo}/issues/{number}")
    body = data.get("body") or ""
    image_urls = extract_image_urls(body)
    # Also check first page of comments for images
    try:
        comments = gh.get(f"/repos/{owner}/{repo}/issues/{number}/comments", params={"per_page": "10"})
        for comment in comments:
            comment_body = comment.get("body") or ""
            image_urls.extend(extract_image_urls(comment_body))
        # Deduplicate while preserving order
        image_urls = list(dict.fromkeys(image_urls))
    except Exception:
        pass
    return {
        "number": data["number"],
        "title": data.get("title") or "",
        "body": body,
        "state": data.get("state"),
        "labels": [lbl["name"] if isinstance(lbl, dict) else lbl
                   for lbl in data.get("labels", [])],
        "author": (data.get("user") or {}).get("login"),
        "url": data.get("html_url"),
        "is_pull_request": "pull_request" in data,
        "image_urls": image_urls,
    }


def fetch_file_content(gh, owner, repo, path, branch="main"):
    """Fetch raw text content of a specific file from GitHub."""
    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
    headers = {}
    if gh.token:
        headers["Authorization"] = f"Bearer {gh.token}"
    try:
        resp = gh.session.get(raw_url, headers=headers, timeout=TIMEOUT)
        if resp.status_code == 200:
            return resp.text
    except Exception:
        pass
    try:
        data = gh.get(f"/repos/{owner}/{repo}/contents/{path}", params={"ref": branch})
        if isinstance(data, dict) and "content" in data:
            import base64
            return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
    except Exception:
        pass
    return None



def fetch_file_paths(gh, owner, repo, ref):
    """Return (sorted file paths, truncated flag) for every blob under ref."""
    data = gh.get(f"/repos/{owner}/{repo}/git/trees/{ref}", params={"recursive": "1"})
    if not data.get("truncated"):
        return sorted(e["path"] for e in data["tree"] if e["type"] == "blob"), False

    # The recursive listing caps out (~100k entries / 7MB) on huge repos.
    # Fall back to walking subtrees: each directory is fetched recursively,
    # descending further only where that listing is itself truncated.
    log("Tree listing truncated by GitHub; walking subtrees (more API calls)...")
    paths, still_truncated = [], False
    root = gh.get(f"/repos/{owner}/{repo}/git/trees/{ref}")
    stack = [("", e) for e in root["tree"]]
    while stack:
        prefix, entry = stack.pop()
        full = f"{prefix}{entry['path']}"
        if entry["type"] == "blob":
            paths.append(full)
        elif entry["type"] == "tree":
            sub = gh.get(f"/repos/{owner}/{repo}/git/trees/{entry['sha']}",
                         params={"recursive": "1"})
            if not sub.get("truncated"):
                paths.extend(f"{full}/{e['path']}" for e in sub["tree"] if e["type"] == "blob")
            else:
                flat = gh.get(f"/repos/{owner}/{repo}/git/trees/{entry['sha']}")
                if flat.get("truncated"):
                    still_truncated = True
                stack.extend((f"{full}/", e) for e in flat["tree"])
        # type == "commit" is a submodule; it has no files in this repo.
    return sorted(paths), still_truncated


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Fetch a GitHub issue and its repo's file list as JSON."
    )
    parser.add_argument("issue_url", help="e.g. https://github.com/owner/repo/issues/123")
    parser.add_argument("--branch", help="branch/ref to list files from "
                        "(default: the repo's default branch)")
    parser.add_argument("-o", "--output", help="write JSON to this file instead of stdout")
    parser.add_argument("--compact", action="store_true", help="no indentation in output")
    args = parser.parse_args(argv)

    try:
        owner, repo, number = parse_issue_url(args.issue_url)
    except ValueError as e:
        log(str(e))
        return 2

    gh = GitHubClient(os.environ.get("GITHUB_TOKEN") or None)
    try:
        repo_info = gh.get(f"/repos/{owner}/{repo}")
        # GitHub redirects renamed repos; use the canonical name from here on.
        owner, repo = repo_info["owner"]["login"], repo_info["name"]
        branch = args.branch or repo_info.get("default_branch") or "main"

        issue = fetch_issue(gh, owner, repo, number)
        files, truncated = fetch_file_paths(gh, owner, repo, branch)
    except GitHubError as e:
        log(str(e))
        return 1

    if truncated:
        log("WARNING: some directories were still truncated; file list is incomplete.")

    result = {
        "repo": {
            "owner": owner,
            "name": repo,
            "full_name": f"{owner}/{repo}",
            "branch": branch,
            "clone_url": repo_info.get("clone_url"),
            "language": repo_info.get("language"),
        },
        "issue": issue,
        "files": files,
        "file_count": len(files),
        "files_truncated": truncated,
    }

    text = json.dumps(result, indent=None if args.compact else 2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        log(f"Wrote {args.output} ({len(files)} files)")
    else:
        # Windows consoles default to cp1252; force UTF-8 so emoji in issues don't crash.
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
