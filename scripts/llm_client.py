"""
llm_client.py - Provider-agnostic LLM interface for Patch-Pilot.

Supports:
1. Local open-weight models via Ollama (default: http://localhost:11434)
2. OpenAI-compatible local/remote servers (vLLM, LM Studio, etc.)
3. Google Gemini API (optional fallback via GEMINI_API_KEY)
4. Offline Mock provider (for reliable offline demos and automated evaluation)
"""

import json
import os
import sys
import requests

DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_OPENAI_BASE = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
DEFAULT_MODEL = os.environ.get("PATCH_PILOT_MODEL", "qwen2.5-coder:7b")


class LLMClient:
    def __init__(self, provider=None, model=None, api_key=None, base_url=None,
                 timeout=int(os.environ.get("PATCH_PILOT_TIMEOUT", "300"))):
        self.provider = provider or os.environ.get("PATCH_PILOT_PROVIDER") or self._detect_provider()
        self.model = model or DEFAULT_MODEL
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or (DEFAULT_OPENAI_BASE if self.provider == "openai" else DEFAULT_OLLAMA_URL)
        self.timeout = timeout

    def _detect_provider(self):
        """Auto-detect the available model provider, prioritizing open-weight local providers."""
        # Check if Ollama is running locally
        try:
            resp = requests.get(f"{DEFAULT_OLLAMA_URL}/api/tags", timeout=1.5)
            if resp.status_code == 200:
                return "ollama"
        except Exception:
            pass

        # Check if OpenAI compatible base URL or key is provided
        if os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_KEY"):
            return "openai"

        # Check Gemini API key
        if os.environ.get("GEMINI_API_KEY"):
            return "gemini"

        # Fallback to ollama as the open-source standard
        return "ollama"

    def complete(self, prompt, system_prompt=None, temperature=0.2):
        """Send a prompt to the configured model provider and return the response text."""
        if self.provider == "ollama":
            return self._complete_ollama(prompt, system_prompt, temperature)
        elif self.provider == "openai":
            return self._complete_openai(prompt, system_prompt, temperature)
        elif self.provider == "gemini":
            return self._complete_gemini(prompt, system_prompt, temperature)
        elif self.provider == "mock":
            return self._complete_mock(prompt, system_prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _complete_ollama(self, prompt, system_prompt, temperature):
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt or "",
            "stream": False,
            "options": {
                "temperature": temperature,
            }
        }
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except requests.ConnectionError:
            raise ConnectionError(
                f"Could not connect to Ollama at {self.base_url}. "
                "Ensure Ollama is running (`ollama serve`) or specify --provider gemini / --provider mock."
            )
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}")

    def _complete_openai(self, prompt, system_prompt, temperature):
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise RuntimeError(f"OpenAI-compatible request failed: {e}")

    def _complete_gemini(self, prompt, system_prompt, temperature):
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required for Gemini provider.")

        model_name = self.model if "gemini" in self.model else "gemini-1.5-flash"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.api_key}"
        
        contents = []
        full_text = f"System: {system_prompt}\n\nUser: {prompt}" if system_prompt else prompt
        contents.append({"parts": [{"text": full_text}]})

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
            }
        }
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            candidates = data.get("candidates", [])
            if candidates:
                return candidates[0]["content"]["parts"][0]["text"].strip()
            return ""
        except Exception as e:
            raise RuntimeError(f"Gemini API request failed: {e}")

    def _complete_mock(self, prompt, system_prompt):
        """Mock provider returning clean, deterministic outputs for testing & demo fallback."""
        if "Output ONLY a raw, valid JSON array" in prompt or "Identify the 1 to 3 files" in prompt:
            # Triage step
            return json.dumps(["src/parser.py", "tests/test_parser.py"])
        
        # Patch step
        return (
            "### Reproduction Test (TDD)\n"
            "```python\n"
            "def test_parse_empty_input():\n"
            "    from src.parser import parse_input\n"
            "    result = parse_input('')\n"
            "    assert result == {}, f'Expected empty dict, got {result}'\n"
            "```\n\n"
            "### Surgical Patch\n"
            "```diff\n"
            "--- a/src/parser.py\n"
            "+++ b/src/parser.py\n"
            "@@ -4,3 +4,5 @@\n"
            " def parse_input(data):\n"
            "     \"\"\"Parses JSON input data.\"\"\"\n"
            "+    if not data or not data.strip():\n"
            "+        return {}\n"
            "     return json.loads(data)\n"
            "```\n\n"
            "### Maintainer Summary\n"
            "- **Root Cause:** Function threw JSONDecodeError on empty string instead of returning empty mapping.\n"
            "- **Fix:** Guard against empty/whitespace-only input before parsing.\n"
            "- **Safety:** Zero external dependencies added, scope strictly limited to 1 file."
        )

