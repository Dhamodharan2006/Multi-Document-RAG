"""Groq LLM client wrapper with model routing, retry, and logging."""

from __future__ import annotations

import asyncio
import json
import time

from groq import Groq
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings

# Module-level singleton
_llm_client: LLMClient | None = None


class LLMClient:
    """Wrapper around the Groq Python SDK for LLM generation.

    Provides text generation, JSON generation with parsing/retry,
    and model routing based on question complexity.
    """

    def __init__(self) -> None:
        """Initialize the Groq client with the configured API key."""
        self.client = Groq(api_key=settings.groq_api_key)
        self.primary_model = settings.primary_model
        self.reasoning_model = settings.reasoning_model
        logger.info(
            "Groq LLM client initialized (primary={}, reasoning={})",
            self.primary_model,
            self.reasoning_model,
        )

    def route_model(self, question_type: str) -> str:
        """Select the appropriate model based on question complexity.

        Args:
            question_type: Either 'reasoning', 'compare', or 'standard'.

        Returns:
            The model name string to use.
        """
        if question_type in ("reasoning", "compare"):
            logger.debug("Routing to reasoning model: {}", self.reasoning_model)
            return self.reasoning_model
        logger.debug("Routing to primary model: {}", self.primary_model)
        return self.primary_model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        reraise=True,
    )
    def _generate_sync(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system_message: str | None = None,
    ) -> tuple[str, dict]:
        """Synchronous generation call to Groq API with retry.

        Args:
            prompt: The user prompt text.
            model: Model name to use.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            system_message: Optional system message.

        Returns:
            Tuple of (response_text, usage_dict).
        """
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        latency_ms = int((time.perf_counter() - start) * 1000)

        text = response.choices[0].message.content or ""
        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0,
        }

        logger.info(
            "LLM call: model={}, latency={}ms, tokens={}",
            model,
            latency_ms,
            usage.get("total_tokens", 0),
        )
        return text, usage

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system_message: str | None = None,
    ) -> str:
        """Generate text asynchronously using Groq API.

        Args:
            prompt: The user prompt text.
            model: Model name to use. Defaults to primary model.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            system_message: Optional system message.

        Returns:
            Generated text response string.
        """
        model = model or self.primary_model
        text, _ = await asyncio.to_thread(
            self._generate_sync,
            prompt,
            model,
            temperature,
            max_tokens,
            system_message,
        )
        return text

    async def generate_json(
        self,
        prompt: str,
        model: str | None = None,
    ) -> dict:
        """Generate a JSON response and parse it.

        Retries up to 2 times if JSON parsing fails, instructing
        the model to fix its output.

        Args:
            prompt: The prompt requesting JSON output.
            model: Model name to use. Defaults to primary model.

        Returns:
            Parsed JSON as a Python dict.

        Raises:
            ValueError: If JSON parsing fails after all retries.
        """
        model = model or self.primary_model

        for attempt in range(3):
            text = await self.generate(
                prompt=prompt,
                model=model,
                temperature=0.1,
                max_tokens=1024,
            )

            try:
                # Try to extract JSON from the response
                parsed = _extract_json(text)
                return parsed
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(
                    "JSON parse failed (attempt {}/3): {}",
                    attempt + 1,
                    str(e)[:100],
                )
                if attempt < 2:
                    prompt = (
                        f"Your previous response was not valid JSON. "
                        f"Error: {str(e)}\n\n"
                        f"Original request: {prompt}\n\n"
                        f"Please respond with ONLY valid JSON, no other text."
                    )

        raise ValueError(f"Failed to get valid JSON after 3 attempts from {model}")

    def check_connection(self) -> bool:
        """Check if the Groq API is reachable.

        Returns:
            True if a test call succeeds, False otherwise.
        """
        try:
            self.client.chat.completions.create(
                model=self.primary_model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            return True
        except Exception:
            return False


def _extract_json(text: str) -> dict:
    """Extract and parse JSON from a text response.

    Handles cases where the model wraps JSON in markdown code blocks
    or includes extra text.

    Args:
        text: Raw text response that should contain JSON.

    Returns:
        Parsed JSON dictionary.

    Raises:
        ValueError: If no valid JSON can be extracted.
    """
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code blocks
    import re

    json_patterns = [
        r"```json\s*\n?(.*?)\n?\s*```",
        r"```\s*\n?(.*?)\n?\s*```",
        r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",
    ]

    for pattern in json_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1) if match.lastindex else match.group(0))
            except (json.JSONDecodeError, IndexError):
                continue

    raise ValueError(f"Could not extract valid JSON from response: {text[:200]}")


def get_llm_client() -> LLMClient:
    """Get or create the singleton LLMClient instance.

    Returns:
        The shared LLMClient instance.
    """
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
