from __future__ import annotations

import os
from typing import Any


class OpenAIClientError(RuntimeError):
    """Raised when an OpenAI API request cannot be completed."""


class OpenAITextClient:
    """
    Minimal wrapper around the OpenAI Responses API.

    Keeping the SDK behind this wrapper allows tests to use a fake client
    without making real API requests.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = (
            model
            or os.getenv("OPENAI_MODEL")
            or "gpt-4o"
        )

        if client is not None:
            self._client = client
            return

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise OpenAIClientError(
                "The OpenAI SDK is not installed. "
                'Run: pip install -e ".[dev]"'
            ) from exc

        try:
            if api_key:
                self._client = OpenAI(
                    api_key=api_key
                )
            else:
                # The SDK reads OPENAI_API_KEY from the environment.
                self._client = OpenAI()
        except Exception as exc:
            raise OpenAIClientError(
                f"Failed to initialize OpenAI client: {exc}"
            ) from exc

    def generate_text(
        self,
        *,
        instructions: str,
        input_text: str,
    ) -> str:
        """Generate one plain-text response."""

        try:
            response = self._client.responses.create(
                model=self.model,
                instructions=instructions,
                input=input_text,
            )
        except Exception as exc:
            raise OpenAIClientError(
                f"OpenAI request failed for model "
                f"{self.model!r}: {exc}"
            ) from exc

        output_text = getattr(
            response,
            "output_text",
            None,
        )

        if not isinstance(output_text, str):
            raise OpenAIClientError(
                "OpenAI response did not contain output_text"
            )

        output_text = output_text.strip()

        if not output_text:
            raise OpenAIClientError(
                "OpenAI response contained empty output_text"
            )

        return output_text