"""Gemini story-provider contract and Google Gen AI adapter."""

from google import genai
from google.genai import errors, types
from pydantic import ValidationError

from comiccraft.errors import (
    GeminiConfigurationError,
    GeminiProviderError,
    GeminiRateLimitError,
    GeminiServiceUnavailableError,
    InvalidGeminiResponseError,
    MissingGeminiApiKeyError,
)
from comiccraft.models.story import ComicInput, GeneratedStoryContent
from comiccraft.providers.contracts import StoryProvider

SDK_MAX_ATTEMPTS = 4
SDK_RETRY_INITIAL_DELAY_SECONDS = 0.5
SDK_RETRY_MAX_DELAY_SECONDS = 2.0


class GoogleGeminiStoryProvider:
    """Generate structured comic story content using Google's Gen AI SDK."""

    def __init__(self, api_key: str | None, model: str) -> None:
        self._api_key = api_key
        self._model = model

    def generate_story(self, user_input: ComicInput) -> GeneratedStoryContent:
        if not self._api_key:
            raise MissingGeminiApiKeyError(
                "Set GEMINI_API_KEY in your environment or .env file to generate stories."
            )

        prompt = self._build_prompt(user_input)
        try:
            with genai.Client(
                api_key=self._api_key,
                http_options=types.HttpOptions(
                    retry_options=types.HttpRetryOptions(
                        attempts=SDK_MAX_ATTEMPTS,
                        initial_delay=SDK_RETRY_INITIAL_DELAY_SECONDS,
                        max_delay=SDK_RETRY_MAX_DELAY_SECONDS,
                        exp_base=2.0,
                        http_status_codes=list(range(500, 600)),
                    )
                ),
            ) as client:
                response = client.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=GeneratedStoryContent,
                        temperature=0.8,
                    ),
                )
        except errors.APIError as exc:
            message = str(exc).lower()
            if exc.code in (401, 403) or (
                exc.code == 400
                and any(marker in message for marker in ("api_key_invalid", "api key not valid", "invalid api key"))
            ):
                raise GeminiConfigurationError(
                    "The Gemini API key was rejected."
                ) from exc
            if exc.code == 429:
                raise GeminiRateLimitError(
                    "Gemini rate limit reached. Please try again shortly."
                ) from exc
            if isinstance(exc.code, int) and 500 <= exc.code <= 599:
                raise GeminiServiceUnavailableError(
                    "Gemini AI is currently experiencing high demand. Please try again in a few moments to generate your comic."
                ) from exc
            raise GeminiProviderError(
                f"Gemini API request failed with status {exc.code}."
            ) from exc
        except (GeminiRateLimitError, GeminiProviderError):
            raise
        except Exception as exc:
            raise GeminiProviderError("The Gemini request could not be completed.") from exc

        try:
            parsed_response = response.parsed
            if parsed_response is None:
                response_text = response.text
                if not response_text:
                    raise ValueError("Gemini returned an empty response.")
                return GeneratedStoryContent.model_validate_json(response_text)
            return GeneratedStoryContent.model_validate(parsed_response)
        except (ValidationError, ValueError, TypeError, AttributeError) as exc:
            raise InvalidGeminiResponseError(
                "Gemini returned content that does not match the required five-panel story format."
            ) from exc

    @staticmethod
    def _build_prompt(user_input: ComicInput) -> str:
        return (
            "Create a cohesive comic story as structured JSON. The response schema "
            "has separate outline and narration sections. Include exactly five panels, "
            "numbered 1 through 5 in order, in both sections. Give each outline panel "
            "a concise title, scene description, and detailed image-generation prompt. "
            "For each matching panel, write narration and any character dialogue. "
            "Keep character, setting, tone, and art style consistent.\n\n"
            f"User comic input:\n{user_input.model_dump_json(indent=2)}"
        )