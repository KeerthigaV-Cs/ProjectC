"""Hosted Gemini image-generation adapter."""

import base64

import httpx
from google import genai
from google.genai import errors
from pydantic import ValidationError

from comiccraft.errors import (
    ImageProviderError,
    ImageProviderRateLimitError,
    ImageProviderServiceUnavailableError,
    ImageProviderTimeoutError,
    InvalidImageResponseError,
    MissingImageApiKeyError,
)
from comiccraft.models.images import GeneratedImageData


class GeminiImageGenerationProvider:
    """Generate one hosted image using the configurable Gemini image model."""

    def __init__(self, api_key: str | None, model: str, timeout_seconds: float = 90) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout_seconds = timeout_seconds

    def generate_image(self, prompt: str) -> GeneratedImageData:
        if not self._api_key:
            raise MissingImageApiKeyError(
                "Set IMAGE_API_KEY in your environment or .env file to generate images."
            )

        try:
            with genai.Client(api_key=self._api_key) as client:
                interaction = client.interactions.create(
                    model=self._model,
                    input=prompt,
                    response_format={
                        "type": "image",
                        "mime_type": "image/jpeg",
                        "aspect_ratio": "3:2",
                        "image_size": "1K",
                    },
                    timeout=self._timeout_seconds,
                )
        except errors.APIError as exc:
            if exc.code == 429:
                raise ImageProviderRateLimitError(
                    "Image provider rate limit reached. Please try again shortly."
                ) from exc
            if isinstance(exc.code, int) and 500 <= exc.code <= 599:
                raise ImageProviderServiceUnavailableError(
                    "The image provider is temporarily unavailable."
                ) from exc
            raise ImageProviderError(
                f"Image provider request failed with status {exc.code}."
            ) from exc
        except httpx.TimeoutException as exc:
            raise ImageProviderTimeoutError(
                "Image provider request timed out. Please try again."
            ) from exc
        except httpx.RequestError as exc:
            raise ImageProviderTimeoutError(
                "The image provider request could not reach the service."
            ) from exc
        except Exception as exc:
            if (
                any(base.__name__ == "RateLimitError" for base in type(exc).__mro__)
                or getattr(exc, "code", None) == 429
                or getattr(exc, "status_code", None) == 429
            ):
                raise ImageProviderRateLimitError(
                    "Image provider rate limit reached. Please try again shortly."
                ) from exc
            raise ImageProviderError("The image provider request could not be completed.") from exc

        try:
            output_image = interaction.output_image
            if output_image is None:
                raise ValueError("The image provider returned no encoded image data.")
            encoded_data = output_image.data
            if isinstance(encoded_data, bytes):
                image_bytes = encoded_data
            elif isinstance(encoded_data, str):
                image_bytes = base64.b64decode(encoded_data, validate=True)
            else:
                raise ValueError("The image provider returned no encoded image data.")
            return GeneratedImageData(
                data=image_bytes,
                mime_type=output_image.mime_type or "image/jpeg",
            )
        except (ValidationError, ValueError, TypeError, AttributeError) as exc:
            raise InvalidImageResponseError(
                "The image provider returned an invalid image response."
            ) from exc