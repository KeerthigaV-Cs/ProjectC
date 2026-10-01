"""Hugging Face image-generation adapter."""

from io import BytesIO
import logging
import re

import httpx
from huggingface_hub import InferenceClient

from comiccraft.errors import (
    ImageProviderCreditsExhaustedError,
    ImageProviderError,
    ImageProviderRateLimitError,
    ImageProviderServiceUnavailableError,
    ImageProviderTimeoutError,
    InvalidImageResponseError,
    MissingHFTokenError,
)
from comiccraft.models.images import GeneratedImageData

logger = logging.getLogger(__name__)


class HuggingFaceImageGenerationProvider:
    """Generate one hosted image using a configurable Hugging Face model."""

    def __init__(
        self,
        token: str | None,
        model: str | None,
        client_factory: type[InferenceClient] | None = None,
    ) -> None:
        self._token = token
        self._model = model
        self._client_factory = client_factory or InferenceClient

    def generate_image(self, prompt: str) -> GeneratedImageData:
        if not self._token:
            raise MissingHFTokenError(
                "Set HF_TOKEN in your environment or .env file to generate images with Hugging Face."
            )
        if not self._model:
            raise ImageProviderError(
                "Set HF_IMAGE_MODEL in your environment or .env file to use Hugging Face for image generation."
            )

        try:
            client = self._client_factory(token=self._token)
            output = client.text_to_image(prompt, model=self._model)
            image_bytes = self._coerce_image_bytes(output)
            if not image_bytes:
                raise InvalidImageResponseError(
                    "The Hugging Face image provider returned an empty image."
                )
            mime_type = self._detect_mime_type(image_bytes)
            return GeneratedImageData(data=image_bytes, mime_type=mime_type)
        except InvalidImageResponseError:
            raise
        except Exception as exc:
            status_code = self._status_code(exc)
            safe_message = self._safe_error_message(str(exc))
            logger.error(
                "Hugging Face image generation failed; provider=huggingface; "
                "model=%s; exception_type=%s; status_code=%s; error=%s",
                self._model,
                type(exc).__name__,
                status_code,
                safe_message,
            )
            if self._is_timeout(exc):
                raise ImageProviderTimeoutError(
                    "The Hugging Face image provider request timed out."
                ) from exc
            if isinstance(exc, httpx.RequestError):
                raise ImageProviderTimeoutError(
                    "The Hugging Face image provider request could not reach the service."
                ) from exc
            if status_code in (401, 403):
                raise MissingHFTokenError(
                    "The Hugging Face access token was rejected."
                ) from exc
            if status_code == 402:
                raise ImageProviderCreditsExhaustedError(
                    "The Hugging Face image-generation credits are exhausted."
                ) from exc
            if status_code == 429 or self._is_rate_limit(exc):
                raise ImageProviderRateLimitError(
                    "The Hugging Face image provider rate limit was reached."
                ) from exc
            if status_code is not None and 500 <= status_code <= 599:
                raise ImageProviderServiceUnavailableError(
                    "The Hugging Face image service is temporarily unavailable."
                ) from exc
            raise ImageProviderError(
                "The Hugging Face image provider request could not be completed."
            ) from exc

    @staticmethod
    def _status_code(error: Exception) -> int | None:
        pending: list[BaseException] = [error]
        visited: set[int] = set()
        while pending:
            current = pending.pop()
            if id(current) in visited:
                continue
            visited.add(id(current))
            response = getattr(current, "response", None)
            for candidate in (
                getattr(response, "status_code", None),
                getattr(current, "status_code", None),
                getattr(current, "code", None),
            ):
                if isinstance(candidate, int) and 100 <= candidate <= 599:
                    return candidate
            for chained_error in (current.__cause__, current.__context__):
                if chained_error is not None:
                    pending.append(chained_error)
        return None

    @staticmethod
    def _is_timeout(error: Exception) -> bool:
        return isinstance(error, httpx.TimeoutException) or any(
            "Timeout" in base.__name__ for base in type(error).__mro__
        )

    @staticmethod
    def _is_rate_limit(error: Exception) -> bool:
        return any("RateLimit" in base.__name__ for base in type(error).__mro__)

    def _safe_error_message(self, message: str) -> str:
        if self._token:
            message = message.replace(self._token, "<redacted>")
        message = re.sub(
            r"(?i)(authorization\s*[:=]\s*)(?:bearer\s+)?[^\s,;]+",
            r"\1<redacted>",
            message,
        )
        return re.sub(
            r"(?i)(api[_-]?key|access[_-]?token|token|secret|password)\s*[:=]\s*[^\s,;]+",
            r"\1=<redacted>",
            message,
        )

    @staticmethod
    def _coerce_image_bytes(payload: object) -> bytes:
        if isinstance(payload, (bytes, bytearray)):
            return bytes(payload)
        if isinstance(payload, dict):
            for key in ("data", "bytes", "image", "content"):
                if key in payload and isinstance(payload[key], (bytes, bytearray)):
                    return bytes(payload[key])
        if hasattr(payload, "read"):
            image_bytes = payload.read()
            if isinstance(image_bytes, (bytes, bytearray)):
                return bytes(image_bytes)
            raise InvalidImageResponseError(
                "The Hugging Face image provider returned invalid image bytes."
            )
        if hasattr(payload, "save"):
            buffer = BytesIO()
            format_name = getattr(payload, "format", None) or "PNG"
            payload.save(buffer, format=format_name)
            return buffer.getvalue()
        raise InvalidImageResponseError("The Hugging Face image provider returned no image bytes.")

    @staticmethod
    def _detect_mime_type(image_bytes: bytes) -> str:
        if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if image_bytes.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
            return "image/webp"
        raise InvalidImageResponseError(
            "The Hugging Face image provider returned an unsupported image format."
        )
