"""Application errors surfaced by story generation."""


class FallbackEligibleError(Exception):
    """Raised when a temporary provider error allows a fallback to be attempted."""


class MissingGeminiApiKeyError(Exception):
    """Raised when Gemini generation is requested without an API key."""


class GeminiRateLimitError(FallbackEligibleError):
    """Raised when Gemini rejects a request because of rate limits."""


class GeminiProviderError(Exception):
    """Raised when the Gemini provider request fails."""


class GeminiConfigurationError(GeminiProviderError):
    """Raised when Gemini rejects an invalid API key."""


class GeminiServiceUnavailableError(FallbackEligibleError, GeminiProviderError):
    """Raised when Gemini remains unavailable after transient-error retries."""


class InvalidGeminiResponseError(Exception):
    """Raised when Gemini returns content that fails the story schema."""


class MissingImageApiKeyError(Exception):
    """Raised when image generation is requested without provider credentials."""


class MissingHFTokenError(Exception):
    """Raised when a Hugging Face provider is selected without a token."""


class MissingHFStoryModelError(Exception):
    """Raised when Hugging Face story generation has no model configured."""


class HuggingFaceStoryProviderError(Exception):
    """Raised when a Hugging Face story-generation request fails."""


class InvalidHuggingFaceStoryResponseError(Exception):
    """Raised when Hugging Face returns content that fails the story schema."""


class ImageProviderRateLimitError(FallbackEligibleError):
    """Raised when the image provider rejects a request due to rate limits."""


class ImageProviderTimeoutError(FallbackEligibleError):
    """Raised when the hosted image provider times out."""


class ImageProviderError(Exception):
    """Raised when the hosted image provider request fails."""


class ImageProviderCreditsExhaustedError(ImageProviderError):
    """Raised when an image provider has no remaining credits."""


class ImageProviderServiceUnavailableError(ImageProviderError):
    """Raised when an image provider returns a temporary server error."""


class InvalidImageResponseError(Exception):
    """Raised when the image provider returns invalid image data."""


class UnsupportedImageProviderError(Exception):
    """Raised when IMAGE_PROVIDER names an unregistered provider."""


class UnsupportedStoryProviderError(Exception):
    """Raised when STORY_PROVIDER names an unregistered provider."""


class ProviderFallbackError(RuntimeError):
    """Raised when the primary provider fails and the fallback provider also fails."""

    def __init__(self, primary_error: Exception, fallback_error: Exception) -> None:
        self.primary_error = primary_error
        self.fallback_error = fallback_error
        if isinstance(primary_error, GeminiRateLimitError):
            self.status_code = 429
            message = (
                "Gemini API quota is currently unavailable and the Hugging Face fallback also failed. "
                "Please check your AI provider configuration or try again later."
            )
        elif isinstance(primary_error, GeminiServiceUnavailableError):
            self.status_code = 503
            message = (
                "Gemini is temporarily unavailable and the Hugging Face fallback also failed. "
                "Please check your AI provider configuration or try again later."
            )
        elif isinstance(primary_error, ImageProviderRateLimitError):
            self.status_code = 429
            if isinstance(fallback_error, (ImageProviderError, MissingHFTokenError)):
                message = (
                    "The primary image provider was rate limited, and the Hugging Face "
                    f"image fallback failed: {fallback_error}"
                )
            else:
                message = (
                    "The primary image provider failed and the Hugging Face fallback also failed. "
                    "Please check your AI provider configuration or try again later."
                )
        elif isinstance(primary_error, ImageProviderTimeoutError):
            self.status_code = 504
            message = (
                "The primary image provider failed and the Hugging Face fallback also failed. "
                "Please check your AI provider configuration or try again later."
            )
        else:
            self.status_code = 502
            message = (
                "The primary provider failed and the fallback provider also failed. "
                "Please check your AI provider configuration or try again later."
            )
        super().__init__(message)


class ImageStorageError(Exception):
    """Raised when generated image data cannot be saved locally."""


class ComicPDFExportError(Exception):
    """Raised when a comic PDF cannot be created from saved panel images."""


class MissingComicPanelImageError(ComicPDFExportError):
    """Raised when a saved image for one of the comic panels is missing."""


class ComicPDFNotFoundError(Exception):
    """Raised when a requested comic PDF does not exist or has an invalid name."""