"""Select a hosted image provider from environment-backed settings."""

from comiccraft.config import Settings
from comiccraft.errors import UnsupportedImageProviderError
from comiccraft.providers.gemini_image import GeminiImageGenerationProvider
from comiccraft.providers.huggingface_image import HuggingFaceImageGenerationProvider
from comiccraft.providers.image_contracts import ImageGenerationProvider
from comiccraft.providers.local_image import LocalImageGenerationProvider


def create_image_provider(
    settings: Settings, provider_name: str | None = None
) -> ImageGenerationProvider:
    """Create the configured image provider adapter."""
    resolved_name = (provider_name or settings.image_provider or "local").strip().lower()
    if resolved_name == "gemini":
        return GeminiImageGenerationProvider(
            api_key=settings.image_api_key,
            model=settings.image_model or "gemini-3.1-flash-image",
        )
    if resolved_name == "huggingface":
        return HuggingFaceImageGenerationProvider(
            token=settings.hf_token,
            model=settings.hf_image_model or "black-forest-labs/FLUX.1-schnell",
        )
    if resolved_name == "local":
        return LocalImageGenerationProvider()
    raise UnsupportedImageProviderError(
        f"Unsupported IMAGE_PROVIDER '{resolved_name}'. Currently supported: gemini, huggingface, and local."
    )