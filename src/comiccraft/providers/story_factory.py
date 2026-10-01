"""Factory for story-generation providers."""

from comiccraft.config import Settings
from comiccraft.errors import UnsupportedStoryProviderError
from comiccraft.models.story import ComicInput, GeneratedStoryContent
from comiccraft.providers.contracts import StoryProvider
from comiccraft.providers.gemini import GoogleGeminiStoryProvider
from comiccraft.providers.huggingface import HuggingFaceStoryProvider


def create_story_provider(
    settings: Settings, provider_name: str | None = None
) -> StoryProvider:
    """Create the configured story provider adapter."""
    resolved_name = (provider_name or settings.story_provider or "gemini").strip().lower()
    if resolved_name == "gemini":
        return GoogleGeminiStoryProvider(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model or "gemini-3.8-flash",
        )
    if resolved_name == "huggingface":
        return HuggingFaceStoryProvider(
            token=settings.hf_token,
            model=settings.hf_story_model,
        )
    raise UnsupportedStoryProviderError(
        f"Unsupported STORY_PROVIDER '{resolved_name}'. Currently supported: gemini and huggingface."
    )
