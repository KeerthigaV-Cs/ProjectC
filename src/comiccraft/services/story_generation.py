"""Orchestrate validated comic story generation."""

import logging

from pydantic import ValidationError

from comiccraft.errors import (
    FallbackEligibleError,
    InvalidGeminiResponseError,
    ProviderFallbackError,
)
from comiccraft.models.story import (
    ComicInput,
    ComicPanel,
    ComicStory,
    GeneratedStoryContent,
)
from comiccraft.providers.contracts import StoryProvider

logger = logging.getLogger(__name__)


class StoryGenerationService:
    """Combine provider-generated outline and narration into comic panels."""

    def __init__(
        self, provider: StoryProvider, fallback_provider: StoryProvider | None = None
    ) -> None:
        self._provider = provider
        self._fallback_provider = fallback_provider

    def generate(self, user_input: ComicInput) -> ComicStory:
        try:
            return self._generate_with_provider(self._provider, user_input)
        except FallbackEligibleError as exc:
            if self._fallback_provider is None:
                raise
            logger.warning(
                "Primary story provider failed with a fallback-eligible error; trying fallback provider."
            )
            try:
                return self._generate_with_provider(self._fallback_provider, user_input)
            except Exception as fallback_exc:
                raise ProviderFallbackError(exc, fallback_exc) from fallback_exc
        except (ValidationError, KeyError) as exc:
            raise InvalidGeminiResponseError(
                "Gemini returned content that does not match the required five-panel story format."
            ) from exc

    def _generate_with_provider(
        self, provider: StoryProvider, user_input: ComicInput
    ) -> ComicStory:
        generated = GeneratedStoryContent.model_validate(provider.generate_story(user_input))
        narration_by_panel = {panel.panel_number: panel for panel in generated.narration.panels}
        panels = [
            ComicPanel(
                **outline_panel.model_dump(),
                narration=narration_by_panel[outline_panel.panel_number].narration,
                dialogue=narration_by_panel[outline_panel.panel_number].dialogue,
            )
            for outline_panel in generated.outline.panels
        ]
        return ComicStory(user_input=user_input, panels=panels)