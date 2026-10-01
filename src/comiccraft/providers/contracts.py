"""Interfaces implemented by model providers."""

from typing import Protocol

from comiccraft.models.story import ComicInput, GeneratedStoryContent


class StoryProvider(Protocol):
    """Provider contract consumed by the story-generation service."""

    def generate_story(self, user_input: ComicInput) -> GeneratedStoryContent:
        """Return a validated five-panel outline and narration."""