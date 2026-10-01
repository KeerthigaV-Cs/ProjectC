"""Interfaces implemented by image-generation providers."""

from typing import Protocol

from comiccraft.models.images import GeneratedImageData


class ImageGenerationProvider(Protocol):
    """Replaceable interface for hosted image-generation providers."""

    def generate_image(self, prompt: str) -> GeneratedImageData:
        """Return image bytes and their MIME type for a text prompt."""