"""Generate and store one image for each panel in a comic story."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from shutil import rmtree
from typing import Callable
from uuid import uuid4

from pydantic import ValidationError

from comiccraft.errors import (
    FallbackEligibleError,
    ImageStorageError,
    InvalidImageResponseError,
    ProviderFallbackError,
)
from comiccraft.models.images import (
    ComicImageSet,
    ComicPanelImage,
    GeneratedImageData,
)
from comiccraft.models.story import ComicStory
from comiccraft.providers.image_contracts import ImageGenerationProvider
from comiccraft.providers.local_image import LocalImageGenerationProvider

PACKAGE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIRECTORY = PACKAGE_DIR / "static" / "panels"
IMAGE_EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
}
logger = logging.getLogger(__name__)


class ImageGenerationService:
    """Keep provider calls and generated-file storage out of HTTP routes."""

    def __init__(
        self,
        provider: ImageGenerationProvider,
        output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
        fallback_provider: ImageGenerationProvider | None = None,
    ) -> None:
        self._provider = provider
        self._output_directory = output_directory
        self._fallback_provider = fallback_provider

    def generate_for_story(
        self,
        story: ComicStory,
        progress_callback: Callable[[int], None] | None = None,
    ) -> ComicImageSet:
        try:
            return self._generate_for_story_with_provider(
                self._provider, story, progress_callback
            )
        except FallbackEligibleError as exc:
            if self._fallback_provider is None:
                raise
            logger.warning(
                "Primary image provider failed with a fallback-eligible error; trying fallback provider."
            )
            try:
                return self._generate_for_story_with_provider(
                    self._fallback_provider, story, progress_callback
                )
            except Exception as fallback_exc:
                raise ProviderFallbackError(exc, fallback_exc) from fallback_exc

    def _generate_for_story_with_provider(
        self,
        provider: ImageGenerationProvider,
        story: ComicStory,
        progress_callback: Callable[[int], None] | None = None,
    ) -> ComicImageSet:
        try:
            prompts = []
            for panel in story.panels:
                prompt = panel.image_generation_prompt
                if isinstance(provider, LocalImageGenerationProvider):
                    prompt = (
                        f"Panel number: {panel.panel_number}\n"
                        f"Character: {story.user_input.character_name}\n"
                        f"Scene: {panel.scene_description}\n"
                        f"Image prompt: {prompt}"
                    )
                prompts.append(prompt)
            with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
                futures = {
                    executor.submit(provider.generate_image, prompt): panel.panel_number
                    for panel, prompt in zip(story.panels, prompts, strict=True)
                }
                completed_images = {}
                next_panel_number = 1
                for future in as_completed(futures):
                    panel_number = futures[future]
                    completed_images[panel_number] = future.result()
                    while next_panel_number in completed_images:
                        if progress_callback is not None:
                            progress_callback(next_panel_number)
                        next_panel_number += 1
                generated_images = [
                    completed_images[panel_number]
                    for panel_number in range(1, len(prompts) + 1)
                ]
            generated_images = [
                GeneratedImageData.model_validate(image) for image in generated_images
            ]
        except ValidationError as exc:
            raise InvalidImageResponseError(
                "The image provider returned invalid image data."
            ) from exc

        story_id = uuid4().hex
        story_directory = self._output_directory / story_id
        try:
            story_directory.mkdir(parents=True, exist_ok=False)
            panels = []
            for panel, image in zip(story.panels, generated_images, strict=True):
                extension = IMAGE_EXTENSIONS[image.mime_type]
                filename = f"panel-{panel.panel_number}{extension}"
                (story_directory / filename).write_bytes(image.data)
                panels.append(
                    ComicPanelImage(
                        panel_number=panel.panel_number,
                        image_url=f"/static/panels/{story_id}/{filename}",
                    )
                )
            return ComicImageSet(story_id=story_id, panels=panels)
        except (OSError, ValidationError) as exc:
            rmtree(story_directory, ignore_errors=True)
            raise ImageStorageError("Generated panel images could not be saved.") from exc