"""Validated data contracts for generated comic images."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

SupportedImageMimeType = Literal["image/png", "image/jpeg", "image/webp"]


class GeneratedImageData(BaseModel):
    """Raw image bytes returned by a hosted image provider."""

    data: bytes = Field(min_length=1)
    mime_type: SupportedImageMimeType


class ComicPanelImage(BaseModel):
    """Stored image URL associated with one comic panel."""

    panel_number: int = Field(ge=1, le=5)
    image_url: str


class ComicImageSet(BaseModel):
    """Stored image URLs for all five ordered comic panels."""

    story_id: str
    panels: list[ComicPanelImage] = Field(min_length=5, max_length=5)

    @model_validator(mode="after")
    def validate_panel_order(self) -> "ComicImageSet":
        if [panel.panel_number for panel in self.panels] != [1, 2, 3, 4, 5]:
            raise ValueError("Image panels must be numbered 1 through 5 in order.")
        return self