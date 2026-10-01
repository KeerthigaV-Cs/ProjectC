"""Validated data contracts for comic story generation."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

StoryText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ComicInput(BaseModel):
    """User-provided direction for a generated comic."""

    model_config = ConfigDict(extra="forbid")

    story_prompt: Annotated[StoryText, Field(max_length=2000)]
    character_name: Annotated[StoryText, Field(max_length=80)]
    setting: Annotated[StoryText, Field(max_length=120)]
    tone: Annotated[StoryText, Field(max_length=80)]
    art_style: Annotated[StoryText, Field(max_length=80)]


class ComicOutlinePanel(BaseModel):
    """Visual and story beat for a single panel."""

    panel_number: int = Field(ge=1, le=5)
    title: Annotated[StoryText, Field(max_length=120)]
    scene_description: Annotated[StoryText, Field(max_length=1000)]
    image_generation_prompt: Annotated[StoryText, Field(max_length=1000)]


class ComicOutline(BaseModel):
    """Exactly five ordered panel outlines."""

    panels: list[ComicOutlinePanel] = Field(min_length=5, max_length=5)

    @model_validator(mode="after")
    def validate_panel_order(self) -> "ComicOutline":
        if [panel.panel_number for panel in self.panels] != [1, 2, 3, 4, 5]:
            raise ValueError("Outline panels must be numbered 1 through 5 in order.")
        return self


class DialogueLine(BaseModel):
    """One character's line of dialogue."""

    character: Annotated[StoryText, Field(max_length=80)]
    line: Annotated[StoryText, Field(max_length=500)]


class PanelNarration(BaseModel):
    """Narration and dialogue for one outline panel."""

    panel_number: int = Field(ge=1, le=5)
    narration: Annotated[StoryText, Field(max_length=1000)]
    dialogue: list[DialogueLine] = Field(default_factory=list, max_length=8)


class ComicNarration(BaseModel):
    """Exactly five ordered narration entries."""

    panels: list[PanelNarration] = Field(min_length=5, max_length=5)

    @model_validator(mode="after")
    def validate_panel_order(self) -> "ComicNarration":
        if [panel.panel_number for panel in self.panels] != [1, 2, 3, 4, 5]:
            raise ValueError("Narration panels must be numbered 1 through 5 in order.")
        return self


class GeneratedStoryContent(BaseModel):
    """Structured Gemini response before outline and narration are combined."""

    outline: ComicOutline
    narration: ComicNarration


class ComicPanel(BaseModel):
    """Complete, display-ready story data for one panel."""

    panel_number: int = Field(ge=1, le=5)
    title: Annotated[StoryText, Field(max_length=120)]
    scene_description: Annotated[StoryText, Field(max_length=1000)]
    image_generation_prompt: Annotated[StoryText, Field(max_length=1000)]
    narration: Annotated[StoryText, Field(max_length=1000)]
    dialogue: list[DialogueLine] = Field(default_factory=list, max_length=8)


class ComicStory(BaseModel):
    """Complete story containing exactly five ordered comic panels."""

    user_input: ComicInput
    panels: list[ComicPanel] = Field(min_length=5, max_length=5)

    @model_validator(mode="after")
    def validate_panel_order(self) -> "ComicStory":
        if [panel.panel_number for panel in self.panels] != [1, 2, 3, 4, 5]:
            raise ValueError("Comic panels must be numbered 1 through 5 in order.")
        return self