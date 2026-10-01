"""Environment-backed application settings."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str | None = None
    gemini_model: str | None = "gemini-3.8-flash"
    story_provider: str | None = "gemini"
    story_fallback_provider: str | None = "huggingface"
    image_provider: str | None = "local"
    image_fallback_provider: str | None = "huggingface"
    image_api_key: str | None = None
    image_model: str | None = "gemini-3.1-flash-image"
    hf_token: str | None = None
    hf_story_model: str | None = None
    hf_image_model: str | None = "black-forest-labs/FLUX.1-schnell"


def get_settings() -> Settings:
    """Read provider credentials and model selections from the environment."""
    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY") or None,
        gemini_model=os.getenv("GEMINI_MODEL") or "gemini-3.8-flash",
        story_provider=os.getenv("STORY_PROVIDER") or "gemini",
        story_fallback_provider=os.getenv("STORY_FALLBACK_PROVIDER") or "huggingface",
        image_provider=os.getenv("IMAGE_PROVIDER") or "local",
        image_fallback_provider=os.getenv("IMAGE_FALLBACK_PROVIDER") or "huggingface",
        image_api_key=os.getenv("IMAGE_API_KEY") or None,
        image_model=os.getenv("IMAGE_MODEL") or "gemini-3.1-flash-image",
        hf_token=os.getenv("HF_TOKEN") or None,
        hf_story_model=os.getenv("HF_STORY_MODEL") or None,
        hf_image_model=os.getenv("HF_IMAGE_MODEL") or "black-forest-labs/FLUX.1-schnell",
    )