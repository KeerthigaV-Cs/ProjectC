import json
import re
from html import unescape
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from google import genai
from PIL import Image

from comiccraft.errors import (
    GeminiConfigurationError,
    GeminiProviderError,
    GeminiRateLimitError,
    GeminiServiceUnavailableError,
    HuggingFaceStoryProviderError,
    ImageProviderError,
    ImageProviderCreditsExhaustedError,
    ImageProviderRateLimitError,
    ImageProviderServiceUnavailableError,
    ImageProviderTimeoutError,
    InvalidGeminiResponseError,
    InvalidHuggingFaceStoryResponseError,
    MissingGeminiApiKeyError,
    MissingHFTokenError,
    MissingHFStoryModelError,
    MissingImageApiKeyError,
    ProviderFallbackError,
)
from comiccraft.main import app
from comiccraft.models.images import ComicImageSet
from comiccraft.models.story import ComicInput, ComicStory
from comiccraft.routes import (
    get_image_generation_service,
    get_pdf_export_service,
    get_story_generation_service,
)
from comiccraft.errors import ComicPDFExportError
from comiccraft.providers.local_image import LocalImageGenerationProvider
from comiccraft.services.image_generation import ImageGenerationService
from comiccraft.services.pdf_export import ComicPDFExportService


def make_story() -> ComicStory:
    user_input = ComicInput(
        story_prompt="A brave fox finds a lost star.",
        character_name="Pip",
        setting="Enchanted forest",
        tone="Whimsical",
        art_style="Watercolor comic",
    )
    return ComicStory(
        user_input=user_input,
        panels=[
            {
                "panel_number": number,
                "title": f"The {number}th turn",
                "scene_description": f"Scene description {number}.",
                "image_generation_prompt": f"Illustration prompt {number}.",
                "narration": f"Narration for panel {number}.",
                "dialogue": [
                    {"character": "Pip", "line": f"Dialogue for panel {number}."}
                ],
            }
            for number in range(1, 6)
        ],
    )


def make_images() -> ComicImageSet:
    return ComicImageSet(
        story_id="test-story",
        panels=[
            {
                "panel_number": number,
                "image_url": f"/static/panels/test-story/panel-{number}.png",
            }
            for number in range(1, 6)
        ],
    )


def make_form_data() -> dict[str, str]:
    return {
        "story_prompt": "A brave fox finds a lost star.",
        "character_name": "Pip",
        "setting": "Enchanted forest",
        "tone": "Whimsical",
        "art_style": "Watercolor comic",
    }


def make_daily_quota_error() -> GeminiRateLimitError:
    quota_error = GeminiRateLimitError("Gemini rate limit reached.")
    quota_error.__cause__ = genai.errors.APIError(
        429,
        {
            "error": {
                "message": "You exceeded your current quota.",
                "status": "RESOURCE_EXHAUSTED",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                        "violations": [
                            {
                                "quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                                "quotaValue": "20",
                                "quotaDimensions": {"model": "gemini-3.8-flash"},
                            }
                        ],
                    }
                ],
            }
        },
    )
    return quota_error


@pytest.fixture
def preview_client(tmp_path: Path):
    story_service = MagicMock()
    story_service.generate.return_value = make_story()
    image_service = MagicMock()
    image_service.generate_for_story.return_value = make_images()
    panel_directory = tmp_path / "panels"
    story_directory = panel_directory / "test-story"
    story_directory.mkdir(parents=True)
    for number in range(1, 6):
        Image.new("RGB", (120, 80), color=(number * 20, 100, 140)).save(
            story_directory / f"panel-{number}.png"
        )
    pdf_export_service = ComicPDFExportService(
        panel_directory=panel_directory,
        export_directory=tmp_path / "exports",
    )
    app.dependency_overrides[get_story_generation_service] = lambda: story_service
    app.dependency_overrides[get_image_generation_service] = lambda: image_service
    app.dependency_overrides[get_pdf_export_service] = lambda: pdf_export_service
    with TestClient(app) as test_client:
        yield test_client, story_service, image_service
    app.dependency_overrides.pop(get_story_generation_service, None)
    app.dependency_overrides.pop(get_image_generation_service, None)
    app.dependency_overrides.pop(get_pdf_export_service, None)


def test_home_page_contains_comic_form_and_options(preview_client) -> None:
    test_client, _, _ = preview_client
    response = test_client.get("/")

    assert response.status_code == 200
    assert 'action="/generate/progress"' in response.text
    assert 'class="generate-button-label" aria-live="polite" aria-atomic="true">Generate comic</span>' in response.text
    assert "Generation includes:" not in response.text
    assert "generation-details" not in response.text
    assert "submit-status" not in response.text
    form_script = Path(__file__).parents[1].joinpath(
        "src/comiccraft/static/js/comic-form.js"
    ).read_text(encoding="utf-8")
    for stage in (
        "Preparing story",
        "Generating 5-panel story",
        "Generating narration and dialogue",
        "Creating illustrations",
        "Creating Panel ${data.panel_number} of 5",
        "Creating comic panels",
        "Assembling comic",
        "Preparing preview",
        "Preparing PDF",
        "✓ Comic ready",
    ):
        assert stage in form_script
    assert 'setInterval(() => {' in form_script
    assert '".".repeat(dotCount)' in form_script
    assert 'submitButton.classList.add("is-generating")' in form_script
    assert 'submitButton.classList.remove("is-generating")' in form_script
    comic_css = Path(__file__).parents[1].joinpath(
        "src/comiccraft/static/css/comic.css"
    ).read_text(encoding="utf-8")
    assert ".generate-button.is-generating" in comic_css
    assert "animation: generate-button-glow 2.4s ease-in-out infinite" in comic_css
    for field_name in make_form_data():
        assert f'name="{field_name}"' in response.text
    for field_id, field_name in (
        ("setting", "setting"),
        ("tone", "tone"),
        ("art-style", "art_style"),
    ):
        assert f'id="{field_id}" class="combo-input" name="{field_name}"' in response.text
    for option in (
        "Enchanted forest",
        "Space station",
        "Coastal town",
        "Mountain village",
        "Whimsical",
        "Funny",
        "Dramatic",
        "Adventurous",
        "Watercolor comic",
        "Anime",
        "Bold ink comic",
        "Pixel art",
    ):
        assert f'data-value="{option}"' in response.text
    art_style_positions = [
        response.text.index(f'data-value="{style}"')
        for style in ("Pixel art", "Bold ink comic", "Anime", "Watercolor comic")
    ]
    assert art_style_positions == sorted(art_style_positions)
    assert "Choose a setting" not in response.text
    assert "Choose a tone" not in response.text
    assert "Choose an art style" not in response.text


def test_progress_route_streams_real_stages_and_preview(preview_client) -> None:
    test_client, _, image_service = preview_client

    def generate_images(story, progress_callback):
        for panel_number in range(1, 6):
            progress_callback(panel_number)
        return make_images()

    image_service.generate_for_story.side_effect = generate_images
    response = test_client.post("/generate/progress", data=make_form_data())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = re.findall(r"event: ([a-z]+)\ndata: (.*?)\n\n", response.text)
    event_names = [name for name, _ in events]
    assert event_names == [
        "preparing",
        "story",
        "narration",
        "images",
        "panel",
        "panel",
        "panel",
        "panel",
        "panel",
        "panels",
        "assembly",
        "preview",
        "pdf",
        "complete",
        "result",
    ]
    panel_numbers = [
        json.loads(data)["panel_number"] for name, data in events if name == "panel"
    ]
    assert panel_numbers == [1, 2, 3, 4, 5]
    result_html = json.loads(events[-1][1])["html"]
    assert "Completed: Your comic is ready!" in result_html
    assert len(re.findall(r'<article class="comic-panel"', result_html)) == 5


def test_progress_route_streams_error_and_returns_existing_error_page(preview_client) -> None:
    test_client, story_service, _ = preview_client
    story_service.generate.side_effect = MissingGeminiApiKeyError("missing")

    response = test_client.post("/generate/progress", data=make_form_data())

    events = re.findall(r"event: ([a-z]+)\ndata: (.*?)\n\n", response.text)
    assert [name for name, _ in events] == [
        "preparing",
        "story",
        "error",
    ]
    error_data = json.loads(events[-1][1])
    assert error_data == {
        "success": False,
        "error_code": "GEMINI_CONFIGURATION_ERROR",
        "message": "Gemini AI is not configured correctly. Please check your Gemini API key in the project settings.",
    }


def test_generate_route_calls_services_and_renders_five_panels(preview_client) -> None:
    test_client, story_service, image_service = preview_client
    response = test_client.post("/generate", data=make_form_data())

    assert response.status_code == 200
    assert len(re.findall(r'<article class="comic-panel"', response.text)) == 5
    assert response.text.count('class="comic-image"') == 5
    assert "The 1th turn" in response.text
    assert "Scene description 1." in response.text
    assert "Narration for panel 1." in response.text
    assert "Dialogue for panel 1." in response.text
    assert "Completed: Your comic is ready!" in response.text
    download_match = re.search(
        r'<a(?=[^>]*data-pdf-download)[^>]*href="([^"]+)"', response.text
    )
    assert download_match is not None
    assert "data-success-url=" in response.text
    assert "/static/js/comic-export.js" in response.text
    download_response = test_client.get(download_match.group(1))
    assert download_response.status_code == 200
    assert download_response.headers["content-type"] == "application/pdf"
    assert download_response.headers["content-disposition"] == (
        'attachment; filename="Pip_The_1th_turn.pdf"'
    )
    success_match = re.search(r'data-success-url="([^"]+)"', response.text)
    assert success_match is not None
    success_response = test_client.get(unescape(success_match.group(1)))
    assert success_response.status_code == 200
    assert "EXPORT COMPLETE" in success_response.text
    assert "Pip_The_1th_turn.pdf" in success_response.text
    assert "Download again" in success_response.text
    story_service.generate.assert_called_once()
    image_service.generate_for_story.assert_called_once_with(make_story())


def test_offline_local_images_flow_through_preview_and_pdf_download(tmp_path) -> None:
    story_service = MagicMock()
    story_service.generate.return_value = make_story()
    panel_directory = tmp_path / "panels"
    export_directory = tmp_path / "exports"
    image_service = ImageGenerationService(
        LocalImageGenerationProvider(), output_directory=panel_directory
    )
    pdf_service = ComicPDFExportService(panel_directory, export_directory)
    app.dependency_overrides[get_story_generation_service] = lambda: story_service
    app.dependency_overrides[get_image_generation_service] = lambda: image_service
    app.dependency_overrides[get_pdf_export_service] = lambda: pdf_service
    try:
        with TestClient(app) as test_client:
            response = test_client.post("/generate", data=make_form_data())

            assert response.status_code == 200
            assert len(re.findall(r'<article class="comic-panel"', response.text)) == 5
            assert len(list(panel_directory.rglob("panel-*.png"))) == 5
            download_match = re.search(
                r'<a(?=[^>]*data-pdf-download)[^>]*href="([^"]+)"', response.text
            )
            assert download_match is not None
            download_response = test_client.get(download_match.group(1))
            assert download_response.status_code == 200
            assert download_response.headers["content-type"] == "application/pdf"
            success_match = re.search(r'data-success-url="([^"]+)"', response.text)
            assert success_match is not None
            success_response = test_client.get(unescape(success_match.group(1)))
            assert success_response.status_code == 200
            assert "EXPORT COMPLETE" in success_response.text
    finally:
        app.dependency_overrides.pop(get_story_generation_service, None)
        app.dependency_overrides.pop(get_image_generation_service, None)
        app.dependency_overrides.pop(get_pdf_export_service, None)


def test_generate_route_shows_a_clear_pdf_export_error(preview_client, monkeypatch) -> None:
    test_client, _, _ = preview_client
    pdf_service = app.dependency_overrides[get_pdf_export_service]()
    monkeypatch.setattr(
        pdf_service,
        "create_pdf",
        MagicMock(side_effect=ComicPDFExportError("PDF generation failed.")),
    )

    response = test_client.post("/generate", data=make_form_data())

    assert response.status_code == 500
    assert "PDF generation failed." in response.text
    assert "RuntimeError" not in response.text
    assert "Generate comic" in response.text
    assert "Generation includes:" not in response.text


def test_generate_route_explains_when_a_panel_image_is_missing(
    preview_client, monkeypatch
) -> None:
    test_client, _, _ = preview_client
    pdf_service = app.dependency_overrides[get_pdf_export_service]()
    monkeypatch.setattr(
        pdf_service,
        "create_pdf",
        MagicMock(
            side_effect=ComicPDFExportError(
                "The saved image for panel 3 is missing."
            )
        ),
    )

    response = test_client.post("/generate", data=make_form_data())

    assert response.status_code == 500
    assert "The saved image for panel 3 is missing." in response.text


def test_generate_route_renders_field_validation_errors(preview_client) -> None:
    test_client, story_service, image_service = preview_client
    invalid_form = make_form_data()
    invalid_form["story_prompt"] = "   "
    response = test_client.post("/generate", data=invalid_form)

    assert response.status_code == 422
    assert 'role="alert"' in response.text
    assert "Please provide a story prompt and character name before generating your comic." in response.text
    assert "at least 1 character" in response.text
    story_service.generate.assert_not_called()
    image_service.generate_for_story.assert_not_called()


@pytest.mark.parametrize(
    ("stage", "error", "status_code", "message"),
    [
        ("story", MissingGeminiApiKeyError("missing"), 503, "Gemini AI is not configured correctly. Please check your Gemini API key in the project settings."),
        ("story", GeminiConfigurationError("private key detail"), 503, "Gemini AI is not configured correctly. Please check your Gemini API key in the project settings."),
        (
            "story",
            MissingHFStoryModelError("missing model"),
            503,
            "We couldn't create the story at the moment because the AI story services are unavailable. Please try again later.",
        ),
        ("image", MissingImageApiKeyError("missing"), 503, "We couldn't generate the comic illustrations. Please try again."),
        ("story", GeminiProviderError("provider"), 503, "We couldn't create the story at the moment because the AI story services are unavailable. Please try again later."),
        (
            "story",
            GeminiRateLimitError("Gemini rate limit reached. Please try again shortly."),
            429,
            "Gemini AI usage limit has been reached. Please wait for the quota to reset or use an available fallback service.",
        ),
        (
            "story",
            make_daily_quota_error(),
            429,
            "Gemini AI usage limit has been reached. Please wait for the quota to reset or use an available fallback service.",
        ),
        (
            "story",
            GeminiServiceUnavailableError(
                "Gemini AI is currently experiencing high demand. Please try again in a few moments to generate your comic."
            ),
            503,
            "Gemini AI is temporarily unavailable. Please try generating your comic again in a few moments.",
        ),
        ("story", InvalidGeminiResponseError("invalid"), 503, "We couldn't create the story at the moment because the AI story services are unavailable. Please try again later."),
        ("story", ProviderFallbackError(GeminiRateLimitError("primary quota"), HuggingFaceStoryProviderError("private fallback response")), 503, "We couldn't create the story at the moment because the AI story services are unavailable. Please try again later."),
        (
            "story",
            HuggingFaceStoryProviderError("private internal detail"),
            503,
            "We couldn't create the story at the moment because the AI story services are unavailable. Please try again later.",
        ),
        (
            "story",
            InvalidHuggingFaceStoryResponseError("private internal detail"),
            503,
            "We couldn't create the story at the moment because the AI story services are unavailable. Please try again later.",
        ),
        ("image", MissingHFTokenError("missing"), 503, "Hugging Face image generation is not configured correctly. Please check the Hugging Face access token in the project settings."),
        ("image", ImageProviderCreditsExhaustedError("private 402 response"), 402, "Hugging Face image-generation credits have been used up. Please add Hugging Face credits or switch to another image provider."),
        ("image", ImageProviderRateLimitError("limited"), 429, "Hugging Face image generation is temporarily rate-limited. Please wait a moment and try again."),
        ("image", ImageProviderServiceUnavailableError("private 503 response"), 503, "Hugging Face image generation is temporarily unavailable. Please try again in a few moments."),
        ("image", ImageProviderTimeoutError("timeout"), 504, "Image generation took too long to respond. Please check your internet connection and try again."),
        ("image", ImageProviderError("provider"), 502, "We couldn't generate the comic illustrations. Please try again."),
        ("image", RuntimeError("private server detail"), 500, "Something went wrong while creating your comic. Please try again."),
        ("story", RuntimeError("primary and fallback unavailable"), 500, "Something went wrong while creating your comic. Please try again."),
    ],
)
def test_generate_route_renders_provider_errors(
    preview_client, stage, error, status_code, message
) -> None:
    test_client, story_service, image_service = preview_client
    if stage == "story":
        story_service.generate.side_effect = error
    else:
        image_service.generate_for_story.side_effect = error
    response = test_client.post("/generate", data=make_form_data())

    assert response.status_code == status_code
    rendered_text = unescape(response.text)
    assert message in rendered_text
    assert "Pip" in response.text
    assert "private" not in rendered_text


def test_daily_quota_message_does_not_claim_gemini_is_busy(preview_client) -> None:
    test_client, story_service, _ = preview_client
    story_service.generate.side_effect = make_daily_quota_error()

    response = test_client.post("/generate", data=make_form_data())

    assert response.status_code == 429
    assert "Gemini AI usage limit has been reached." in response.text
    assert "Gemini API daily quota reached." not in response.text


def test_generate_route_logs_redacted_stage_diagnostics(preview_client, caplog, monkeypatch) -> None:
    import comiccraft.routes as routes
    from comiccraft.config import Settings

    gemini_key = "test-gemini-secret"
    image_key = "test-image-secret"
    monkeypatch.setattr(
        routes,
        "get_settings",
        lambda: Settings(gemini_key, "test-model", "gemini", image_key, "image-model"),
    )
    test_client, story_service, _ = preview_client
    error = GeminiProviderError(f"Gemini API request failed with status 403; api_key={gemini_key}")
    provider_error = RuntimeError(
        f"Authorization: Bearer {image_key}; headers={{'x-api-key': '{gemini_key}'}}"
    )
    provider_error.code = 403
    error.__cause__ = provider_error
    story_service.generate.side_effect = error

    with caplog.at_level("ERROR", logger="comiccraft.routes"):
        response = test_client.post("/generate", data=make_form_data())

    assert response.status_code == 503
    assert "We couldn't create the story at the moment because the AI story services are unavailable. Please try again later." in unescape(response.text)
    assert "STORY generation failed" in caplog.text
    assert "GeminiProviderError" in caplog.text
    assert "status_code=403" in caplog.text
    assert "<redacted>" in caplog.text
    assert gemini_key not in caplog.text
    assert image_key not in caplog.text
