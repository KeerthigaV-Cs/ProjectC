import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient
from google import genai
from google.genai import types
from pydantic import ValidationError

from comiccraft.errors import (
    GeminiConfigurationError,
    GeminiProviderError,
    GeminiRateLimitError,
    GeminiServiceUnavailableError,
    HuggingFaceStoryProviderError,
    InvalidHuggingFaceStoryResponseError,
    InvalidGeminiResponseError,
    InvalidHuggingFaceStoryResponseError,
    MissingGeminiApiKeyError,
    MissingHFStoryModelError,
    ProviderFallbackError,
)
from comiccraft.models.story import (
    ComicInput,
    ComicNarration,
    ComicOutline,
    GeneratedStoryContent,
)
from comiccraft.providers import gemini
from comiccraft.providers.gemini import GoogleGeminiStoryProvider
from comiccraft.providers.huggingface import HuggingFaceStoryProvider
from comiccraft.routes import get_story_generation_service
from comiccraft.services.story_generation import StoryGenerationService
from comiccraft.main import app


def make_input() -> ComicInput:
    return ComicInput(
        story_prompt="A fox finds a lost star.",
        character_name="Pip",
        setting="A moonlit forest",
        tone="Whimsical",
        art_style="Watercolor comic",
    )


def make_generated_content() -> GeneratedStoryContent:
    return GeneratedStoryContent(
        outline={
            "panels": [
                {
                    "panel_number": number,
                    "title": f"Panel {number}",
                    "scene_description": f"Scene {number}",
                    "image_generation_prompt": f"Illustration prompt {number}",
                }
                for number in range(1, 6)
            ]
        },
        narration={
            "panels": [
                {
                    "panel_number": number,
                    "narration": f"Narration {number}",
                    "dialogue": [{"character": "Pip", "line": f"Line {number}"}],
                }
                for number in range(1, 6)
            ]
        },
    )


def make_huggingface_provider(response_content: object):
    client = MagicMock()
    client.chat_completion.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=response_content))]
    )
    provider = HuggingFaceStoryProvider(
        token="testing-only",
        model="test-model",
        client_factory=MagicMock(return_value=client),
    )
    return provider, client


def make_mocked_sdk_attempts(monkeypatch, statuses: list[int]) -> list[httpx.Request]:
    attempts: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        status_code = statuses[min(len(attempts) - 1, len(statuses) - 1)]
        if status_code == 200:
            content = json.dumps(make_generated_content().model_dump())
            body = {
                "candidates": [
                    {
                        "content": {"parts": [{"text": content}], "role": "model"},
                        "finishReason": "STOP",
                        "index": 0,
                    }
                ]
            }
        else:
            body = {
                "error": {
                    "code": status_code,
                    "message": "Mock provider error",
                    "status": "UNAVAILABLE" if status_code >= 500 else "INVALID_ARGUMENT",
                }
            }
        return httpx.Response(status_code, json=body, request=request)

    sdk_client_factory = gemini.genai.Client

    def create_mocked_client(*, api_key: str, http_options: types.HttpOptions):
        http_options.retry_options.initial_delay = 0.001
        http_options.retry_options.max_delay = 0.001
        http_options.httpx_client = httpx.Client(
            transport=httpx.MockTransport(respond)
        )
        return sdk_client_factory(api_key=api_key, http_options=http_options)

    monkeypatch.setattr(gemini.genai, "Client", create_mocked_client)
    return attempts


def test_comic_input_strips_text_and_rejects_blank_fields() -> None:
    comic_input = ComicInput(
        story_prompt="  A fox  ",
        character_name=" Pip ",
        setting=" Forest ",
        tone=" Funny ",
        art_style=" Ink ",
    )
    assert comic_input.story_prompt == "A fox"
    assert comic_input.character_name == "Pip"

    with pytest.raises(ValidationError):
        ComicInput(
            story_prompt="   ",
            character_name="Pip",
            setting="Forest",
            tone="Funny",
            art_style="Ink",
        )


@pytest.mark.parametrize("model_type", [ComicOutline, ComicNarration])
def test_outline_and_narration_require_exactly_five_ordered_panels(model_type) -> None:
    content = make_generated_content()
    panel_data = (
        [panel.model_dump() for panel in content.outline.panels]
        if model_type is ComicOutline
        else [panel.model_dump() for panel in content.narration.panels]
    )

    with pytest.raises(ValidationError):
        model_type(panels=panel_data[:4])
    panel_data[0]["panel_number"] = 2
    with pytest.raises(ValidationError):
        model_type(panels=panel_data)


def test_service_combines_outline_and_narration_into_five_panels() -> None:
    provider = MagicMock()
    provider.generate_story.return_value = make_generated_content()

    story = StoryGenerationService(provider).generate(make_input())

    assert len(story.panels) == 5
    assert [panel.panel_number for panel in story.panels] == [1, 2, 3, 4, 5]
    assert story.panels[0].image_generation_prompt == "Illustration prompt 1"
    assert story.panels[0].narration == "Narration 1"
    assert story.panels[0].dialogue[0].line == "Line 1"


def test_service_rejects_mismatched_provider_content() -> None:
    content = make_generated_content().model_dump()
    content["narration"]["panels"][0]["panel_number"] = 2
    provider = MagicMock()
    provider.generate_story.return_value = content

    with pytest.raises(InvalidGeminiResponseError):
        StoryGenerationService(provider).generate(make_input())


def test_provider_requires_api_key_before_creating_client() -> None:
    provider = GoogleGeminiStoryProvider(api_key=None, model="test-model")

    with pytest.raises(MissingGeminiApiKeyError):
        provider.generate_story(make_input())


def test_provider_uses_configured_model_and_structured_response(monkeypatch) -> None:
    client = MagicMock()
    client.__enter__.return_value = client
    client.models.generate_content.return_value = SimpleNamespace(
        parsed=make_generated_content(), text="{}"
    )
    client_factory = MagicMock(return_value=client)
    monkeypatch.setattr(gemini.genai, "Client", client_factory)
    provider = GoogleGeminiStoryProvider(api_key="testing-only", model="test-model")

    response = provider.generate_story(make_input())

    assert response == make_generated_content()
    client_factory.assert_called_once()
    client_kwargs = client_factory.call_args.kwargs
    assert client_kwargs["api_key"] == "testing-only"
    retry_options = client_kwargs["http_options"].retry_options
    assert retry_options.attempts == 4
    assert retry_options.initial_delay == 0.5
    assert retry_options.max_delay == 2.0
    assert retry_options.exp_base == 2.0
    assert retry_options.http_status_codes == list(range(500, 600))
    call = client.models.generate_content.call_args
    assert call.kwargs["model"] == "test-model"
    assert call.kwargs["config"].response_schema is GeneratedStoryContent


def test_provider_maps_rate_limit_error(monkeypatch) -> None:
    attempts = make_mocked_sdk_attempts(monkeypatch, [429])

    with pytest.raises(GeminiRateLimitError):
        GoogleGeminiStoryProvider("testing-only", "test-model").generate_story(
            make_input()
        )
    assert len(attempts) == 1


def test_provider_maps_invalid_api_key_to_configuration_error(monkeypatch) -> None:
    attempts = make_mocked_sdk_attempts(monkeypatch, [403])

    with pytest.raises(GeminiConfigurationError):
        GoogleGeminiStoryProvider("testing-only", "test-model").generate_story(
            make_input()
        )

    assert len(attempts) == 1


def test_provider_retries_503_once_then_succeeds(monkeypatch) -> None:
    attempts = make_mocked_sdk_attempts(monkeypatch, [503, 200])

    result = GoogleGeminiStoryProvider("testing-only", "test-model").generate_story(
        make_input()
    )

    assert result == make_generated_content()
    assert len(attempts) == 2


def test_provider_stops_after_retry_limit_for_repeated_503(monkeypatch) -> None:
    attempts = make_mocked_sdk_attempts(monkeypatch, [503, 503, 503, 503])

    with pytest.raises(
        GeminiServiceUnavailableError,
        match="Gemini AI is currently experiencing high demand",
    ):
        GoogleGeminiStoryProvider("testing-only", "test-model").generate_story(
            make_input()
        )

    assert len(attempts) == 4


def test_provider_does_not_retry_permanent_api_errors(monkeypatch) -> None:
    attempts = make_mocked_sdk_attempts(monkeypatch, [400])

    with pytest.raises(GeminiProviderError, match="status 400") as error:
        GoogleGeminiStoryProvider("testing-only", "test-model").generate_story(
            make_input()
        )

    assert not isinstance(error.value, GeminiServiceUnavailableError)
    assert len(attempts) == 1


def test_provider_rejects_invalid_structured_response(monkeypatch) -> None:
    client = MagicMock()
    client.__enter__.return_value = client
    client.models.generate_content.return_value = SimpleNamespace(
        parsed=None, text='{"outline": {"panels": []}, "narration": {"panels": []}}'
    )
    monkeypatch.setattr(gemini.genai, "Client", MagicMock(return_value=client))

    with pytest.raises(InvalidGeminiResponseError):
        GoogleGeminiStoryProvider("testing-only", "test-model").generate_story(
            make_input()
        )


def test_huggingface_story_provider_normalizes_list_based_chat_completion() -> None:
    client = MagicMock()
    expected = make_generated_content()
    hf_response = expected.model_dump()
    hf_response["outline"] = hf_response["outline"]["panels"]
    hf_response["narration"] = hf_response["narration"]["panels"]
    client.chat_completion.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(hf_response))
            )
        ]
    )
    provider = HuggingFaceStoryProvider(
        token="testing-only", model="google/gemma-3-4b-it", client_factory=MagicMock(return_value=client)
    )

    result = provider.generate_story(make_input())

    assert result == expected
    assert len(result.outline.panels) == 5
    assert len(result.narration.panels) == 5
    assert result.outline.panels[0].title == "Panel 1"
    assert result.outline.panels[0].scene_description == "Scene 1"
    assert result.outline.panels[0].image_generation_prompt == "Illustration prompt 1"
    assert result.narration.panels[0].narration == "Narration 1"
    assert result.narration.panels[0].dialogue[0].line == "Line 1"
    call = client.chat_completion.call_args
    assert call.kwargs["model"] == "google/gemma-3-4b-it"
    assert call.kwargs["messages"][0]["role"] == "user"
    assert "A fox finds a lost star." in call.kwargs["messages"][0]["content"]
    assert call.kwargs["max_tokens"] == 1800


def test_huggingface_story_provider_rejects_non_five_panel_response() -> None:
    invalid_content = make_generated_content().model_dump()
    invalid_content["outline"] = invalid_content["outline"]["panels"][:-1]
    invalid_content["narration"] = invalid_content["narration"]["panels"]
    client = MagicMock()
    client.chat_completion.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(invalid_content))
            )
        ]
    )
    provider = HuggingFaceStoryProvider(
        token="testing-only", model="google/gemma-3-4b-it", client_factory=MagicMock(return_value=client)
    )

    with pytest.raises(InvalidHuggingFaceStoryResponseError, match="five-panel story"):
        provider.generate_story(make_input())


def test_huggingface_story_provider_maps_chat_completion_failure() -> None:
    client = MagicMock()
    client.chat_completion.side_effect = RuntimeError("mock HF failure")
    provider = HuggingFaceStoryProvider(
        token="testing-only", model="google/gemma-3-4b-it", client_factory=MagicMock(return_value=client)
    )

    with pytest.raises(HuggingFaceStoryProviderError, match="Hugging Face story generation failed"):
        provider.generate_story(make_input())


def test_huggingface_story_provider_requires_a_model_before_client_creation() -> None:
    client_factory = MagicMock()
    provider = HuggingFaceStoryProvider(
        token="testing-only", model=None, client_factory=client_factory
    )

    with pytest.raises(MissingHFStoryModelError, match="HF_STORY_MODEL"):
        provider.generate_story(make_input())

    client_factory.assert_not_called()


@pytest.mark.parametrize(
    "response_content",
    [
        pytest.param(make_generated_content().model_dump(), id="object"),
        pytest.param(json.dumps(make_generated_content().model_dump()), id="canonical-json"),
    ],
)
def test_huggingface_story_provider_accepts_canonical_response(response_content) -> None:
    provider, _ = make_huggingface_provider(response_content)

    assert provider.generate_story(make_input()) == make_generated_content()


@pytest.mark.parametrize(
    ("response_format", "expected_text"),
    [
        ("outline-list", "Panel 1"),
        ("narration-list", "Narration 1"),
        ("narration-panels-wrapper", "Narration 1"),
        ("image-prompt-alias", "Illustration prompt 1"),
        ("dialogue-aliases", "Line 1"),
    ],
)
def test_huggingface_story_provider_normalizes_common_schema_variations(
    response_format, expected_text
) -> None:
    content = make_generated_content().model_dump()
    if response_format == "outline-list":
        content["outline"] = content["outline"]["panels"]
    elif response_format == "narration-list":
        content["narration"] = content["narration"]["panels"]
    elif response_format == "narration-panels-wrapper":
        content["narration"] = {"panels": content["narration"]["panels"]}
    elif response_format == "image-prompt-alias":
        for panel in content["outline"]["panels"]:
            panel["image_prompt"] = panel.pop("image_generation_prompt")
    elif response_format == "dialogue-aliases":
        for panel in content["narration"]["panels"]:
            panel["dialogue"] = [
                {"speaker": line["character"], "text": line["line"]}
                for line in panel["dialogue"]
            ]
    provider, _ = make_huggingface_provider(json.dumps(content))

    result = provider.generate_story(make_input())

    if response_format == "image-prompt-alias":
        assert result.outline.panels[0].image_generation_prompt == expected_text
    elif response_format == "dialogue-aliases":
        assert result.narration.panels[0].dialogue[0].line == expected_text
    elif response_format.startswith("narration"):
        assert result.narration.panels[0].narration == expected_text
    else:
        assert result.outline.panels[0].title == expected_text


@pytest.mark.parametrize(
    ("dialogue_value", "expected_lines"),
    [
        ([{"character": "Pip", "line": "Hello"}], ["Hello"]),
        ("Hello", ["Hello"]),
        ("", []),
        ("   \t", []),
        (["Hello", "Hi"], ["Hello", "Hi"]),
        (None, []),
    ],
)
def test_huggingface_story_provider_normalizes_dialogue(
    dialogue_value, expected_lines
) -> None:
    content = make_generated_content().model_dump()
    content["narration"]["panels"][0]["dialogue"] = dialogue_value
    provider, _ = make_huggingface_provider(json.dumps(content))

    result = provider.generate_story(make_input())

    first_panel_dialogue = result.narration.panels[0].dialogue
    assert [line.line for line in first_panel_dialogue] == expected_lines
    if dialogue_value == "Hello" or dialogue_value == ["Hello", "Hi"]:
        assert all(line.character == "Pip" for line in first_panel_dialogue)


def test_huggingface_story_provider_accepts_fenced_json_with_surrounding_text() -> None:
    content = json.dumps(make_generated_content().model_dump())
    provider, _ = make_huggingface_provider(
        f"Here is the comic:\n```json\n{content}\n```\nEnjoy!"
    )

    assert provider.generate_story(make_input()) == make_generated_content()


def test_huggingface_story_provider_accepts_json_encoded_as_a_string() -> None:
    content = json.dumps(make_generated_content().model_dump())
    provider, _ = make_huggingface_provider(json.dumps(content))

    assert provider.generate_story(make_input()) == make_generated_content()


@pytest.mark.parametrize("panel_count", [4, 6])
def test_huggingface_story_provider_rejects_non_five_panel_sections(panel_count) -> None:
    content = make_generated_content().model_dump()
    content["outline"]["panels"] = content["outline"]["panels"][:panel_count]
    content["narration"]["panels"] = content["narration"]["panels"][:panel_count]
    if panel_count == 6:
        content["outline"]["panels"].append(
            {**content["outline"]["panels"][-1], "panel_number": 6}
        )
        content["narration"]["panels"].append(
            {**content["narration"]["panels"][-1], "panel_number": 6}
        )
    provider, _ = make_huggingface_provider(json.dumps(content))

    with pytest.raises(InvalidHuggingFaceStoryResponseError, match="exactly five panels"):
        provider.generate_story(make_input())


@pytest.mark.parametrize(
    "response_content",
    [
        "not a story, just prose",
        '{"outline": [broken json], "narration": []}',
        (
            json.dumps(make_generated_content().model_dump())
            + "\n"
            + json.dumps(make_generated_content().model_dump())
        ),
        json.dumps({"outline": "not an outline", "narration": {"panels": []}}),
    ],
    ids=["prose", "malformed-json", "multiple-story-objects", "unsupported-shape"],
)
def test_huggingface_story_provider_rejects_malformed_or_ambiguous_response(
    response_content
) -> None:
    provider, _ = make_huggingface_provider(response_content)

    with pytest.raises(InvalidHuggingFaceStoryResponseError):
        provider.generate_story(make_input())


def test_huggingface_story_provider_rejects_missing_required_panel_fields() -> None:
    content = make_generated_content().model_dump()
    del content["outline"]["panels"][0]["image_generation_prompt"]
    provider, _ = make_huggingface_provider(json.dumps(content))

    with pytest.raises(InvalidHuggingFaceStoryResponseError, match="image_generation_prompt"):
        provider.generate_story(make_input())


def test_huggingface_story_provider_rejects_object_dialogue_values() -> None:
    content = make_generated_content().model_dump()
    content["narration"]["panels"][0]["dialogue"] = {"line": "Hello"}
    provider, _ = make_huggingface_provider(json.dumps(content))

    with pytest.raises(InvalidHuggingFaceStoryResponseError, match="dialogue"):
        provider.generate_story(make_input())


def test_story_falls_back_from_gemini_429_to_mocked_huggingface_provider() -> None:
    primary = MagicMock()
    primary.generate_story.side_effect = GeminiRateLimitError("Gemini quota reached.")
    content = make_generated_content().model_dump()
    content["narration"]["panels"][0]["dialogue"] = "Beep boop initializing"
    fallback, client = make_huggingface_provider(json.dumps(content))

    story = StoryGenerationService(primary, fallback_provider=fallback).generate(make_input())

    assert len(story.panels) == 5
    assert story.panels[0].dialogue[0].character == "Pip"
    assert story.panels[0].dialogue[0].line == "Beep boop initializing"
    primary.generate_story.assert_called_once()
    client.chat_completion.assert_called_once()


def test_story_wraps_invalid_huggingface_response_after_gemini_429() -> None:
    primary = MagicMock()
    primary.generate_story.side_effect = GeminiRateLimitError("Gemini quota reached.")
    fallback, client = make_huggingface_provider("not valid JSON")

    with pytest.raises(ProviderFallbackError) as error:
        StoryGenerationService(primary, fallback_provider=fallback).generate(make_input())

    assert isinstance(error.value.fallback_error, InvalidHuggingFaceStoryResponseError)
    primary.generate_story.assert_called_once()
    client.chat_completion.assert_called_once()


@pytest.fixture
def api_client():
    provider = MagicMock()
    provider.generate_story.return_value = make_generated_content()
    app.dependency_overrides[get_story_generation_service] = lambda: StoryGenerationService(
        provider
    )
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.pop(get_story_generation_service, None)


def test_story_falls_back_to_huggingface_when_gemini_is_rate_limited() -> None:
    primary = MagicMock()
    primary.generate_story.side_effect = GeminiRateLimitError("Gemini rate limit reached.")
    fallback = MagicMock()
    fallback.generate_story.return_value = make_generated_content()

    story = StoryGenerationService(primary, fallback_provider=fallback).generate(make_input())

    assert len(story.panels) == 5
    assert primary.generate_story.call_count == 1
    assert fallback.generate_story.call_count == 1


def test_story_raises_combined_error_when_both_providers_fail() -> None:
    primary = MagicMock()
    primary.generate_story.side_effect = GeminiRateLimitError("Gemini quota reached.")
    fallback = MagicMock()
    fallback.generate_story.side_effect = ValueError("HF down")

    with pytest.raises(ProviderFallbackError, match="both providers failed|fallback"):
        StoryGenerationService(primary, fallback_provider=fallback).generate(make_input())


def test_story_endpoint_returns_five_combined_panels(api_client) -> None:
    response = api_client.post("/generate-comic/json", json=make_input().model_dump())

    assert response.status_code == 200
    assert len(response.json()["panels"]) == 5
    assert response.json()["panels"][0]["narration"] == "Narration 1"
    assert response.json()["panels"][0]["image_generation_prompt"] == "Illustration prompt 1"


def test_story_endpoint_returns_safe_quota_message(api_client) -> None:
    provider_error = genai.errors.APIError(
        429,
        {
            "error": {
                "message": "You exceeded your current quota.",
                "status": "RESOURCE_EXHAUSTED",
                "details": [
                    {
                        "violations": [
                            {
                                "quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                                "quotaValue": "20",
                                "quotaDimensions": {"model": "gemini-3.8-flash"},
                            }
                        ]
                    }
                ],
            }
        },
    )
    quota_error = GeminiRateLimitError("Gemini rate limit reached. Please try again shortly.")
    quota_error.__cause__ = provider_error
    service = MagicMock()
    service.generate.side_effect = quota_error
    app.dependency_overrides[get_story_generation_service] = lambda: service

    response = api_client.post("/generate-comic/json", json=make_input().model_dump())

    assert response.status_code == 429
    assert response.json() == {
        "success": False,
        "error_code": "GEMINI_RATE_LIMITED",
        "message": "Gemini AI usage limit has been reached. Please wait for the quota to reset or use an available fallback service.",
    }


@pytest.mark.parametrize(
    ("error", "status_code"),
    [
        (MissingGeminiApiKeyError("No Gemini API key is configured."), 503),
        (MissingHFStoryModelError("Set HF_STORY_MODEL."), 503),
        (GeminiRateLimitError("Gemini rate limit reached."), 429),
        (GeminiProviderError("Gemini request failed."), 503),
        (InvalidGeminiResponseError("Invalid Gemini response."), 503),
        (HuggingFaceStoryProviderError("private provider detail"), 503),
        (InvalidHuggingFaceStoryResponseError("private response detail"), 503),
    ],
)
def test_story_endpoint_maps_generation_errors(api_client, error, status_code) -> None:
    service = MagicMock()
    service.generate.side_effect = error
    app.dependency_overrides[get_story_generation_service] = lambda: service

    response = api_client.post("/generate-comic/json", json=make_input().model_dump())

    assert response.status_code == status_code
    assert response.json()["success"] is False
    assert response.json()["error_code"]
    assert response.json()["message"]
    assert "private" not in response.json()["message"]


def test_story_endpoint_validates_input(api_client) -> None:
    response = api_client.post("/generate-comic/json", json={"story_prompt": "A fox"})

    assert response.status_code == 422