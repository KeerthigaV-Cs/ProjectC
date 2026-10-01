import base64
from io import BytesIO
from threading import Barrier, Lock
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
import requests
from PIL import Image
from fastapi.testclient import TestClient
from google import genai
from huggingface_hub.errors import HfHubHTTPError

from comiccraft.config import Settings, get_settings
from comiccraft.errors import (
    ImageProviderCreditsExhaustedError,
    ImageProviderError,
    ImageProviderRateLimitError,
    ImageProviderServiceUnavailableError,
    ImageProviderTimeoutError,
    InvalidImageResponseError,
    MissingImageApiKeyError,
    MissingHFTokenError,
    ProviderFallbackError,
    UnsupportedImageProviderError,
)
from comiccraft.models.images import GeneratedImageData
from comiccraft.models.story import ComicInput, ComicStory
from comiccraft.providers import gemini_image
from comiccraft.providers import huggingface_image, image_factory
from comiccraft.providers.gemini_image import GeminiImageGenerationProvider
from comiccraft.providers.huggingface_image import HuggingFaceImageGenerationProvider
from comiccraft.providers.image_factory import create_image_provider
from comiccraft.providers.local_image import LocalImageGenerationProvider
from comiccraft import routes
from comiccraft.routes import get_image_generation_service
from comiccraft.services.image_generation import ImageGenerationService
from comiccraft.main import app


class RateLimitError(Exception):
    code = 429


def make_story() -> ComicStory:
    user_input = ComicInput(
        story_prompt="A fox finds a lost star.",
        character_name="Pip",
        setting="A moonlit forest",
        tone="Whimsical",
        art_style="Watercolor comic",
    )
    return ComicStory(
        user_input=user_input,
        panels=[
            {
                "panel_number": number,
                "title": f"Panel {number}",
                "scene_description": f"Scene {number}",
                "image_generation_prompt": f"Illustration prompt {number}",
                "narration": f"Narration {number}",
                "dialogue": [],
            }
            for number in range(1, 6)
        ],
    )


def make_image_response(data: bytes = b"jpeg-data", mime_type: str = "image/jpeg"):
    return SimpleNamespace(
        output_image=SimpleNamespace(
            data=base64.b64encode(data).decode("ascii"),
            mime_type=mime_type,
        )
    )


def test_image_provider_requests_jpeg_and_extracts_data(monkeypatch) -> None:
    client = MagicMock()
    client.__enter__.return_value = client
    client.interactions.create.return_value = make_image_response()
    client_factory = MagicMock(return_value=client)
    monkeypatch.setattr(gemini_image.genai, "Client", client_factory)

    result = GeminiImageGenerationProvider("test-only-key", "test-image-model").generate_image(
        "A fox in a moonlit forest"
    )

    assert result == GeneratedImageData(data=b"jpeg-data", mime_type="image/jpeg")
    client_factory.assert_called_once_with(api_key="test-only-key")
    call = client.interactions.create.call_args.kwargs
    assert call["model"] == "test-image-model"
    assert call["input"] == "A fox in a moonlit forest"
    assert call["timeout"] == 90
    assert call["response_format"] == {
        "type": "image",
        "mime_type": "image/jpeg",
        "aspect_ratio": "3:2",
        "image_size": "1K",
    }
    assert "generation_config" not in call


def test_image_provider_accepts_raw_output_image_bytes(monkeypatch) -> None:
    image_bytes = b"\x89PNG\r\n\x1a\nmock-png-data"
    client = MagicMock()
    client.__enter__.return_value = client
    client.interactions.create.return_value = SimpleNamespace(
        output_image=SimpleNamespace(data=image_bytes, mime_type="image/png")
    )
    monkeypatch.setattr(gemini_image.genai, "Client", MagicMock(return_value=client))

    result = GeminiImageGenerationProvider("test-only-key", "test-image-model").generate_image(
        "A fox in a moonlit forest"
    )

    assert result == GeneratedImageData(data=image_bytes, mime_type="image/png")


def test_image_provider_requires_api_key() -> None:
    with pytest.raises(MissingImageApiKeyError):
        GeminiImageGenerationProvider(None, "test-image-model").generate_image("fox")


def test_image_provider_maps_rate_limits(monkeypatch) -> None:
    client = MagicMock()
    client.__enter__.return_value = client
    client.interactions.create.side_effect = genai.errors.APIError(
        429, {"error": {"message": "Too many requests", "status": "RESOURCE_EXHAUSTED"}}
    )
    monkeypatch.setattr(gemini_image.genai, "Client", MagicMock(return_value=client))

    with pytest.raises(ImageProviderRateLimitError):
        GeminiImageGenerationProvider("test-only-key", "test-image-model").generate_image(
            "fox"
        )


def test_image_provider_maps_sdk_rate_limit_error_and_preserves_cause(monkeypatch) -> None:
    client = MagicMock()
    client.__enter__.return_value = client
    rate_limit_error = RateLimitError("429 rate limit exceeded")
    client.interactions.create.side_effect = rate_limit_error
    monkeypatch.setattr(gemini_image.genai, "Client", MagicMock(return_value=client))

    with pytest.raises(ImageProviderRateLimitError) as error:
        GeminiImageGenerationProvider("test-only-key", "test-image-model").generate_image(
            "fox"
        )

    assert error.value.__cause__ is rate_limit_error


def test_image_provider_maps_other_api_errors(monkeypatch) -> None:
    client = MagicMock()
    client.__enter__.return_value = client
    client.interactions.create.side_effect = genai.errors.APIError(
        503, {"error": {"message": "Unavailable", "status": "UNAVAILABLE"}}
    )
    monkeypatch.setattr(gemini_image.genai, "Client", MagicMock(return_value=client))

    with pytest.raises(ImageProviderError):
        GeminiImageGenerationProvider("test-only-key", "test-image-model").generate_image(
            "fox"
        )


def test_image_provider_maps_timeouts(monkeypatch) -> None:
    client = MagicMock()
    client.__enter__.return_value = client
    client.interactions.create.side_effect = httpx.ReadTimeout("timed out")
    monkeypatch.setattr(gemini_image.genai, "Client", MagicMock(return_value=client))

    with pytest.raises(ImageProviderTimeoutError):
        GeminiImageGenerationProvider("test-only-key", "test-image-model").generate_image(
            "fox"
        )


def test_image_provider_maps_network_errors_to_timeout(monkeypatch) -> None:
    client = MagicMock()
    client.__enter__.return_value = client
    network_error = httpx.ConnectError(
        "connection failed", request=httpx.Request("POST", "https://example.invalid")
    )
    client.interactions.create.side_effect = network_error
    monkeypatch.setattr(gemini_image.genai, "Client", MagicMock(return_value=client))

    with pytest.raises(ImageProviderTimeoutError) as error:
        GeminiImageGenerationProvider("test-only-key", "test-image-model").generate_image(
            "fox"
        )

    assert error.value.__cause__ is network_error


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(output_image=None),
        SimpleNamespace(output_image=SimpleNamespace(data="not-base64", mime_type="image/jpeg")),
        make_image_response(b"image-data", "text/plain"),
    ],
)
def test_image_provider_rejects_invalid_responses(monkeypatch, response) -> None:
    client = MagicMock()
    client.__enter__.return_value = client
    client.interactions.create.return_value = response
    monkeypatch.setattr(gemini_image.genai, "Client", MagicMock(return_value=client))

    with pytest.raises(InvalidImageResponseError):
        GeminiImageGenerationProvider("test-only-key", "test-image-model").generate_image(
            "fox"
        )


def test_image_provider_factory_uses_environment_settings() -> None:
    provider = create_image_provider(
        Settings(
            gemini_api_key=None,
            gemini_model=None,
            image_provider="gemini",
            image_api_key="test-only-key",
            image_model="test-image-model",
        )
    )

    assert isinstance(provider, GeminiImageGenerationProvider)
    assert provider._model == "test-image-model"


def test_image_provider_factory_rejects_unknown_provider() -> None:
    with pytest.raises(UnsupportedImageProviderError):
        create_image_provider(
            Settings(
                gemini_api_key=None,
                gemini_model=None,
                image_provider="unknown",
                image_api_key="test-only-key",
                image_model="model",
            )
        )


def test_local_image_provider_generates_comic_style_png() -> None:
    provider = LocalImageGenerationProvider()

    result = provider.generate_image(
        "Panel number: 3\nCharacter: Pip\nScene: Pip discovers a glowing star."
    )

    assert result.mime_type == "image/png"
    assert result.data.startswith(b"\x89PNG\r\n\x1a\n")
    with Image.open(BytesIO(result.data)) as image:
        assert image.size == (768, 512)
        assert image.format == "PNG"


def test_local_provider_selection_never_constructs_or_calls_external_clients(monkeypatch) -> None:
    hosted_provider_factory = MagicMock(side_effect=AssertionError("Hosted provider constructed"))
    gemini_client_factory = MagicMock(side_effect=AssertionError("Gemini client constructed"))
    hf_client_factory = MagicMock(side_effect=AssertionError("HF client constructed"))
    monkeypatch.setattr(image_factory, "GeminiImageGenerationProvider", hosted_provider_factory)
    monkeypatch.setattr(image_factory, "HuggingFaceImageGenerationProvider", hosted_provider_factory)
    monkeypatch.setattr(gemini_image.genai, "Client", gemini_client_factory)
    monkeypatch.setattr(huggingface_image, "InferenceClient", hf_client_factory)
    provider = create_image_provider(Settings(image_provider="local"))

    assert isinstance(provider, LocalImageGenerationProvider)
    assert provider.generate_image("A fox visits a moonlit forest.").mime_type == "image/png"
    hosted_provider_factory.assert_not_called()
    gemini_client_factory.assert_not_called()
    hf_client_factory.assert_not_called()


def test_local_image_provider_generates_and_stores_all_five_panels(tmp_path) -> None:
    class RecordingLocalProvider(LocalImageGenerationProvider):
        def __init__(self):
            self.prompts = []

        def generate_image(self, prompt: str) -> GeneratedImageData:
            self.prompts.append(prompt)
            return super().generate_image(prompt)

    provider = RecordingLocalProvider()
    service = ImageGenerationService(provider, output_directory=tmp_path)
    completed_panels = []

    result = service.generate_for_story(
        make_story(), progress_callback=completed_panels.append
    )

    assert [panel.panel_number for panel in result.panels] == [1, 2, 3, 4, 5]
    assert completed_panels == [1, 2, 3, 4, 5]
    assert len(provider.prompts) == 5
    for number, prompt in enumerate(provider.prompts, start=1):
        assert f"Panel number: {number}" in prompt
        assert "Character: Pip" in prompt
        assert f"Scene: Scene {number}" in prompt
        image_path = tmp_path / result.story_id / f"panel-{number}.png"
        assert image_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert result.panels[number - 1].image_url.endswith(f"panel-{number}.png")


def test_image_provider_factory_reads_local_selection_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("IMAGE_PROVIDER", "local")
    settings = get_settings()

    assert settings.image_provider == "local"
    assert isinstance(create_image_provider(settings), LocalImageGenerationProvider)


def test_huggingface_image_provider_uses_configured_model_and_returns_image_bytes() -> None:
    image_bytes = b"\x89PNG\r\n\x1a\nmock-png-data"
    client = MagicMock()
    client.text_to_image.return_value = image_bytes
    client_factory = MagicMock(return_value=client)
    provider = HuggingFaceImageGenerationProvider(
        token="test-only-token",
        model="black-forest-labs/FLUX.1-schnell",
        client_factory=client_factory,
    )

    result = provider.generate_image("A fox in a moonlit forest")

    assert result == GeneratedImageData(data=image_bytes, mime_type="image/png")
    client_factory.assert_called_once_with(token="test-only-token")
    client.text_to_image.assert_called_once_with(
        "A fox in a moonlit forest", model="black-forest-labs/FLUX.1-schnell"
    )


def test_huggingface_image_provider_requires_token_before_client_creation() -> None:
    client_factory = MagicMock()
    provider = HuggingFaceImageGenerationProvider(
        token=None, model="test-model", client_factory=client_factory
    )

    with pytest.raises(MissingHFTokenError):
        provider.generate_image("fox")

    client_factory.assert_not_called()


def test_huggingface_image_provider_preserves_and_safely_logs_api_failure(caplog) -> None:
    client = MagicMock()
    provider_error = RuntimeError(
        "authorization: Bearer secret-token; API_KEY=another-secret; model unavailable"
    )
    client.text_to_image.side_effect = provider_error
    provider = HuggingFaceImageGenerationProvider(
        token="secret-token",
        model="black-forest-labs/FLUX.1-schnell",
        client_factory=MagicMock(return_value=client),
    )

    with pytest.raises(ImageProviderError) as error:
        provider.generate_image("fox")

    assert "secret-token" not in str(error.value)
    assert error.value.__cause__ is provider_error
    assert "exception_type=RuntimeError" in caplog.text
    assert "provider=huggingface" in caplog.text
    assert "model=black-forest-labs/FLUX.1-schnell" in caplog.text
    assert "model unavailable" in caplog.text
    assert "secret-token" not in caplog.text
    assert "another-secret" not in caplog.text
    assert "authorization: <redacted>" in caplog.text


def test_huggingface_image_provider_maps_and_logs_rate_limit(caplog) -> None:
    provider_error = RuntimeError("Too many requests")
    provider_error.response = SimpleNamespace(status_code=429)
    client = MagicMock()
    client.text_to_image.side_effect = provider_error
    provider = HuggingFaceImageGenerationProvider(
        token="test-token", model="test-model", client_factory=MagicMock(return_value=client)
    )

    with pytest.raises(ImageProviderRateLimitError) as error:
        provider.generate_image("fox")

    assert error.value.__cause__ is provider_error
    assert "status_code=429" in caplog.text
    assert "Too many requests" in caplog.text


@pytest.mark.parametrize(
    ("status_code", "expected_error"),
    [
        (402, ImageProviderCreditsExhaustedError),
        (429, ImageProviderRateLimitError),
        (503, ImageProviderServiceUnavailableError),
        (401, MissingHFTokenError),
        (403, MissingHFTokenError),
    ],
)
def test_huggingface_image_provider_maps_http_statuses(
    status_code, expected_error
) -> None:
    provider_error = RuntimeError("private provider response with sensitive details")
    provider_error.response = SimpleNamespace(status_code=status_code)
    client = MagicMock()
    client.text_to_image.side_effect = provider_error
    provider = HuggingFaceImageGenerationProvider(
        token="test-token", model="test-model", client_factory=MagicMock(return_value=client)
    )

    with pytest.raises(expected_error) as error:
        provider.generate_image("fox")

    assert error.value.__cause__ is provider_error
    assert "private provider response" not in str(error.value)


def test_huggingface_sdk_http_error_maps_402_credits_without_api_call() -> None:
    response = requests.Response()
    response.status_code = 402
    response.request = requests.Request(
        "POST", "https://example.invalid/inference"
    ).prepare()
    provider_error = HfHubHTTPError(
        "Payment Required: monthly credits exhausted", response=response
    )
    wrapped_error = RuntimeError("Inference client wrapped the HTTP error")
    wrapped_error.__cause__ = provider_error
    client = MagicMock()
    client.text_to_image.side_effect = wrapped_error
    provider = HuggingFaceImageGenerationProvider(
        token="test-token", model="test-model", client_factory=MagicMock(return_value=client)
    )

    with pytest.raises(ImageProviderCreditsExhaustedError) as error:
        provider.generate_image("fox")

    assert error.value.__cause__ is wrapped_error
    assert "monthly credits exhausted" not in str(error.value)


def test_huggingface_image_provider_maps_network_errors_to_timeout() -> None:
    provider_error = httpx.ConnectError(
        "connection failed", request=httpx.Request("POST", "https://example.invalid")
    )
    client = MagicMock()
    client.text_to_image.side_effect = provider_error
    provider = HuggingFaceImageGenerationProvider(
        token="test-token", model="test-model", client_factory=MagicMock(return_value=client)
    )

    with pytest.raises(ImageProviderTimeoutError) as error:
        provider.generate_image("fox")

    assert error.value.__cause__ is provider_error


def test_huggingface_image_provider_maps_timeout_and_preserves_cause() -> None:
    from huggingface_hub.errors import InferenceTimeoutError

    provider_error = InferenceTimeoutError("Inference request timed out")
    client = MagicMock()
    client.text_to_image.side_effect = provider_error
    provider = HuggingFaceImageGenerationProvider(
        token="test-token", model="test-model", client_factory=MagicMock(return_value=client)
    )

    with pytest.raises(ImageProviderTimeoutError) as error:
        provider.generate_image("fox")

    assert error.value.__cause__ is provider_error


@pytest.mark.parametrize("response", [None, b""])
def test_huggingface_image_provider_rejects_empty_response(response) -> None:
    client = MagicMock()
    client.text_to_image.return_value = response
    provider = HuggingFaceImageGenerationProvider(
        token="test-token", model="test-model", client_factory=MagicMock(return_value=client)
    )

    with pytest.raises(InvalidImageResponseError):
        provider.generate_image("fox")


def test_huggingface_image_provider_requires_configured_model() -> None:
    client_factory = MagicMock()
    provider = HuggingFaceImageGenerationProvider(
        token="test-token", model=None, client_factory=client_factory
    )

    with pytest.raises(ImageProviderError, match="HF_IMAGE_MODEL"):
        provider.generate_image("fox")

    client_factory.assert_not_called()


def test_huggingface_image_provider_stores_images_for_all_five_panels(tmp_path) -> None:
    image_bytes = b"\x89PNG\r\n\x1a\nmock-png-data"
    image_output = SimpleNamespace(
        format="PNG",
        save=lambda buffer, format: buffer.write(image_bytes),
    )
    client = MagicMock()
    client.text_to_image.return_value = image_output
    provider = HuggingFaceImageGenerationProvider(
        token="test-token",
        model="black-forest-labs/FLUX.1-schnell",
        client_factory=MagicMock(return_value=client),
    )
    service = ImageGenerationService(provider, output_directory=tmp_path)

    result = service.generate_for_story(make_story())

    assert [panel.panel_number for panel in result.panels] == [1, 2, 3, 4, 5]
    assert client.text_to_image.call_count == 5
    for number, panel in enumerate(result.panels, start=1):
        path = tmp_path / result.story_id / f"panel-{number}.png"
        assert path.read_bytes() == image_bytes
        assert panel.image_url == f"/static/panels/{result.story_id}/panel-{number}.png"
    assert [call.kwargs["model"] for call in client.text_to_image.call_args_list] == [
        "black-forest-labs/FLUX.1-schnell"
    ] * 5


def test_settings_use_huggingface_image_provider_and_model_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("IMAGE_PROVIDER", "huggingface")
    monkeypatch.setenv("HF_IMAGE_MODEL", "black-forest-labs/FLUX.1-schnell")

    settings = get_settings()

    assert settings.image_provider == "huggingface"
    assert settings.hf_image_model == "black-forest-labs/FLUX.1-schnell"


def test_image_provider_factory_and_service_do_not_construct_gemini_for_hf(monkeypatch) -> None:
    settings = Settings(
        image_provider="huggingface",
        hf_token="test-token",
        hf_image_model="black-forest-labs/FLUX.1-schnell",
    )
    gemini_factory = MagicMock(side_effect=AssertionError("Gemini must not be constructed"))
    monkeypatch.setattr(image_factory, "GeminiImageGenerationProvider", gemini_factory)
    provider = create_image_provider(settings)
    assert isinstance(provider, HuggingFaceImageGenerationProvider)
    assert provider._model == "black-forest-labs/FLUX.1-schnell"
    gemini_factory.assert_not_called()

    service_factory = MagicMock(wraps=create_image_provider)
    monkeypatch.setattr(routes, "get_settings", lambda: settings)
    monkeypatch.setattr(routes, "create_image_provider", service_factory)

    service = routes.get_image_generation_service()

    service_factory.assert_called_once_with(settings, "huggingface")
    assert isinstance(service._provider, HuggingFaceImageGenerationProvider)
    assert service._fallback_provider is None
    gemini_factory.assert_not_called()


def test_image_service_generates_and_saves_five_images(tmp_path) -> None:
    barrier = Barrier(5)
    state_lock = Lock()
    active_requests = 0
    max_active_requests = 0

    class ConcurrentRecordingProvider:
        def generate_image(self, prompt: str) -> GeneratedImageData:
            nonlocal active_requests, max_active_requests
            number = int(prompt.rsplit(" ", 1)[-1])
            with state_lock:
                active_requests += 1
                max_active_requests = max(max_active_requests, active_requests)
            barrier.wait(timeout=3)
            with state_lock:
                active_requests -= 1
            return GeneratedImageData(
                data=f"panel-{number}".encode(), mime_type="image/jpeg"
            )

    provider = ConcurrentRecordingProvider()
    service = ImageGenerationService(provider, output_directory=tmp_path)

    result = service.generate_for_story(make_story())

    assert [panel.panel_number for panel in result.panels] == [1, 2, 3, 4, 5]
    assert max_active_requests == 5
    for number, panel in enumerate(result.panels, start=1):
        path = tmp_path / result.story_id / f"panel-{number}.jpg"
        assert path.read_bytes() == f"panel-{number}".encode()
        assert panel.image_url.endswith(f"panel-{number}.jpg")


def test_image_service_does_not_leave_partial_output_on_provider_error(tmp_path) -> None:
    provider = MagicMock()

    def generate_image(prompt: str) -> GeneratedImageData:
        if prompt.endswith("2"):
            raise ImageProviderRateLimitError("rate limited")
        return GeneratedImageData(data=prompt.encode(), mime_type="image/jpeg")

    provider.generate_image.side_effect = generate_image
    service = ImageGenerationService(provider, output_directory=tmp_path)

    with pytest.raises(ImageProviderRateLimitError):
        service.generate_for_story(make_story())

    assert list(tmp_path.iterdir()) == []


def test_image_service_falls_back_to_huggingface_on_rate_limit(tmp_path) -> None:
    primary = MagicMock()
    primary.generate_image.side_effect = ImageProviderRateLimitError("rate limited")
    fallback = MagicMock()
    fallback.generate_image.side_effect = [
        GeneratedImageData(data=b"fallback-panel-1", mime_type="image/jpeg")
        for _ in range(5)
    ]

    service = ImageGenerationService(primary, fallback_provider=fallback, output_directory=tmp_path)
    result = service.generate_for_story(make_story())

    assert result.story_id
    assert 1 <= primary.generate_image.call_count <= 5
    assert fallback.generate_image.call_count == 5


def test_gemini_429_uses_huggingface_fallback_for_all_five_panels(
    monkeypatch, tmp_path
) -> None:
    rate_limit_error = RateLimitError("429 rate limit exceeded")
    gemini_client = MagicMock()
    gemini_client.__enter__.return_value = gemini_client
    gemini_client.interactions.create.side_effect = rate_limit_error
    monkeypatch.setattr(
        gemini_image.genai, "Client", MagicMock(return_value=gemini_client)
    )

    image_bytes = b"\x89PNG\r\n\x1a\nmock-png-image"
    hf_client = MagicMock()
    hf_client.text_to_image.return_value = image_bytes
    hf_client_factory = MagicMock(return_value=hf_client)
    monkeypatch.setattr(huggingface_image, "InferenceClient", hf_client_factory)
    monkeypatch.setattr(
        routes,
        "get_settings",
        lambda: Settings(
            image_provider="gemini",
            image_fallback_provider="huggingface",
            image_api_key="test-gemini-key",
            image_model="test-gemini-image-model",
            hf_token="test-hf-token",
            hf_image_model="test-hf-image-model",
        ),
    )
    monkeypatch.setattr(
        routes,
        "ImageGenerationService",
        lambda provider, fallback_provider=None: ImageGenerationService(
            provider,
            output_directory=tmp_path,
            fallback_provider=fallback_provider,
        ),
    )

    service = routes.get_image_generation_service()
    result = service.generate_for_story(make_story())

    assert [panel.panel_number for panel in result.panels] == [1, 2, 3, 4, 5]
    assert 1 <= gemini_client.interactions.create.call_count <= 5
    assert hf_client_factory.call_count == 5
    assert all(
        call.kwargs == {"token": "test-hf-token"}
        for call in hf_client_factory.call_args_list
    )
    assert hf_client.text_to_image.call_count == 5
    for number in range(1, 6):
        assert (tmp_path / result.story_id / f"panel-{number}.png").read_bytes() == image_bytes


def test_image_service_raises_combined_error_when_fallback_also_fails(tmp_path) -> None:
    primary = MagicMock()
    primary.generate_image.side_effect = ImageProviderRateLimitError("rate limited")
    fallback = MagicMock()
    fallback.generate_image.side_effect = ValueError("HF down")

    with pytest.raises(ProviderFallbackError, match="both providers failed|fallback"):
        ImageGenerationService(primary, fallback_provider=fallback, output_directory=tmp_path).generate_for_story(make_story())


def test_image_service_exposes_huggingface_model_configuration_error(tmp_path) -> None:
    primary = MagicMock()
    primary.generate_image.side_effect = ImageProviderRateLimitError("rate limited")
    fallback = HuggingFaceImageGenerationProvider(
        token="test-token", model=None, client_factory=MagicMock()
    )

    with pytest.raises(ProviderFallbackError, match="HF_IMAGE_MODEL") as error:
        ImageGenerationService(
            primary, fallback_provider=fallback, output_directory=tmp_path
        ).generate_for_story(make_story())

    assert isinstance(error.value.fallback_error, ImageProviderError)


@pytest.fixture
def api_client(tmp_path):
    provider = MagicMock()
    provider.generate_image.return_value = GeneratedImageData(
        data=b"panel-data", mime_type="image/jpeg"
    )
    app.dependency_overrides[get_image_generation_service] = lambda: ImageGenerationService(
        provider, output_directory=tmp_path
    )
    with TestClient(app) as test_client:
        yield test_client, provider
    app.dependency_overrides.pop(get_image_generation_service, None)


def test_image_generation_route_returns_five_images(api_client) -> None:
    test_client, provider = api_client
    response = test_client.post("/generate-comic/images", json=make_story().model_dump())

    assert response.status_code == 200
    assert len(response.json()["panels"]) == 5
    assert provider.generate_image.call_count == 5


def test_image_generation_route_validates_story_input(api_client) -> None:
    test_client, provider = api_client
    response = test_client.post("/generate-comic/images", json={"panels": []})

    assert response.status_code == 422
    provider.generate_image.assert_not_called()


@pytest.mark.parametrize(
    ("error", "status_code"),
    [
        (MissingImageApiKeyError("No image API key is configured."), 503),
        (MissingHFTokenError("private missing token detail"), 503),
        (ImageProviderCreditsExhaustedError("private 402 response"), 402),
        (ImageProviderRateLimitError("Image rate limit reached."), 429),
        (ImageProviderServiceUnavailableError("private 503 response"), 503),
        (ImageProviderTimeoutError("Image request timed out."), 504),
        (ImageProviderError("Image request failed."), 502),
        (InvalidImageResponseError("Invalid image response."), 502),
    ],
)
def test_image_generation_route_maps_provider_errors(api_client, error, status_code) -> None:
    test_client, provider = api_client
    provider.generate_image.side_effect = error

    response = test_client.post("/generate-comic/images", json=make_story().model_dump())

    assert response.status_code == status_code
    assert response.json()["success"] is False
    assert response.json()["error_code"]
    assert response.json()["message"]
    assert "private" not in response.json()["message"]
    if isinstance(error, ImageProviderCreditsExhaustedError):
        assert response.json()["message"] == (
            "Hugging Face image-generation credits have been used up. Please add Hugging Face credits or switch to another image provider."
        )