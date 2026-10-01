from fastapi.testclient import TestClient

from comiccraft.config import get_settings
from comiccraft.main import app

client = TestClient(app)


def test_home_page_renders() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "ComicCraft" in response.text
    assert "COMIC STUDIO" in response.text
    assert 'action="/generate/progress"' in response.text
    assert "/static/css/styles.css" in response.text


def test_stylesheet_is_served() -> None:
    response = client.get("/static/css/styles.css")

    assert response.status_code == 200
    assert "--paper:" in response.text


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_provider_settings_come_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-gemini-model")
    monkeypatch.setenv("IMAGE_PROVIDER", "test-image-provider")
    monkeypatch.setenv("IMAGE_API_KEY", "test-image-key")
    monkeypatch.setenv("IMAGE_MODEL", "test-image-model")

    settings = get_settings()

    assert settings.gemini_api_key == "test-gemini-key"
    assert settings.gemini_model == "test-gemini-model"
    assert settings.image_provider == "test-image-provider"
    assert settings.image_api_key == "test-image-key"
    assert settings.image_model == "test-image-model"


def test_image_provider_defaults_to_local(monkeypatch) -> None:
    monkeypatch.delenv("IMAGE_PROVIDER", raising=False)

    assert get_settings().image_provider == "local"