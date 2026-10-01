from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from PIL import Image
import pytest
from pypdf import PdfReader

from comiccraft.errors import (
    ComicPDFExportError,
    ComicPDFNotFoundError,
    MissingComicPanelImageError,
)
from comiccraft.main import app
from comiccraft.models.images import ComicImageSet
from comiccraft.models.story import ComicInput, ComicStory, DialogueLine
from comiccraft.routes import get_pdf_export_service
from comiccraft.services.pdf_export import ComicPDFExportService


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
                "title": f"Panel title {number}",
                "scene_description": f"Scene caption {number}.",
                "image_generation_prompt": f"Image prompt {number}.",
                "narration": f"Narration text {number}.",
                "dialogue": [
                    {"character": "Pip", "line": f"Dialogue line {number}."}
                ],
            }
            for number in range(1, 6)
        ],
    )


def make_image_set() -> ComicImageSet:
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


def create_panel_images(panel_directory: Path) -> None:
    story_directory = panel_directory / "test-story"
    story_directory.mkdir(parents=True)
    for number in range(1, 6):
        Image.new("RGB", (120, 80), color=(number * 20, 100, 140)).save(
            story_directory / f"panel-{number}.png"
        )


@pytest.fixture
def export_service(tmp_path: Path) -> ComicPDFExportService:
    panel_directory = tmp_path / "panels"
    create_panel_images(panel_directory)
    return ComicPDFExportService(
        panel_directory=panel_directory,
        export_directory=tmp_path / "exports",
    )


def test_pdf_contains_five_pages_with_images_and_panel_content(export_service) -> None:
    pdf_path = export_service.create_pdf(make_story(), make_image_set())

    assert pdf_path.is_file()
    reader = PdfReader(pdf_path)
    assert len(reader.pages) == 5
    for number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        assert f"Panel title {number}" in text
        assert f"Scene caption {number}." in text
        assert f"Narration text {number}." in text
        assert f"Pip: Dialogue line {number}." in text
        assert len(page.images) == 1


def test_pdf_preserves_unicode_story_text(export_service) -> None:
    story = make_story()
    first_panel = story.panels[0].model_copy(
        update={
            "title": "Maya’s discovery",
            "scene_description": "“A mysterious robot” appears in a café.",
            "narration": "A — new adventure begins.",
            "dialogue": [DialogueLine(character="Maya", line="Let’s explore the café.")],
        }
    )
    unicode_story = story.model_copy(
        update={"panels": [first_panel, *story.panels[1:]]}
    )

    pdf_path = export_service.create_pdf(unicode_story, make_image_set())

    extracted_text = PdfReader(pdf_path).pages[0].extract_text()
    assert "Maya’s discovery" in extracted_text
    assert "“A mysterious robot”" in extracted_text
    assert "A — new adventure" in extracted_text
    assert "café" in extracted_text
    assert "Let’s explore the café." in extracted_text


def test_pdf_export_uses_a_unique_filename(export_service) -> None:
    first_pdf = export_service.create_pdf(make_story(), make_image_set())
    second_pdf = export_service.create_pdf(make_story(), make_image_set())

    assert first_pdf != second_pdf
    assert first_pdf.name == "Pip_Panel_title_1.pdf"
    assert second_pdf.name == "Pip_Panel_title_1_2.pdf"
    assert first_pdf.is_file()
    assert second_pdf.is_file()


def test_pdf_filename_sanitizes_and_limits_story_information(export_service) -> None:
    story = make_story()
    user_input = story.user_input.model_copy(
        update={"character_name": "Maya//__Robot"}
    )
    first_panel = story.panels[0].model_copy(
        update={"title": "The: Mysterious? AI* Robot"}
    )
    story = story.model_copy(
        update={
            "user_input": user_input,
            "panels": [first_panel, *story.panels[1:]],
        }
    )

    pdf_path = export_service.create_pdf(story, make_image_set())

    assert pdf_path.name == "Maya_Robot_The_Mysterious_AI_Robot.pdf"
    assert not any(character in pdf_path.name for character in '<>:"/\\|?*')

    long_user_input = story.user_input.model_copy(
        update={"character_name": "M" * 80}
    )
    long_title = story.panels[0].model_copy(update={"title": "T" * 120})
    long_story = story.model_copy(
        update={
            "user_input": long_user_input,
            "panels": [long_title, *story.panels[1:]],
        }
    )
    long_pdf_path = export_service.create_pdf(long_story, make_image_set())

    assert len(long_pdf_path.stem) == 120


def test_pdf_export_reports_a_missing_panel_image(export_service, tmp_path) -> None:
    (tmp_path / "panels" / "test-story" / "panel-3.png").unlink()

    with pytest.raises(MissingComicPanelImageError, match="panel 3 is missing"):
        export_service.create_pdf(make_story(), make_image_set())


def test_pdf_export_reports_an_unavailable_export_directory(tmp_path) -> None:
    export_directory = tmp_path / "not-a-directory"
    export_directory.write_text("occupied")
    panel_directory = tmp_path / "panels"
    create_panel_images(panel_directory)
    service = ComicPDFExportService(panel_directory, export_directory)

    with pytest.raises(ComicPDFExportError, match="export directory is unavailable"):
        service.create_pdf(make_story(), make_image_set())


def test_pdf_export_reports_a_generation_failure(export_service, monkeypatch) -> None:
    monkeypatch.setattr(
        ComicPDFExportService,
        "_build_pdf",
        staticmethod(MagicMock(side_effect=RuntimeError("private internal detail"))),
    )

    with pytest.raises(ComicPDFExportError, match="PDF generation failed") as error:
        export_service.create_pdf(make_story(), make_image_set())

    assert "private internal detail" not in str(error.value)


@pytest.mark.parametrize(("image_format", "extension"), [("JPEG", "jpg"), ("WEBP", "webp")])
def test_pdf_export_accepts_hosted_image_formats(
    tmp_path: Path, image_format: str, extension: str
) -> None:
    panel_directory = tmp_path / "panels"
    story_directory = panel_directory / "test-story"
    story_directory.mkdir(parents=True)
    for number in range(1, 6):
        Image.new("RGB", (120, 80), color=(number * 20, 100, 140)).save(
            story_directory / f"panel-{number}.{extension}", format=image_format
        )
    image_set = ComicImageSet(
        story_id="test-story",
        panels=[
            {
                "panel_number": number,
                "image_url": f"/static/panels/test-story/panel-{number}.{extension}",
            }
            for number in range(1, 6)
        ],
    )
    service = ComicPDFExportService(panel_directory, tmp_path / "exports")

    pdf_path = service.create_pdf(make_story(), image_set)

    assert len(PdfReader(pdf_path).pages) == 5


def test_pdf_export_rejects_invalid_story_data(export_service) -> None:
    with pytest.raises(ComicPDFExportError, match="story or panel image data is invalid"):
        export_service.create_pdf({"panels": []}, make_image_set())


def test_pdf_download_route_serves_the_generated_file(export_service):
    pdf_path = export_service.create_pdf(make_story(), make_image_set())
    app.dependency_overrides[get_pdf_export_service] = lambda: export_service
    try:
        with TestClient(app) as client:
            response = client.get(f"/download-comic/{pdf_path.name}")
    finally:
        app.dependency_overrides.pop(get_pdf_export_service, None)

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.headers["content-disposition"] == (
        f'attachment; filename="{pdf_path.name}"'
    )
    assert response.content == pdf_path.read_bytes()


def test_pdf_download_route_returns_404_for_unknown_or_unsafe_filename():
    with TestClient(app) as client:
        missing_response = client.get("/download-comic/Pip_Missing_Comic.pdf")
        unsafe_response = client.get("/download-comic/..%2F.env")

    assert missing_response.status_code == 404
    assert unsafe_response.status_code == 404


def test_export_success_route_renders_for_an_existing_pdf(export_service):
    pdf_path = export_service.create_pdf(make_story(), make_image_set())
    app.dependency_overrides[get_pdf_export_service] = lambda: export_service
    try:
        with TestClient(app) as client:
            response = client.get(
                "/export-success", params={"filename": pdf_path.name}
            )
    finally:
        app.dependency_overrides.pop(get_pdf_export_service, None)

    assert response.status_code == 200
    assert "EXPORT COMPLETE" in response.text
    assert pdf_path.name in response.text
    assert "Download again" in response.text


def test_export_success_route_returns_404_for_a_missing_pdf():
    with TestClient(app) as client:
        response = client.get(
            "/export-success",
            params={"filename": "Pip_Missing_Comic.pdf"},
        )

    assert response.status_code == 404


def test_export_service_rejects_missing_export_file(export_service):
    with pytest.raises(ComicPDFNotFoundError):
        export_service.get_export_path("Pip_Missing_Comic.pdf")