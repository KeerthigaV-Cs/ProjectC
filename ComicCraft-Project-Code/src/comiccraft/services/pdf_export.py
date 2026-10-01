"""Create downloadable PDF comics from saved panel images."""

from io import BytesIO
from pathlib import Path
import re
from urllib.parse import urlsplit

from fpdf import FPDF
from fpdf.enums import XPos, YPos
from PIL import Image
from pydantic import ValidationError

from comiccraft.errors import (
    ComicPDFExportError,
    ComicPDFNotFoundError,
    MissingComicPanelImageError,
)
from comiccraft.models.images import ComicImageSet, ComicPanelImage
from comiccraft.models.story import ComicPanel, ComicStory

PACKAGE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PANEL_DIRECTORY = PACKAGE_DIR / "static" / "panels"
DEFAULT_EXPORT_DIRECTORY = PACKAGE_DIR / "static" / "exports"
MAX_STORY_ID_LENGTH = 64
MAX_FILENAME_STEM_LENGTH = 120
UNICODE_FONT_PATH = PACKAGE_DIR / "static" / "fonts" / "NotoSans-VF.ttf"


class ComicPDFExportService:
    """Build and store one five-page PDF using an existing image set."""

    def __init__(
        self,
        panel_directory: Path = DEFAULT_PANEL_DIRECTORY,
        export_directory: Path = DEFAULT_EXPORT_DIRECTORY,
    ) -> None:
        self._panel_directory = panel_directory
        self._export_directory = export_directory

    def create_pdf(
        self, story: ComicStory | dict, image_set: ComicImageSet | dict
    ) -> Path:
        try:
            comic_story = ComicStory.model_validate(story)
            comic_images = ComicImageSet.model_validate(image_set)
        except ValidationError as exc:
            raise ComicPDFExportError(
                "Comic story or panel image data is invalid."
            ) from exc

        if len(comic_story.panels) != 5 or len(comic_images.panels) != 5:
            raise ComicPDFExportError("A comic PDF requires exactly five panels.")

        image_data = [
            self._load_panel_image(comic_images.story_id, image)
            for image in comic_images.panels
        ]
        try:
            pdf_content = self._build_pdf(comic_story.panels, image_data)
        except ComicPDFExportError:
            raise
        except Exception as exc:
            raise ComicPDFExportError("PDF generation failed.") from exc

        try:
            self._export_directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ComicPDFExportError(
                "The PDF export directory is unavailable."
            ) from exc

        filename_stem = self._filename_stem(comic_story)
        for suffix_number in range(1, 10000):
            suffix = "" if suffix_number == 1 else f"_{suffix_number}"
            stem = filename_stem[: MAX_FILENAME_STEM_LENGTH - len(suffix)].rstrip("_-")
            filename = f"{stem}{suffix}.pdf"
            export_path = self._export_directory / filename
            try:
                with export_path.open("xb") as export_file:
                    export_file.write(pdf_content)
                return export_path
            except FileExistsError:
                continue
            except OSError as exc:
                try:
                    export_path.unlink(missing_ok=True)
                except OSError:
                    pass
                raise ComicPDFExportError("The generated PDF could not be saved.") from exc

        raise ComicPDFExportError("A unique PDF filename could not be created.")

    def get_export_path(self, filename: str) -> Path:
        if not re.fullmatch(r"[\w-]{1,120}\.pdf", filename):
            raise ComicPDFNotFoundError("The requested comic PDF was not found.")

        export_path = self._export_directory / filename
        try:
            export_path.resolve().relative_to(self._export_directory.resolve())
        except ValueError as exc:
            raise ComicPDFNotFoundError(
                "The requested comic PDF was not found."
            ) from exc
        if not export_path.is_file():
            raise ComicPDFNotFoundError("The requested comic PDF was not found.")
        return export_path

    @staticmethod
    def _filename_stem(story: ComicStory) -> str:
        def safe_part(value: str, fallback: str) -> str:
            cleaned = re.sub(r"[^\w-]+", "_", value, flags=re.UNICODE)
            cleaned = re.sub(r"_+", "_", cleaned).strip("_-")
            return cleaned or fallback

        character = safe_part(story.user_input.character_name, "ComicCraft")
        title = safe_part(story.panels[0].title, "ComicCraft")
        stem = f"{character}_{title}"
        return stem[:MAX_FILENAME_STEM_LENGTH].rstrip("_-") or "ComicCraft"

    def _load_panel_image(
        self, story_id: str, panel_image: ComicPanelImage
    ) -> tuple[BytesIO, int, int]:
        if not re.fullmatch(rf"[A-Za-z0-9_-]{{1,{MAX_STORY_ID_LENGTH}}}", story_id):
            raise ComicPDFExportError("Comic image data is invalid.")

        parsed_url = urlsplit(panel_image.image_url)
        image_match = re.fullmatch(
            rf"/static/panels/{re.escape(story_id)}/panel-{panel_image.panel_number}\.(?:png|jpg|webp)",
            parsed_url.path,
        )
        if (
            parsed_url.scheme
            or parsed_url.netloc
            or parsed_url.query
            or parsed_url.fragment
            or image_match is None
        ):
            raise ComicPDFExportError("Comic panel image data is invalid.")

        image_path = self._panel_directory / story_id / image_match.group(0).rsplit("/", 1)[-1]
        try:
            image_path.resolve().relative_to(self._panel_directory.resolve())
        except ValueError as exc:
            raise ComicPDFExportError("Comic panel image data is invalid.") from exc
        if not image_path.is_file():
            raise MissingComicPanelImageError(
                f"The saved image for panel {panel_image.panel_number} is missing."
            )

        try:
            with Image.open(image_path) as image:
                image.load()
                width, height = image.size
                rgb_image = image.convert("RGB")
                image_buffer = BytesIO()
                rgb_image.save(image_buffer, format="PNG")
                image_buffer.seek(0)
                return image_buffer, width, height
        except Exception as exc:
            raise ComicPDFExportError(
                f"The saved image for panel {panel_image.panel_number} could not be read."
            ) from exc

    @staticmethod
    def _build_pdf(
        panels: list[ComicPanel], image_data: list[tuple[BytesIO, int, int]]
    ) -> bytes:
        pdf = FPDF(format="A4")
        pdf.add_font(
            "NotoSans",
            fname=UNICODE_FONT_PATH,
            variations={"wght": 400},
        )
        pdf.add_font(
            "NotoSans",
            style="B",
            fname=UNICODE_FONT_PATH,
            variations={"wght": 700},
        )
        pdf.set_margins(15, 15, 15)
        pdf.set_auto_page_break(auto=False)

        for panel, (image_buffer, image_width, image_height) in zip(
            panels, image_data, strict=True
        ):
            pdf.add_page()
            pdf.set_font("NotoSans", "B", 16)
            pdf.multi_cell(0, 8, f"Panel {panel.panel_number}: {panel.title}")

            max_image_width = 180
            max_image_height = 120
            scale = min(max_image_width / image_width, max_image_height / image_height)
            display_width = image_width * scale
            display_height = image_height * scale
            image_x = (210 - display_width) / 2
            image_y = pdf.get_y() + 4
            pdf.image(
                image_buffer,
                x=image_x,
                y=image_y,
                w=display_width,
                h=display_height,
            )
            pdf.set_y(image_y + display_height + 7)

            _add_text_section(pdf, "Scene description", panel.scene_description)
            _add_text_section(pdf, "Narration", panel.narration)
            if panel.dialogue:
                dialogue = "\n".join(
                    f"{line.character}: {line.line}" for line in panel.dialogue
                )
                _add_text_section(pdf, "Dialogue", dialogue)

            if pdf.get_y() > pdf.h - pdf.b_margin:
                raise ComicPDFExportError(
                    f"Panel {panel.panel_number} text is too long to fit on one PDF page."
                )

        return bytes(pdf.output())


def _add_text_section(pdf: FPDF, label: str, value: str) -> None:
    pdf.set_font("NotoSans", "B", 10)
    pdf.cell(0, 6, label, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("NotoSans", "", 10)
    pdf.multi_cell(0, 5, value)
    pdf.ln(2)