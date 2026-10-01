"""Free, deterministic placeholder image provider for local development."""

from io import BytesIO
import textwrap

from PIL import Image, ImageDraw, ImageFont

from comiccraft.models.images import GeneratedImageData

IMAGE_SIZE = (768, 512)
INK = (35, 39, 48)


class LocalImageGenerationProvider:
    """Draw a simple comic panel without contacting a hosted image API."""

    def generate_image(self, prompt: str) -> GeneratedImageData:
        details = self._read_prompt_details(prompt)
        panel_number = details["panel_number"]
        character = details["character"]
        scene = details["scene"]

        image = Image.new("RGB", IMAGE_SIZE, (255, 247, 225))
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        draw.rounded_rectangle(
            (12, 12, 756, 500), radius=18, fill=(218, 239, 248), outline=INK, width=8
        )
        draw.ellipse((610, 56, 680, 126), fill=(255, 198, 74), outline=INK, width=3)
        draw.polygon(
            [(20, 330), (175, 205), (330, 330), (480, 190), (748, 338), (748, 390), (20, 390)],
            fill=(116, 177, 119),
            outline=INK,
        )
        draw.ellipse((330, 212, 424, 306), fill=(245, 190, 145), outline=INK, width=4)
        draw.rounded_rectangle(
            (327, 294, 427, 375), radius=24, fill=(239, 119, 91), outline=INK, width=4
        )
        draw.line((337, 322, 294, 350), fill=INK, width=7)
        draw.line((417, 322, 460, 350), fill=INK, width=7)
        draw.line((352, 370, 337, 394), fill=INK, width=7)
        draw.line((402, 370, 417, 394), fill=INK, width=7)
        draw.ellipse((354, 247, 363, 256), fill=INK)
        draw.ellipse((391, 247, 400, 256), fill=INK)
        draw.arc((365, 255, 391, 280), start=10, end=165, fill=INK, width=3)

        draw.text((42, 38), f"COMICCRAFT  /  PANEL {panel_number}", fill=INK, font=font)
        if character:
            draw.text((42, 62), f"Featuring: {character}", fill=INK, font=font)
        draw.rounded_rectangle(
            (34, 410, 734, 474), radius=12, fill=(255, 255, 255), outline=INK, width=3
        )
        caption_lines = textwrap.wrap(scene, width=88)[:2] or ["A new adventure begins!"]
        for line_number, line in enumerate(caption_lines):
            draw.text((52, 426 + line_number * 18), line, fill=INK, font=font)

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return GeneratedImageData(data=buffer.getvalue(), mime_type="image/png")

    @staticmethod
    def _read_prompt_details(prompt: str) -> dict[str, str | int]:
        details: dict[str, str | int] = {
            "panel_number": 1,
            "character": "",
            "scene": prompt.strip(),
        }
        for line in prompt.splitlines():
            label, separator, value = line.partition(":")
            if not separator:
                continue
            key = label.strip().lower()
            value = value.strip()
            if key == "panel number":
                try:
                    details["panel_number"] = int(value)
                except ValueError:
                    pass
            elif key == "character":
                details["character"] = value
            elif key == "scene":
                details["scene"] = value
        scene = str(details["scene"]).encode("ascii", errors="ignore").decode("ascii")
        character = str(details["character"]).encode("ascii", errors="ignore").decode("ascii")
        details["scene"] = scene[:160]
        details["character"] = character[:48]
        return details