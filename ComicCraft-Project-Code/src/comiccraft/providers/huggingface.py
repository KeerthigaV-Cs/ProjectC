"""Hugging Face story-provider adapter."""

import json
from copy import deepcopy
from typing import Any, get_args, get_origin

from huggingface_hub import InferenceClient
from pydantic import ValidationError

from comiccraft.errors import (
    HuggingFaceStoryProviderError,
    InvalidHuggingFaceStoryResponseError,
    MissingHFTokenError,
    MissingHFStoryModelError,
)
from comiccraft.models.story import (
    ComicInput,
    GeneratedStoryContent,
    PanelNarration,
)


class HuggingFaceStoryProvider:
    """Generate structured comic story content through Hugging Face."""

    def __init__(
        self,
        token: str | None,
        model: str | None,
        client_factory: type[InferenceClient] | None = None,
    ) -> None:
        self._token = token
        self._model = model
        self._client_factory = client_factory or InferenceClient

    def generate_story(self, user_input: ComicInput) -> GeneratedStoryContent:
        if not self._token:
            raise MissingHFTokenError(
                "Set HF_TOKEN in your environment or .env file to generate stories with Hugging Face."
            )
        if not self._model:
            raise MissingHFStoryModelError(
                "Set HF_STORY_MODEL in your environment or .env file to use Hugging Face for story generation."
            )

        prompt = self._build_prompt(user_input)
        try:
            client = self._client_factory(token=self._token)
            response = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=self._model,
                max_tokens=1800,
                temperature=0.8,
            )
        except Exception as exc:
            raise HuggingFaceStoryProviderError(
                f"Hugging Face story generation failed: {exc}"
            ) from exc

        try:
            response_text = response.choices[0].message.content
            payload = self._parse_response_payload(response_text)
            normalized = self._normalize_story_payload(payload, user_input)
            return GeneratedStoryContent.model_validate(normalized)
        except (ValidationError, TypeError, ValueError, KeyError) as exc:
            raise InvalidHuggingFaceStoryResponseError(
                "Hugging Face returned content that does not match the required five-panel story format: "
                f"{exc}"
            ) from exc
        except (AttributeError, IndexError) as exc:
            raise InvalidHuggingFaceStoryResponseError(
                "Hugging Face returned an invalid chat-completion response."
            ) from exc

    @staticmethod
    def _parse_response_payload(raw_response: object) -> dict[str, Any]:
        """Extract a JSON story object from common chat-model output wrappers."""
        if isinstance(raw_response, dict):
            return deepcopy(raw_response)
        if not isinstance(raw_response, str):
            raise TypeError("Hugging Face returned no text or object content.")

        text = raw_response.strip()
        if not text:
            raise ValueError("Hugging Face returned an empty response.")

        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, dict):
            return decoded
        if isinstance(decoded, str):
            return HuggingFaceStoryProvider._parse_response_payload(decoded)

        decoder = json.JSONDecoder()
        candidates: list[tuple[int, int, dict[str, Any]]] = []
        for index, character in enumerate(text):
            if character != "{":
                continue
            try:
                candidate, end = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and (
                "outline" in candidate or "narration" in candidate
            ):
                candidates.append((index, index + end, candidate))

        outermost_candidates = [
            candidate
            for candidate in candidates
            if not any(
                other_start <= candidate[0]
                and candidate[1] <= other_end
                and (other_start, other_end) != (candidate[0], candidate[1])
                for other_start, other_end, _ in candidates
            )
        ]

        if len(outermost_candidates) == 1:
            return outermost_candidates[0][2]
        if len(outermost_candidates) > 1:
            raise ValueError("Hugging Face response contains multiple possible story objects.")
        raise ValueError("Hugging Face returned no safely extractable JSON story object.")

    @staticmethod
    def _normalize_story_payload(
        payload: dict[str, Any], user_input: ComicInput
    ) -> dict[str, Any]:
        """Adapt safe HF shape differences to the application's canonical models."""
        normalized = deepcopy(payload)
        outline_source = normalized.get("outline")
        if outline_source is None and isinstance(normalized.get("panels"), list):
            potential_outline = normalized["panels"]
            if all(
                isinstance(panel, dict)
                and any(key in panel for key in ("title", "scene_description", "image_generation_prompt", "image_prompt"))
                for panel in potential_outline
            ):
                outline_source = potential_outline

        outline_panels = HuggingFaceStoryProvider._extract_panels(
            outline_source, "outline"
        )
        narration_panels = HuggingFaceStoryProvider._extract_panels(
            normalized.get("narration"), "narration"
        )
        HuggingFaceStoryProvider._require_five_panels(outline_panels, "outline")
        HuggingFaceStoryProvider._require_five_panels(narration_panels, "narration")

        normalized["outline"] = {
            "panels": [
                HuggingFaceStoryProvider._normalize_outline_panel(panel)
                for panel in outline_panels
            ]
        }
        normalized["narration"] = {
            "panels": [
                HuggingFaceStoryProvider._normalize_narration_panel(panel, user_input)
                for panel in narration_panels
            ]
        }
        return normalized

    @staticmethod
    def _extract_panels(section: object, section_name: str) -> list[object]:
        if isinstance(section, dict):
            section = section.get("panels")
        if not isinstance(section, list):
            raise ValueError(f"Hugging Face {section_name} must be a panel list or contain 'panels'.")
        return section

    @staticmethod
    def _require_five_panels(panels: list[object], section_name: str) -> None:
        if len(panels) != 5:
            raise ValueError(
                f"Hugging Face {section_name} must contain exactly five panels; received {len(panels)}."
            )

    @staticmethod
    def _normalize_outline_panel(panel: object) -> dict[str, Any]:
        if not isinstance(panel, dict):
            raise TypeError("Hugging Face outline panels must be objects.")
        normalized = deepcopy(panel)
        if "image_generation_prompt" not in normalized and "image_prompt" in normalized:
            normalized["image_generation_prompt"] = normalized.pop("image_prompt")
        return normalized

    @staticmethod
    def _normalize_narration_panel(
        panel: object, user_input: ComicInput
    ) -> dict[str, Any]:
        if not isinstance(panel, dict):
            raise TypeError("Hugging Face narration panels must be objects.")
        normalized = deepcopy(panel)
        if "narration" not in normalized and "text" in normalized:
            normalized["narration"] = normalized.pop("text")

        dialogue = normalized.get("dialogue", [])
        if dialogue is None:
            dialogue = []
        elif isinstance(dialogue, str):
            dialogue = [] if not dialogue.strip() else [dialogue]
        if not isinstance(dialogue, list):
            raise TypeError("Hugging Face dialogue must be a string, list, or null.")

        dialogue_model = get_args(PanelNarration.model_fields["dialogue"].annotation)[0]
        normalized_dialogue = []
        for line in dialogue:
            if isinstance(line, str):
                normalized_dialogue.append(
                    {"character": user_input.character_name, "line": line}
                )
            elif isinstance(line, dict):
                line = deepcopy(line)
                if "character" not in line and "speaker" in line:
                    line["character"] = line.pop("speaker")
                if "line" not in line and "text" in line:
                    line["line"] = line.pop("text")
                normalized_dialogue.append(line)
            else:
                raise TypeError("Hugging Face dialogue entries must be strings or objects.")

        if get_origin(PanelNarration.model_fields["dialogue"].annotation) is not list:
            raise TypeError("The canonical dialogue field is not a list.")
        if dialogue_model.__name__ != "DialogueLine":
            raise TypeError("The canonical dialogue entry model is unsupported.")
        normalized["dialogue"] = normalized_dialogue
        return normalized

    @staticmethod
    def _build_prompt(user_input: ComicInput) -> str:
        return (
            "Create a cohesive comic story as structured JSON with exactly two top-level keys: "
            "'outline' and 'narration'. The outline section must contain five panels numbered 1-5, "
            "each with title, scene_description, and image_generation_prompt. The narration section must contain "
            "five panels numbered 1-5, each with narration and dialogue. Return only valid JSON and no extra text.\n\n"
            f"User comic input:\n{user_input.model_dump_json(indent=2)}"
        )
