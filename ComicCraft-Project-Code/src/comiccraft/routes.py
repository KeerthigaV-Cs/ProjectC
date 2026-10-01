"""HTTP routes for the ComicCraft app shell."""

import asyncio
import json
import logging
from pathlib import Path
import re
import traceback
from typing import Callable

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

from comiccraft.config import get_settings
from comiccraft.errors import (
    ComicPDFExportError,
    ComicPDFNotFoundError,
    GeminiConfigurationError,
    GeminiProviderError,
    GeminiRateLimitError,
    GeminiServiceUnavailableError,
    HuggingFaceStoryProviderError,
    ImageProviderCreditsExhaustedError,
    ImageProviderError,
    ImageProviderRateLimitError,
    ImageProviderServiceUnavailableError,
    ImageProviderTimeoutError,
    ImageStorageError,
    InvalidImageResponseError,
    InvalidGeminiResponseError,
    InvalidHuggingFaceStoryResponseError,
    MissingGeminiApiKeyError,
    MissingHFTokenError,
    MissingHFStoryModelError,
    MissingImageApiKeyError,
    ProviderFallbackError,
    UnsupportedImageProviderError,
)
from comiccraft.models.images import ComicImageSet
from comiccraft.models.story import ComicInput, ComicStory
from comiccraft.providers.image_factory import create_image_provider
from comiccraft.providers.story_factory import create_story_provider
from comiccraft.services.image_generation import ImageGenerationService
from comiccraft.services.pdf_export import ComicPDFExportService
from comiccraft.services.story_generation import StoryGenerationService

PACKAGE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=PACKAGE_DIR / "templates")
router = APIRouter()
logger = logging.getLogger(__name__)
def _log_generation_failure(stage: str, exc: Exception) -> None:
    settings = get_settings()
    sensitive_values = tuple(
        value
        for value in (settings.gemini_api_key, settings.image_api_key, settings.hf_token)
        if value
    )

    def redact(text: str) -> str:
        for secret in sensitive_values:
            text = text.replace(secret, "<redacted>")
        text = re.sub(
            r"(?is)(headers?\s*[:=]\s*)(\{.*?\}|\[.*?\])",
            r"\1<redacted>",
            text,
        )
        text = re.sub(
            r"(?i)(authorization\s*[:=]\s*)(?:bearer\s+)?[^\s,;]+",
            r"\1<redacted>",
            text,
        )
        text = re.sub(
            r"(?i)(x-goog-api-key|x-api-key|api[_-]?key|access[_-]?token|token|secret|password)\s*[:=]\s*[^\s,;]+",
            r"\1=<redacted>",
            text,
        )
        return re.sub(r"\bAIza[0-9A-Za-z_-]{20,}\b", "<redacted>", text)

    status_code = None
    current: BaseException | None = exc
    while current is not None:
        candidate = getattr(current, "code", None)
        if isinstance(candidate, int) and 100 <= candidate <= 599:
            status_code = candidate
            break
        current = current.__cause__ or current.__context__

    diagnostic = (
        f"{stage} generation failed; exception_type={type(exc).__name__}; "
        f"provider_error={redact(str(exc))}"
    )
    if status_code is not None:
        diagnostic += f"; status_code={status_code}"
    diagnostic += "\n" + redact(
        "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    )
    logger.error("%s", diagnostic)


def render_home(
    request: Request,
    form_data: dict[str, str] | None = None,
    field_errors: dict[str, str] | None = None,
    error_message: str | None = None,
    status_code: int = 200,
    error_code: str | None = None,
) -> HTMLResponse:
    response = templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "form_data": form_data or {},
            "field_errors": field_errors or {},
            "error_message": error_message,
        },
        status_code=status_code,
    )
    if error_code and error_message:
        response.headers["X-ComicCraft-Error-Code"] = error_code
        response.headers["X-ComicCraft-Error-Message"] = error_message
    return response


GEMINI_CONFIGURATION_MESSAGE = (
    "Gemini AI is not configured correctly. Please check your Gemini API key in the project settings."
)
GEMINI_RATE_LIMIT_MESSAGE = (
    "Gemini AI usage limit has been reached. Please wait for the quota to reset or use an available fallback service."
)
GEMINI_UNAVAILABLE_MESSAGE = (
    "Gemini AI is temporarily unavailable. Please try generating your comic again in a few moments."
)
HF_CONFIGURATION_MESSAGE = (
    "Hugging Face image generation is not configured correctly. Please check the Hugging Face access token in the project settings."
)
HF_CREDITS_MESSAGE = (
    "Hugging Face image-generation credits have been used up. Please add Hugging Face credits or switch to another image provider."
)
HF_RATE_LIMIT_MESSAGE = (
    "Hugging Face image generation is temporarily rate-limited. Please wait a moment and try again."
)
HF_UNAVAILABLE_MESSAGE = (
    "Hugging Face image generation is temporarily unavailable. Please try again in a few moments."
)
IMAGE_TIMEOUT_MESSAGE = (
    "Image generation took too long to respond. Please check your internet connection and try again."
)
IMAGE_FAILURE_MESSAGE = "We couldn't generate the comic illustrations. Please try again."
STORY_FAILURE_MESSAGE = (
    "We couldn't create the story at the moment because the AI story services are unavailable. Please try again later."
)
INVALID_INPUT_MESSAGE = (
    "Please provide a story prompt and character name before generating your comic."
)
UNEXPECTED_ERROR_MESSAGE = (
    "Something went wrong while creating your comic. Please try again."
)


def _image_error_details(exc: Exception) -> tuple[str, str, int] | None:
    if isinstance(exc, ProviderFallbackError):
        for nested_error in (exc.fallback_error, exc.primary_error):
            details = _image_error_details(nested_error)
            if details is not None:
                return details
        return "IMAGE_GENERATION_FAILED", IMAGE_FAILURE_MESSAGE, 502
    if isinstance(exc, MissingHFTokenError):
        return "HF_TOKEN_INVALID", HF_CONFIGURATION_MESSAGE, 503
    if isinstance(exc, ImageProviderCreditsExhaustedError):
        return "HF_CREDITS_EXHAUSTED", HF_CREDITS_MESSAGE, 402
    if isinstance(exc, ImageProviderRateLimitError):
        return "HF_RATE_LIMITED", HF_RATE_LIMIT_MESSAGE, 429
    if isinstance(exc, ImageProviderServiceUnavailableError):
        return "HF_SERVICE_UNAVAILABLE", HF_UNAVAILABLE_MESSAGE, 503
    if isinstance(exc, ImageProviderTimeoutError):
        return "IMAGE_GENERATION_TIMEOUT", IMAGE_TIMEOUT_MESSAGE, 504
    if isinstance(exc, (ImageProviderError, InvalidImageResponseError)):
        return "IMAGE_GENERATION_FAILED", IMAGE_FAILURE_MESSAGE, 502
    return None


def _generation_error_details(
    exc: Exception, stage: str
) -> tuple[str, str, int]:
    if stage == "STORY":
        if isinstance(exc, (MissingGeminiApiKeyError, GeminiConfigurationError)):
            return "GEMINI_CONFIGURATION_ERROR", GEMINI_CONFIGURATION_MESSAGE, 503
        if isinstance(exc, GeminiRateLimitError):
            return "GEMINI_RATE_LIMITED", GEMINI_RATE_LIMIT_MESSAGE, 429
        if isinstance(exc, GeminiServiceUnavailableError):
            return "GEMINI_SERVICE_UNAVAILABLE", GEMINI_UNAVAILABLE_MESSAGE, 503
        if isinstance(exc, ProviderFallbackError):
            return "STORY_SERVICES_UNAVAILABLE", STORY_FAILURE_MESSAGE, 503
        if isinstance(
            exc,
            (
                GeminiProviderError,
                InvalidGeminiResponseError,
                HuggingFaceStoryProviderError,
                InvalidHuggingFaceStoryResponseError,
                MissingHFTokenError,
                MissingHFStoryModelError,
            ),
        ):
            return "STORY_SERVICES_UNAVAILABLE", STORY_FAILURE_MESSAGE, 503
    else:
        image_details = _image_error_details(exc)
        if image_details is not None:
            return image_details
        if isinstance(exc, UnsupportedImageProviderError):
            return "IMAGE_PROVIDER_UNAVAILABLE", IMAGE_FAILURE_MESSAGE, 503
        if isinstance(exc, MissingImageApiKeyError):
            return "IMAGE_GENERATION_FAILED", IMAGE_FAILURE_MESSAGE, 503
    return "UNEXPECTED_SERVER_ERROR", UNEXPECTED_ERROR_MESSAGE, 500


def _generation_error_response(exc: Exception, stage: str) -> JSONResponse:
    _log_generation_failure(stage, exc)
    error_code, message, status_code = _generation_error_details(exc, stage)
    return JSONResponse(
        status_code=status_code,
        content={"success": False, "error_code": error_code, "message": message},
    )


def get_story_generation_service() -> StoryGenerationService:
    settings = get_settings()
    primary_name = (settings.story_provider or "gemini").strip().lower()
    primary_provider = create_story_provider(settings, primary_name)
    fallback_provider = None
    if primary_name == "gemini":
        fallback_name = (settings.story_fallback_provider or "huggingface").strip().lower()
        if fallback_name != "gemini":
            fallback_provider = create_story_provider(settings, fallback_name)
    return StoryGenerationService(primary_provider, fallback_provider=fallback_provider)


def get_image_generation_service() -> ImageGenerationService:
    try:
        settings = get_settings()
        primary_name = (settings.image_provider or "local").strip().lower()
        primary_provider = create_image_provider(settings, primary_name)
        fallback_provider = None
        if primary_name == "gemini":
            fallback_name = (settings.image_fallback_provider or "huggingface").strip().lower()
            if fallback_name != "gemini":
                fallback_provider = create_image_provider(settings, fallback_name)
        return ImageGenerationService(primary_provider, fallback_provider=fallback_provider)
    except UnsupportedImageProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def get_pdf_export_service() -> ComicPDFExportService:
    return ComicPDFExportService()


@router.get("/", name="home")
async def home(request: Request):
    return render_home(request)


@router.get("/health", name="health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@router.get("/download-comic/{filename}", name="download_comic_pdf")
def download_comic_pdf(
    filename: str,
    service: ComicPDFExportService = Depends(get_pdf_export_service),
) -> FileResponse:
    try:
        export_path = service.get_export_path(filename)
    except ComicPDFNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(
        export_path,
        media_type="application/pdf",
        filename=export_path.name,
    )


@router.get("/export-success", response_class=HTMLResponse, name="export_success")
def export_success(
    request: Request,
    filename: str,
    service: ComicPDFExportService = Depends(get_pdf_export_service),
) -> HTMLResponse:
    try:
        export_path = service.get_export_path(filename)
    except ComicPDFNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return templates.TemplateResponse(
        request=request,
        name="export_success.html",
        context={
            "filename": export_path.name,
            "download_url": request.url_for(
                "download_comic_pdf", filename=export_path.name
            ),
        },
    )


@router.post("/generate-comic/json", response_model=ComicStory, name="generate_comic_json")
def generate_comic_json(
    user_input: ComicInput,
    service: StoryGenerationService = Depends(get_story_generation_service),
) -> ComicStory | JSONResponse:
    try:
        return service.generate(user_input)
    except Exception as exc:
        return _generation_error_response(exc, "STORY")


@router.post(
    "/generate-comic/images",
    response_model=ComicImageSet,
    name="generate_comic_images",
)
def generate_comic_images(
    story: ComicStory,
    service: ImageGenerationService = Depends(get_image_generation_service),
) -> ComicImageSet | JSONResponse:
    try:
        return service.generate_for_story(story)
    except Exception as exc:
        return _generation_error_response(exc, "IMAGE")


def _generate_comic_preview(
    request: Request,
    story_prompt: str,
    character_name: str,
    setting: str,
    tone: str,
    art_style: str,
    story_service: StoryGenerationService,
    image_service: ImageGenerationService,
    pdf_export_service: ComicPDFExportService,
    progress_callback: Callable[..., None] | None = None,
) -> HTMLResponse:
    form_data = {
        "story_prompt": story_prompt,
        "character_name": character_name,
        "setting": setting,
        "tone": tone,
        "art_style": art_style,
    }
    try:
        user_input = ComicInput.model_validate(form_data)
    except ValidationError as exc:
        field_errors = {
            str(error["loc"][-1]): error["msg"]
            for error in exc.errors()
            if error.get("loc")
        }
        return render_home(
            request,
            form_data=form_data,
            field_errors=field_errors,
            error_message=INVALID_INPUT_MESSAGE,
            status_code=422,
            error_code="INVALID_USER_INPUT",
        )

    generation_stage = "STORY"
    try:
        if progress_callback is not None:
            progress_callback("preparing")
            progress_callback("story")
        story = story_service.generate(user_input)
        generation_stage = "IMAGE"
        if progress_callback is not None:
            progress_callback("narration")
            progress_callback("images")
            image_set = image_service.generate_for_story(
                story,
                progress_callback=lambda panel_number: progress_callback(
                    "panel", panel_number=panel_number
                ),
            )
        else:
            image_set = image_service.generate_for_story(story)
    except Exception as exc:
        _log_generation_failure(generation_stage, exc)
        error_code, error_message, status_code = _generation_error_details(
            exc, generation_stage
        )
        return render_home(
            request,
            form_data=form_data,
            error_message=error_message,
            status_code=status_code,
            error_code=error_code,
        )

    if progress_callback is not None:
        progress_callback("panels")

    image_urls = {panel.panel_number: panel.image_url for panel in image_set.panels}
    panels = [
        {
            **panel.model_dump(),
            "image_url": image_urls[panel.panel_number],
        }
        for panel in story.panels
    ]
    if progress_callback is not None:
        progress_callback("assembly")
        progress_callback("preview")
        progress_callback("pdf")

    try:
        pdf_path = pdf_export_service.create_pdf(story, image_set)
    except ComicPDFExportError as exc:
        _log_generation_failure("PDF", exc)
        return render_home(
            request,
            form_data=form_data,
            error_message=str(exc),
            status_code=500,
        )

    return templates.TemplateResponse(
        request=request,
        name="comic_preview.html",
        context={
            "panels": panels,
            "user_input": user_input,
            "download_url": request.url_for(
                "download_comic_pdf", filename=pdf_path.name
            ),
            "download_filename": pdf_path.name,
            "success_url": request.url_for("export_success").include_query_params(
                filename=pdf_path.name
            ),
        },
    )


@router.post("/generate", response_class=HTMLResponse, name="generate_comic_preview")
def generate_comic_preview(
    request: Request,
    story_prompt: str = Form(default=""),
    character_name: str = Form(default=""),
    setting: str = Form(default=""),
    tone: str = Form(default=""),
    art_style: str = Form(default=""),
    story_service: StoryGenerationService = Depends(get_story_generation_service),
    image_service: ImageGenerationService = Depends(get_image_generation_service),
    pdf_export_service: ComicPDFExportService = Depends(get_pdf_export_service),
) -> HTMLResponse:
    return _generate_comic_preview(
        request,
        story_prompt,
        character_name,
        setting,
        tone,
        art_style,
        story_service,
        image_service,
        pdf_export_service,
    )


@router.post("/generate/progress", name="generate_comic_progress")
async def generate_comic_progress(
    request: Request,
    story_prompt: str = Form(default=""),
    character_name: str = Form(default=""),
    setting: str = Form(default=""),
    tone: str = Form(default=""),
    art_style: str = Form(default=""),
    story_service: StoryGenerationService = Depends(get_story_generation_service),
    image_service: ImageGenerationService = Depends(get_image_generation_service),
    pdf_export_service: ComicPDFExportService = Depends(get_pdf_export_service),
) -> StreamingResponse:
    async def event_stream():
        events: asyncio.Queue[tuple[str, dict[str, object]]] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def publish(event: str, **data: object) -> None:
            loop.call_soon_threadsafe(events.put_nowait, (event, data))

        def generate() -> None:
            form_data = {
                "story_prompt": story_prompt,
                "character_name": character_name,
                "setting": setting,
                "tone": tone,
                "art_style": art_style,
            }
            try:
                result = _generate_comic_preview(
                    request,
                    story_prompt,
                    character_name,
                    setting,
                    tone,
                    art_style,
                    story_service,
                    image_service,
                    pdf_export_service,
                    progress_callback=publish,
                )
                if result.status_code >= 400:
                    publish(
                        "error",
                        success=False,
                        error_code=result.headers.get(
                            "X-ComicCraft-Error-Code", "UNEXPECTED_SERVER_ERROR"
                        ),
                        message=result.headers.get(
                            "X-ComicCraft-Error-Message", UNEXPECTED_ERROR_MESSAGE
                        ),
                    )
                else:
                    publish("complete")
                    publish("result", html=result.body.decode("utf-8"))
            except Exception as exc:
                _log_generation_failure("COMIC", exc)
                publish(
                    "error",
                    success=False,
                    error_code="UNEXPECTED_SERVER_ERROR",
                    message=UNEXPECTED_ERROR_MESSAGE,
                )
            publish("end")

        worker = asyncio.create_task(asyncio.to_thread(generate))
        while True:
            item = await events.get()
            event, data = item
            if event == "end":
                break
            yield f"event: {event}\ndata: {json.dumps(data)}\n\n"
        await worker

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )