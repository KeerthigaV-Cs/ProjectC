"""FastAPI application entry point."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from comiccraft.routes import router

PACKAGE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="ComicCraft", version="0.1.0")
app.mount(
    "/static",
    StaticFiles(directory=PACKAGE_DIR / "static"),
    name="static",
)
app.include_router(router)