# ComicCraft

ComicCraft is a FastAPI web app that turns a short story idea into a five-panel comic preview and a downloadable PDF.

## Overview

ComicCraft helps a user describe a comic concept, generate a story and panel structure, create illustrations, preview the result in the browser, and export the final comic as a PDF.

## Features

- Story generation with Gemini as the default provider
- Hugging Face story fallback for temporary service interruptions or quota issues
- Local image generation by default, with optional hosted image providers
- Five-panel comic preview rendered in the browser
- PDF export using fpdf2
- User-friendly validation and provider error handling without exposing secrets or stack traces
- Offline, mocked test suite for local validation

## Technology Stack

### Frontend
- HTML
- CSS
- JavaScript

### Backend
- Python
- FastAPI
- Jinja2 templates
- Pydantic validation

### AI providers
- Gemini for story generation
- Hugging Face fallback story provider
- Hugging Face FLUX.1-schnell for optional image generation
- Local deterministic image generation as the default for no-cost local development

### PDF generation
- fpdf2

## System Architecture

User input
↓
FastAPI app
↓
Story generation
↓
Gemini primary provider
↓
Hugging Face fallback if eligible
↓
Five-panel story structure
↓
Image generation
↓
Five illustrations
↓
Comic preview in browser
↓
PDF export

## Project Structure

```text
comic-craft/
├── src/
│   └── comiccraft/
│       ├── main.py
│       ├── config.py
│       ├── routes.py
│       ├── errors.py
│       ├── models/
│       ├── providers/
│       ├── services/
│       ├── static/
│       └── templates/
├── tests/
├── .env.example
├── .gitignore
├── README.md
├── pyproject.toml
├── requirements.txt
└── .venv/   (local environment; not committed)
```

## Prerequisites

- Python 3.11 or newer
- Git
- Internet access
- A Gemini API key for Gemini story generation
- A Hugging Face access token for Hugging Face providers
- Optional: Hugging Face credits for FLUX image generation

> Hosted AI providers may impose usage limits, quota limits, or credits. ComicCraft does not guarantee unlimited free generations.

## Installation

### Windows PowerShell

```powershell
git clone <repository-url>
cd comic-craft
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
```

If PowerShell blocks script activation, use:

```powershell
.venv\Scripts\activate.bat
```

### macOS / Linux

```bash
git clone <repository-url>
cd comic-craft
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

### VS Code setup

1. Open VS Code
2. Choose File → Open Folder
3. Select the cloned `comic-craft` folder
4. Open the integrated terminal
5. Activate the virtual environment
6. Run the app with the command below

## Environment Variables

Create a local `.env` file from the example and add your own credentials. Never place real secrets in the repository or in the README.

```env
GEMINI_API_KEY=
GEMINI_MODEL=gemini-3.8-flash
STORY_PROVIDER=gemini
STORY_FALLBACK_PROVIDER=huggingface

IMAGE_PROVIDER=huggingface
IMAGE_FALLBACK_PROVIDER=huggingface
IMAGE_API_KEY=
IMAGE_MODEL=gemini-3.1-flash-image

HF_TOKEN=
HF_STORY_MODEL=
HF_IMAGE_MODEL=black-forest-labs/FLUX.1-schnell
```

### Configuration table

| Variable | Purpose | Required |
| --- | --- | --- |
| `GEMINI_API_KEY` | Gemini authentication for story generation | Yes for Gemini story use |
| `GEMINI_MODEL` | Gemini story model | Yes |
| `STORY_PROVIDER` | Primary story provider | Yes |
| `STORY_FALLBACK_PROVIDER` | Fallback story provider for temporary Gemini issues | Yes |
| `IMAGE_PROVIDER` | Image provider selection | Yes |
| `IMAGE_FALLBACK_PROVIDER` | Fallback image provider | Yes |
| `HF_TOKEN` | Hugging Face authentication | Yes for Hugging Face providers |
| `HF_STORY_MODEL` | Hugging Face story model | Only when using HF story generation |
| `HF_IMAGE_MODEL` | Hugging Face image model | Yes for Hugging Face image generation |
| `IMAGE_API_KEY` | Gemini image authentication | Only for Gemini image provider |
| `IMAGE_MODEL` | Gemini image model | Only for Gemini image provider |

## Gemini API Setup

1. Open Google AI Studio or the Gemini API console.
2. Create an API key.
3. Add it to the `GEMINI_API_KEY` value in your local `.env` file.
4. Keep the key in your local machine environment only; do not commit it.

## Hugging Face Setup

1. Sign in to Hugging Face.
2. Create an access token from the Access Tokens page.
3. Ensure the token has access to the model and provider usage needed for your selected workflow.
4. Add the token to `HF_TOKEN` in `.env`.

> Hugging Face Inference Providers may require available credits for model usage. If you see a 402 response or a credit exhaustion message, that is an account/provider limit, not a ComicCraft code bug.

## Running the Application

From the project root, start the app:

```bash
uvicorn comiccraft.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

The health endpoint is available at:

```text
http://127.0.0.1:8000/health
```

## Using ComicCraft

1. Enter a story prompt, main character, setting, tone, and art style.
2. Click Generate comic.
3. Review the generated five-panel story and illustrations.
4. Download the PDF when ready.
5. Start another comic from the form when needed.

## Running Tests

Run the automated tests:

```bash
pytest
```

All tests should pass.

## Troubleshooting

### Python not found
- Install Python 3.11+ and ensure it is on your PATH.
- On Windows, try `py` instead of `python`.

### pip not found
- Upgrade pip with: `python -m pip install --upgrade pip`
- Reopen the terminal after installation changes.

### Virtual environment activation fails
- On Windows PowerShell: use `.venv\Scripts\Activate.ps1`
- If blocked, use `.venv\Scripts\activate.bat`
- On macOS/Linux: `source .venv/bin/activate`

### Missing API key
- Confirm `.env` exists and contains `GEMINI_API_KEY` or `HF_TOKEN` as needed.
- Do not leave keys blank when the provider is enabled.

### Gemini quota or rate limit
- A Gemini quota or rate-limit response often means the provider limit has been reached.
- Wait and retry later or switch to the configured fallback story provider.

### Gemini temporary 503 service error
- This indicates a transient provider outage.
- Try again later or use the configured fallback provider.

### Hugging Face token problem
- Check that `HF_TOKEN` is set in `.env`.
- Recreate the token if it has expired or been revoked.

### Hugging Face 402 credits exhausted
- If the service indicates monthly credits are exhausted, this is a usage limit on the provider account, not a ComicCraft bug.
- Add credits or switch to a different provider setup.

### Hugging Face 429 rate limit
- Wait a few minutes and retry.
- Avoid repeatedly hammering the API with retries.

### Image generation service unavailable
- Confirm your selected provider is configured correctly.
- Check network access and provider status.
- If using Hugging Face, verify the token and model are valid.

### Port 8000 already in use
- Stop the process using port 8000 or launch the app on another port.
- Example: `uvicorn comiccraft.main:app --reload --port 8001`

### Browser cannot connect
- Confirm the app is running in the terminal.
- Check the URL: `http://127.0.0.1:8000`
- Ensure no firewall or port-forwarding setting is blocking local access.

### Dependency installation failure
- Upgrade pip.
- Ensure Python 3.11+ is active.
- Reinstall dependencies with `pip install -r requirements.txt`.

## API / AI Provider Notes

- Gemini is the default story provider.
- Hugging Face is used as the default story fallback when appropriate.
- Local image generation is the default, which avoids external image-generation charges.
- Hugging Face FLUX.1-schnell is the default hosted image model when AI image generation is enabled.
- Provider availability, quotas, and price limits are controlled by the external service and may change without notice.

## Security

- Never commit `.env`, tokens, API keys, generated PDFs, or generated panel images.
- Keep all credentials in a local `.env` file or operating-system environment variables.
- The repository includes a safe `.env.example` with placeholders only.
- `.gitignore` excludes runtime artifacts and local secrets.

## Future Enhancements

- Additional provider integrations
- More export formats
- Better panel-level editing and regeneration
- Improved error analytics and user guidance

## Release Readiness Notes

This project is prepared for a fresh clone from GitHub. A new user should only need to:

1. Clone the repository
2. Create a virtual environment
3. Install dependencies
4. Copy `.env.example` to `.env`
5. Add their own credentials
6. Run `uvicorn comiccraft.main:app --reload`
7. Open the app in a browser

No local machine-specific paths or developer-only setup steps are required for the normal project workflow.