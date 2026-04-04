# Development Plan: WeatherWear Bot

## Overview

WeatherWear is a Telegram bot that accepts street photos, classifies the weather type using a ViT model, and provides clothing recommendations via a Qwen LLM agent. The target users are people who struggle with deciding what to wear based on current weather conditions.

## Architecture

### Core Components

1. **Telegram Bot** (`bot/`) — aiogram-based bot that accepts photos and returns clothing advice.
2. **ViT Classifier** (`classifier/`) — FastAPI microservice running the `prithivMLmods/Weather-Image-Classification` model (SigLIP2, 5 weather types: sunny, cloudy, rainy, snowy, foggy).
3. **Qwen LLM** — Cloud-hosted LLM that generates clothing recommendations based on the classified weather type.

### Data Flow

```
User sends photo → Bot → POST /classify → ViT → {weather_type}
                                                    ↓
User ← clothing advice ← LLM (Qwen) ← prompt with weather type
```

## Implementation Phases

### Phase 1: Scaffold (current)

Create the basic project structure with:

- `bot/` — Telegram bot entry point with `--test` mode
- `classifier/` — ViT classification service scaffold
- `docker-compose.yml` — orchestration for both services
- Handler architecture: start, help, health, photo (placeholders)
- Configuration via environment variables (pydantic-settings)

**Deliverables**: `bot/bot.py`, `bot/handlers/`, `bot/config.py`, `classifier/app.py`, `docker-compose.yml`

### Phase 2: ViT Classifier Service

Implement the actual model loading and classification:

- Load `prithivMLmods/Weather-Image-Classification` from HuggingFace at startup
- `POST /classify` endpoint — accepts image, returns `{weather_type, confidence}`
- `GET /health` endpoint for service health checks
- Docker container with PyTorch + transformers

**Deliverables**: `classifier/model_loader.py`, `classifier/app.py`, `classifier/Dockerfile`

### Phase 3: Bot + Classifier Integration

Connect the bot to the classifier service:

- `bot/services/vit_classifier.py` — HTTP client (httpx) to call the classifier
- `bot/handlers/photo_handler.py` — download photo from Telegram, send to classifier, format response

**Deliverables**: Working photo classification flow in test mode

### Phase 4: LLM Agent Integration

Connect the bot to Qwen LLM for clothing recommendations:

- `bot/services/llm_client.py` — OpenAI-compatible client for Qwen API
- System prompt: clothing advisor role
- Tool: `get_clothing_recommendation(weather_type)`
- Multi-step reasoning: photo → classifier → weather type → LLM → advice

**Deliverables**: Full end-to-end flow working in test mode

### Phase 5: Deployment (current)

Containerize and deploy on VM:

- Both services in Docker Compose (context: root workspace)
- Health checks and restart policies
- `model_cache` volume for HuggingFace model persistence
- Environment variables from `.env.docker.secret`
- Deploy steps: clone repo → create `.env.docker.secret` → `docker compose up --build -d`
- Verify in Telegram

### Phase 6: Polish

- Inline keyboard buttons for quick actions
- Rich formatting (emoji, tables)
- Error handling and fallbacks
- Documentation and troubleshooting guide

## Testing Strategy

1. **Test mode**: `uv run bot.py --test "/start"` — verify handlers without Telegram
2. **Classifier tests**: Unit tests for model predictions on sample images
3. **Integration tests**: End-to-end flow with mocked services
4. **E2E tests**: Real Telegram interaction after deployment

## Configuration

Environment variables (see `.env.bot.example`):

- `BOT_TOKEN` — Telegram bot token
- `CLASSIFIER_API_BASE_URL` — URL of the ViT classifier service
- `LLM_API_BASE_URL` — Qwen API base URL
- `LLM_API_KEY` — Qwen API key
- `LLM_API_MODEL` — Model name (default: qwen3-coder-plus)

## Success Criteria

- Bot accepts a photo and returns a weather-appropriate clothing recommendation
- Classifier correctly identifies weather types with >80% accuracy
- LLM responses are concise (<3 sentences) and actionable
- All services containerized and deployable via `docker compose up`
- `--test` mode works for all command handlers
