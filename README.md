# WeatherWear

AI-powered Telegram bot that analyzes street photos and tells you exactly what to wear.

## Demo

<p align="center">
  <img src="https://img.shields.io/badge/Classifier-SigLIP2%20(93M%20params)-blue" alt="Classifier"/>
  <img src="https://img.shields.io/badge/LLM-Qwen%20(qwen3--coder--plus)-green" alt="LLM"/>
  <img src="https://img.shields.io/badge/TTA-3%20augmentations-orange" alt="TTA"/>
  <img src="https://img.shields.io/badge/Inference-ONNX%20Runtime-purple" alt="ONNX"/>
</p>

### How it works

```
User sends photo → 👀 "Analyzing..." → Weather classification → AI clothing advice
```

**Example output:**

```
🌧️ Weather: rainy (confidence 87%)
Photo analysis: Dim lighting — early morning or evening. Cool color tones detected.
Low contrast — fog, haze, or mist. Muted colors — dull lighting or heavy cloud.

Clothing recommendation:

What I see: overcast street with cool blue tones and low visibility.

Wear this:
- Top: waterproof windbreaker with hood
- Bottom: quick-dry synthetic trousers
- Footwear: rubber rain boots
- Accessories: large stick umbrella
- Layers: fleece sweater underneath

Tips: put your phone in a waterproof case, swap fabric bag for water-resistant backpack, take spare socks.
```

> **Screenshots:** After deployment, send a photo to the bot in Telegram and screenshot the response. Add it here as `docs/screenshot-1.png`.

## Product Context

### End Users

People who look out the window and can't decide what to wear. Commuters, students, anyone who needs fast, practical clothing advice from a photo — without storing their images anywhere.

### Problem

You look outside — is it sunny enough for shorts? Is that fog or just overcast? Do you need a jacket? Checking a weather app tells you the temperature but not what it actually looks like on your street right now.

### Solution

Take a photo of your street. WeatherWear analyzes it with a vision model (SigLIP2, 93M parameters) using Test-Time Augmentation for accuracy, then an LLM generates a specific clothing recommendation — exact items (top, bottom, footwear, accessories), not vague "dress warmly" advice.

## Features

### Implemented ✅

| Feature | Description |
|---------|-------------|
| **Photo analysis** | Send any street photo → get weather classification (sunny, cloudy, rainy, snowy, foggy, night) |
| **AI clothing advice** | LLM generates unique, detailed recommendation every time (specific items, not generic) |
| **TTA (Test-Time Augmentation)** | 3 augmentations per image for higher accuracy (original, flip, center crop) |
| **ONNX Runtime** | 2-3x faster CPU inference compared to PyTorch |
| **Visual analysis** | Brightness, color temperature, contrast, saturation — all sent to LLM as context |
| **Snow detection heuristic** | Fixes model confusion between snow and sunny |
| **Text weather input** | Describe weather in text ("raining heavily") → get advice without photo |
| **Inline keyboard** | Quick weather-type buttons with pre-written advice |
| **Loading indicator** | Bot replies with 👀 immediately when photo is received |
| **High randomness** | Temperature 1.2 + frequency/presence penalty — every response unique |
| **Fallback handling** | If LLM is down, bot still responds with classifier analysis |
| **Docker deployment** | bot + classifier as separate containers, health checks, auto-restart |

### Not Yet Implemented 🔲

| Feature | Description |
|---------|-------------|
| User history | Per-user request log with past photos and recommendations |
| Analytics dashboard | Weather distribution, request volume over time |
| Geolocation + real weather API | OpenWeatherMap integration as second opinion |
| Multi-language support | Russian, Spanish, etc. |
| Outfit image generation | DALL-E / Stable Diffusion generates example outfit |
| Budget-aware suggestions | User sets budget, bot suggests specific items from shops |

## Usage

### In Telegram

1. Find the bot on Telegram (via its username from @BotFather)
2. Send `/start` to see the welcome message
3. Send a **photo** of the street
4. Wait for the 👀 reaction → receive weather classification + clothing advice

### Commands

| Command | Description |
|---------|-------------|
| `/start` | Welcome message + inline keyboard |
| `/help` | List of all commands |
| `/health` | Service status (classifier + LLM) |
| `/weather` | Quick reference for all weather types |

### Text Input

You can also describe weather in text:
- "it's raining and windy" → rainy advice
- "snowy and freezing" → snowy advice
- "foggy morning" → foggy advice

## Deployment

### Requirements

- **OS:** Ubuntu 24.04 (or any Linux with Docker)
- **Docker** 24+ and **Docker Compose** v2+
- **RAM:** 2 GB minimum (classifier uses ~800MB)
- **Disk:** 2 GB for model cache
- **Network:** Outbound internet (downloads model from HuggingFace on first start)

### What's Included

| Service | Port | Description |
|---------|------|-------------|
| `classifier` | 8001:8000 | SigLIP2 model with ONNX Runtime, TTA |
| `bot` | — | Telegram bot (aiogram), connects to classifier + LLM |

### Step-by-Step

#### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/se-toolkit-hackathon.git
cd se-toolkit-hackathon
```

#### 2. Create environment file

```bash
cp .env.docker.secret.example .env.docker.secret
nano .env.docker.secret
```

Fill in the values:

```env
# Telegram Bot — get token from @BotFather
BOT_TOKEN=your-bot-token-here

# LLM (Qwen) — your LLM API endpoint
LLM_API_BASE_URL=http://your-llm-host:8080/v1
LLM_API_KEY=your-api-key-here
LLM_API_MODEL=qwen3-coder-plus

# Secondary LLM (fallback, optional)
LLM_API_BASE_URL_2=http://your-llm-host:8080/v1
LLM_API_MODEL_2=coder-model
```

#### 3. Build and start

```bash
docker compose --env-file .env.docker.secret up --build -d
```

First run downloads the SigLIP2 model (~350 MB) from HuggingFace. This takes 30-60 seconds.

#### 4. Verify

```bash
# Check all services are running
docker compose --env-file .env.docker.secret ps

# Check classifier is healthy (model loaded)
docker compose --env-file .env.docker.secret logs classifier --tail 10

# Check bot is polling
docker compose --env-file .env.docker.secret logs bot --tail 10

# Test classifier directly
curl http://localhost:8001/health
```

#### 5. Test in Telegram

1. Open Telegram and find your bot
2. Send `/start` — you should see the welcome message with inline keyboard
3. Send a photo of the street
4. You should see 👀 reaction, then the weather classification + clothing advice

### Troubleshooting

| Symptom | Solution |
|---------|----------|
| Bot container exits immediately | Check logs: `docker compose logs bot` — usually missing `BOT_TOKEN` |
| Classifier returns 503 | Model is still loading. Wait 60s and retry. |
| LLM fails | Ensure `LLM_API_BASE_URL` is reachable from inside Docker container |
| Out of memory | Classifier needs ~800MB RAM. Ensure VM has at least 2GB. |
| Model download fails | Check internet connection. First start downloads ~350MB from HuggingFace. |

### Stopping

```bash
docker compose --env-file .env.docker.secret down
```

### Updating

```bash
git pull
docker compose --env-file .env.docker.secret up --build -d
```

## License

MIT — see [LICENSE](LICENSE).
