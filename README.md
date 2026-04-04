# WeatherWear — Telegram Bot for Clothing Recommendations

## Product Brief

> Telegram-бот, который принимает фото улицы, классифицирует погоду (sunny, cloudy, rainy, snowy, foggy) и даёт рекомендации по одежде через LLM-агент.

**Целевая аудитория:** Люди, которые не могут определиться, что надеть. Быстрый совет по фото улицы без сохранения изображений.

## Architecture

```
┌──────────────┐      ┌──────────────────────────┐      ┌──────────────────────┐
│  Telegram    │─────▶│  WeatherWear Bot         │      │  Qwen LLM API        │
│  User        │◀────│  (aiogram)               │─────▶│  (qwen3-coder-plus)  │
└──────────────┘      │         │                │      └──────────────────────┘
                      │         │                │
                      │         ▼                │
                      │  ┌───────────────────┐   │
                      │  │ SigLIP2 Classif.  │   │
                      │  │  (prithivMLmods/  │   │
                      │  │   Weather-Image-  │   │
                      │  │   Classification) │   │
                      │  └───────────────────┘   │
                      └──────────────────────────┘
```

## Requirements

### P0 — Must Have

1. **Приём фото:** Бот принимает фотографию через Telegram
2. **Классификация:** ViT-модель классифицирует погоду в один из 6 типов: `sunny`, `cloudy`, `rainy`, `snowy`, `foggy`, `night`
3. **Рекомендация:** LLM (Qwen) получает метку погоды и возвращает краткий совет по одежде
4. **Тестовый режим:** `cd bot && uv run bot.py --test "<command>"` работает без Telegram
5. **Error handling:** Падение модели/LLM — дружелюбное сообщение, не креш

### P1 — Should Have

1. **Inline-кнопки:** Быстрый выбор действий
2. **История:** Контекст диалога (multi-turn)
3. **Кэширование:** Повторные запросы к LLM не тратят токены

### P2 — Nice to Have

1. **Rich formatting:** Красивое форматирование ответа (эмодзи, таблицы)
2. **Мульти-фото:** Анализ нескольких фото подряд
3. **Геолокация:** Учёт местоположения пользователя

### P3 — Deployment

1. Бот контейнеризован (Dockerfile)
2. Docker Compose: bot + ViT-сервис
3. Развёрнут на VM
4. README документирует деплой

---

## Implementation Plan

### Phase 1: Scaffold & Project Structure

**Цель:** Создать рабочую структуру проекта с тестовым режимом.

```
weatherwear/
├── bot/                          # Telegram-бот
│   ├── bot.py                    # Entry point (Telegram + --test mode)
│   ├── config.py                 # Загрузка env-переменных
│   ├── pyproject.toml            # Зависимости бота
│   ├── Dockerfile                # Контейнеризация
│   ├── handlers/                 # Обработчики команд
│   │   ├── __init__.py
│   │   ├── start.py              # /start — приветствие
│   │   ├── help.py               # /help — список команд
│   │   └── photo_handler.py      # Обработка фото
│   └── services/                 # Сервисы (API-клиенты)
│       ├── __init__.py
│       ├── vit_classifier.py     # Клиент к ViT-сервису
│       └── llm_client.py         # Клиент к Qwen LLM
├── classifier/                   # ViT-сервис классификации
│   ├── app.py                    # FastAPI-сервис
│   ├── model_loader.py           # Загрузка модели с HuggingFace
│   ├── requirements.txt          # Зависимости (transformers, torch, pillow)
│   ├── Dockerfile                # Контейнеризация
│   └── tests/
│       └── test_classifier.py
├── docker-compose.yml            # Оркестрация
├── .env.bot.example              # Пример переменных для бота
├── .env.docker.secret            # Секреты для деплоя
└── README.md                     # Документация
```

**Deliverables:**

- `bot/bot.py` — entry point с `--test` режимом
- `bot/handlers/` — заглушки обработчиков
- `bot/config.py` — загрузка env
- `bot/pyproject.toml` — зависимости (aiogram, httpx, pydantic-settings)
- `bot/PLAN.md` — план разработки

**Verify:**

```bash
cd bot && uv run bot.py --test "/start"    # → welcome message
cd bot && uv run bot.py --test "/help"     # → список команд
```

---

### Phase 2: ViT Classifier Service

**Цель:** FastAPI-сервис, который классифицирует фото погоды.

**Технологии:** `transformers`, `torch`, `Pillow`, `FastAPI`, `uvicorn`

**Модель:** [prithivMLmods/Weather-Image-Classification](https://huggingface.co/prithivMLmods/Weather-Image-Classification) — SigLIP2-модель (93M параметров), 5 классов.

**Эндпоинты:**

| Метод | Путь | Описание |
|-------|------|----------|
| POST | `/classify` | Принимает фото (multipart/form-data), возвращает `{weather_type, confidence}` |
| GET | `/health` | Проверка работоспособности |

**Deliverables:**

- `classifier/app.py` — FastAPI-приложение
- `classifier/model_loader.py` — загрузка модели при старте
- `classifier/Dockerfile` — контейнер с PyTorch + transformers
- `classifier/requirements.txt` — зависимости

**Verify:**

```bash
cd classifier && uvicorn app:app --reload
curl -X POST http://localhost:8001/classify -F "photo=@sunny_street.jpg"
# → {"weather_type": "sunny", "confidence": 0.92}
```

---

### Phase 3: Bot + Classifier Integration

**Цель:** Бот принимает фото, отправляет в classifier, получает метку погоды.

**Deliverables:**

- `bot/services/vit_classifier.py` — HTTP-клиент к classifier-сервису
- `bot/handlers/photo_handler.py` — логика обработки фото
- `bot/bot.py` — обновлённый entry point

**Поток данных:**

```
User отправляет фото → bot.py → photo_handler → vit_classifier → POST /classify → {weather_type}
```

**Verify:**

```bash
cd bot && uv run bot.py --test "/classify_test sunny.jpg"
# → "Detected: sunny (confidence: 0.92)"
```

---

### Phase 4: LLM Agent Integration (Qwen)

**Цель:** Бот отправляет метку погоды в Qwen LLM и получает рекомендацию по одежде.

**LLM Config:**

- `LLM_API_BASE_URL=http://10.93.25.110:42005/v1`
- `LLM_API_MODEL=qwen3-coder-plus`
- OpenAI-совместимый API

**System Prompt:**

```
You are a weather-aware clothing advisor. Given a weather type (sunny/cloudy/rainy/snowy/foggy/night),
provide a short, practical clothing recommendation. Keep it under 3 sentences.
Be friendly and specific.
```

**Tool для LLM:**

| Tool | Описание |
|------|----------|
| `get_clothing_recommendation(weather_type)` | Возвращает совет по одежде для данной погоды |

**Deliverables:**

- `bot/services/llm_client.py` — OpenAI-совместимый клиент к Qwen
- `bot/handlers/photo_handler.py` — обновлённый: фото → classifier → LLM → ответ
- `bot/config.py` — LLM credentials из env

**Поток данных:**

```
Фото → ViT → {weather_type: "rainy"}
       → LLM System: "You are a clothing advisor..."
       → LLM User: "Weather: rainy"
       → LLM Response: "It's raining! Wear a waterproof jacket..."
       → Bot → User
```

**Verify:**

```bash
cd bot && uv run bot.py --test "/photo rainy_test.jpg"
# → "🌧️ Raining today! I'd recommend: waterproof jacket, umbrella, waterproof boots..."
```

---

### Phase 5: Docker Compose & Deployment

**Цель:** Контейнеризация и деплой на VM.

**docker-compose.yml** — два сервиса:

| Сервис | Контекст | Порт | Описание |
|--------|----------|------|----------|
| `classifier` | `.` | 8001:8000 | SigLIP2 модель, healthcheck |
| `bot` | `.` | — | Telegram-бот, зависит от classifier |

**Deploy на VM:**

```bash
# 1. Клонировать репозиторий на VM
git clone <repo-url> ~/weatherwear
cd ~/weatherwear

# 2. Создать .env.docker.secret
cp .env.docker.secret.example .env.docker.secret
# Заполнить BOT_TOKEN, LLM_API_BASE_URL, LLM_API_KEY

# 3. Запустить
docker compose --env-file .env.docker.secret up --build -d

# 4. Проверить
docker compose --env-file .env.docker.secret ps
docker compose --env-file .env.docker.secret logs classifier --tail 20
docker compose --env-file .env.docker.secret logs bot --tail 20
```

**Verify:**

- `/start` — welcome message
- Отправить фото улицы → рекомендация по одежде
- `/health` — статус classifier + LLM
- `curl http://localhost:8001/health` — classifier снаружи

---

### Phase 6: Polish & Documentation

**Цель:** Финальная полировка и документация.

**Deliverables:**

- Inline-кнопки для быстрых действий
- Красивое форматирование ответов
- Обновлённый README с полным описанием
- Troubleshooting секция

---

## Environment Variables

### .env.bot.example

```env
# Telegram Bot
BOT_TOKEN=<bot-token-from-BotFather>

# ViT Classifier
CLASSIFIER_API_BASE_URL=http://localhost:8001

# LLM (Qwen)
LLM_API_BASE_URL=http://<vm-ip>:42005/v1
LLM_API_KEY=<qwen-api-key>
LLM_API_MODEL=qwen3-coder-plus
```

### .env.docker.secret

```env
# Telegram Bot
BOT_TOKEN=<your-bot-token>

# ViT Classifier (Docker networking)
CLASSIFIER_API_BASE_URL=http://classifier:8000

# LLM (Qwen) — используем host.docker.internal если LLM на хосте
LLM_API_BASE_URL=http://10.93.25.110:42005/v1
LLM_API_KEY=sk-8774a89b791548be9cbe1113e78ab2ef
LLM_API_MODEL=qwen3-coder-plus
```

---

## Local Development

### Run Classifier

```bash
# Install dependencies
uv sync

# Start the classifier service
uv run uvicorn classifier.app:app --host 127.0.0.1 --port 8001

# Test (in another terminal)
curl http://127.0.0.1:8001/health
curl -X POST http://127.0.0.1:8001/classify -F "photo=@your_photo.jpg"
```

> **Note:** At first start, the model (~350MB) downloads from HuggingFace. This takes 30-60 seconds.

### Run Bot in Test Mode

```bash
uv run -m bot --test "/start"
uv run -m bot --test "/help"
uv run -m bot --test "/health"
```

### Run Bot in Telegram Mode

```bash
# Make sure .env.bot.secret has BOT_TOKEN
uv run -m bot
```

---

## Troubleshooting

| Symptom | Solution |
|---------|----------|
| Classifier не отвечает | Проверить `docker compose logs classifier` — модель загружается при первом старте (~1 мин) |
| LLM запросы фейлятся | Убедиться что `LLM_API_BASE_URL` доступен из контейнера |
| Бот не подключается | Проверить `BOT_TOKEN` в `.env.docker.secret` |
| Out of memory на VM | ViT-модель требует ~2GB RAM; уменьшить batch size или использовать CPU-only |

---

## Weather Types & Clothing Mapping

| Model Label | Display Name | Описание | Пример рекомендации |
|-------------|-------------|----------|---------------------|
| `sun/clear` | `sunny` | Ясно, солнечно | Лёгкая одежда, SPF, sunglasses |
| `cloudy/overcast` | `cloudy` | Облачно, пасмурно | Лёгкая куртка, может быть зонт |
| `rain/storm` | `rainy` | Дождь, гроза | Waterproof jacket, umbrella, boots |
| `snow/frosty` | `snowy` | Снег, мороз | Тёплая куртка, шапка, перчатки |
| `foggy/hazy` | `foggy` | Туман, дымка | Видимая одежда, слои |

> **Note:** Модель не определяет `night`. Ночные фото обычно классифицируются как `cloudy` или `foggy`.
> При необходимости можно добавить проверку экспозиции фото для определения ночи.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Bot | Python 3.14, aiogram 3.x, httpx, pydantic-settings |
| Classifier | FastAPI, SigLIP2 (google/siglip2-base), PyTorch, Pillow |
| Model | prithivMLmods/Weather-Image-Classification (93M params, 5 classes) |
| LLM | Qwen (qwen3-coder-plus) via OpenAI-compatible API |
| Deploy | Docker, Docker Compose |
| Dev Tools | uv, ruff, pyright, pytest |
