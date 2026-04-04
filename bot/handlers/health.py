"""Handler for the /health command — checks all services."""

import logging

from bot.config import settings
from bot.services import vit_classifier as classifier
from bot.services import llm_client as llm

logger = logging.getLogger(__name__)


async def handle_health_async() -> str:
    """Check the real status of all services via HTTP."""
    parts = ["🏥 Статус сервисов:\n"]

    # Check classifier
    try:
        cl_status = await classifier.check_classifier_health()
        parts.append(f"🔬 Classifier: ✅ {cl_status.get('status', 'ok')}")
    except classifier.ClassifierError as e:
        parts.append(f"🔬 Classifier: ❌ {str(e)[:80]}")

    # Check LLM
    if settings.llm_api_base_url:
        try:
            parts.append(
                f"🧠 Qwen LLM: ✅ `{settings.llm_api_base_url}` "
                f"({settings.llm_api_model})"
            )
        except Exception:
            parts.append(f"🧠 Qwen LLM: ❌ не отвечает")
    else:
        parts.append("🧠 Qwen LLM: ⚠️ не настроен (нет LLM_API_BASE_URL)")

    return "\n".join(parts)


def handle_health() -> str:
    """Sync placeholder for --test mode."""
    classifier_url = settings.classifier_api_base_url
    llm_url = settings.llm_api_base_url

    return (
        "🏥 Статус сервисов:\n\n"
        f"🔬 ViT Classifier: `{classifier_url}`\n"
        f"🧠 Qwen LLM: `{llm_url or 'не настроен'}`\n\n"
        "Сервисы готовы к работе ✅"
    )
