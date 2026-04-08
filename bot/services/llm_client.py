"""HTTP client for the Qwen LLM service (text-only).

Sends detailed classifier analysis and gets a unique clothing recommendation.
"""

import logging

import httpx

from bot.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert stylist and weather-aware clothing advisor. "
    "You will receive a DETAILED AI analysis of a street photo including "
    "weather classification, brightness, color temperature, contrast, "
    "and saturation.\n\n"
    "Your task: use this analysis to give a SPECIFIC, ACTIONABLE "
    "clothing recommendation.\n\n"
    "ALWAYS format your response like this:\n\n"
    "What I see: [1 short sentence summarizing the scene from the analysis]\n\n"
    "Wear this:\n"
    "- Top: [specific item]\n"
    "- Bottom: [specific item]\n"
    "- Footwear: [specific item]\n"
    "- Accessories: [specific items]\n"
    "- Layers: [specific items if needed]\n\n"
    "Tips: [2-3 practical tips]\n\n"
    "Rules:\n"
    "- Be VERY specific — name actual clothing items\n"
    "- Use short, clear phrases — no rambling\n"
    "- Reference the visual evidence (brightness, colors, contrast)\n"
    "- ALWAYS respond in English\n"
    "- Do NOT use markdown symbols like **, *, _, `\n"
    "- Each response must be unique — vary your word choice"
)


class LLMError(Exception):
    """Raised when all LLM services fail."""


async def get_clothing_recommendation(image_context: str) -> str:
    """Get clothing recommendation from classifier analysis.

    Args:
        image_context: Full analysis string from the classifier.

    Returns:
        Clothing recommendation text (English, clean, no markdown).

    Raises:
        LLMError: If all LLM services fail.
    """
    if settings.llm_api_base_url and settings.llm_api_key:
        try:
            url = f"{settings.llm_api_base_url}/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {settings.llm_api_key}",
            }
            payload = {
                "model": settings.llm_api_model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": (
                        f"Here is the visual analysis of the street photo:\n\n"
                        f"{image_context}\n\n"
                        f"What should I wear right now?"
                    )},
                ],
                "temperature": 1.2,
                "max_tokens": 300,
                "top_p": 0.95,
                "frequency_penalty": 0.3,
                "presence_penalty": 0.3,
            }
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                choices = data.get("choices", [])
                if not choices:
                    raise ValueError(f"Empty choices: {data}")
                content = choices[0].get("message", {}).get("content", "")
                if not content:
                    raise ValueError(f"Empty message: {data}")
                logger.info("LLM (%s) responded: %s",
                           settings.llm_api_model, content[:80])
                return content.strip()
        except Exception as e:
            logger.warning("LLM (%s) failed: %s", settings.llm_api_model, e)

    # Fallback to secondary model
    if settings.llm_api_base_url_2 and settings.llm_api_key:
        try:
            url = f"{settings.llm_api_base_url_2}/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {settings.llm_api_key}",
            }
            payload = {
                "model": settings.llm_api_model_2,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": (
                        f"Here is the visual analysis:\n\n{image_context}\n\n"
                        f"What should I wear?"
                    )},
                ],
                "temperature": 1.2,
                "max_tokens": 300,
            }
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                choices = data.get("choices", [])
                if not choices:
                    raise ValueError(f"Empty choices: {data}")
                content = choices[0].get("message", {}).get("content", "")
                if not content:
                    raise ValueError(f"Empty message: {data}")
                logger.info("Secondary LLM responded: %s", content[:80])
                return content.strip()
        except Exception as e:
            logger.warning("Secondary LLM failed: %s", e)

    raise LLMError(
        f"All LLM services failed. "
        f"Primary: {settings.llm_api_model}, "
        f"Secondary: {settings.llm_api_model_2}"
    )
