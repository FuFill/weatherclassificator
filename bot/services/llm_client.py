"""HTTP client for the Qwen LLM service (OpenAI-compatible API).

Sends weather-based prompts and receives clothing recommendations.
"""

import logging

import httpx

from bot.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a weather-aware clothing advisor. Given a weather type "
    "(sunny, cloudy, rainy, snowy, foggy, night), provide a short, practical "
    "clothing recommendation. Keep it under 3 sentences. Be friendly and specific. "
    "Use emojis sparingly. Respond in the same language as the user's message. "
    "If the weather is 'night', assume the person is going out at night and "
    "recommend warm, visible clothing."
)


class LLMError(Exception):
    """Raised when the LLM service is unreachable or returns an error."""


async def get_clothing_recommendation(weather_type: str, user_message: str = "") -> str:
    """Ask the LLM for a clothing recommendation based on weather.

    Args:
        weather_type: Classified weather (sunny / cloudy / rainy / snowy / foggy).
        user_message: Optional additional context from the user.

    Returns:
        Clothing recommendation text.

    Raises:
        LLMError: If the LLM service fails.
    """
    url = f"{settings.llm_api_base_url}/chat/completions"

    user_content = f"The weather is: **{weather_type}**."
    if user_message:
        user_content += f"\n\nUser context: {user_message}"

    payload = {
        "model": settings.llm_api_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.7,
        "max_tokens": 150,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.llm_api_key}",
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()

        data = response.json()
        reply = data["choices"][0]["message"]["content"].strip()

        logger.info("LLM recommendation for %s: %s", weather_type, reply[:100])
        return reply

    except httpx.ConnectError as e:
        raise LLMError(
            f"Cannot connect to LLM service at {url}. "
            f"Check LLM_API_BASE_URL and network."
        ) from e
    except httpx.TimeoutException as e:
        raise LLMError("LLM service timed out. Try again in a moment.") from e
    except (KeyError, IndexError) as e:
        raise LLMError(f"Unexpected LLM response format: {response.text[:200]}") from e
    except httpx.HTTPStatusError as e:
        raise LLMError(
            f"LLM returned HTTP {e.response.status_code}: {e.response.text[:200]}"
        ) from e
