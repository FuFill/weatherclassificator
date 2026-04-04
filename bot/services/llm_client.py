"""HTTP client for the Qwen LLM service (OpenAI-compatible API).

Generates unique, contextual clothing recommendations based on
detailed image analysis from the ViT classifier.
"""

import logging

import httpx

from bot.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a weather-aware clothing advisor. You will receive detailed "
    "visual analysis of a street photo including weather classification, "
    "brightness, color temperature, contrast, and saturation.\n\n"
    "Your task: give a SHORT, PRACTICAL clothing recommendation based on "
    "the visual evidence. Be specific and contextual — mention what you "
    "observed from the photo analysis (e.g., 'looks like a bright sunny day', "
    "'dark image suggests it is night time', 'low contrast means fog').\n\n"
    "Rules:\n"
    "- Keep it under 3 sentences\n"
    "- Be friendly and specific\n"
    "- Reference the visual evidence when possible\n"
    "- ALWAYS respond in Russian\n"
    "- Do NOT use markdown, bold, italics, or any special symbols like **, *, _, `\n"
    "- Do NOT use HTML tags or code blocks\n"
    "- Each response should be slightly different from the previous ones"
)


class LLMError(Exception):
    """Raised when the LLM service is unreachable or returns an error."""


async def get_clothing_recommendation(image_context: str) -> str:
    """Ask the LLM for a clothing recommendation based on full image analysis.

    Args:
        image_context: Full analysis string from the classifier including
                       weather type, confidence, brightness, colors, contrast, etc.

    Returns:
        Clothing recommendation text (Russian, clean, no markdown).

    Raises:
        LLMError: If the LLM service fails.
    """
    url = f"{settings.llm_api_base_url}/chat/completions"

    user_content = (
        f"Here is the visual analysis of the street photo:\n\n"
        f"{image_context}\n\n"
        f"What should I wear right now?"
    )

    payload = {
        "model": settings.llm_api_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.9,
        "max_tokens": 200,
        "top_p": 0.95,
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

        logger.info("LLM recommendation: %s", reply[:100])
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
