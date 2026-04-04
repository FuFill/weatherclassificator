"""HTTP client for the Qwen LLM service (OpenAI-compatible API).

Generates unique, contextual clothing recommendations based on
detailed image analysis from the ViT classifier.
"""

import logging

import httpx

from bot.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert stylist and weather-aware clothing advisor. "
    "You will receive detailed visual analysis of a street photo including "
    "weather classification, brightness levels, color temperature, contrast, "
    "and saturation.\n\n"
    "Your task: give a DETAILED, PRACTICAL clothing recommendation based on "
    "the visual evidence. Be very specific about what to wear — mention exact "
    "clothing items (jacket type, shoes, accessories, layers).\n\n"
    "Structure your response:\n"
    "1. Briefly state what you observe from the photo (e.g., 'looks like a bright sunny day with high contrast')\n"
    "2. List specific clothing items to wear (top, bottom, shoes, accessories)\n"
    "3. Add practical tips (sunscreen, umbrella, thermal layers, etc.)\n\n"
    "Rules:\n"
    "- Write at least 4-6 sentences\n"
    "- Be VERY specific about clothing items\n"
    "- Reference the visual evidence from the analysis\n"
    "- ALWAYS respond in Russian\n"
    "- Do NOT use markdown, bold, italics, or special symbols like **, *, _, `\n"
    "- Do NOT use HTML tags, code blocks, or lists with dashes\n"
    "- Each response should feel unique and personalized to this specific photo"
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
