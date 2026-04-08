"""HTTP client for the Qwen LLM service with dual-model fallback.

Primary model is tried first. If it fails, falls back to secondary model.
If both fail, raises LLMError for the caller to handle.
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
    "- ALWAYS respond in English\n"
    "- Do NOT use markdown, bold, italics, or special symbols like **, *, _, `\n"
    "- Do NOT use HTML tags, code blocks, or lists with dashes\n"
    "- Each response must feel unique and personalized — never repeat the same phrasing\n"
    "- Vary your sentence structure and word choice between responses\n"
    "- Sometimes mention specific clothing brands generically (e.g., 'a windbreaker like a North Face shell')\n"
)


class LLMError(Exception):
    """Raised when all LLM services fail."""


def _build_payload(model: str, context: str) -> dict:
    """Build the OpenAI-compatible chat completion payload."""
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Here is the visual analysis of the street photo:\n\n"
                f"{context}\n\n"
                f"What should I wear right now?"
            )},
        ],
        "temperature": 1.2,
        "max_tokens": 300,
        "top_p": 0.95,
        "frequency_penalty": 0.3,
        "presence_penalty": 0.3,
    }


async def _call_llm(base_url: str, api_key: str, model: str, context: str) -> str:
    """Make a single LLM API call.

    Returns:
        Clean response text.

    Raises:
        Exception with details on failure.
    """
    url = f"{base_url}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = _build_payload(model, context)

    async with httpx.AsyncClient(timeout=45.0) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

        choices = data.get("choices", [])
        if not choices:
            raise ValueError(f"LLM returned empty choices response: {data}")

        message = choices[0].get("message", {})
        content = message.get("content", "")
        if not content:
            raise ValueError(f"LLM returned empty message: {data}")

        return content.strip()


async def get_clothing_recommendation(image_context: str) -> str:
    """Get clothing recommendation with dual-model fallback.

    Tries primary model first. If it fails (HTTP error, timeout, etc.),
    tries secondary model. If both fail, raises LLMError.

    Args:
        image_context: Full analysis string from the classifier.

    Returns:
        Clothing recommendation text (Russian, clean, no markdown).

    Raises:
        LLMError: If both primary and secondary LLM services fail.
    """
    # Try primary
    if settings.llm_api_base_url and settings.llm_api_key:
        try:
            reply = await _call_llm(
                settings.llm_api_base_url,
                settings.llm_api_key,
                settings.llm_api_model,
                image_context,
            )
            logger.info("Primary LLM (%s) responded: %s",
                       settings.llm_api_model, reply[:80])
            return reply
        except Exception as e:
            logger.warning("Primary LLM (%s) failed: %s",
                          settings.llm_api_model, e)

    # Try secondary
    if settings.llm_api_base_url_2 and settings.llm_api_key:
        try:
            reply = await _call_llm(
                settings.llm_api_base_url_2,
                settings.llm_api_key,
                settings.llm_api_model_2,
                image_context,
            )
            logger.info("Secondary LLM (%s) responded: %s",
                       settings.llm_api_model_2, reply[:80])
            return reply
        except Exception as e:
            logger.warning("Secondary LLM (%s) failed: %s",
                          settings.llm_api_model_2, e)

    # Both failed
    raise LLMError(
        f"Both LLM services failed. Primary: {settings.llm_api_model} "
        f"({settings.llm_api_base_url}), "
        f"Secondary: {settings.llm_api_model_2} "
        f"({settings.llm_api_base_url_2 or 'not configured'})"
    )
