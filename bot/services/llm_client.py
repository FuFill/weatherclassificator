"""HTTP client for the Qwen LLM service with vision support.

Sends both the image (base64) and classifier analysis to the LLM
so it can see the photo directly AND use the ViT predictions.
"""

import base64
import logging

import httpx

from bot.config import settings

logger = logging.getLogger(__name__)

# Vision model for image understanding
VISION_MODEL = "vision-model"

SYSTEM_PROMPT = (
    "You are an expert stylist and weather-aware clothing advisor. "
    "You will receive a street photo AND a detailed AI analysis of it "
    "including weather classification, brightness, color temperature, contrast, "
    "and saturation.\n\n"
    "Your task: look at the photo AND read the analysis, then give a SPECIFIC, "
    "ACTIONABLE clothing recommendation.\n\n"
    "ALWAYS format your response like this:\n\n"
    "What I see: [1 short sentence about what you observe in the photo]\n\n"
    "Wear this:\n"
    "- Top: [specific item]\n"
    "- Bottom: [specific item]\n"
    "- Footwear: [specific item]\n"
    "- Accessories: [specific items]\n"
    "- Layers: [specific items if needed]\n\n"
    "Tips: [2-3 practical tips like sunscreen, umbrella, thermal wear, etc.]\n\n"
    "Rules:\n"
    "- Be VERY specific — name actual clothing items, not vague suggestions\n"
    "- Use short, clear phrases — no long rambling sentences\n"
    "- Combine what you SEE in the photo with the classifier data\n"
    "- ALWAYS respond in English\n"
    "- Do NOT use markdown symbols like **, *, _, `\n"
    "- Each response must be unique — vary your word choice every time"
)


class LLMError(Exception):
    """Raised when all LLM services fail."""


def _image_to_base64(image_bytes: bytes) -> str:
    """Convert image bytes to base64 data URL."""
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


async def _call_vision_llm(base_url: str, api_key: str, image_b64: str, text_context: str) -> str:
    """Call the vision model with both image and text.

    Returns:
        Clean response text.
    """
    url = f"{base_url}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    payload = {
        "model": VISION_MODEL,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_b64},
                    },
                    {
                        "type": "text",
                        "text": (
                            f"Here is the AI analysis of this street photo:\n\n"
                            f"{text_context}\n\n"
                            f"What should I wear right now?"
                        ),
                    },
                ],
            },
        ],
        "temperature": 1.2,
        "max_tokens": 300,
        "top_p": 0.95,
        "frequency_penalty": 0.3,
        "presence_penalty": 0.3,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise ValueError(f"LLM returned empty choices: {data}")
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if not content:
            raise ValueError(f"LLM returned empty message: {data}")
        return content.strip()


async def get_clothing_recommendation(image_context: str, image_bytes: bytes = None) -> str:
    """Get clothing recommendation with vision model if image available.

    If image_bytes is provided, sends it to the vision model so LLM can
    see the photo directly. Falls back to text-only if vision fails
    or no image provided.

    Args:
        image_context: Full analysis string from the classifier.
        image_bytes: Optional raw image bytes for vision model.

    Returns:
        Clothing recommendation text (English, clean, no markdown).

    Raises:
        LLMError: If all LLM services fail.
    """
    # Try vision model first (if we have the image)
    if image_bytes and settings.llm_api_base_url and settings.llm_api_key:
        try:
            image_b64 = _image_to_base64(image_bytes)
            reply = await _call_vision_llm(
                settings.llm_api_base_url,
                settings.llm_api_key,
                image_b64,
                image_context,
            )
            logger.info("Vision LLM responded: %s", reply[:80])
            return reply
        except Exception as e:
            logger.warning("Vision LLM failed, falling back to text-only: %s", e)

    # Fallback: text-only LLM call
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
            async with httpx.AsyncClient(timeout=45.0) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                choices = data.get("choices", [])
                if not choices:
                    raise ValueError(f"Empty choices: {data}")
                content = choices[0].get("message", {}).get("content", "")
                if not content:
                    raise ValueError(f"Empty message: {data}")
                logger.info("Text LLM (%s) responded: %s",
                           settings.llm_api_model, content[:80])
                return content.strip()
        except Exception as e:
            logger.error("Text LLM also failed: %s", e)

    raise LLMError(
        f"All LLM services failed. Vision: {VISION_MODEL}, "
        f"Text: {settings.llm_api_model}"
    )
