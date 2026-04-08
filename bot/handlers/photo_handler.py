"""Handler for photo messages — classifies weather and gives clothing advice.

Flow:
  1. Receive photo from user
  2. Download image bytes from Telegram
  3. Send to ViT classifier → full image analysis
  4. Try Qwen LLM → unique contextual recommendation
  5. If LLM fails, return detailed classifier analysis summary
  6. Return clean text response to user
"""

import logging

from bot.services import llm_client as llm
from bot.services import vit_classifier as classifier

logger = logging.getLogger(__name__)

WEATHER_EMOJI = {
    "sunny": "☀️",
    "cloudy": "☁️",
    "rainy": "🌧️",
    "snowy": "❄️",
    "foggy": "🌫️",
    "night": "🌙",
}


def _clean_response(text: str) -> str:
    """Remove markdown artifacts and ensure clean plain text."""
    text = text.replace("**", "").replace("*", "")
    text = text.replace("__", "").replace("_", "")
    text = text.replace("`", "")
    text = text.replace("<|", "").replace("|>", "")
    import re
    # Replace multiple spaces (but NOT newlines) with single space
    text = re.sub(r" {2,}", " ", text)
    # Remove trailing whitespace on each line
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()


async def handle_photo_async(file_bytes: bytes, user_message: str = "") -> str:
    """Process a photo and return a UNIQUE clothing recommendation from LLM.

    Output format:
      🌧️ Weather: rainy (confidence 87%)
      Photo analysis: Dim lighting — early morning or evening.

      Clothing recommendation:
      [LLM-generated advice]

    Args:
        file_bytes: Raw image bytes from Telegram.
        user_message: Optional text sent with the photo.

    Returns:
        Formatted text recommendation (no markdown, English only).
    """
    try:
        result = await classifier.classify_image(file_bytes)
        weather_type = result["weather_type"]
        confidence = result["confidence"]
        context = result["context_for_llm"]

        if user_message:
            context += f"\nUser also said: {user_message}"

        # Build photo analysis summary from visual features
        features = result.get("visual_features", {})
        brightness = features.get("brightness", {})
        color_temp = features.get("color_temperature", {})
        contrast = features.get("contrast", {})
        saturation = features.get("saturation", {})

        analysis_parts = []
        b_val = brightness.get("brightness_0_255", 0)
        if b_val > 200:
            analysis_parts.append("Very bright lighting — likely midday sun")
        elif b_val > 120:
            analysis_parts.append("Normal daylight illumination")
        elif b_val > 45:
            analysis_parts.append("Dim lighting — early morning or evening")
        else:
            analysis_parts.append("Dark image — nighttime or heavy overcast")

        if color_temp.get("dominant_tone") == "warm":
            analysis_parts.append("warm color tones detected")
        elif color_temp.get("dominant_tone") == "cool":
            analysis_parts.append("cool color tones detected")

        c_level = contrast.get("contrast_level", "medium")
        if c_level in ("very_low", "low"):
            analysis_parts.append("low contrast — fog, haze, or mist")
        elif c_level == "high":
            analysis_parts.append("high contrast — clear visibility")

        if saturation.get("is_vivid"):
            analysis_parts.append("vivid colors — good visibility")
        elif saturation.get("is_muted"):
            analysis_parts.append("muted colors — dull lighting or heavy cloud")

        analysis_str = ". ".join(analysis_parts).capitalize() + "."

        # LLM generates unique recommendation — no cached/pre-written answers
        emoji = WEATHER_EMOJI.get(weather_type, "🌤️")
        try:
            recommendation = await llm.get_clothing_recommendation(context)
            recommendation = _clean_response(recommendation)
            return (
                f"{emoji} Weather: {weather_type} (confidence {confidence:.0%})\n"
                f"Photo analysis: {analysis_str}\n\n"
                f"Clothing recommendation:\n{recommendation}"
            )

        except llm.LLMError as e:
            logger.error("LLM failed: %s", e)
            return (
                f"{emoji} Weather: {weather_type} (confidence {confidence:.0%})\n"
                f"Photo analysis: {analysis_str}\n\n"
                f"AI recommendation is temporarily unavailable. "
                f"Dress for {weather_type} weather!"
            )

    except classifier.ClassifierError as e:
        logger.error("Classifier error: %s", e)
        return (
            "Could not analyze the photo. "
            "Make sure it's a street photo and try again."
        )

    except Exception as e:
        import traceback
        logger.error("Unexpected error: %s", traceback.format_exc())
        return (
            "An error occurred while processing the photo. "
            "Try again or send a different photo."
        )


def handle_photo() -> str:
    """Sync placeholder for --test mode."""
    return (
        "Photo received! Analyzing weather and finding clothing advice...\n\n"
        "This is a stub — connect the classifier and LLM for real functionality."
    )
