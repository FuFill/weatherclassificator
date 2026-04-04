"""Weather classifier with enhanced visual analysis.

Model: SiglipForImageClassification (prithivMLmods/Weather-Image-Classification)
  - SigLIP2 base (google/siglip2-base-patch16-224), ~93M params
  - 5 classes: cloudy/overcast, foggy/hazy, rain/storm, snow/frosty, sun/clear

Accuracy is enhanced through detailed visual analysis:
  - Brightness analysis (night/day detection)
  - Color temperature (warm/cool tones)
  - Contrast level (clear/hazy)
  - Saturation (vivid/muted)

These features are sent to the LLM alongside the model prediction
for more accurate, contextual clothing recommendations.
"""

import logging
import statistics

import torch
from PIL import Image
from transformers import AutoImageProcessor, SiglipForImageClassification

logger = logging.getLogger(__name__)

MODEL_NAME = "prithivMLmods/Weather-Image-Classification"

WEATHER_DISPLAY = {
    "sun/clear": "sunny",
    "cloudy/overcast": "cloudy",
    "rain/storm": "rainy",
    "snow/frosty": "snowy",
    "foggy/hazy": "foggy",
}

NIGHT_BRIGHTNESS_THRESHOLD = 45
WARM_COLOR_THRESHOLD = 0.15


def _analyze_brightness(image: Image.Image) -> dict:
    """Analyze overall image brightness (0-255)."""
    gray = image.convert("L")
    pixels = list(gray.getdata())
    avg_brightness = statistics.mean(pixels)
    return {
        "brightness_0_255": round(avg_brightness, 1),
        "is_dark": avg_brightness < NIGHT_BRIGHTNESS_THRESHOLD,
        "is_very_bright": avg_brightness > 200,
    }


def _analyze_color_temperature(image: Image.Image) -> dict:
    """Analyze warm vs cool color balance."""
    rgb = image.convert("RGB")
    pixels = list(rgb.getdata())
    sampled = pixels[::10] if len(pixels) > 1000 else pixels

    warm_count = 0
    blue_count = 0
    total = len(sampled)

    for r, g, b in sampled:
        if r + g + b == 0:
            continue
        warmth = (r - (g + b) / 2) / 255
        blueness = (b - (r + g) / 2) / 255
        if warmth > WARM_COLOR_THRESHOLD:
            warm_count += 1
        if blueness > WARM_COLOR_THRESHOLD:
            blue_count += 1

    warmth_ratio = warm_count / total if total > 0 else 0
    blueness_ratio = blue_count / total if total > 0 else 0

    avg_r = statistics.mean([p[0] for p in sampled])
    avg_g = statistics.mean([p[1] for p in sampled])
    avg_b = statistics.mean([p[2] for p in sampled])

    return {
        "warmth_ratio": round(warmth_ratio, 3),
        "blueness_ratio": round(blueness_ratio, 3),
        "dominant_tone": "warm" if warmth_ratio > blueness_ratio else "cool",
        "avg_red": round(avg_r, 1),
        "avg_green": round(avg_g, 1),
        "avg_blue": round(avg_b, 1),
    }


def _analyze_contrast(image: Image.Image) -> dict:
    """Analyze contrast level (hazy vs clear)."""
    gray = image.convert("L")
    pixels = list(gray.getdata())
    sampled = pixels[::10] if len(pixels) > 1000 else pixels
    std_dev = statistics.pstdev(sampled) if sampled else 0

    if std_dev < 20:
        level = "very_low"
    elif std_dev < 40:
        level = "low"
    elif std_dev < 70:
        level = "medium"
    else:
        level = "high"

    return {
        "std_deviation": round(std_dev, 1),
        "contrast_level": level,
    }


def _analyze_saturation(image: Image.Image) -> dict:
    """Analyze color saturation (vivid vs muted)."""
    rgb = image.convert("RGB")
    pixels = list(rgb.getdata())
    sampled = pixels[::10] if len(pixels) > 1000 else pixels

    saturations = []
    for r, g, b in sampled:
        r_f, g_f, b_f = r / 255.0, g / 255.0, b / 255.0
        max_c = max(r_f, g_f, b_f)
        min_c = min(r_f, g_f, b_f)
        s = (max_c - min_c) / max_c if max_c > 0 else 0
        saturations.append(s)

    avg_saturation = statistics.mean(saturations) if saturations else 0

    return {
        "avg_saturation": round(avg_saturation, 3),
        "is_vivid": avg_saturation > 0.4,
        "is_muted": avg_saturation < 0.15,
    }


class WeatherClassifier:
    """Weather classifier with detailed visual analysis for LLM context."""

    def __init__(self) -> None:
        logger.info("Loading SigLIP2 model: %s", MODEL_NAME)
        self.processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        self.model = SiglipForImageClassification.from_pretrained(MODEL_NAME)
        self.model.eval()
        self.id2label = self.model.config.id2label
        logger.info("Model loaded (%d classes).", len(self.id2label))

    def predict(self, image: Image.Image) -> dict:
        """Classify weather with enhanced visual analysis.

        Returns:
            dict with weather_type, confidence, all_scores, visual_features, context_for_llm
        """
        # ML classification
        inputs = self.processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)

        predicted_idx = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][predicted_idx].item()

        raw_label = self.id2label.get(str(predicted_idx), self.id2label.get(predicted_idx, "unknown"))
        display_name = WEATHER_DISPLAY.get(raw_label, raw_label)

        # All probabilities
        all_scores = {}
        for i in range(len(probs[0])):
            raw = self.id2label.get(str(i), self.id2label.get(i, ""))
            disp = WEATHER_DISPLAY.get(raw, raw)
            all_scores[disp] = round(probs[0][i].item(), 4)

        # Visual analysis
        brightness = _analyze_brightness(image)
        color_temp = _analyze_color_temperature(image)
        contrast = _analyze_contrast(image)
        saturation = _analyze_saturation(image)

        # Night detection overrides model
        is_night = brightness["is_dark"]
        if is_night:
            display_name = "night"
            confidence = max(confidence, 0.8)
            all_scores["night"] = round(confidence, 4)

        # Build LLM context
        context_parts = []
        context_parts.append(f"Weather classification: {display_name}")
        context_parts.append(f"Confidence: {confidence:.0%}")

        top3 = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        context_parts.append(
            "Top weather possibilities: "
            + ", ".join(f"{w}({p:.0%})" for w, p in top3)
        )

        context_parts.append(f"Brightness: {brightness['brightness_0_255']}/255")
        if brightness["is_very_bright"]:
            context_parts.append("Image is very bright — likely midday sun")
        if brightness["is_dark"]:
            context_parts.append("Image is dark — nighttime or heavy overcast")

        if color_temp["dominant_tone"] == "warm":
            context_parts.append(
                f"Warm color tones (warmth: {color_temp['warmth_ratio']}) — "
                "possibly sunrise, sunset, or warm lighting"
            )
        elif color_temp["dominant_tone"] == "cool":
            context_parts.append(
                f"Cool color tones (blueness: {color_temp['blueness_ratio']}) — "
                "possibly overcast, winter, or shade"
            )

        context_parts.append(f"Contrast: {contrast['contrast_level']}")
        if contrast["contrast_level"] in ("very_low", "low"):
            context_parts.append("Low contrast suggests fog, haze, or mist")

        if saturation["is_vivid"]:
            context_parts.append("Colors are vivid and saturated — clear visibility")
        elif saturation["is_muted"]:
            context_parts.append("Colors are muted/desaturated — dull lighting or fog")

        context_summary = ". ".join(context_parts) + "."

        return {
            "weather_type": display_name,
            "is_night": is_night,
            "confidence": round(confidence, 4),
            "all_scores": all_scores,
            "visual_features": {
                "brightness": brightness,
                "color_temperature": color_temp,
                "contrast": contrast,
                "saturation": saturation,
            },
            "context_for_llm": context_summary,
        }
