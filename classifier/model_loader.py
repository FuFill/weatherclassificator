"""Weather classifier with extended capabilities.

Loads the prithivMLmods/Weather-Image-Classification model from HuggingFace
and provides enhanced predictions:
  - 5 weather types from the model (sunny, cloudy, rainy, snowy, foggy)
  - Night detection via image brightness
  - Time-of-day hint (daytime/nighttime)
  - Color temperature (warm/cool) for better LLM recommendations

Model: SiglipForImageClassification (SigLIP2 base) — ~93M params.
Labels: cloudy/overcast, foggy/hazy, rain/storm, snow/frosty, sun/clear
"""

import colorsys
import statistics

import torch
from PIL import Image
from transformers import AutoImageProcessor, SiglipForImageClassification

MODEL_NAME = "prithivMLmods/Weather-Image-Classification"

# Human-friendly names for the LLM prompt
WEATHER_DISPLAY = {
    "sun/clear": "sunny",
    "cloudy/overcast": "cloudy",
    "rain/storm": "rainy",
    "snow/frosty": "snowy",
    "foggy/hazy": "foggy",
}

# Thresholds for extended analysis
NIGHT_BRIGHTNESS_THRESHOLD = 45  # Below this → night
WARM_COLOR_THRESHOLD = 0.15     # Above this → warm tones (sunset/sunrise)


def _analyze_brightness(image: Image.Image) -> dict:
    """Analyze overall image brightness.

    Returns:
        dict with avg_brightness (0-255), is_night (bool)
    """
    # Convert to grayscale and calculate average pixel value
    gray = image.convert("L")  # L = 8-bit grayscale
    pixels = list(gray.getdata())
    avg_brightness = statistics.mean(pixels)

    return {
        "avg_brightness": round(avg_brightness, 2),
        "is_night": avg_brightness < NIGHT_BRIGHTNESS_THRESHOLD,
    }


def _analyze_color_temperature(image: Image.Image) -> dict:
    """Analyze color temperature (warm vs cool tones).

    Warm tones (orange/red) → sunset/sunrise
    Cool tones (blue/gray) → overcast/normal day

    Returns:
        dict with warmth (0-1), is_warm (bool)
    """
    rgb = image.convert("RGB")
    pixels = list(rgb.getdata())

    # Sample every 10th pixel for speed
    sampled = pixels[::10] if len(pixels) > 1000 else pixels

    warm_count = 0
    total = len(sampled)

    for r, g, b in sampled:
        if r + g + b == 0:
            continue
        # Calculate warmth: ratio of red channel vs green+blue
        warmth = (r - (g + b) / 2) / 255
        if warmth > WARM_COLOR_THRESHOLD:
            warm_count += 1

    warmth_ratio = warm_count / total if total > 0 else 0

    return {
        "warmth_ratio": round(warmth_ratio, 4),
        "is_warm": warmth_ratio > 0.2,  # >20% warm pixels
    }


def _analyze_contrast(image: Image.Image) -> dict:
    """Analyze image contrast (clear vs hazy).

    Low contrast often indicates fog, haze, or mist.

    Returns:
        dict with contrast_level (low/medium/high)
    """
    gray = image.convert("L")
    pixels = list(gray.getdata())

    # Sample for speed
    sampled = pixels[::10] if len(pixels) > 1000 else pixels

    if not sampled:
        return {"contrast_level": "medium"}

    std_dev = statistics.pstdev(sampled)

    if std_dev < 20:
        return {"contrast_level": "low"}
    elif std_dev < 60:
        return {"contrast_level": "medium"}
    else:
        return {"contrast_level": "high"}


class WeatherClassifier:
    """Wrapper around the SigLIP2 weather classification model
    with extended image analysis capabilities."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = SiglipForImageClassification.from_pretrained(model_name)
        self.model.eval()

    def predict(self, image: Image.Image) -> dict:
        """Classify an image and return weather type with extended analysis.

        Combines ML model prediction with heuristic image analysis:
          - SigLIP2 model for weather type
          - Brightness analysis for night detection
          - Color temperature for warm/cool tones
          - Contrast analysis for clear/hazy distinction

        Args:
            image: PIL Image object

        Returns:
            dict with keys:
              - weather_type (str): sunny / cloudy / rainy / snowy / foggy
              - is_night (bool): whether the image appears to be nighttime
              - confidence (float): prediction confidence
              - analysis (dict): extended analysis (brightness, warmth, contrast)
              - all_scores (dict): probabilities for all model classes
        """
        # ML model prediction
        inputs = self.processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)

        # Top prediction
        predicted_idx = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][predicted_idx].item()

        id2label = self.model.config.id2label
        raw_label = id2label.get(str(predicted_idx), id2label.get(predicted_idx, "unknown"))
        display_name = WEATHER_DISPLAY.get(raw_label, raw_label)

        # Extended analysis
        brightness = _analyze_brightness(image)
        color_temp = _analyze_color_temperature(image)
        contrast = _analyze_contrast(image)

        # Override to night if very dark
        is_night = brightness["is_night"]
        if is_night:
            display_name = "night"
            confidence = max(confidence, 0.8)  # High confidence for night

        # All class probabilities
        all_scores = {}
        for i in range(len(probs[0])):
            raw = id2label.get(str(i), id2label.get(i, f"class_{i}"))
            disp = WEATHER_DISPLAY.get(raw, raw)
            all_scores[disp] = round(probs[0][i].item(), 4)

        # Add night to all_scores if detected
        if is_night:
            all_scores["night"] = confidence

        return {
            "weather_type": display_name,
            "raw_label": raw_label,
            "is_night": is_night,
            "confidence": round(confidence, 4),
            "analysis": {
                "brightness": brightness,
                "warmth": color_temp,
                "contrast": contrast,
            },
            "all_scores": all_scores,
        }
