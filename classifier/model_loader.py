"""SigLIP2 model loader for weather classification.

Loads the prithivMLmods/Weather-Image-Classification model from HuggingFace
and provides a predict function.

Model: SiglipForImageClassification (SigLIP2 base) — ~93M params.
Labels: cloudy/overcast, foggy/hazy, rain/storm, snow/frosty, sun/clear
"""

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


class WeatherClassifier:
    """Wrapper around the SigLIP2 weather classification model."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = SiglipForImageClassification.from_pretrained(model_name)
        self.model.eval()

    def predict(self, image: Image.Image) -> dict:
        """Classify an image and return weather type with confidence.

        Args:
            image: PIL Image object

        Returns:
            dict with keys:
              - weather_type (str): human-friendly name (sunny/cloudy/rainy/snowy/foggy)
              - raw_label (str): original model label
              - confidence (float): prediction confidence
              - all_scores (dict): all classes with their probabilities
        """
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

        # All class probabilities
        all_scores = {}
        for i in range(len(probs[0])):
            raw = id2label.get(str(i), id2label.get(i, f"class_{i}"))
            disp = WEATHER_DISPLAY.get(raw, raw)
            all_scores[disp] = round(probs[0][i].item(), 4)

        return {
            "weather_type": display_name,
            "raw_label": raw_label,
            "confidence": round(confidence, 4),
            "all_scores": all_scores,
        }
