"""Export SigLIP2 weather model to ONNX format for 2-3x faster CPU inference.

Usage:
    uv run python classifier/export_onnx.py

This produces: classifier/models/weather-classifier.onnx
"""

import os
from pathlib import Path

import torch
from transformers import AutoImageProcessor, SiglipForImageClassification

MODEL_NAME = "prithivMLmods/Weather-Image-Classification"
OUTPUT_DIR = Path(__file__).parent / "models"
OUTPUT_PATH = OUTPUT_DIR / "weather-classifier.onnx"


def main():
    print(f"Loading model: {MODEL_NAME}")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = SiglipForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Exporting to ONNX: {OUTPUT_PATH}")
    torch.onnx.export(
        model,
        dummy_input,
        str(OUTPUT_PATH),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )

    # Save processor config alongside
    processor.save_pretrained(OUTPUT_DIR)

    file_size = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"Exported successfully: {file_size:.1f} MB")

    # Verify by loading
    import onnx
    onnx_model = onnx.load(str(OUTPUT_PATH))
    onnx.checker.check_model(onnx_model)
    print("ONNX model verification: PASSED")


if __name__ == "__main__":
    main()
