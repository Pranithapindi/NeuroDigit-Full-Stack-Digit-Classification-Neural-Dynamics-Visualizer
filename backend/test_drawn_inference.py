import numpy as np
import io
import base64
from PIL import Image, ImageDraw
from backend.services.prediction_service import prediction_service

def test_drawn_digit_inference():
    print("\n--- Testing Drawn Digit Inference ---")
    
    # Draw sample digits on 280x280 canvas like a human user would
    test_cases = [
        # (Digit label, points list for stroke)
        (1, [(140, 40), (140, 240)]),  # Vertical stroke for 1
        (0, [(140, 50), (100, 90), (100, 190), (140, 230), (180, 190), (180, 90), (140, 50)]), # Oval for 0
        (7, [(90, 60), (190, 60), (130, 240)]), # Top bar + diagonal for 7
    ]

    for digit, points in test_cases:
        # Create 280x280 image with black background and white strokes
        img = Image.new("L", (280, 280), 0)
        draw = ImageDraw.Draw(img)
        draw.line(points, fill=255, width=22, joint="curve")
        
        # Convert to base64
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
        
        # Test with Perceptron
        res = prediction_service.predict_from_base64("perceptron_model", b64)
        pred = res["predicted_digit"]
        top_3 = res["top_3"]
        print(f"Target: {digit} | Predicted: {pred} | Top 3: {top_3}")

if __name__ == "__main__":
    test_drawn_digit_inference()
