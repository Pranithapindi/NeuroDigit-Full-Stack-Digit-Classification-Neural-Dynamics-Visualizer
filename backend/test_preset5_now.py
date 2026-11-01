import numpy as np
import io, base64
from PIL import Image, ImageDraw
from backend.services.prediction_service import prediction_service

# Test new 5 preset
img = Image.new("L", (280, 280), 0)
d = ImageDraw.Draw(img)

# Top bar
d.line([(100, 65), (175, 65)], fill=255, width=24)
# Neck
d.line([(105, 65), (100, 135)], fill=255, width=24)
# Belly loop
pts = [
    (100, 135), (145, 130), (180, 145), (185, 180), (165, 215), (120, 225), (85, 210)
]
d.line(pts, fill=255, width=24, joint="curve")

buf = io.BytesIO()
img.save(buf, format="PNG")
b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

print("--- Current saved models on new preset 5 ---")
for m in ["perceptron_model", "adaline_model", "hebb_model"]:
    res = prediction_service.predict_from_base64(m, b64)
    print(f"Model: {m:18} | Pred: {res['predicted_digit']} | Top 3: {res['top_3']}")
