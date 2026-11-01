import numpy as np
import io, base64
from PIL import Image, ImageDraw
from backend.services.prediction_service import prediction_service

# Proper MNIST-style handwritten 5
img = Image.new("L", (280, 280), 0)
d = ImageDraw.Draw(img)

# 1. Top horizontal bar
d.line([(100, 60), (175, 60)], fill=255, width=24)
# 2. Vertical stem
d.line([(105, 60), (105, 135)], fill=255, width=24)
# 3. Belly curve starting with top horizontal junction and curving down
belly = [
    (105, 135), (145, 130), (180, 145), (185, 180), (165, 215), (120, 225), (85, 210)
]
d.line(belly, fill=255, width=24, joint="curve")

buf = io.BytesIO()
img.save(buf, format="PNG")
b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

for m in ["perceptron_model", "adaline_model", "hebb_model"]:
    res = prediction_service.predict_from_base64(m, b64)
    print(f"Model: {m:18} | Pred: {res['predicted_digit']} | Top 3: {res['top_3']}")
