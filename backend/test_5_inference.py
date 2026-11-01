import numpy as np
import io
import base64
from PIL import Image, ImageDraw
from backend.services.prediction_service import prediction_service
from backend.dataset.preprocessing import preprocess_image_array

# Let's test the exact preset from DigitRecognition.jsx:
# ctx.moveTo(180, 65);
# ctx.lineTo(110, 65);
# ctx.lineTo(105, 130);
# ctx.arc(140, 165, 46, -Math.PI * 0.6, Math.PI * 0.75);

img = Image.new("L", (280, 280), 0)
draw = ImageDraw.Draw(img)
# Draw preset 5
points = [(180, 65), (110, 65), (105, 130)]
draw.line(points, fill=255, width=24, joint="curve")
# Draw arc for loop
# PIL arc: bbox [left, top, right, bottom], start angle, end angle (in degrees clockwise from 3 o'clock)
# arc center (140, 165), r=46 => bbox [94, 119, 186, 211]
# -0.6 * pi rad = -108 deg = 252 deg. 0.75 * pi rad = 135 deg.
# In PIL: start=252, end=135
draw.arc([94, 119, 186, 211], start=250, end=135, fill=255, width=24)

# Also test a human drawn 5
img2 = Image.new("L", (280, 280), 0)
draw2 = ImageDraw.Draw(img2)
# Top horizontal bar
draw2.line([(100, 65), (180, 65)], fill=255, width=24)
# Vertical neck
draw2.line([(105, 65), (100, 135)], fill=255, width=24)
# Bottom belly loop
# Arc from (100, 135) curving right and down to (105, 220)
arc_points = [
    (100, 135), (135, 125), (175, 145), (180, 180), (160, 215), (120, 225), (85, 210)
]
draw2.line(arc_points, fill=255, width=24, joint="curve")

for name, im in [("Preset 5", img), ("Human style 5", img2)]:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
    
    print(f"\n--- Testing {name} ---")
    for m in ["perceptron_model", "adaline_model", "hebb_model"]:
        res = prediction_service.predict_from_base64(m, b64)
        print(f"Model: {m:18} | Pred: {res['predicted_digit']} | Top 3: {res['top_3']}")
