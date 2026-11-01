import numpy as np
from PIL import Image
import io
import base64

def array_to_image_bytes(arr_2d, format="PNG"):
    if np.max(arr_2d) <= 1.0:
        arr_2d = arr_2d * 255.0
    arr_uint8 = np.clip(arr_2d, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(arr_uint8)
    buf = io.BytesIO()
    pil_img.save(buf, format=format)
    return buf.getvalue()
