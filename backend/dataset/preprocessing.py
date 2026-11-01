import numpy as np
import base64
import io
from PIL import Image, ImageFilter

def preprocess_image_array(img_2d):
    """
    Standard LeCun MNIST preprocessing pipeline for 2D images:
    1. Grayscale float array in [0, 255].
    2. Detect background intensity (invert if light background).
    3. Extract tight bounding box of active digit pixels.
    4. Size-normalize into 20x20 box preserving aspect ratio with anti-aliasing.
    5. Center in 28x28 grid by pixel Center of Mass.
    6. Normalize intensity to [0.0, 1.0].
    """
    img = np.array(img_2d, dtype=np.float32)

    # 1. Background detection: check border pixels
    border_pixels = np.concatenate([img[0, :], img[-1, :], img[:, 0], img[:, -1]])
    if np.mean(border_pixels) > 127.5:
        # Invert: white background with dark ink -> black background with white ink
        img = 255.0 - img

    # Clean low-intensity noise
    img[img < 25.0] = 0.0

    # 2. Extract bounding box
    rows = np.any(img > 25.0, axis=1)
    cols = np.any(img > 25.0, axis=0)

    if not np.any(rows) or not np.any(cols):
        # Empty image: return zeros
        return np.zeros((28, 28), dtype=np.float32), np.zeros(784, dtype=np.float32)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    cropped = img[rmin:rmax+1, cmin:cmax+1]

    # Convert to PIL Image for high quality resampling
    pil_cropped = Image.fromarray(cropped.astype(np.uint8))
    # Soft anti-aliasing filter to smooth stroke edges and match genuine MNIST gradient profile
    if max(cropped.shape) > 40:
        pil_cropped = pil_cropped.filter(ImageFilter.GaussianBlur(radius=1.0))

    # 3. Fit inside 20x20 box while preserving aspect ratio
    w, h = pil_cropped.size
    if w >= h:
        new_w = 20
        new_h = max(1, int(round((h * 20.0) / w)))
    else:
        new_h = 20
        new_w = max(1, int(round((w * 20.0) / h)))

    resized_pil = pil_cropped.resize((new_w, new_h), Image.Resampling.BILINEAR)
    resized_arr = np.array(resized_pil, dtype=np.float32)

    # 4. Center in 28x28 grid using Center of Mass
    canvas = np.zeros((28, 28), dtype=np.float32)
    
    cy, cx = center_of_mass(resized_arr)
    # Position center of mass at ~14.0 (MNIST standard)
    paste_y = int(round(14.0 - cy))
    paste_x = int(round(14.0 - cx))

    # Clamp safely inside [1, 27 - dim] so digits do not touch extreme edges
    paste_y = max(1, min(27 - new_h, paste_y))
    paste_x = max(1, min(27 - new_w, paste_x))

    canvas[paste_y:paste_y+new_h, paste_x:paste_x+new_w] = resized_arr

    # Fine adjustment: check canvas center of mass and fine-shift to exactly 14.0
    final_cy, final_cx = center_of_mass(canvas)
    if not np.isnan(final_cy) and not np.isnan(final_cx):
        shift_y = int(round(14.0 - final_cy))
        shift_x = int(round(14.0 - final_cx))
        shift_y = max(-3, min(3, shift_y))
        shift_x = max(-3, min(3, shift_x))
        if shift_y != 0 or shift_x != 0:
            canvas = shift_image(canvas, shift_y, shift_x)

    # Normalize to [0.0, 1.0]
    normalized_28x28 = np.clip(canvas / 255.0, 0.0, 1.0)
    flattened_784 = normalized_28x28.reshape(-1)

    return normalized_28x28, flattened_784

def center_of_mass(img):
    total = np.sum(img)
    if total <= 1e-5:
        return img.shape[0] / 2.0, img.shape[1] / 2.0
    y_coords, x_coords = np.indices(img.shape)
    cy = np.sum(y_coords * img) / total
    cx = np.sum(x_coords * img) / total
    return float(cy), float(cx)

def shift_image(img, sy, sx):
    shifted = np.zeros_like(img)
    src_y1 = max(0, -sy)
    src_y2 = min(img.shape[0], img.shape[0] - sy)
    src_x1 = max(0, -sx)
    src_x2 = min(img.shape[1], img.shape[1] - sx)

    dst_y1 = max(0, sy)
    dst_y2 = min(img.shape[0], img.shape[0] + sy)
    dst_x1 = max(0, sx)
    dst_x2 = min(img.shape[1], img.shape[1] + sx)

    h = min(src_y2 - src_y1, dst_y2 - dst_y1)
    w = min(src_x2 - src_x1, dst_x2 - dst_x1)

    if h > 0 and w > 0:
        shifted[dst_y1:dst_y1+h, dst_x1:dst_x1+w] = img[src_y1:src_y1+h, src_x1:src_x1+w]
    return shifted

def decode_base64_image(base64_str):
    """
    Decodes a base64 encoded image string (e.g. data:image/png;base64,...)
    into a PIL Grayscale Image, handling transparent alpha properly.
    """
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    img_data = base64.b64decode(base64_str)
    pil_img = Image.open(io.BytesIO(img_data))
    
    # If transparent canvas, composite onto solid black background
    if pil_img.mode in ("RGBA", "LA") or "transparency" in pil_img.info:
        bg = Image.new("RGBA", pil_img.size, (0, 0, 0, 255))
        bg.alpha_composite(pil_img.convert("RGBA"))
        return bg.convert("L")
    return pil_img.convert("L")

def encode_array_to_base64(arr_2d):
    """
    Encodes a 28x28 numpy array (values [0, 1] or [0, 255]) to base64 PNG string.
    """
    if np.max(arr_2d) <= 1.0:
        arr_2d = arr_2d * 255.0
    arr_uint8 = np.clip(arr_2d, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(arr_uint8)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")
