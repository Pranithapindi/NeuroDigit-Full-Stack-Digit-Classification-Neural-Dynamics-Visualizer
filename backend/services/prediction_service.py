import numpy as np
import io
from PIL import Image
from pathlib import Path
from backend.config import MODELS_DIR
from backend.algorithms.hebb import HebbianClassifier
from backend.algorithms.perceptron import PerceptronClassifier
from backend.algorithms.adaline import AdalineClassifier
from backend.dataset.preprocessing import preprocess_image_array, decode_base64_image, encode_array_to_base64

class PredictionService:
    def __init__(self):
        self.cached_models = {}

    def get_model(self, model_id):
        """Loads or returns cached model instance."""
        model_id = model_id.strip()
        if not model_id.endswith(".json"):
            filename = f"{model_id}.json"
        else:
            filename = model_id

        filepath = MODELS_DIR / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Model '{model_id}' not found. Please train this model first in the Train Models tab.")

        # Check mtime to invalidate cache if retrained
        mtime = filepath.stat().st_mtime
        if model_id in self.cached_models:
            cached_mtime, model_instance = self.cached_models[model_id]
            if cached_mtime == mtime:
                return model_instance

        # Detect algorithm type from filename or file content
        import json
        with open(filepath, "r") as f:
            data = json.load(f)
        algo = data.get("model_type", "perceptron").lower()

        if algo == "hebb":
            model = HebbianClassifier()
        elif algo == "perceptron":
            model = PerceptronClassifier()
        elif algo == "adaline":
            model = AdalineClassifier()
        else:
            model = PerceptronClassifier()

        model.load(filepath)
        self.cached_models[model_id] = (mtime, model)
        return model

    def predict_from_base64(self, model_id, base64_image_str):
        """Processes base64 drawing canvas image and runs inference."""
        if not base64_image_str or len(base64_image_str.strip()) < 20:
            raise ValueError("Canvas is empty. Please draw a digit (0-9) on the canvas first.")

        pil_img = decode_base64_image(base64_image_str)
        return self._process_and_predict(model_id, pil_img)

    def predict_from_bytes(self, model_id, file_bytes):
        """Processes uploaded image file bytes and runs inference."""
        if not file_bytes or len(file_bytes) == 0:
            raise ValueError("Uploaded file is empty.")

        try:
            pil_img = Image.open(io.BytesIO(file_bytes)).convert("L")
        except Exception as e:
            raise ValueError(f"Invalid image format. Supported formats: PNG, JPG, JPEG. ({e})")

        return self._process_and_predict(model_id, pil_img)

    def _process_and_predict(self, model_id, pil_img):
        raw_arr = np.array(pil_img, dtype=np.float32)
        
        # Preprocess into centered 28x28 normalized MNIST format
        norm_28x28, flat_784 = preprocess_image_array(raw_arr)

        # Check if the canvas / image was essentially blank
        if np.max(flat_784) < 0.1 or np.sum(flat_784 > 0.1) < 5:
            raise ValueError("The drawn or uploaded image appears to be blank. Please draw a clear digit stroke.")

        # Load model and run inference
        model = self.get_model(model_id)
        prediction_result = model.predict_scores(flat_784)

        # Encode preprocessed 28x28 image for visual confirmation
        preview_b64 = encode_array_to_base64(norm_28x28)

        # Explain model details
        algo_name = model.__class__.__name__.replace("Classifier", "")
        
        return {
            "model_id": model_id,
            "algorithm": algo_name,
            "predicted_digit": prediction_result["predicted_digit"],
            "raw_scores": prediction_result["raw_scores"],
            "normalized_scores": prediction_result["normalized_scores"],
            "top_3": prediction_result["top_3"],
            "preview_image_b64": preview_b64,
            "input_dimensions": [28, 28],
            "note": "Scores are linear decision outputs normalized via softmax for relative visualization. Linear models do not output calibrated probabilities."
        }

prediction_service = PredictionService()
