import sys
from pathlib import Path

# Add backend directory and root directory to python path
backend_dir = Path(__file__).resolve().parent
root_dir = backend_dir.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(backend_dir))

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from backend.config import PORT, HOST
from backend.dataset.mnist_loader import mnist_dataset
from backend.services.training_service import training_service
from backend.services.prediction_service import prediction_service
from backend.services.evaluation_service import evaluation_service

app = Flask(__name__)
# Enable CORS for all routes so Vite frontend at localhost:5173 can query seamlessly
CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.route("/", methods=["GET"])
def index():
    """Root route with API status and frontend navigation link."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Digit Recognition API - Backend Online</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #090d16; color: #f1f5f9; padding: 40px 20px; text-align: center; margin: 0; }
            .card { max-width: 620px; margin: 0 auto; background: #131b2e; border: 1px solid #1e293b; border-radius: 16px; padding: 32px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }
            h1 { color: #38bdf8; margin-top: 0; font-size: 26px; }
            .badge { display: inline-block; background: #064e3b; color: #34d399; padding: 6px 14px; border-radius: 999px; font-weight: 600; font-size: 14px; margin-bottom: 20px; border: 1px solid #059669; }
            p { color: #94a3b8; line-height: 1.6; font-size: 15px; }
            .btn { display: inline-block; margin-top: 20px; background: linear-gradient(135deg, #0ea5e9, #6366f1); color: #fff; padding: 12px 28px; border-radius: 10px; text-decoration: none; font-weight: 600; font-size: 16px; transition: opacity 0.2s; }
            .btn:hover { opacity: 0.9; }
            .endpoints { text-align: left; background: #0b1120; border-radius: 8px; padding: 14px; margin-top: 24px; font-family: monospace; font-size: 13px; color: #cbd5e1; }
            .endpoints div { margin: 6px 0; }
            .method { color: #38bdf8; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="card">
            <div class="badge">&#9679; Backend API Server Running</div>
            <h1>Handwritten Digit Recognition API</h1>
            <p>You have accessed the Flask REST API directly on <strong>port 5000</strong>. The interactive visual user interface is hosted on the frontend application.</p>
            <a class="btn" href="http://localhost:5173" target="_blank">Open Frontend Application (localhost:5173) &rarr;</a>
            <div class="endpoints">
                <div style="color: #64748b; margin-bottom: 8px; font-weight: bold;">AVAILABLE REST ENDPOINTS:</div>
                <div><span class="method">GET</span>  <a href="/api/health" style="color:#38bdf8;">/api/health</a> - Health check</div>
                <div><span class="method">GET</span>  <a href="/api/models" style="color:#38bdf8;">/api/models</a> - List trained models</div>
                <div><span class="method">GET</span>  <a href="/api/compare" style="color:#38bdf8;">/api/compare</a> - Model comparative analytics</div>
                <div><span class="method">POST</span> /api/predict - Digit classification</div>
                <div><span class="method">POST</span> /api/train - Train custom models</div>
            </div>
        </div>
    </body>
    </html>
    """, 200, {"Content-Type": "text/html"}

@app.route("/api/health", methods=["GET"])
def health_check():
    """Check backend availability and dataset status."""
    return jsonify({
        "status": "healthy",
        "service": "Handwritten Digit Recognition API",
        "python_version": sys.version,
        "dataset_ready": mnist_dataset.is_loaded
    }), 200

@app.route("/api/dataset/info", methods=["GET"])
def get_dataset_info():
    """Return dataset size, image dimensions, class distribution, and pixel statistics."""
    try:
        info = mnist_dataset.get_info()
        return jsonify({"success": True, "data": info}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/dataset/samples", methods=["GET"])
def get_dataset_samples():
    """Return sample preview images and intensity stats for each digit."""
    try:
        samples_per_class = int(request.args.get("per_class", 2))
        samples = mnist_dataset.get_sample_images(samples_per_class=samples_per_class)
        return jsonify({"success": True, "samples": samples}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/train", methods=["POST"])
def start_train():
    """
    Train the selected learning rule (Hebb, Perceptron, or ADALINE) with validated hyperparameters.
    Payload:
        algorithm: 'hebb' | 'perceptron' | 'adaline'
        dataset_size: int (e.g. 500 to 5000)
        epochs: int (e.g. 1 to 50)
        learning_rate: float
        random_seed: int
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        algorithm = data.get("algorithm", "perceptron").lower()
        dataset_size = int(data.get("dataset_size", 1200))
        epochs = int(data.get("epochs", 15))
        learning_rate = float(data.get("learning_rate", 0.01))
        random_seed = int(data.get("random_seed", 42))

        # Hyperparameter bounds validation
        if algorithm not in ["hebb", "perceptron", "adaline"]:
            return jsonify({"success": False, "error": f"Invalid algorithm '{algorithm}'. Must be 'hebb', 'perceptron', or 'adaline'."}), 400

        dataset_size = max(100, min(10000, dataset_size))
        epochs = max(1, min(100, epochs))
        learning_rate = max(0.00001, min(1.0, learning_rate))

        job_id = training_service.start_training(
            algorithm=algorithm,
            dataset_size=dataset_size,
            epochs=epochs,
            learning_rate=learning_rate,
            random_state=random_seed
        )

        return jsonify({
            "success": True,
            "job_id": job_id,
            "message": f"Training job started for {algorithm.upper()} with {dataset_size} samples for {epochs} epochs."
        }), 202

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route("/api/train/status/<job_id>", methods=["GET"])
def get_train_status(job_id):
    """Return training progress, current epoch, loss/accuracy history, and logs."""
    status = training_service.get_status(job_id)
    if not status:
        return jsonify({"success": False, "error": f"Job ID '{job_id}' not found."}), 404
    return jsonify({"success": True, "job": status}), 200

@app.route("/api/models", methods=["GET"])
def list_models():
    """List all available trained models and their test accuracies."""
    try:
        models = evaluation_service.list_models()
        return jsonify({"success": True, "models": models}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/predict", methods=["POST"])
def predict_digit():
    """
    Accepts drawn canvas (base64) or uploaded image file.
    Returns predicted digit, class decision scores, normalized scores, and top 3 candidates.
    """
    try:
        model_id = request.form.get("model_id") or request.args.get("model_id")
        
        # Check if json body with base64 image
        if request.is_json:
            json_data = request.get_json()
            model_id = json_data.get("model_id", model_id)
            base64_image = json_data.get("image")
            if not model_id:
                model_id = "perceptron_model"
            result = prediction_service.predict_from_base64(model_id, base64_image)
            return jsonify({"success": True, "result": result}), 200

        # Check if file upload (multipart/form-data)
        if "file" in request.files:
            file = request.files["file"]
            if not model_id:
                model_id = "perceptron_model"
            file_bytes = file.read()
            result = prediction_service.predict_from_bytes(model_id, file_bytes)
            return jsonify({"success": True, "result": result}), 200

        return jsonify({"success": False, "error": "No image provided. Send a base64 'image' field in JSON or upload a 'file' in multipart form."}), 400

    except ValueError as ve:
        return jsonify({"success": False, "error": str(ve)}), 400
    except FileNotFoundError as fnf:
        return jsonify({"success": False, "error": str(fnf)}), 404
    except Exception as e:
        return jsonify({"success": False, "error": f"Prediction failed: {str(e)}"}), 500

@app.route("/api/evaluate/<model_id>", methods=["GET"])
def evaluate_model(model_id):
    """Return test metrics, confusion matrix, and per-class results for a specific model."""
    try:
        eval_data = evaluation_service.get_model_evaluation(model_id)
        return jsonify({"success": True, "data": eval_data}), 200
    except FileNotFoundError as fnf:
        return jsonify({"success": False, "error": str(fnf)}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/compare", methods=["GET"])
def compare_models():
    """Return comparative metrics and side-by-side analysis for all evaluated models."""
    try:
        comparison = evaluation_service.get_comparison()
        return jsonify({"success": True, "data": comparison}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/export/results", methods=["GET"])
def export_results_csv():
    """Export comparison results as a downloadable CSV."""
    try:
        csv_content = evaluation_service.generate_csv()
        return Response(
            csv_content,
            mimetype="text/csv",
            headers={"Content-disposition": "attachment; filename=model_comparison_results.csv"}
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    print(f"Starting Handwritten Digit Recognition API on http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=False)
