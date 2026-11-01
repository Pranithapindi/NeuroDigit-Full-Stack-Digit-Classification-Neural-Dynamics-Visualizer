import threading
import time
import uuid
import numpy as np
from pathlib import Path
from backend.config import MODELS_DIR, DEFAULT_EPOCHS, DEFAULT_LEARNING_RATES, DEFAULT_TRAIN_SIZE, DEFAULT_TEST_SIZE
from backend.dataset.mnist_loader import mnist_dataset
from backend.algorithms.hebb import HebbianClassifier
from backend.algorithms.perceptron import PerceptronClassifier
from backend.algorithms.adaline import AdalineClassifier
from backend.utils.metrics import compute_classification_metrics

class TrainingService:
    def __init__(self):
        self.jobs = {}
        self.lock = threading.Lock()

    def start_training(self, algorithm, dataset_size=1200, epochs=None, learning_rate=None, random_state=42):
        """
        Starts a training job asynchronously in a background thread.
        Returns job_id immediately.
        """
        algorithm = algorithm.lower().strip()
        if algorithm not in ["hebb", "perceptron", "adaline"]:
            raise ValueError(f"Unknown algorithm: {algorithm}. Must be 'hebb', 'perceptron', or 'adaline'.")

        if epochs is None:
            epochs = DEFAULT_EPOCHS.get(algorithm, 15)
        if learning_rate is None:
            learning_rate = DEFAULT_LEARNING_RATES.get(algorithm, 0.01)

        job_id = f"job_{algorithm}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        model_id = f"{algorithm}_model"

        job_record = {
            "job_id": job_id,
            "model_id": model_id,
            "algorithm": algorithm,
            "status": "queued",
            "progress": 0.0,
            "current_epoch": 0,
            "total_epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "dataset_size": int(dataset_size),
            "random_state": int(random_state),
            "latest_metrics": {},
            "history": [],
            "logs": [f"Job {job_id} queued for {algorithm.upper()} algorithm."],
            "test_metrics": None,
            "start_time": None,
            "end_time": None,
            "duration": None,
            "error": None
        }

        with self.lock:
            self.jobs[job_id] = job_record

        # Launch background thread
        thread = threading.Thread(
            target=self._run_job,
            args=(job_id, algorithm, dataset_size, epochs, learning_rate, random_state, model_id),
            daemon=True
        )
        thread.start()

        return job_id

    def _run_job(self, job_id, algorithm, dataset_size, epochs, learning_rate, random_state, model_id):
        try:
            with self.lock:
                self.jobs[job_id]["status"] = "training"
                self.jobs[job_id]["start_time"] = time.time()
                self.jobs[job_id]["logs"].append(f"Loading training subset of size {dataset_size} (Seed: {random_state})...")

            # Load dataset subsets
            X_train, y_train = mnist_dataset.get_train_subset(n_samples=dataset_size, random_state=random_state)
            X_test, y_test = mnist_dataset.get_test_subset(n_samples=DEFAULT_TEST_SIZE, random_state=random_state)

            with self.lock:
                self.jobs[job_id]["logs"].append(f"Loaded {len(X_train)} training samples and {len(X_test)} test samples.")
                self.jobs[job_id]["logs"].append(f"Initializing {algorithm.upper()} model...")

            # Select and instantiate model
            if algorithm == "hebb":
                model = HebbianClassifier(input_dim=784, num_classes=10, learning_rate=learning_rate, random_state=random_state)
            elif algorithm == "perceptron":
                model = PerceptronClassifier(input_dim=784, num_classes=10, learning_rate=learning_rate, random_state=random_state)
            elif algorithm == "adaline":
                model = AdalineClassifier(input_dim=784, num_classes=10, learning_rate=learning_rate, random_state=random_state)

            def progress_callback(cur_epoch, tot_epochs, metrics_dict):
                with self.lock:
                    pct = round((cur_epoch / tot_epochs) * 100.0, 1)
                    self.jobs[job_id]["current_epoch"] = cur_epoch
                    self.jobs[job_id]["progress"] = pct
                    self.jobs[job_id]["latest_metrics"] = metrics_dict
                    self.jobs[job_id]["history"].append(metrics_dict)
                    msg = f"Epoch {cur_epoch}/{tot_epochs} - Acc: {metrics_dict.get('accuracy', 0):.4f}, Loss/Error: {metrics_dict.get('loss', 0):.4f}"
                    self.jobs[job_id]["logs"].append(msg)

            # Train model
            with self.lock:
                self.jobs[job_id]["logs"].append(f"Beginning training across {epochs} epochs...")
            model.fit(X_train, y_train, epochs=epochs, callback=progress_callback)

            # Evaluate on independent test set
            with self.lock:
                self.jobs[job_id]["logs"].append("Training complete. Evaluating on independent test set...")

            y_pred_test = model.predict(X_test)
            test_metrics = compute_classification_metrics(y_test, y_pred_test, num_classes=10)

            # Save model to disk
            model_path = MODELS_DIR / f"{model_id}.json"
            model.save(model_path)

            # Save metadata and evaluation results
            metadata_path = MODELS_DIR / f"{model_id}_eval.json"
            import json
            eval_data = {
                "model_id": model_id,
                "algorithm": algorithm,
                "dataset_size": dataset_size,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "random_state": random_state,
                "training_time": model.training_time,
                "test_metrics": test_metrics,
                "history": model.history,
                "saved_at": time.time()
            }
            with open(metadata_path, "w") as f:
                json.dump(eval_data, f, indent=2)

            with self.lock:
                end_time = time.time()
                self.jobs[job_id]["status"] = "completed"
                self.jobs[job_id]["progress"] = 100.0
                self.jobs[job_id]["end_time"] = end_time
                self.jobs[job_id]["duration"] = round(end_time - self.jobs[job_id]["start_time"], 2)
                self.jobs[job_id]["test_metrics"] = test_metrics
                self.jobs[job_id]["logs"].append(f"Model saved successfully as '{model_id}'. Test Accuracy: {test_metrics['accuracy'] * 100:.2f}%.")

        except Exception as e:
            import traceback
            err_trace = traceback.format_exc()
            print(f"[Training Error] {err_trace}")
            with self.lock:
                self.jobs[job_id]["status"] = "failed"
                self.jobs[job_id]["error"] = str(e)
                self.jobs[job_id]["logs"].append(f"Error during training: {str(e)}")

    def get_status(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                return None
            return dict(self.jobs[job_id])

training_service = TrainingService()
