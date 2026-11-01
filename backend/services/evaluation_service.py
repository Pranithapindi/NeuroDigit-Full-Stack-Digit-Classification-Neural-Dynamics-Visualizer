import json
import csv
import io
import time
from pathlib import Path
from backend.config import MODELS_DIR
from backend.dataset.mnist_loader import mnist_dataset
from backend.utils.metrics import compute_classification_metrics

class EvaluationService:
    def list_models(self):
        """Returns metadata of all trained models currently saved."""
        models = []
        for file in MODELS_DIR.glob("*_model.json"):
            model_id = file.stem
            eval_file = MODELS_DIR / f"{model_id}_eval.json"
            
            algo = "unknown"
            accuracy = None
            training_time = None
            dataset_size = None
            epochs = None
            
            if eval_file.exists():
                try:
                    with open(eval_file, "r") as f:
                        meta = json.load(f)
                    algo = meta.get("algorithm", model_id.split("_")[0])
                    accuracy = meta.get("test_metrics", {}).get("accuracy")
                    training_time = meta.get("training_time")
                    dataset_size = meta.get("dataset_size")
                    epochs = meta.get("epochs")
                except Exception:
                    pass
            else:
                try:
                    with open(file, "r") as f:
                        data = json.load(f)
                    algo = data.get("model_type", "unknown")
                    training_time = data.get("training_time")
                except Exception:
                    pass

            models.append({
                "model_id": model_id,
                "algorithm": algo,
                "display_name": algo.upper() if algo != "adaline" else "ADALINE (Delta)",
                "accuracy": accuracy,
                "training_time": training_time,
                "dataset_size": dataset_size,
                "epochs": epochs,
                "modified_time": file.stat().st_mtime
            })
            
        models.sort(key=lambda m: m["modified_time"], reverse=True)
        return models

    def get_model_evaluation(self, model_id):
        """Returns detailed evaluation results and confusion matrix for a specific model."""
        eval_file = MODELS_DIR / f"{model_id}_eval.json"
        if not eval_file.exists():
            # If eval file not found, try loading the model and re-evaluating on test set
            from backend.services.prediction_service import prediction_service
            try:
                model = prediction_service.get_model(model_id)
                X_test, y_test = mnist_dataset.get_test_subset(500)
                y_pred = model.predict(X_test)
                metrics = compute_classification_metrics(y_test, y_pred)
                return {
                    "model_id": model_id,
                    "algorithm": model.__class__.__name__,
                    "test_metrics": metrics,
                    "history": model.history,
                    "training_time": model.training_time
                }
            except Exception as e:
                raise FileNotFoundError(f"Model evaluation not found for '{model_id}': {e}")

        with open(eval_file, "r") as f:
            return json.load(f)

    def get_comparison(self):
        """
        Gathers real experimental comparison data for Hebb, Perceptron, and ADALINE.
        Returns side-by-side metric cards, comparison table, error curves, and per-class stats.
        """
        algorithms = ["hebb", "perceptron", "adaline"]
        algo_details = {
            "hebb": {
                "name": "Hebbian Learning",
                "model": "Single-Layer Linear Associator",
                "update_mechanism": "w_new = w_old + eta * x * t (Correlation reinforcement)",
                "error_objective": "None (Reinforces co-occurrence; no error feedback)",
                "convergence_behavior": "Weights grow monotonically; does not minimize misclassification",
                "advantages": "Biologically plausible, computationally simple (one-shot or additive), no backpropagation",
                "limitations": "Fails when input patterns are correlated or non-orthogonal; lower accuracy on digits"
            },
            "perceptron": {
                "name": "Perceptron Learning Rule",
                "model": "Rosenblatt Threshold Perceptron (10 OvR)",
                "update_mechanism": "w_new = w_old + eta * (t - y_hat) * x (Threshold error driven)",
                "error_objective": "Minimizes misclassification count (Perceptron Criterion)",
                "convergence_behavior": "Guaranteed to converge in finite steps IF linearly separable; oscillates otherwise",
                "advantages": "Simple error correction, updates only on misclassification, fast convergence on separable data",
                "limitations": "Cannot solve non-linearly separable problems; step function has zero gradient everywhere"
            },
            "adaline": {
                "name": "Delta Learning Rule (ADALINE)",
                "model": "Adaptive Linear Neuron / Widrow-Hoff LMS",
                "update_mechanism": "w_new = w_old + eta * (t - y_linear) * x (Continuous output error)",
                "error_objective": "Minimizes Mean Squared Error (MSE) via gradient descent",
                "convergence_behavior": "Asymptotically converges to Wiener-Hopf optimal least-squares solution",
                "advantages": "Smooth gradient descent on continuous linear outputs, finds best-fit hyperplane even if non-separable",
                "limitations": "Sensitive to learning rate choice (can diverge if eta too high); requires feature scaling"
            }
        }

        comparison_table = []
        models_data = {}
        all_curves = {}

        for algo in algorithms:
            model_id = f"{algo}_model"
            eval_file = MODELS_DIR / f"{model_id}_eval.json"
            meta = None
            
            if eval_file.exists():
                try:
                    with open(eval_file, "r") as f:
                        meta = json.load(f)
                except Exception:
                    pass

            info = algo_details[algo]
            
            if meta and "test_metrics" in meta:
                acc = meta["test_metrics"]["accuracy"]
                macro_f1 = meta["test_metrics"]["macro_f1"]
                macro_prec = meta["test_metrics"]["macro_precision"]
                macro_rec = meta["test_metrics"]["macro_recall"]
                training_time = meta.get("training_time", 0.0)
                history = meta.get("history", {})
                per_class = meta["test_metrics"].get("per_class", [])
                cm = meta["test_metrics"].get("confusion_matrix", [])
                cm_records = meta["test_metrics"].get("confusion_matrix_records", [])
            else:
                acc = None
                macro_f1 = None
                macro_prec = None
                macro_rec = None
                training_time = None
                history = {}
                per_class = []
                cm = []
                cm_records = []

            models_data[algo] = {
                "algorithm": algo,
                "display_name": info["name"],
                "model_name": info["model"],
                "accuracy": acc,
                "macro_f1": macro_f1,
                "macro_precision": macro_prec,
                "macro_recall": macro_rec,
                "training_time": training_time,
                "history": history,
                "per_class": per_class,
                "confusion_matrix": cm,
                "confusion_matrix_records": cm_records,
                "is_trained": bool(acc is not None)
            }

            comparison_table.append({
                "learning_rule": info["name"],
                "model": info["model"],
                "update_mechanism": info["update_mechanism"],
                "error_objective": info["error_objective"],
                "convergence_behavior": info["convergence_behavior"],
                "accuracy": f"{acc * 100:.2f}%" if acc is not None else "Not Trained",
                "training_time": f"{training_time:.2f}s" if training_time is not None else "N/A",
                "advantages": info["advantages"],
                "limitations": info["limitations"]
            })

        return {
            "models": models_data,
            "comparison_table": comparison_table,
            "generated_at": time.time()
        }

    def generate_csv(self):
        """Generates CSV string of comparative experimental results."""
        comp = self.get_comparison()
        output = io.StringIO()
        fieldnames = [
            "Learning Rule", "Model", "Update Mechanism", "Error Objective",
            "Convergence Behavior", "Accuracy", "Training Time", "Advantages", "Limitations"
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in comp["comparison_table"]:
            writer.writerow({
                "Learning Rule": row["learning_rule"],
                "Model": row["model"],
                "Update Mechanism": row["update_mechanism"],
                "Error Objective": row["error_objective"],
                "Convergence Behavior": row["convergence_behavior"],
                "Accuracy": row["accuracy"],
                "Training Time": row["training_time"],
                "Advantages": row["advantages"],
                "Limitations": row["limitations"]
            })
        return output.getvalue()

evaluation_service = EvaluationService()
