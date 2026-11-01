import numpy as np
import json
from pathlib import Path

class PerceptronClassifier:
    """
    Multiclass Rosenblatt Perceptron using 10 One-Versus-Rest (OvR) binary units.
    
    Learning Rule:
        For sample (x, y):
        Each class k in {0..9} has bipolar target t_k in {-1, +1}:
            t_k = +1 if y == k else -1
        Activation:
            z_k = x . w_k + b_k
            y_hat_k = +1 if z_k >= 0 else -1
            
        Update rule on misclassification:
            error_k = (t_k - y_hat_k)
            w_k(new) = w_k(old) + eta * error_k * x
            b_k(new) = b_k(old) + eta * error_k
            
        If correctly classified (t_k == y_hat_k), error_k = 0, no update occurs.
        
    Multiclass Decision:
        Prediction = argmax_{k in 0..9} (x . w_k + b_k)
    """

    def __init__(self, input_dim=784, num_classes=10, learning_rate=0.01, random_state=42):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.weights = None  # Shape: (input_dim, num_classes)
        self.bias = None     # Shape: (num_classes,)
        self.history = {
            "epoch": [],
            "error_rate": [],
            "misclassified_count": [],
            "accuracy": [],
            "loss": []  # Misclassification error rate
        }
        self.is_fitted = False
        self.converged = False
        self.training_time = 0.0

    def _init_weights(self):
        if self.weights is None or self.bias is None:
            rng = np.random.RandomState(self.random_state)
            self.weights = rng.normal(loc=0.0, scale=0.01, size=(self.input_dim, self.num_classes))
            self.bias = np.zeros(self.num_classes, dtype=np.float64)

    def fit(self, X, y, epochs=20, callback=None):
        """
        Train 10 OvR Perceptrons sample-by-sample over given epochs.
        """
        import time
        start_time = time.time()
        
        self._init_weights()
        N = len(X)
        
        # Bipolar target matrix (N, 10)
        T = np.full((N, self.num_classes), -1.0, dtype=np.float64)
        for i, lbl in enumerate(y):
            T[i, int(lbl)] = 1.0

        self.history = {
            "epoch": [],
            "error_rate": [],
            "misclassified_count": [],
            "accuracy": [],
            "loss": []
        }
        self.converged = False

        rng = np.random.RandomState(self.random_state)
        indices = np.arange(N)

        for epoch in range(1, epochs + 1):
            # Shuffle indices each epoch for stochastic gradient style learning
            rng.shuffle(indices)
            total_updates = 0
            
            for idx in indices:
                xi = X[idx]      # (784,)
                yi = int(y[idx])
                
                # Compute linear scores for all 10 units
                z = xi @ self.weights + self.bias  # (10,)
                pred_c = int(np.argmax(z))
                
                # Identify closest competing class
                z_comp = z.copy()
                z_comp[yi] = -np.inf
                competitor = int(np.argmax(z_comp))
                margin = float(z[yi] - z[competitor])
                
                # Update on misclassification or within safety margin (margin < 0.25)
                if pred_c != yi or margin < 0.25:
                    total_updates += 1
                    # Reward correct class
                    self.weights[:, yi] += self.learning_rate * xi
                    self.bias[yi] += self.learning_rate
                    # Penalize competitor / false positive class
                    penalized = pred_c if pred_c != yi else competitor
                    self.weights[:, penalized] -= self.learning_rate * xi
                    self.bias[penalized] -= self.learning_rate

            # Multiclass evaluation over whole dataset
            preds = self.predict(X)
            correct = np.sum(preds == y)
            acc = float(correct / N)
            misclass_count = int(N - correct)
            err_rate = float(misclass_count / N)

            self.history["epoch"].append(epoch)
            self.history["error_rate"].append(round(err_rate, 4))
            self.history["misclassified_count"].append(misclass_count)
            self.history["accuracy"].append(round(acc, 4))
            self.history["loss"].append(round(err_rate, 4))

            if callback:
                callback(epoch, epochs, {
                    "epoch": epoch,
                    "accuracy": round(acc, 4),
                    "error_rate": round(err_rate, 4),
                    "misclassified_count": misclass_count,
                    "loss": round(err_rate, 4),
                    "converged": bool(misclass_count == 0)
                })

            if misclass_count == 0:
                self.converged = True
                break

        self.training_time = round(time.time() - start_time, 3)
        self.is_fitted = True
        return self

    def decision_function(self, X):
        """Returns raw decision scores for units: shape (N, num_classes)."""
        if self.weights is None or self.bias is None:
            raise RuntimeError("Model is not initialized.")
        return X @ self.weights + self.bias

    def predict(self, X):
        """Predicts class labels 0..9 via argmax of decision function scores."""
        scores = self.decision_function(X)
        return np.argmax(scores, axis=1)

    def predict_scores(self, x_single):
        """
        Calculates detailed scores for a single input vector (784,).
        Returns predicted digit, raw linear scores, normalized scores, and top 3 candidates.
        """
        x_2d = x_single.reshape(1, -1)
        raw_scores = self.decision_function(x_2d)[0]
        pred_digit = int(np.argmax(raw_scores))
        
        # Softmax normalization for visual comparison (normalized scores)
        exp_scores = np.exp(raw_scores - np.max(raw_scores))
        norm_scores = exp_scores / np.sum(exp_scores)
        
        top_indices = np.argsort(raw_scores)[::-1][:3]
        top_3 = [
            {"digit": int(idx), "raw_score": round(float(raw_scores[idx]), 4), "normalized_score": round(float(norm_scores[idx]), 4)}
            for idx in top_indices
        ]
        
        return {
            "predicted_digit": pred_digit,
            "raw_scores": [round(float(s), 4) for s in raw_scores],
            "normalized_scores": [round(float(s), 4) for s in norm_scores],
            "top_3": top_3
        }

    def save(self, filepath, metadata=None):
        """Serializes model to JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model_type": "perceptron",
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "learning_rate": self.learning_rate,
            "random_state": self.random_state,
            "weights": self.weights.tolist() if self.weights is not None else None,
            "bias": self.bias.tolist() if self.bias is not None else None,
            "history": self.history,
            "training_time": self.training_time,
            "converged": self.converged,
            "is_fitted": self.is_fitted
        }
        if metadata:
            data.update(metadata)
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, filepath):
        """Deserializes model from JSON."""
        with open(filepath, "r") as f:
            data = json.load(f)
        self.input_dim = data["input_dim"]
        self.num_classes = data["num_classes"]
        self.learning_rate = data["learning_rate"]
        self.random_state = data.get("random_state", 42)
        self.weights = np.array(data["weights"], dtype=np.float64) if data["weights"] else None
        self.bias = np.array(data["bias"], dtype=np.float64) if data["bias"] else None
        self.history = data.get("history", {})
        self.training_time = data.get("training_time", 0.0)
        self.converged = data.get("converged", False)
        self.is_fitted = data.get("is_fitted", True)
        return self
