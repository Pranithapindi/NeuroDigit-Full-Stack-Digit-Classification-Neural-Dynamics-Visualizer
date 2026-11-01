import numpy as np
import json
from pathlib import Path

class HebbianClassifier:
    """
    Multiclass Supervised Hebbian Learning Classifier using One-Versus-Rest (OvR).
    
    Learning Rule:
        w_k(new) = w_k(old) + eta * x * t_k
        b_k(new) = b_k(old) + eta * t_k
        
    Targets:
        Bipolar target encoding: +1 for the target digit class, -1 for all other 9 classes.
        
    Mathematical Principle:
        Hebbian learning strengthens synaptic weights in proportion to the correlation
        between the input pattern and the desired target ("cells that fire together, wire together").
        It does not compute an error feedback signal or perform gradient descent.
    """
    
    def __init__(self, input_dim=784, num_classes=10, learning_rate=0.01, random_state=42):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.weights = None
        self.bias = None
        self.history = {
            "epoch": [],
            "error_rate": [],
            "accuracy": [],
            "loss": []  # For Hebb, classification error is logged as surrogate
        }
        self.is_fitted = False
        self.training_time = 0.0

    def _init_weights(self):
        # Traditional Hebbian learning initializes weights to zero
        if self.weights is None or self.bias is None:
            self.weights = np.zeros((self.input_dim, self.num_classes), dtype=np.float64)
            self.bias = np.zeros(self.num_classes, dtype=np.float64)

    def _to_bipolar(self, y):
        """Converts integer labels (0-9) to bipolar {-1, +1} target matrix of shape (N, num_classes)."""
        N = len(y)
        T = np.full((N, self.num_classes), -1.0, dtype=np.float64)
        for i, label in enumerate(y):
            T[i, int(label)] = 1.0
        return T

    def fit(self, X, y, epochs=1, callback=None):
        """
        Train the Hebbian model using the standard Hebbian learning rule:
            w_k(new) = w_k(old) + eta * x * t_k
            b_k(new) = b_k(old) + eta * t_k
        
        Args:
            X: np.ndarray of shape (N, input_dim), normalized to [0, 1]
            y: np.ndarray of shape (N,), integer class labels 0..9
            epochs: int, number of training passes
            callback: optional callable for progress reporting callback(epoch, total_epochs, metrics)
        """
        import time
        start_time = time.time()
        
        self._init_weights()
        T = self._to_bipolar(y)
        N = len(X)
        
        self.history = {
            "epoch": [],
            "error_rate": [],
            "accuracy": [],
            "loss": []
        }

        for epoch in range(1, epochs + 1):
            self.weights += self.learning_rate * (X.T @ T)
            self.bias += self.learning_rate * np.sum(T, axis=0)

            # Evaluate training metrics
            preds = self.predict(X)
            correct = np.sum(preds == y)
            acc = float(correct / N)
            err_rate = float(1.0 - acc)
            
            self.history["epoch"].append(epoch)
            self.history["error_rate"].append(round(err_rate, 4))
            self.history["accuracy"].append(round(acc, 4))
            self.history["loss"].append(round(err_rate, 4))
            
            if callback:
                callback(epoch, epochs, {
                    "epoch": epoch,
                    "accuracy": round(acc, 4),
                    "error_rate": round(err_rate, 4),
                    "loss": round(err_rate, 4)
                })

        self.training_time = round(time.time() - start_time, 3)
        self.is_fitted = True
        return self

    def decision_function(self, X):
        """Returns balanced associative correlation scores of shape (N, num_classes)."""
        if self.weights is None or self.bias is None:
            raise RuntimeError("Model is not initialized.")
        return X @ self.weights + self.bias

    def predict(self, X):
        """Predicts class labels 0..9 via argmax of decision function."""
        scores = self.decision_function(X)
        return np.argmax(scores, axis=1)

    def predict_scores(self, x_single):
        """
        Calculates scores for a single sample x_single (784,).
        Returns:
            dict with 'predicted_digit', 'scores' (raw), 'normalized_scores' (softmax/min-max),
            'top_3' candidate list.
        """
        x_2d = x_single.reshape(1, -1)
        raw_scores = self.decision_function(x_2d)[0]
        pred_digit = int(np.argmax(raw_scores))
        
        # Softmax normalization for visual comparison (clearly documented as normalized score)
        exp_scores = np.exp(raw_scores - np.max(raw_scores))
        norm_scores = exp_scores / np.sum(exp_scores)
        
        top_indices = np.argsort(raw_scores)[::-1][:3]
        top_3 = [
            {"digit": int(idx), "raw_score": float(raw_scores[idx]), "normalized_score": round(float(norm_scores[idx]), 4)}
            for idx in top_indices
        ]
        
        return {
            "predicted_digit": pred_digit,
            "raw_scores": [round(float(s), 4) for s in raw_scores],
            "normalized_scores": [round(float(s), 4) for s in norm_scores],
            "top_3": top_3
        }

    def save(self, filepath, metadata=None):
        """Saves model weights, bias, config, and history to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model_type": "hebb",
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "learning_rate": self.learning_rate,
            "random_state": self.random_state,
            "weights": self.weights.tolist() if self.weights is not None else None,
            "bias": self.bias.tolist() if self.bias is not None else None,
            "history": self.history,
            "training_time": self.training_time,
            "is_fitted": self.is_fitted
        }
        if metadata:
            data.update(metadata)
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, filepath):
        """Loads model weights, bias, and config from disk."""
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
        self.is_fitted = data.get("is_fitted", True)
        return self
