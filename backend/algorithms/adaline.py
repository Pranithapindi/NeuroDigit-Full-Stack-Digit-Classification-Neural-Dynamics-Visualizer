import numpy as np
import json
from pathlib import Path

class AdalineClassifier:
    """
    Multiclass ADALINE (Adaptive Linear Neuron) using the Delta (Widrow-Hoff LMS) Rule.
    
    Architecture:
        10 continuous linear output units (one for each digit 0..9).
        Linear activation during learning:
            y_linear_k = x . w_k + b_k
            
    Error Objective:
        Minimizes Mean Squared Error (MSE):
            J(W, b) = (1 / (2 * N)) * sum_{i=1}^N sum_{k=0}^9 (t_{i,k} - y_linear_{i,k})^2
            
    Delta Update Rule:
        For sample (x, t) where t in {-1, +1}^10:
            error_k = (t_k - y_linear_k)
            w_k(new) = w_k(old) + eta * error_k * x
            b_k(new) = b_k(old) + eta * error_k
            
    Key Contrast with Perceptron:
        - Perceptron: Error is computed AFTER applying the step/threshold function.
        - ADALINE: Error is computed directly on the CONTINUOUS linear activation
                   BEFORE any thresholding.
                   
    Multiclass Decision at Inference:
        Prediction = argmax_{k in 0..9} (x . w_k + b_k)
    """

    def __init__(self, input_dim=784, num_classes=10, learning_rate=0.0005, random_state=42):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.weights = None  # Shape: (input_dim, num_classes)
        self.bias = None     # Shape: (num_classes,)
        self.history = {
            "epoch": [],
            "mse": [],
            "error_rate": [],
            "accuracy": [],
            "loss": []       # MSE is the primary loss
        }
        self.is_fitted = False
        self.converged = False
        self.training_time = 0.0

    def _init_weights(self):
        if self.weights is None or self.bias is None:
            rng = np.random.RandomState(self.random_state)
            scale = 1.0 / np.sqrt(self.input_dim)
            self.weights = rng.normal(loc=0.0, scale=scale, size=(self.input_dim, self.num_classes))
            self.bias = np.zeros(self.num_classes, dtype=np.float64)

    def fit(self, X, y, epochs=25, callback=None, batch_size=32):
        """
        Train ADALINE using stochastic/mini-batch Delta rule to minimize MSE.
        """
        import time
        start_time = time.time()
        
        self._init_weights()
        N = len(X)
        
        # Zero-mean balanced target encoding (+1.0 for true digit, -1/9 for other 9)
        T = np.full((N, self.num_classes), -1.0 / (self.num_classes - 1.0), dtype=np.float64)
        for i, lbl in enumerate(y):
            T[i, int(lbl)] = 1.0

        self.history = {
            "epoch": [],
            "mse": [],
            "error_rate": [],
            "accuracy": [],
            "loss": []
        }
        self.converged = False

        rng = np.random.RandomState(self.random_state)
        prev_mse = float("inf")
        l2_reg = 0.0005

        for epoch in range(1, epochs + 1):
            indices = np.arange(N)
            rng.shuffle(indices)
            
            # Mini-batch gradient descent for numerical stability and fast execution
            for start_idx in range(0, N, batch_size):
                end_idx = min(start_idx + batch_size, N)
                batch_indices = indices[start_idx:end_idx]
                
                xb = X[batch_indices]  # (B, 784)
                tb = T[batch_indices]  # (B, 10)
                
                # Continuous linear outputs (no thresholding during training)
                yb_linear = xb @ self.weights + self.bias  # (B, 10)
                
                # Continuous Delta error: e = target - linear_output
                errors = tb - yb_linear  # (B, 10)
                
                # Gradient update with L2 weight decay to prevent class weight explosion
                B = len(batch_indices)
                dW = (xb.T @ errors) / B - l2_reg * self.weights
                db = np.mean(errors, axis=0)
                
                self.weights += self.learning_rate * dW
                self.bias += self.learning_rate * db

            # Full evaluation at end of epoch
            all_linear = X @ self.weights + self.bias
            full_errors = T - all_linear
            # MSE = (1 / (2 * N)) * sum(errors^2)
            epoch_mse = float(np.mean(np.square(full_errors)) * 0.5)
            
            preds = np.argmax(all_linear, axis=1)
            correct = np.sum(preds == y)
            acc = float(correct / N)
            err_rate = float(1.0 - acc)

            self.history["epoch"].append(epoch)
            self.history["mse"].append(round(epoch_mse, 5))
            self.history["error_rate"].append(round(err_rate, 4))
            self.history["accuracy"].append(round(acc, 4))
            self.history["loss"].append(round(epoch_mse, 5))

            if callback:
                callback(epoch, epochs, {
                    "epoch": epoch,
                    "mse": round(epoch_mse, 5),
                    "loss": round(epoch_mse, 5),
                    "accuracy": round(acc, 4),
                    "error_rate": round(err_rate, 4),
                    "converged": bool(abs(prev_mse - epoch_mse) < 1e-5)
                })

            if abs(prev_mse - epoch_mse) < 1e-5:
                self.converged = True
            prev_mse = epoch_mse

        self.training_time = round(time.time() - start_time, 3)
        self.is_fitted = True
        return self

    def decision_function(self, X):
        """Returns continuous linear outputs: shape (N, 10), column-normalized for balanced class scale."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        col_norms = np.linalg.norm(self.weights, axis=0, keepdims=True)
        col_norms = np.maximum(col_norms, 1e-8)
        norm_w = self.weights / col_norms
        norm_b = self.bias / col_norms.squeeze()
        return X @ norm_w + norm_b

    def predict(self, X):
        """Predicts class labels 0..9 via argmax of linear outputs."""
        scores = self.decision_function(X)
        return np.argmax(scores, axis=1)

    def predict_scores(self, x_single):
        """
        Calculates scores for a single input vector (784,).
        Returns predicted digit, raw linear scores, normalized scores, and top 3 candidates.
        """
        x_2d = x_single.reshape(1, -1)
        raw_scores = self.decision_function(x_2d)[0]
        pred_digit = int(np.argmax(raw_scores))
        
        # Softmax normalization for visual comparison (normalized scores, not calibrated probabilities)
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
            "model_type": "adaline",
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
