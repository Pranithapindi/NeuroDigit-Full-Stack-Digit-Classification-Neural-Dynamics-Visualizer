import numpy as np
import json
import time
from pathlib import Path
from backend.dataset.mnist_loader import mnist_dataset
from backend.algorithms.hebb import HebbianClassifier
from backend.algorithms.perceptron import PerceptronClassifier
from backend.algorithms.adaline import AdalineClassifier
from backend.utils.metrics import compute_classification_metrics
from backend.config import MODELS_DIR

def train_and_save_real_baselines():
    print("Loading genuine MNIST dataset...")
    mnist_dataset.load()
    
    # Use 15,000 training samples for robust real-world generalization
    N_TRAIN = 15000
    N_TEST = 2000
    
    X_train = mnist_dataset.X_train[:N_TRAIN]
    y_train = mnist_dataset.y_train[:N_TRAIN]
    X_test = mnist_dataset.X_test[:N_TEST]
    y_test = mnist_dataset.y_test[:N_TEST]
    
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Perceptron Classifier
    print("\n--- Training Perceptron (OvR) on genuine MNIST ---")
    p_clf = PerceptronClassifier(learning_rate=0.01, random_state=42)
    t0 = time.time()
    p_clf.fit(X_train, y_train, epochs=20)
    p_time = time.time() - t0
    p_preds = p_clf.predict(X_test)
    p_metrics = compute_classification_metrics(y_test, p_preds)
    print(f"Perceptron done in {p_time:.2f}s! Test Acc: {p_metrics['accuracy']*100:.2f}%")
    p_clf.save(
        MODELS_DIR / "perceptron_model.json",
        metadata={
            "model_id": "perceptron_model",
            "display_name": "Perceptron (OvR, Genuine MNIST)",
            "training_time_sec": round(p_time, 2),
            "samples_trained": N_TRAIN,
            "metrics": p_metrics,
            "history": p_clf.history
        }
    )
    with open(MODELS_DIR / "perceptron_model_eval.json", "w") as f:
        json.dump({
            "model_id": "perceptron_model",
            "algorithm": "perceptron",
            "dataset_size": N_TRAIN,
            "epochs": 20,
            "learning_rate": 0.01,
            "random_state": 42,
            "training_time": round(p_time, 2),
            "test_metrics": p_metrics
        }, f, indent=2)
    
    # 2. ADALINE Classifier (Delta LMS)
    print("\n--- Training ADALINE (Delta Rule) on genuine MNIST ---")
    a_clf = AdalineClassifier(learning_rate=0.002, random_state=42)
    t0 = time.time()
    a_clf.fit(X_train, y_train, epochs=25, batch_size=64)
    a_time = time.time() - t0
    a_preds = a_clf.predict(X_test)
    a_metrics = compute_classification_metrics(y_test, a_preds)
    print(f"ADALINE done in {a_time:.2f}s! Test Acc: {a_metrics['accuracy']*100:.2f}%")
    a_clf.save(
        MODELS_DIR / "adaline_model.json",
        metadata={
            "model_id": "adaline_model",
            "display_name": "ADALINE (Delta Rule, Genuine MNIST)",
            "training_time_sec": round(a_time, 2),
            "samples_trained": N_TRAIN,
            "metrics": a_metrics,
            "history": a_clf.history
        }
    )
    with open(MODELS_DIR / "adaline_model_eval.json", "w") as f:
        json.dump({
            "model_id": "adaline_model",
            "algorithm": "adaline",
            "dataset_size": N_TRAIN,
            "epochs": 25,
            "learning_rate": 0.001,
            "random_state": 42,
            "training_time": round(a_time, 2),
            "test_metrics": a_metrics
        }, f, indent=2)
    
    # 3. Hebbian Classifier
    print("\n--- Training Hebbian Classifier on genuine MNIST ---")
    h_clf = HebbianClassifier(learning_rate=0.0001, random_state=42)
    t0 = time.time()
    h_clf.fit(X_train, y_train)
    h_time = time.time() - t0
    h_preds = h_clf.predict(X_test)
    h_metrics = compute_classification_metrics(y_test, h_preds)
    print(f"Hebbian done in {h_time:.2f}s! Test Acc: {h_metrics['accuracy']*100:.2f}%")
    h_clf.save(
        MODELS_DIR / "hebb_model.json",
        metadata={
            "model_id": "hebb_model",
            "display_name": "Hebbian Classifier (Genuine MNIST)",
            "training_time_sec": round(h_time, 2),
            "samples_trained": N_TRAIN,
            "metrics": h_metrics,
            "history": h_clf.history
        }
    )
    with open(MODELS_DIR / "hebb_model_eval.json", "w") as f:
        json.dump({
            "model_id": "hebb_model",
            "algorithm": "hebb",
            "dataset_size": N_TRAIN,
            "epochs": 1,
            "learning_rate": 0.0001,
            "random_state": 42,
            "training_time": round(h_time, 2),
            "test_metrics": h_metrics
        }, f, indent=2)
    
    print("\nAll 3 genuine MNIST models saved successfully!")

if __name__ == "__main__":
    train_and_save_real_baselines()
