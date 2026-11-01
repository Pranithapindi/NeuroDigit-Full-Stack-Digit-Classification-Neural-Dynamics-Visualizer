import pytest
import numpy as np
from backend.algorithms.hebb import HebbianClassifier
from backend.algorithms.perceptron import PerceptronClassifier
from backend.algorithms.adaline import AdalineClassifier
from backend.dataset.preprocessing import preprocess_image_array
from backend.utils.metrics import compute_classification_metrics

def test_hebbian_weight_update():
    """Verify Hebbian rule: w_new = w_old + eta * x * t"""
    model = HebbianClassifier(input_dim=4, num_classes=2, learning_rate=0.5)
    X = np.array([[1.0, 0.0, 1.0, 0.0]], dtype=np.float64)
    y = np.array([0], dtype=np.int64)
    model.fit(X, y, epochs=1)
    
    # Class 0 target is +1, Class 1 target is -1
    # For Class 0: w[:, 0] should be 0.5 * [1, 0, 1, 0] * (+1) = [0.5, 0, 0.5, 0]
    # For Class 1: w[:, 1] should be 0.5 * [1, 0, 1, 0] * (-1) = [-0.5, 0, -0.5, 0]
    np.testing.assert_allclose(model.weights[:, 0], [0.5, 0.0, 0.5, 0.0])
    np.testing.assert_allclose(model.weights[:, 1], [-0.5, 0.0, -0.5, 0.0])
    np.testing.assert_allclose(model.bias, [0.5, -0.5])

def test_perceptron_weight_update_on_error():
    """Verify Perceptron updates only when prediction != target"""
    model = PerceptronClassifier(input_dim=2, num_classes=2, learning_rate=0.1, random_state=42)
    # Set weights to zeros
    model.weights = np.zeros((2, 2))
    model.bias = np.zeros(2)
    
    # Sample belonging to class 0
    x = np.array([1.0, 1.0])
    y = np.array([0])
    
    # At start, z = 0 >= 0 -> step(z) = +1 for both units.
    # Unit 0: target = +1, y_hat = +1 => error = 0 (no change!)
    # Unit 1: target = -1, y_hat = +1 => error = -2 (penalized!)
    model.fit(np.array([x]), y, epochs=1)
    
    # Unit 0 should have no error update (weight stays ~0)
    # Unit 1 should decrease weight
    assert model.weights[0, 1] < 0.0
    assert model.bias[1] < 0.0

def test_adaline_delta_rule_minimizes_mse():
    """Verify ADALINE reduces MSE across epochs on linearly separable data"""
    rng = np.random.RandomState(42)
    X = rng.normal(size=(50, 10))
    # Linearly separable target based on feature 0
    y = np.where(X[:, 0] > 0, 1, 0)
    
    model = AdalineClassifier(input_dim=10, num_classes=2, learning_rate=0.01, random_state=42)
    model.fit(X, y, epochs=15)
    
    mses = model.history["mse"]
    # MSE should decrease from epoch 1 to epoch 15
    assert mses[-1] < mses[0]
    assert model.is_fitted is True

def test_preprocessing_dimensions():
    """Verify preprocessor outputs exact 28x28 array and 784 vector"""
    # Create a 100x100 synthetic drawing with a digit stroke in the center
    canvas = np.zeros((100, 100), dtype=np.float32)
    canvas[30:70, 48:52] = 255.0  # Vertical line resembling digit '1'
    
    norm_28x28, flat_784 = preprocess_image_array(canvas)
    assert norm_28x28.shape == (28, 28)
    assert flat_784.shape == (784,)
    assert 0.0 <= np.min(norm_28x28) <= np.max(norm_28x28) <= 1.0

def test_classification_metrics_calculation():
    """Verify metrics calculation logic with known values"""
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 1, 0, 1, 2]  # 5 correct out of 6
    
    metrics = compute_classification_metrics(y_true, y_pred, num_classes=3)
    assert metrics["accuracy"] == round(5 / 6, 4)
    assert metrics["total_evaluated"] == 6
    assert metrics["total_correct"] == 5
    assert len(metrics["per_class"]) == 3
    assert len(metrics["confusion_matrix"]) == 3
