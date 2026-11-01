import numpy as np
from backend.dataset.mnist_loader import mnist_dataset
from backend.algorithms.perceptron import PerceptronClassifier
from backend.algorithms.adaline import AdalineClassifier
from backend.algorithms.hebb import HebbianClassifier
from backend.config import MODELS_DIR

mnist_dataset.load()

# Find true 5s in test set
idx_5 = np.where(mnist_dataset.y_test == 5)[0][:10]

p_model = PerceptronClassifier().load(MODELS_DIR / "perceptron_model.json")
a_model = AdalineClassifier().load(MODELS_DIR / "adaline_model.json")
h_model = HebbianClassifier().load(MODELS_DIR / "hebb_model.json")

print("Testing 10 genuine MNIST test set 5s:")
for i in idx_5:
    x = mnist_dataset.X_test[i]
    p_pred = p_model.predict(x.reshape(1, -1))[0]
    a_pred = a_model.predict(x.reshape(1, -1))[0]
    h_pred = h_model.predict(x.reshape(1, -1))[0]
    print(f"Index {i:4d} | Perceptron: {p_pred} | Adaline: {a_pred} | Hebb: {h_pred}")
