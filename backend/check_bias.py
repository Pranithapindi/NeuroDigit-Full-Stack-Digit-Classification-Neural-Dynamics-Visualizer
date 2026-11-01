import json
import numpy as np

with open('backend/models_saved/perceptron_model.json') as f:
    d = json.load(f)

bias = np.array(d['bias'])
weights = np.array(d['weights'])
print("Biases for digits 0..9:")
for i, b in enumerate(bias):
    print(f"  Digit {i}: bias = {b:.4f}, weight norm = {np.linalg.norm(weights[:, i]):.4f}")
