import numpy as np
import json

with open('backend/models_saved/perceptron_model.json') as f:
    d = json.load(f)

weights = np.array(d['weights']) # (784, 10)
bias = np.array(d['bias'])

w3 = weights[:, 3].reshape(28, 28)
w5 = weights[:, 5].reshape(28, 28)

print("Positive weights for 3:       Positive weights for 5:")
for r in range(28):
    s3 = "".join(["#" if w > 0.05 else "." for w in w3[r]])
    s5 = "".join(["#" if w > 0.05 else "." for w in w5[r]])
    print(f"{s3}    {s5}")
