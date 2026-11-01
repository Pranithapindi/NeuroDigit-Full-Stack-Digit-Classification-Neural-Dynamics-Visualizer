import numpy as np
from backend.dataset.mnist_loader import mnist_dataset

mnist_dataset.load()
X_tr = mnist_dataset.X_train[:15000]
y_tr = mnist_dataset.y_train[:15000]

c5 = np.mean(X_tr[y_tr == 5], axis=0).reshape(28, 28)
c3 = np.mean(X_tr[y_tr == 3], axis=0).reshape(28, 28)

diff = c5 - c3
print("Where 5 has pixels that 3 DOES NOT have (+) vs where 3 has pixels that 5 doesn't (-):")
for r in range(28):
    s = "".join(["5" if diff[r, c] > 0.15 else ("3" if diff[r, c] < -0.15 else ".") for c in range(28)])
    print(s)
