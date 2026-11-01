import numpy as np
from backend.dataset.mnist_loader import mnist_dataset

mnist_dataset.load()
mean_5 = np.mean(mnist_dataset.X_train[mnist_dataset.y_train == 5], axis=0).reshape(28, 28)
mean_3 = np.mean(mnist_dataset.X_train[mnist_dataset.y_train == 3], axis=0).reshape(28, 28)

print("Mean 5                               Mean 3")
for r in range(28):
    s5 = "".join(["#" if p > 0.25 else "." for p in mean_5[r]])
    s3 = "".join(["#" if p > 0.25 else "." for p in mean_3[r]])
    print(f"{s5}    {s3}")
