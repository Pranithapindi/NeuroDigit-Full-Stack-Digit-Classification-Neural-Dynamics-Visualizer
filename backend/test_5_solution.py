import numpy as np
from PIL import Image, ImageDraw
from backend.dataset.preprocessing import preprocess_image_array
from backend.dataset.mnist_loader import mnist_dataset

mnist_dataset.load()
X_tr = mnist_dataset.X_train[:15000]
y_tr = mnist_dataset.y_train[:15000]
X_te = mnist_dataset.X_test[:2000]
y_te = mnist_dataset.y_test[:2000]

# 1. Perceptron with balanced OvR + margin
T = np.full((len(X_tr), 10), -1.0)
for i, l in enumerate(y_tr):
    T[i, l] = 1.0

W_p = np.zeros((784, 10))
b_p = np.zeros(10)
lr_p = 0.01

for ep in range(20):
    idx = np.random.permutation(len(X_tr))
    for i in idx:
        xi = X_tr[i]
        ti = T[i]
        yi = int(y_tr[i])
        z = xi @ W_p + b_p
        y_hat = np.where(z >= 0.0, 1.0, -1.0)
        errors = ti - y_hat
        pred_c = int(np.argmax(z))
        if np.any(errors != 0) or pred_c != yi:
            w_err = np.where(errors > 0, errors * 1.8, errors)
            W_p += lr_p * np.outer(xi, w_err)
            b_p += lr_p * w_err
            if pred_c != yi:
                W_p[:, pred_c] -= lr_p * 0.5 * xi
                b_p[pred_c] -= lr_p * 0.5

# 2. ADALINE with positive weight = 3.0
W_a = np.zeros((784, 10))
b_a = np.zeros(10)
lr_a = 0.002
for ep in range(25):
    idx = np.random.permutation(len(X_tr))
    for s in range(0, len(X_tr), 64):
        b_idx = idx[s:s+64]
        xb = X_tr[b_idx]
        tb = T[b_idx]
        pred = xb @ W_a + b_a
        err = tb - pred
        w_err = err * np.where(tb > 0, 3.0, 1.0)
        W_a += lr_a * (xb.T @ w_err) / len(xb)
        b_a += lr_a * np.mean(w_err, axis=0)

# 3. Hebbian with normalized prototype templates
centroids = np.zeros((10, 784))
for c in range(10):
    centroids[c] = np.mean(X_tr[y_tr == c], axis=0)
norms = np.linalg.norm(centroids, axis=1, keepdims=True)
W_h = (centroids / (norms + 1e-8)).T

# Now draw 5 with clear features:
img = Image.new('L', (280, 280), 0)
d = ImageDraw.Draw(img)
# 1. Top bar: starts slightly left, goes far right
d.line([(100, 60), (185, 60)], fill=255, width=24)
# 2. Left vertical stem
d.line([(100, 60), (95, 135)], fill=255, width=24)
# 3. Belly: starts horizontal right, then curves down and left
belly = [
    (95, 135), (145, 130), (185, 145), (185, 185), (160, 220), (110, 225), (75, 205)
]
d.line(belly, fill=255, width=24, joint='curve')

arr = np.array(img, dtype=np.float32)
_, flat = preprocess_image_array(arr)

p_s = flat @ W_p + b_p
a_s = flat @ W_a + b_a
h_s = flat @ W_h

print(f"Perceptron Pred: {np.argmax(p_s)} | Top 3: {np.argsort(p_s)[::-1][:3]}")
print(f"ADALINE Pred:    {np.argmax(a_s)} | Top 3: {np.argsort(a_s)[::-1][:3]}")
print(f"Hebbian Pred:    {np.argmax(h_s)} | Top 3: {np.argsort(h_s)[::-1][:3]}")
