import numpy as np
import io, base64
from PIL import Image, ImageDraw
from backend.dataset.mnist_loader import mnist_dataset
from backend.utils.metrics import compute_classification_metrics
from backend.dataset.preprocessing import preprocess_image_array

mnist_dataset.load()
X_tr = mnist_dataset.X_train[:15000]
y_tr = mnist_dataset.y_train[:15000]
X_te = mnist_dataset.X_test[:2000]
y_te = mnist_dataset.y_test[:2000]

print("--- 1. Training Multiclass Margin Perceptron ---")
W_p = np.zeros((784, 10))
b_p = np.zeros(10)
lr_p = 0.01

for epoch in range(15):
    indices = np.random.permutation(len(X_tr))
    for i in indices:
        xi = X_tr[i]
        yi = y_tr[i]
        scores = xi @ W_p + b_p
        y_hat = np.argmax(scores)
        if y_hat != yi:
            W_p[:, yi] += lr_p * xi
            b_p[yi] += lr_p
            W_p[:, y_hat] -= lr_p * xi
            b_p[y_hat] -= lr_p

p_preds = np.argmax(X_te @ W_p + b_p, axis=1)
m_p = compute_classification_metrics(y_te, p_preds)
print(f"Perceptron overall acc: {m_p['accuracy']*100:.2f}%")
for c in m_p['per_class']:
    if c['class'] == 5:
        print(f"Perceptron Class 5: acc={c['accuracy']*100:.2f}%, prec={c['precision']*100:.2f}%")

print("\n--- 2. Training Balanced ADALINE (Delta Rule) ---")
W_a = np.zeros((784, 10))
b_a = np.zeros(10)
lr_a = 0.002
batch_size = 64
T = np.full((len(X_tr), 10), -1.0)
for i, l in enumerate(y_tr):
    T[i, l] = 1.0

for epoch in range(25):
    indices = np.random.permutation(len(X_tr))
    for s in range(0, len(X_tr), batch_size):
        b_idx = indices[s:s+batch_size]
        xb = X_tr[b_idx]
        tb = T[b_idx]
        pred = xb @ W_a + b_a
        err = tb - pred
        # Balanced weights for positive vs negative
        w_mask = np.where(tb > 0, 2.5, 1.0)
        w_err = err * w_mask
        W_a += lr_a * (xb.T @ w_err) / len(xb)
        b_a += lr_a * np.mean(w_err, axis=0)

a_preds = np.argmax(X_te @ W_a + b_a, axis=1)
m_a = compute_classification_metrics(y_te, a_preds)
print(f"ADALINE overall acc: {m_a['accuracy']*100:.2f}%")
for c in m_a['per_class']:
    if c['class'] == 5:
        print(f"ADALINE Class 5: acc={c['accuracy']*100:.2f}%, prec={c['precision']*100:.2f}%")

print("\n--- 3. Training Normalized Hebbian Classifier ---")
centroids = np.zeros((10, 784))
for c in range(10):
    centroids[c] = np.mean(X_tr[y_tr == c], axis=0)
norms = np.linalg.norm(centroids, axis=1, keepdims=True)
W_h = (centroids / (norms + 1e-8)).T
b_h = np.zeros(10)

h_preds = np.argmax(X_te @ W_h + b_h, axis=1)
m_h = compute_classification_metrics(y_te, h_preds)
print(f"Hebbian overall acc: {m_h['accuracy']*100:.2f}%")
for c in m_h['per_class']:
    if c['class'] == 5:
        print(f"Hebbian Class 5: acc={c['accuracy']*100:.2f}%, prec={c['precision']*100:.2f}%")

# Now test drawn 5
img = Image.new("L", (280, 280), 0)
d = ImageDraw.Draw(img)
d.line([(95, 60), (180, 60)], fill=255, width=24) # top bar
d.line([(100, 60), (95, 135)], fill=255, width=24) # neck
d.line([(95, 135), (145, 125), (185, 145), (185, 185), (160, 220), (110, 225), (75, 205)], fill=255, width=24, joint="curve")

arr = np.array(img, dtype=np.float32)
_, flat = preprocess_image_array(arr)

p_score = flat @ W_p + b_p
a_score = flat @ W_a + b_a
h_score = flat @ W_h + b_h

print("\n--- Inference on Drawn 5 with Updated Models ---")
print("Perceptron predicted:", np.argmax(p_score), "Top scores:", np.argsort(p_score)[::-1][:3])
print("ADALINE predicted:   ", np.argmax(a_score), "Top scores:", np.argsort(a_score)[::-1][:3])
print("Hebbian predicted:   ", np.argmax(h_score), "Top scores:", np.argsort(h_score)[::-1][:3])
