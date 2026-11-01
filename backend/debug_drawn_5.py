import numpy as np
from PIL import Image, ImageDraw
from backend.dataset.preprocessing import preprocess_image_array
from backend.dataset.mnist_loader import mnist_dataset

mnist_dataset.load()

# 1. Compute mean 28x28 of class 5 and class 3 in MNIST
mean_5 = np.mean(mnist_dataset.X_train[mnist_dataset.y_train == 5], axis=0).reshape(28, 28)
mean_3 = np.mean(mnist_dataset.X_train[mnist_dataset.y_train == 3], axis=0).reshape(28, 28)

# 2. Check center of mass of true MNIST 5s:
cms_5 = []
for x in mnist_dataset.X_train[mnist_dataset.y_train == 5][:200]:
    im = x.reshape(28, 28)
    y_idx, x_idx = np.indices((28, 28))
    total = np.sum(im)
    cy = np.sum(y_idx * im) / total
    cx = np.sum(x_idx * im) / total
    cms_5.append((cy, cx))
cms_5 = np.array(cms_5)
print(f"True MNIST 5s center of mass: mean_cy={np.mean(cms_5[:, 0]):.2f}, mean_cx={np.mean(cms_5[:, 1]):.2f}")

# 3. Now let's draw human-style 5 and check its 28x28
img = Image.new("L", (280, 280), 0)
d = ImageDraw.Draw(img)
# Draw standard 5
d.line([(100, 60), (180, 60)], fill=255, width=24) # top bar
d.line([(110, 60), (105, 130)], fill=255, width=24) # neck
# loop
arc_pts = [(105, 130), (145, 120), (185, 145), (185, 185), (160, 220), (115, 225), (85, 205)]
d.line(arc_pts, fill=255, width=24, joint="curve")

arr = np.array(img, dtype=np.float32)
norm_28, flat = preprocess_image_array(arr)

# Check dot product with mean_5 and mean_3:
dot_5 = np.dot(flat, mean_5.reshape(-1))
dot_3 = np.dot(flat, mean_3.reshape(-1))
print(f"Dot product of drawn 5 with MNIST mean_5: {dot_5:.2f}")
print(f"Dot product of drawn 5 with MNIST mean_3: {dot_3:.2f}")

# Print ASCII art of the preprocessed 28x28 drawn 5
print("\nPreprocessed 28x28 drawn 5:")
for row in norm_28:
    print("".join(["#" if p > 0.3 else "." for p in row]))
