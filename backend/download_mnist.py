import urllib.request
import io
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
CACHE_FILE = DATA_DIR / "mnist_processed.npz"

def download_and_prepare_mnist():
    print("Preparing genuine MNIST dataset...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    
    print(f"Downloading genuine MNIST from {url}...")
    with urllib.request.urlopen(req, timeout=30) as resp:
        buf = io.BytesIO(resp.read())
        npz = np.load(buf)
        x_train, y_train = npz["x_train"], npz["y_train"]
        x_test, y_test = npz["x_test"], npz["y_test"]

    print(f"Downloaded: Train {x_train.shape}, Test {x_test.shape}")
    
    # Normalize to [0.0, 1.0] and flatten to (N, 784)
    # We can use float32
    X_train = (x_train.astype(np.float32) / 255.0).reshape(len(x_train), -1)
    y_train = y_train.astype(np.int64)
    X_test = (x_test.astype(np.float32) / 255.0).reshape(len(x_test), -1)
    y_test = y_test.astype(np.int64)
    
    # Save processed cache
    np.savez_compressed(
        CACHE_FILE,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        source="Genuine NIST/MNIST 70,000 Standard Dataset"
    )
    print(f"Saved to {CACHE_FILE}: {CACHE_FILE.stat().st_size / (1024*1024):.2f} MB")

if __name__ == "__main__":
    download_and_prepare_mnist()
