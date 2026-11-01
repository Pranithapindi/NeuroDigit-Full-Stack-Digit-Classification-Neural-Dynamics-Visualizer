# NeuroDigit: Interactive Handwritten Digit Recognition & Neural Learning Sandbox

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-5-646CFF?style=flat-square&logo=vite&logoColor=white)](https://vitejs.dev/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3.x-38B2AC?style=flat-square&logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![NumPy](https://img.shields.io/badge/NumPy-Vectorized-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-10%20Passing-brightgreen?style=flat-square&logo=pytest&logoColor=white)](pytest.ini)

> **NeuroDigit** is a full-stack machine learning platform and educational workbench designed to implement, train, visualize, and benchmark foundational single-layer artificial neural network (ANN) learning algorithms on the MNIST handwritten digit dataset.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Core Learning Algorithms](#-core-learning-algorithms)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation & Getting Started](#-installation--getting-started)
- [Running the Application](#-running-the-application)
- [Running Unit & Integration Tests](#-running-unit--integration-tests)
- [Experimental Benchmarks](#-experimental-benchmarks)
- [API Reference](#-api-reference)
- [License](#-license)

---

## 🌟 Overview

Before deep convolutional neural networks revolutionized computer vision, single-layer connectionist models established the foundational rules of machine learning. **NeuroDigit** bridges historical theory and modern full-stack web applications by providing an interactive environment to study:

1. **Hebbian Learning** (Donald Hebb, 1949) — Unsupervised / Correlation Associator
2. **Perceptron Learning Rule** (Frank Rosenblatt, 1958) — Error-Correction Threshold Logic
3. **ADALINE / Delta Rule** (Bernard Widrow & Ted Hoff, 1960) — Least Mean Squares (LMS) Gradient Descent

Users can draw digits directly on a digital canvas, train models with custom hyperparameters, monitor real-time loss and accuracy curves, evaluate confusion matrices, and experiment with step-by-step mathematical weight updates.

---

## ⚡ Key Features

- 🎨 **Interactive Drawing Canvas:** In-browser canvas with touch/mouse support, smooth brush stroke interpolation, clear controls, and sample presets (0–9).
- 🔬 **MNIST-Grade Preprocessing:** Automated crop-to-bounding-box, aspect-ratio-preserving resize to 20×20, center-of-mass centering into a 28×28 grayscale tensor, and standard normalization matching the official MNIST preprocessing pipeline.
- ⚙️ **One-vs-Rest (OvR) Multi-Class Classification:** Extends binary single-layer decision units into 10-class multiclass digit recognizers using vectorized NumPy matrix operations.
- 🚀 **Asynchronous Training Engine:** Non-blocking background training jobs with progress polling, live epoch logs, weight norm tracking, and dynamic loss curves.
- 📊 **Model Comparison & Analytics:** Side-by-side performance benchmarks, multi-class confusion matrices, per-digit precision/recall/F1 metrics, and one-click CSV export.
- 🧮 **Interactive Learning Theory Sandbox:** Step-by-step weight update calculator illustrating how synaptic weights evolve across mathematical iterations.
- 📁 **Dataset Explorer:** Visual breakdown of MNIST class distributions, pixel intensities, and sample image previews.

---

## 🧠 Core Learning Algorithms

Each model is implemented from scratch using **NumPy** using One-vs-Rest (OvR) binary classification ensembles:

| Algorithm | Learning Type | Weight Update Formulation | Loss / Objective | Activation |
|---|---|---|---|---|
| **Hebb's Rule** | Correlation-based | $w \leftarrow w + \eta \cdot (x \cdot t)$ | Correlation maximization | Bipolar Sign / Identity |
| **Perceptron Rule** | Error-Correction | $w \leftarrow w + \eta \cdot (t - \hat{y}) \cdot x$ | Misclassification count | Heaviside Step Function |
| **ADALINE (Delta / LMS)** | Gradient Descent | $w \leftarrow w + \eta \cdot (t - y_{\text{linear}}) \cdot x$ | Mean Squared Error (MSE) | Linear (Thresholded at inference) |

---

## 🏗️ System Architecture

```
[ Interactive React Frontend ] 
        │ (HTTP / JSON / Base64 Canvas Payload)
        ▼
[ Flask REST API (Port 5000) ]
        ├── Preprocessing Pipeline (Centering, Crop, Normalize 28x28)
        ├── Asynchronous Job Runner (Background Worker Thread)
        └── Multi-Class Model Engine
                ├── Hebbian OvR Classifier
                ├── Rosenblatt Perceptron OvR
                └── ADALINE LMS OvR Classifier
```

---

## 💻 Tech Stack

### Frontend
- **Framework:** React 18 + Vite 5
- **Styling:** Tailwind CSS 3 (Dark glassmorphism UI theme)
- **Charts & Visualizations:** Recharts
- **Icons:** Lucide React

### Backend
- **Server:** Flask 3.x (with Flask-CORS)
- **Mathematical Computation:** NumPy (vectorized array operations)
- **Image Processing:** Pillow (PIL)
- **Metrics & Validation:** Scikit-learn
- **Testing:** Pytest

---

## 📂 Project Structure

```
SC_CEP/
├── backend/
│   ├── app.py                     # Flask REST API endpoints & CORS setup
│   ├── config.py                  # Server configuration, paths, defaults
│   ├── requirements.txt           # Python package dependencies
│   ├── algorithms/
│   │   ├── hebb.py                # Hebbian OvR classifier
│   │   ├── perceptron.py          # Rosenblatt Perceptron OvR classifier
│   │   └── adaline.py             # ADALINE LMS / Delta Rule classifier
│   ├── dataset/
│   │   ├── mnist_loader.py        # Dataset caching, parsing, and augmentation
│   │   └── preprocessing.py       # 28×28 bounding box, center-of-mass centering
│   ├── services/
│   │   ├── training_service.py    # Background thread job manager
│   │   ├── prediction_service.py  # Canvas & uploaded image inference
│   │   └── evaluation_service.py  # Metrics, comparison tables, CSV export
│   ├── utils/
│   │   ├── metrics.py             # Confusion matrix, accuracy, F1 score
│   │   └── image_utils.py         # Base64 decode, image serialization
│   ├── tests/
│   │   ├── test_algorithms.py     # Unit tests for weight updates & math
│   │   └── test_api.py            # Integration tests for REST endpoints
│   ├── data/                      # Cached MNIST dataset (.npz)
│   └── models_saved/              # Serialized model weights (.json)
├── frontend/
│   ├── src/
│   │   ├── App.jsx                # Layout shell, navigation, backend health check
│   │   ├── main.jsx               # React DOM entry point
│   │   ├── index.css              # Custom styling & glassmorphic tokens
│   │   ├── components/            # Reusable UI components (Navbar, Sidebar, etc.)
│   │   ├── pages/
│   │   │   ├── Home.jsx           # Dashboard overview and metric cards
│   │   │   ├── DigitRecognition.jsx  # Interactive canvas & digit classification
│   │   │   ├── TrainModels.jsx    # Async training console with live logs
│   │   │   ├── ModelComparison.jsx   # Side-by-side charts & confusion matrix
│   │   │   ├── LearningTheory.jsx    # Interactive numerical math sandbox
│   │   │   ├── DatasetExplorer.jsx   # Class distribution & sample previews
│   │   │   └── AboutProject.jsx   # Architecture documentation
│   │   └── services/
│   │       └── api.js             # Client API service wrapper
│   ├── package.json
│   ├── tailwind.config.js
│   └── vite.config.js
├── pytest.ini
└── README.md
```

---

## 🚀 Installation & Getting Started

### Prerequisites
- **Python 3.10+** (Python 3.11 recommended)
- **Node.js 18+** & **npm**
- **Git**

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/neurodigit.git
cd neurodigit
```

### 2. Set Up the Backend
Create and activate a virtual environment (optional but recommended):

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

Install backend dependencies:
```bash
pip install -r backend/requirements.txt
```

### 3. Set Up the Frontend
```bash
cd frontend
npm install
cd ..
```

---

## 🖥️ Running the Application

### 1. Start the Flask Backend Server
In your first terminal:
```bash
python backend/app.py
```
> The API server will start at: **`http://127.0.0.1:5000`**

### 2. Start the Vite Frontend Client
In your second terminal:
```bash
cd frontend
npm run dev
```
> The web application will launch at: **`http://localhost:5173`**

Open **[http://localhost:5173](http://localhost:5173)** in your browser to start using the app.

---

## 🧪 Running Unit & Integration Tests

All backend algorithmic updates, image preprocessing, and API endpoints are covered by comprehensive tests:

```bash
python -m pytest backend/tests/ -v
```

**Test Suite Coverage:**
- ✔️ Hebbian weight correlation correctness
- ✔️ Perceptron mistake-driven update condition
- ✔️ ADALINE MSE monotonic decrease via LMS
- ✔️ Bounding box centering and 28×28 dimension guarantee
- ✔️ Multi-class confusion matrix and classification metrics
- ✔️ Flask REST endpoints (`/api/health`, `/api/train`, `/api/predict`, etc.)

---

## 📊 Experimental Benchmarks

Empirical performance on MNIST test partition using default hyperparameter configurations (1,200 training samples):

| Model Architecture | Test Accuracy | Training Time | Convergence Property |
|---|---|---|---|
| **Hebbian Learning** | ~29.2% | ~0.14s | Fast single-pass correlation; limited linear separability |
| **Perceptron (OvR)** | ~90.4% | ~0.52s | Guaranteed convergence on linearly separable subsets |
| **ADALINE (Delta / LMS)** | ~76.8% | ~0.25s | Smooth convex quadratic MSE minimization |

> *Note: These are linear single-layer classifiers evaluated on 784-dimensional raw pixel space. The study highlights the mathematical trade-offs between continuous error minimization and discrete mistake correction.*

---

## 📡 API Reference

| HTTP Method | Route | Description |
|---|---|---|
| `GET` | `/api/health` | Backend status & Python runtime information |
| `GET` | `/api/dataset/info` | Total samples, class breakdown, and cache status |
| `GET` | `/api/dataset/samples` | Base64-encoded preview samples per digit |
| `POST` | `/api/train` | Spawns an asynchronous background training job |
| `GET` | `/api/train/status/<job_id>` | Polls job progress, epoch logs, and training metrics |
| `GET` | `/api/models` | Lists all trained model checkpoints and configurations |
| `POST` | `/api/predict` | Classifies digit from base64 canvas image or file upload |
| `GET` | `/api/evaluate/<model_id>` | Evaluates model on test split (confusion matrix, F1) |
| `GET` | `/api/compare` | Aggregates side-by-side benchmark comparison data |
| `GET` | `/api/export/results` | Downloads comparative metrics as a `.csv` spreadsheet |

---

## 📄 License

This project is open-source under the [MIT License](LICENSE). Feel free to use, modify, and build upon it for academic and research purposes.
