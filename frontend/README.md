# NeuroDigit — Frontend Web Application

The interactive client interface for **NeuroDigit**, built with **React 18**, **Vite 5**, and **Tailwind CSS 3**.

---

## 🎨 Features & Modules

- **Home Dashboard:** Quick stats overview, system status, and links to all interactive modules.
- **Digit Recognition Studio:** Interactive HTML5 drawing canvas with preset selectors (0–9), brush clearing, stroke preview, and real-time backend inference.
- **Model Training Console:** Trigger asynchronous model training for Hebbian, Perceptron, and ADALINE classifiers with customizable hyperparameters (epochs, learning rate, sample limit) and live logs.
- **Model Comparison & Analytics:** Side-by-side performance charts (Recharts), multiclass confusion matrices, per-digit metrics (Precision, Recall, F1), and CSV export.
- **Learning Theory Sandbox:** Mathematical interactive simulator stepping through manual weight calculations for single inputs.
- **Dataset Explorer:** Visual breakdown of MNIST sample images, class distributions, and pixel statistics.

---

## 🛠️ Tech Stack

- **Framework:** [React 18](https://react.dev/)
- **Build Tool:** [Vite 5](https://vitejs.dev/)
- **Styling:** [Tailwind CSS 3](https://tailwindcss.com/)
- **Data Visualization:** [Recharts](https://recharts.org/)
- **Icons:** [Lucide React](https://lucide.dev/)

---

## 🚀 Development Scripts

```bash
# Install dependencies
npm install

# Start Vite dev server (defaults to http://localhost:5173)
npm run dev

# Create optimized production build
npm run build

# Preview production build locally
npm run preview
```

---

## 🔗 Backend Connection

By default, the client communicates with the Flask REST API at:
`http://127.0.0.1:5000/api`

Configured in `src/services/api.js`.
