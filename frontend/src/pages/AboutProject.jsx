import React from 'react';
import {
  Info, Target, Cpu, Layers, AlertTriangle,
  Rocket, Code, BookOpen, Users, Zap
} from 'lucide-react';

export default function AboutProject() {
  return (
    <div className="space-y-8 max-w-5xl mx-auto pb-16">
      <div>
        <h1 className="text-2xl font-bold text-white tracking-tight flex items-center gap-2">
          <Info className="w-6 h-6 text-brand-cyan" />
          About This Project
        </h1>
        <p className="text-xs text-slate-400 mt-1">
          Academic Soft Computing project: full-stack ML experimentation platform.
        </p>
      </div>

      {/* Problem Statement */}
      <div className="glass-panel p-6 rounded-2xl space-y-3">
        <h2 className="text-base font-bold text-white flex items-center gap-2 border-b border-slate-800 pb-3">
          <Target className="w-5 h-5 text-rose-400" />
          Problem Statement & Objectives
        </h2>
        <p className="text-xs text-slate-300 leading-relaxed">
          Handwritten digit recognition is a fundamental benchmark task in pattern recognition and machine learning.
          While modern deep learning approaches achieve near-perfect accuracy on MNIST, this project intentionally
          implements and compares three foundational classical learning paradigms — <strong className="text-white">Hebbian Learning</strong>,
          the <strong className="text-white">Perceptron Learning Rule</strong>, and the <strong className="text-white">Delta (ADALINE) Learning Rule</strong> —
          to understand their theoretical underpinnings, convergence behavior, and practical limitations on a real-world
          multiclass digit classification task.
        </p>
        <ul className="list-disc list-inside space-y-1.5 text-xs text-slate-300 leading-relaxed">
          <li>Implement all three learning algorithms from scratch using NumPy vectorized operations.</li>
          <li>Train and evaluate on a standardized 28 × 28 handwritten digit dataset.</li>
          <li>Compare learning curves, convergence behavior, computational cost, and accuracy.</li>
          <li>Build an interactive web application for live digit recognition and educational exploration.</li>
          <li>Demonstrate limitations of single-layer linear classifiers versus deeper architectures.</li>
        </ul>
      </div>

      {/* Methodology */}
      <div className="glass-panel p-6 rounded-2xl space-y-4">
        <h2 className="text-base font-bold text-white flex items-center gap-2 border-b border-slate-800 pb-3">
          <Cpu className="w-5 h-5 text-brand-violet" />
          Methodology & System Architecture
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-xs text-slate-300 leading-relaxed">
          <div className="space-y-3">
            <h3 className="font-bold text-white text-sm">Frontend (React + Vite)</h3>
            <ul className="list-disc list-inside space-y-1 text-slate-400">
              <li>React 18 SPA with Vite bundler</li>
              <li>Tailwind CSS v3 — dark theme with violet/cyan accents</li>
              <li>Recharts for dynamic charts and training curves</li>
              <li>HTML5 Canvas API for live digit drawing</li>
              <li>Lucide React for professional iconography</li>
              <li>Async polling for real-time training progress</li>
            </ul>
            <h3 className="font-bold text-white text-sm mt-3">Data Pipeline</h3>
            <ul className="list-disc list-inside space-y-1 text-slate-400">
              <li>28 × 28 grayscale images normalized to [0.0, 1.0]</li>
              <li>Aspect-ratio crop and center-of-mass alignment</li>
              <li>Data augmentation via directional pixel shifts</li>
              <li>Train/test split (80/20) with fixed reproducible seeds</li>
            </ul>
          </div>
          <div className="space-y-3">
            <h3 className="font-bold text-white text-sm">Backend (Python + Flask)</h3>
            <ul className="list-disc list-inside space-y-1 text-slate-400">
              <li>Flask REST API with CORS for frontend integration</li>
              <li>Background thread training with live job polling</li>
              <li>NumPy vectorized OvR multiclass implementations</li>
              <li>Pillow for image decoding and preprocessing</li>
              <li>Scikit-learn only for evaluation metrics computation</li>
              <li>JSON model serialization for persistence</li>
            </ul>
            <h3 className="font-bold text-white text-sm mt-3">API Design</h3>
            <ul className="list-disc list-inside space-y-1 text-slate-400">
              <li><code className="text-emerald-400">GET /api/health</code> — Backend liveness check</li>
              <li><code className="text-emerald-400">POST /api/train</code> — Trigger async training job</li>
              <li><code className="text-emerald-400">GET /api/train/status/:id</code> — Poll progress</li>
              <li><code className="text-emerald-400">POST /api/predict</code> — Canvas/file inference</li>
              <li><code className="text-emerald-400">GET /api/compare</code> — Comparative results</li>
              <li><code className="text-emerald-400">GET /api/export/results</code> — CSV download</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Algorithms */}
      <div className="glass-panel p-6 rounded-2xl space-y-4">
        <h2 className="text-base font-bold text-white flex items-center gap-2 border-b border-slate-800 pb-3">
          <Code className="w-5 h-5 text-brand-cyan" />
          Implemented Algorithms
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
          {[
            {
              title: 'Hebbian Classifier',
              color: 'text-purple-400',
              border: 'border-purple-900/40',
              rule: 'w += η · x · t',
              desc: 'Supervised Hebbian with bipolar targets. One-versus-rest. No error feedback or gradient. Weights grow by co-activation correlation.'
            },
            {
              title: 'Perceptron Classifier',
              color: 'text-emerald-400',
              border: 'border-emerald-900/40',
              rule: 'w += η · (t − ŷ) · x',
              desc: '10 OvR Rosenblatt Perceptrons. Bipolar step activation. Updates only on misclassification. Multiclass decision via argmax of scores.'
            },
            {
              title: 'ADALINE Classifier',
              color: 'text-brand-cyan',
              border: 'border-cyan-900/40',
              rule: 'w += η · (t − y_lin) · x',
              desc: 'Widrow-Hoff LMS with mini-batch gradient descent. Continuous linear output during training. Minimizes MSE. Thresholds at inference.'
            }
          ].map(algo => (
            <div key={algo.title} className={`p-4 rounded-xl bg-dark-900 border ${algo.border} space-y-2`}>
              <span className={`font-bold text-sm ${algo.color} block`}>{algo.title}</span>
              <code className={`text-[11px] font-mono block p-2 bg-dark-950 rounded border border-slate-800 ${algo.color}`}>
                {algo.rule}
              </code>
              <p className="text-slate-400 leading-relaxed">{algo.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Limitations & Future Work */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        <div className="glass-panel p-6 rounded-2xl space-y-3">
          <h2 className="text-sm font-bold text-white flex items-center gap-2 border-b border-slate-800 pb-3">
            <AlertTriangle className="w-4 h-4 text-amber-400" />
            Known Limitations
          </h2>
          <ul className="list-disc list-inside space-y-1.5 text-xs text-slate-400 leading-relaxed">
            <li>All three are <strong className="text-white">single-layer linear classifiers</strong> — cannot learn non-linear decision boundaries.</li>
            <li>Perceptron uses a <strong className="text-white">non-differentiable step</strong> function — no smooth gradient flow.</li>
            <li>Hebbian rule has <strong className="text-white">no convergence guarantee</strong> for non-orthogonal inputs.</li>
            <li>ADALINE is <strong className="text-white">sensitive to learning rate</strong> — too large causes divergence.</li>
            <li>All models significantly underperform multi-layer networks (CNN achieves 99%+ on MNIST).</li>
            <li>Dataset uses resampled 8×8 digits from scikit-learn as fallback (no network required).</li>
          </ul>
        </div>

        <div className="glass-panel p-6 rounded-2xl space-y-3">
          <h2 className="text-sm font-bold text-white flex items-center gap-2 border-b border-slate-800 pb-3">
            <Rocket className="w-4 h-4 text-brand-cyan" />
            Future Enhancements
          </h2>
          <ul className="list-disc list-inside space-y-1.5 text-xs text-slate-400 leading-relaxed">
            <li>Implement <strong className="text-white">multi-layer MLP</strong> with backpropagation for comparison.</li>
            <li>Add <strong className="text-white">RBF kernel SVM</strong> as a non-linear baseline.</li>
            <li>Extend to full 60,000 sample MNIST with OpenML download option.</li>
            <li>WebSocket-based real-time training metrics stream.</li>
            <li>Advanced preprocessing: Gaussian blurring, dilation, morphological thinning.</li>
            <li>Export trained model weights as downloadable JSON artifacts.</li>
            <li>Implement Pocket Algorithm for non-separable Perceptron stability.</li>
          </ul>
        </div>
      </div>

      {/* Tech Stack Summary */}
      <div className="glass-panel p-5 rounded-2xl">
        <div className="flex flex-wrap gap-2 items-center">
          <span className="text-xs font-semibold text-slate-400 mr-2">Tech Stack:</span>
          {[
            'Python 3.11', 'Flask 3.x', 'NumPy', 'Scikit-learn (metrics only)',
            'Pillow', 'React 18', 'Vite 5', 'Tailwind CSS 3', 'Recharts', 'Lucide React'
          ].map(tech => (
            <span
              key={tech}
              className="px-2.5 py-1 text-[11px] rounded-lg bg-dark-800 border border-slate-700 text-slate-300 font-mono"
            >
              {tech}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
