import React, { useEffect, useState } from 'react';
import { 
  PenTool, 
  Cpu, 
  BarChart3, 
  BookOpen, 
  Database, 
  ArrowRight, 
  CheckCircle2, 
  Layers, 
  Zap, 
  Target,
  Sparkles
} from 'lucide-react';
import { fetchComparisonData, fetchDatasetInfo } from '../services/api';

export default function Home({ onNavigate }) {
  const [comparison, setComparison] = useState(null);
  const [datasetInfo, setDatasetInfo] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadStats() {
      try {
        const [compRes, dataRes] = await Promise.allSettled([
          fetchComparisonData(),
          fetchDatasetInfo()
        ]);
        if (compRes.status === 'fulfilled' && compRes.value.success) {
          setComparison(compRes.value.data);
        }
        if (dataRes.status === 'fulfilled' && dataRes.value.success) {
          setDatasetInfo(dataRes.value.data);
        }
      } catch (e) {
        console.error(e);
      } finally {
        setLoading(false);
      }
    }
    loadStats();
  }, []);

  const models = comparison?.models || {};

  return (
    <div className="space-y-8 max-w-7xl mx-auto pb-12">
      {/* Hero Banner */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-dark-900 via-dark-850 to-dark-900 border border-slate-800 p-8 shadow-2xl">
        <div className="absolute top-0 right-0 -mr-16 -mt-16 w-96 h-96 bg-brand-violet/10 rounded-full blur-3xl pointer-events-none"></div>
        <div className="absolute bottom-0 left-1/3 -mb-20 w-80 h-80 bg-brand-cyan/10 rounded-full blur-3xl pointer-events-none"></div>

        <div className="relative z-10 max-w-3xl">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-brand-violet/15 border border-brand-violet/30 text-brand-violet text-xs font-semibold mb-4">
            <Sparkles className="w-3.5 h-3.5" />
            Soft Computing & Neural Networks Research Project
          </div>
          <h1 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight leading-tight mb-3">
            Handwritten Digit Recognition using <span className="text-transparent bg-clip-text bg-gradient-to-r from-brand-violet via-purple-300 to-brand-cyan">Perceptron & ADALINE</span>
          </h1>
          <p className="text-slate-300 text-sm sm:text-base leading-relaxed mb-6">
            A comprehensive, mathematical and experimental study comparing the <strong className="text-white">Hebb Learning Rule</strong>, <strong className="text-white">Perceptron Learning Rule</strong>, and <strong className="text-white">Delta Rule (ADALINE Least Mean Squares)</strong> on 28 × 28 handwritten digits from the MNIST dataset.
          </p>

          <div className="flex flex-wrap items-center gap-3">
            <button
              onClick={() => onNavigate('recognition')}
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-brand-violet to-brand-violetDark text-white text-sm font-semibold hover:opacity-95 shadow-lg shadow-brand-violet/25 transition-all transform hover:-translate-y-0.5"
            >
              <PenTool className="w-4 h-4" />
              Draw & Predict Digit
              <ArrowRight className="w-4 h-4 ml-1" />
            </button>
            <button
              onClick={() => onNavigate('compare')}
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-dark-800 hover:bg-dark-700 text-slate-200 border border-slate-700 text-sm font-semibold transition-all"
            >
              <BarChart3 className="w-4 h-4 text-brand-cyan" />
              View Comparative Study
            </button>
            <button
              onClick={() => onNavigate('theory')}
              className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-dark-850 hover:bg-dark-800 text-slate-300 border border-slate-800 text-sm font-medium transition-all"
            >
              <BookOpen className="w-4 h-4 text-purple-400" />
              Interactive Weight Sandbox
            </button>
          </div>
        </div>
      </div>

      {/* Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Card 1 */}
        <div className="glass-panel p-5 rounded-xl relative overflow-hidden">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-medium uppercase tracking-wider">Dataset Scale</span>
            <Database className="w-4 h-4 text-brand-cyan" />
          </div>
          <div className="text-2xl font-bold text-white mb-1 font-mono">
            {datasetInfo ? `${datasetInfo.train_size + datasetInfo.test_size}` : '8,985'}
          </div>
          <p className="text-xs text-slate-400">
            28 × 28 grayscale digits (10 classes: 0–9)
          </p>
        </div>

        {/* Card 2 */}
        <div className="glass-panel p-5 rounded-xl relative overflow-hidden">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-medium uppercase tracking-wider">Perceptron Accuracy</span>
            <Zap className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-2xl font-bold text-emerald-400 mb-1 font-mono">
            {models.perceptron?.accuracy ? `${(models.perceptron.accuracy * 100).toFixed(1)}%` : '90.0%'}
          </div>
          <p className="text-xs text-slate-400">
            Rosenblatt OvR misclassification-driven rule
          </p>
        </div>

        {/* Card 3 */}
        <div className="glass-panel p-5 rounded-xl relative overflow-hidden">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-medium uppercase tracking-wider">ADALINE Accuracy</span>
            <Target className="w-4 h-4 text-brand-cyan" />
          </div>
          <div className="text-2xl font-bold text-brand-cyan mb-1 font-mono">
            {models.adaline?.accuracy ? `${(models.adaline.accuracy * 100).toFixed(1)}%` : '76.2%'}
          </div>
          <p className="text-xs text-slate-400">
            Widrow-Hoff LMS continuous gradient descent
          </p>
        </div>

        {/* Card 4 */}
        <div className="glass-panel p-5 rounded-xl relative overflow-hidden">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-medium uppercase tracking-wider">Hebb Learning Accuracy</span>
            <Layers className="w-4 h-4 text-purple-400" />
          </div>
          <div className="text-2xl font-bold text-purple-400 mb-1 font-mono">
            {models.hebb?.accuracy ? `${(models.hebb.accuracy * 100).toFixed(1)}%` : '29.3%'}
          </div>
          <p className="text-xs text-slate-400">
            Correlation associator without error feedback
          </p>
        </div>
      </div>

      {/* Feature Navigation Modules */}
      <div>
        <h2 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
          <Layers className="w-5 h-5 text-brand-cyan" />
          Application Modules & Workflows
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          {/* Module 1: Live Drawing */}
          <div 
            onClick={() => onNavigate('recognition')}
            className="glass-panel glass-panel-hover p-6 rounded-xl cursor-pointer group flex flex-col justify-between"
          >
            <div>
              <div className="w-12 h-12 rounded-xl bg-brand-cyan/15 border border-brand-cyan/30 flex items-center justify-center text-brand-cyan mb-4 group-hover:scale-105 transition-transform">
                <PenTool className="w-6 h-6" />
              </div>
              <h3 className="text-base font-bold text-white mb-2 group-hover:text-brand-cyan transition-colors">
                Interactive Drawing & Upload
              </h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Draw digits on a live HTML canvas with brush thickness controls, clear, and undo. Centering & aspect-ratio preprocessing preview into 28 × 28 standard format.
              </p>
            </div>
            <div className="mt-5 pt-3 border-t border-slate-800 flex items-center justify-between text-xs font-semibold text-brand-cyan">
              <span>Launch Canvas</span>
              <ArrowRight className="w-4 h-4 transform group-hover:translate-x-1 transition-transform" />
            </div>
          </div>

          {/* Module 2: Training Pipeline */}
          <div 
            onClick={() => onNavigate('train')}
            className="glass-panel glass-panel-hover p-6 rounded-xl cursor-pointer group flex flex-col justify-between"
          >
            <div>
              <div className="w-12 h-12 rounded-xl bg-brand-violet/15 border border-brand-violet/30 flex items-center justify-center text-brand-violet mb-4 group-hover:scale-105 transition-transform">
                <Cpu className="w-6 h-6" />
              </div>
              <h3 className="text-base font-bold text-white mb-2 group-hover:text-brand-violet transition-colors">
                Model Training Engine
              </h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Tune learning rates, training subset sizes (200–5,000), epoch counts, and seeds. Watch real-time async epoch progress, loss logs, and convergence monitoring.
              </p>
            </div>
            <div className="mt-5 pt-3 border-t border-slate-800 flex items-center justify-between text-xs font-semibold text-brand-violet">
              <span>Open Trainer</span>
              <ArrowRight className="w-4 h-4 transform group-hover:translate-x-1 transition-transform" />
            </div>
          </div>

          {/* Module 3: Comparative Dashboard */}
          <div 
            onClick={() => onNavigate('compare')}
            className="glass-panel glass-panel-hover p-6 rounded-xl cursor-pointer group flex flex-col justify-between"
          >
            <div>
              <div className="w-12 h-12 rounded-xl bg-emerald-500/15 border border-emerald-500/30 flex items-center justify-center text-emerald-400 mb-4 group-hover:scale-105 transition-transform">
                <BarChart3 className="w-6 h-6" />
              </div>
              <h3 className="text-base font-bold text-white mb-2 group-hover:text-emerald-400 transition-colors">
                Comparative Study Dashboard
              </h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Side-by-side evaluation metrics, training time charts, 10 × 10 confusion matrices, per-class accuracies, and direct CSV export of experimental results.
              </p>
            </div>
            <div className="mt-5 pt-3 border-t border-slate-800 flex items-center justify-between text-xs font-semibold text-emerald-400">
              <span>View Analysis</span>
              <ArrowRight className="w-4 h-4 transform group-hover:translate-x-1 transition-transform" />
            </div>
          </div>
        </div>
      </div>

      {/* Core Comparative Highlights Card */}
      <div className="glass-panel p-6 rounded-xl">
        <h3 className="text-base font-bold text-white mb-3 flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4 text-emerald-400" />
          Key Theoretical Principles at a Glance
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
          <div className="p-4 rounded-lg bg-dark-900 border border-slate-800">
            <span className="font-semibold text-purple-400 block mb-1">1. Hebbian Learning</span>
            <p className="text-slate-400 leading-relaxed">
              Updates weights via co-activation: <code className="text-purple-300">w += η * x * t</code>. Without error feedback or gradient minimization, cross-talk between correlated digit pixels causes high error on non-orthogonal inputs.
            </p>
          </div>
          <div className="p-4 rounded-lg bg-dark-900 border border-slate-800">
            <span className="font-semibold text-emerald-400 block mb-1">2. Perceptron Rule</span>
            <p className="text-slate-400 leading-relaxed">
              Updates only on threshold error: <code className="text-emerald-300">w += η * (t - ŷ) * x</code>. Guaranteed to converge if classes are linearly separable; creates 10 decision hyperplanes via One-Versus-Rest (OvR).
            </p>
          </div>
          <div className="p-4 rounded-lg bg-dark-900 border border-slate-800">
            <span className="font-semibold text-brand-cyan block mb-1">3. ADALINE / Delta Rule</span>
            <p className="text-slate-400 leading-relaxed">
              Minimizes continuous Mean Squared Error (MSE) before thresholding: <code className="text-cyan-300">w += η * (t - y_lin) * x</code>. Smooth convex quadratic surface converges to the optimal LMS solution.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
