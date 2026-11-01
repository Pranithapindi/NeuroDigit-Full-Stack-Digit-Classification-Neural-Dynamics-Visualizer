import React, { useState, useEffect } from 'react';
import { 
  BarChart3, 
  Download, 
  RefreshCw, 
  Layers, 
  CheckCircle2, 
  XCircle, 
  Clock, 
  Zap, 
  Target,
  FileSpreadsheet
} from 'lucide-react';
import { 
  BarChart, 
  Bar, 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  Tooltip, 
  Legend, 
  ResponsiveContainer, 
  CartesianGrid, 
  Cell 
} from 'recharts';
import { fetchComparisonData, getExportResultsUrl } from '../services/api';

export default function ModelComparison() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selectedConfusionModel, setSelectedConfusionModel] = useState('perceptron');

  const loadData = async () => {
    setLoading(true);
    try {
      const res = await fetchComparisonData();
      if (res.success) {
        setData(res.data);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const models = data?.models || {};

  // Accuracy comparison bar chart data
  const accuracyChartData = [
    { name: 'Hebb Learning', accuracy: models.hebb?.accuracy ? Number((models.hebb.accuracy * 100).toFixed(1)) : 0, color: '#a855f7' },
    { name: 'Perceptron (OvR)', accuracy: models.perceptron?.accuracy ? Number((models.perceptron.accuracy * 100).toFixed(1)) : 0, color: '#10b981' },
    { name: 'ADALINE (Delta/LMS)', accuracy: models.adaline?.accuracy ? Number((models.adaline.accuracy * 100).toFixed(1)) : 0, color: '#06b6d4' }
  ];

  // Training time chart data
  const timeChartData = [
    { name: 'Hebb', time: models.hebb?.training_time || 0, color: '#a855f7' },
    { name: 'Perceptron', time: models.perceptron?.training_time || 0, color: '#10b981' },
    { name: 'ADALINE', time: models.adaline?.training_time || 0, color: '#06b6d4' }
  ];

  // Epoch error reduction curves (align epochs)
  const maxEpochs = Math.max(
    models.hebb?.history?.epoch?.length || 0,
    models.perceptron?.history?.epoch?.length || 0,
    models.adaline?.history?.epoch?.length || 0
  );

  const errorCurvesData = [];
  for (let ep = 1; ep <= maxEpochs; ep++) {
    const epIdx = ep - 1;
    const item = { epoch: `Ep ${ep}` };

    if (models.hebb?.history?.error_rate?.[epIdx] !== undefined) {
      item.Hebb = Number((models.hebb.history.error_rate[epIdx] * 100).toFixed(1));
    }
    if (models.perceptron?.history?.error_rate?.[epIdx] !== undefined) {
      item.Perceptron = Number((models.perceptron.history.error_rate[epIdx] * 100).toFixed(1));
    }
    if (models.adaline?.history?.error_rate?.[epIdx] !== undefined) {
      item.ADALINE = Number((models.adaline.history.error_rate[epIdx] * 100).toFixed(1));
    }
    errorCurvesData.push(item);
  }

  // ADALINE continuous MSE curve
  const adalineMseData = (models.adaline?.history?.mse || []).map((val, idx) => ({
    epoch: `Ep ${idx + 1}`,
    mse: val
  }));

  // Per-class accuracy comparison across models
  const perClassData = Array.from({ length: 10 }, (_, digit) => {
    const hebbClass = models.hebb?.per_class?.[digit]?.accuracy;
    const percClass = models.perceptron?.per_class?.[digit]?.accuracy;
    const adalClass = models.adaline?.per_class?.[digit]?.accuracy;

    return {
      digit: `Digit ${digit}`,
      Hebb: hebbClass !== undefined ? Number((hebbClass * 100).toFixed(1)) : 0,
      Perceptron: percClass !== undefined ? Number((percClass * 100).toFixed(1)) : 0,
      ADALINE: adalClass !== undefined ? Number((adalClass * 100).toFixed(1)) : 0
    };
  });

  const activeCm = models[selectedConfusionModel]?.confusion_matrix || [];

  return (
    <div className="space-y-8 max-w-7xl mx-auto pb-12">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight flex items-center gap-2">
            <BarChart3 className="w-6 h-6 text-brand-cyan" />
            Comparative Study Dashboard
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Empirical comparative analysis of Hebb, Perceptron, and Delta (ADALINE) learning rules evaluated on MNIST handwritten digits.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={loadData}
            className="px-3.5 py-2 rounded-xl bg-dark-800 hover:bg-dark-750 text-slate-300 text-xs font-semibold border border-slate-700 flex items-center gap-1.5 transition-colors"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            Refresh
          </button>
          <a
            href={getExportResultsUrl()}
            download="model_comparison_results.csv"
            className="px-4 py-2 rounded-xl bg-gradient-to-r from-brand-violet to-brand-cyan hover:opacity-95 text-white text-xs font-bold shadow-lg shadow-brand-violet/20 flex items-center gap-1.5 transition-all"
          >
            <Download className="w-3.5 h-3.5" />
            Export Results (CSV)
          </a>
        </div>
      </div>

      {/* Side-by-Side Comparison Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        {/* Hebb Card */}
        <div className="glass-panel p-6 rounded-2xl border-t-4 border-t-purple-500 relative">
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs font-bold text-purple-400 uppercase tracking-wider">Hebb Learning</span>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-purple-950/60 text-purple-300 border border-purple-800">
              Correlation
            </span>
          </div>
          <h3 className="text-lg font-bold text-white mb-1">Single-Layer Associator</h3>
          <p className="text-xs text-slate-400 mb-4">
            w_new = w_old + η · x · t
          </p>

          <div className="space-y-2 border-t border-slate-800 pt-3 text-xs">
            <div className="flex justify-between">
              <span className="text-slate-400">Test Accuracy:</span>
              <span className="font-mono font-bold text-purple-400">
                {models.hebb?.accuracy ? `${(models.hebb.accuracy * 100).toFixed(1)}%` : 'N/A'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Macro F1-Score:</span>
              <span className="font-mono text-slate-200">
                {models.hebb?.macro_f1 !== undefined ? models.hebb.macro_f1.toFixed(3) : 'N/A'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Training Time:</span>
              <span className="font-mono text-slate-200">
                {models.hebb?.training_time ? `${models.hebb.training_time}s` : 'N/A'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Error Objective:</span>
              <span className="text-[11px] text-slate-300">None (Co-occurrence)</span>
            </div>
          </div>
        </div>

        {/* Perceptron Card */}
        <div className="glass-panel p-6 rounded-2xl border-t-4 border-t-emerald-500 relative">
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs font-bold text-emerald-400 uppercase tracking-wider">Perceptron Rule</span>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-950/60 text-emerald-300 border border-emerald-800">
              Step Error
            </span>
          </div>
          <h3 className="text-lg font-bold text-white mb-1">Rosenblatt OvR</h3>
          <p className="text-xs text-slate-400 mb-4">
            w_new = w_old + η · (t - ŷ) · x
          </p>

          <div className="space-y-2 border-t border-slate-800 pt-3 text-xs">
            <div className="flex justify-between">
              <span className="text-slate-400">Test Accuracy:</span>
              <span className="font-mono font-bold text-emerald-400">
                {models.perceptron?.accuracy ? `${(models.perceptron.accuracy * 100).toFixed(1)}%` : 'N/A'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Macro F1-Score:</span>
              <span className="font-mono text-slate-200">
                {models.perceptron?.macro_f1 !== undefined ? models.perceptron.macro_f1.toFixed(3) : 'N/A'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Training Time:</span>
              <span className="font-mono text-slate-200">
                {models.perceptron?.training_time ? `${models.perceptron.training_time}s` : 'N/A'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Error Objective:</span>
              <span className="text-[11px] text-slate-300">Misclassification count</span>
            </div>
          </div>
        </div>

        {/* ADALINE Card */}
        <div className="glass-panel p-6 rounded-2xl border-t-4 border-t-cyan-500 relative">
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs font-bold text-brand-cyan uppercase tracking-wider">ADALINE / Delta</span>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-950/60 text-cyan-300 border border-cyan-800">
              LMS Gradient
            </span>
          </div>
          <h3 className="text-lg font-bold text-white mb-1">Adaptive Linear Neuron</h3>
          <p className="text-xs text-slate-400 mb-4">
            w_new = w_old + η · (t - y_lin) · x
          </p>

          <div className="space-y-2 border-t border-slate-800 pt-3 text-xs">
            <div className="flex justify-between">
              <span className="text-slate-400">Test Accuracy:</span>
              <span className="font-mono font-bold text-brand-cyan">
                {models.adaline?.accuracy ? `${(models.adaline.accuracy * 100).toFixed(1)}%` : 'N/A'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Macro F1-Score:</span>
              <span className="font-mono text-slate-200">
                {models.adaline?.macro_f1 !== undefined ? models.adaline.macro_f1.toFixed(3) : 'N/A'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Training Time:</span>
              <span className="font-mono text-slate-200">
                {models.adaline?.training_time ? `${models.adaline.training_time}s` : 'N/A'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Error Objective:</span>
              <span className="text-[11px] text-slate-300">Mean Squared Error (MSE)</span>
            </div>
          </div>
        </div>
      </div>

      {/* Charts Section: Row 1 (Accuracy & Training Time) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Accuracy Comparison Chart (7 Cols) */}
        <div className="lg:col-span-7 glass-panel p-6 rounded-2xl space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              <BarChart3 className="w-4 h-4 text-emerald-400" />
              Test Accuracy Comparison (%)
            </h3>
            <span className="text-[11px] text-slate-400 font-mono">Independent Test Split</span>
          </div>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={accuracyChartData} margin={{ top: 10, right: 20, left: -20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                <YAxis domain={[0, 100]} tick={{ fill: '#94a3b8', fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '11px' }}
                  formatter={(val) => [`${val}%`, 'Accuracy']}
                />
                <Bar dataKey="accuracy" radius={[6, 6, 0, 0]}>
                  {accuracyChartData.map((entry, index) => (
                    <Cell key={`acc-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Training Time Comparison (5 Cols) */}
        <div className="lg:col-span-5 glass-panel p-6 rounded-2xl space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              <Clock className="w-4 h-4 text-brand-cyan" />
              Training Execution Time (Seconds)
            </h3>
            <span className="text-[11px] text-slate-400 font-mono">Vectorized NumPy</span>
          </div>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={timeChartData} margin={{ top: 10, right: 20, left: -20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '11px' }}
                  formatter={(val) => [`${val}s`, 'Training Time']}
                />
                <Bar dataKey="time" radius={[6, 6, 0, 0]}>
                  {timeChartData.map((entry, index) => (
                    <Cell key={`time-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Charts Section: Row 2 (Epoch Error Curves & ADALINE MSE) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Epoch Error Curves */}
        <div className="lg:col-span-7 glass-panel p-6 rounded-2xl space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              <Zap className="w-4 h-4 text-purple-400" />
              Epoch-wise Misclassification Error Rate (%)
            </h3>
            <span className="text-[11px] text-slate-400 font-mono">Lower is better</span>
          </div>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={errorCurvesData} margin={{ top: 10, right: 20, left: -20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="epoch" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                <YAxis domain={[0, 100]} tick={{ fill: '#94a3b8', fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '11px' }}
                  formatter={(val) => [`${val}%`, 'Error Rate']}
                />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                <Line type="monotone" dataKey="Hebb" stroke="#a855f7" strokeWidth={2} dot={{ r: 2 }} />
                <Line type="monotone" dataKey="Perceptron" stroke="#10b981" strokeWidth={2} dot={{ r: 2 }} />
                <Line type="monotone" dataKey="ADALINE" stroke="#06b6d4" strokeWidth={2} dot={{ r: 2 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* ADALINE MSE Curve */}
        <div className="lg:col-span-5 glass-panel p-6 rounded-2xl space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              <Target className="w-4 h-4 text-brand-cyan" />
              ADALINE Continuous MSE Loss Curve
            </h3>
            <span className="text-[11px] text-slate-400 font-mono">LMS Gradient Descent</span>
          </div>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={adalineMseData} margin={{ top: 10, right: 20, left: -20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="epoch" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '11px' }}
                  formatter={(val) => [val, 'MSE Loss']}
                />
                <Line type="monotone" dataKey="mse" stroke="#06b6d4" strokeWidth={2} dot={{ r: 2 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Per-Class Accuracy Comparison */}
      <div className="glass-panel p-6 rounded-2xl space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-bold text-white flex items-center gap-2">
            <Layers className="w-4 h-4 text-brand-violet" />
            Per-Class Accuracy Breakdown (Digits 0–9)
          </h3>
          <span className="text-[11px] text-slate-400 font-mono">Class-wise True Positive Rate</span>
        </div>
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={perClassData} margin={{ top: 10, right: 20, left: -20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="digit" tick={{ fill: '#94a3b8', fontSize: 11 }} />
              <YAxis domain={[0, 100]} tick={{ fill: '#94a3b8', fontSize: 11 }} />
              <Tooltip
                contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '11px' }}
                formatter={(val, name) => [`${val}%`, name]}
              />
              <Legend wrapperStyle={{ fontSize: '11px' }} />
              <Bar dataKey="Hebb" fill="#a855f7" radius={[3, 3, 0, 0]} />
              <Bar dataKey="Perceptron" fill="#10b981" radius={[3, 3, 0, 0]} />
              <Bar dataKey="ADALINE" fill="#06b6d4" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* 10x10 Interactive Confusion Matrix */}
      <div className="glass-panel p-6 rounded-2xl space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              <FileSpreadsheet className="w-4 h-4 text-brand-cyan" />
              10 × 10 Confusion Matrix Heatmap
            </h3>
            <p className="text-[11px] text-slate-400 mt-0.5">
              Rows: True Class | Columns: Model Predicted Class
            </p>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-400">Select Model:</span>
            {['hebb', 'perceptron', 'adaline'].map((algo) => (
              <button
                key={algo}
                onClick={() => setSelectedConfusionModel(algo)}
                className={`px-3 py-1 rounded-lg text-xs font-semibold capitalize transition-all ${
                  selectedConfusionModel === algo
                    ? 'bg-brand-cyan/20 text-brand-cyan border border-brand-cyan/40 shadow-sm'
                    : 'bg-dark-800 text-slate-400 hover:text-white border border-slate-700'
                }`}
              >
                {algo}
              </button>
            ))}
          </div>
        </div>

        {activeCm.length > 0 ? (
          <div className="overflow-x-auto pb-2">
            <table className="w-full text-center border-collapse text-xs font-mono">
              <thead>
                <tr className="border-b border-slate-800">
                  <th className="p-2 text-slate-400 font-semibold text-left">True \ Pred</th>
                  {Array.from({ length: 10 }, (_, i) => (
                    <th key={i} className="p-2 text-slate-300 font-bold bg-dark-900/50">
                      {i}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {activeCm.map((row, trueDigit) => (
                  <tr key={trueDigit} className="border-b border-slate-900">
                    <td className="p-2 font-bold text-slate-300 text-left bg-dark-900/50">
                      Digit {trueDigit}
                    </td>
                    {row.map((count, predDigit) => {
                      const isDiagonal = trueDigit === predDigit;
                      // Heatmap intensity
                      const intensity = Math.min(count / 50, 1);
                      let bgColor = 'transparent';
                      let textColor = '#64748b';

                      if (count > 0) {
                        if (isDiagonal) {
                          bgColor = `rgba(16, 185, 129, ${0.15 + intensity * 0.6})`;
                          textColor = '#34d399';
                        } else {
                          bgColor = `rgba(244, 63, 94, ${0.1 + intensity * 0.5})`;
                          textColor = '#fda4af';
                        }
                      }

                      return (
                        <td
                          key={predDigit}
                          style={{ backgroundColor: bgColor, color: textColor }}
                          className={`p-2 transition-colors font-mono ${
                            isDiagonal ? 'font-bold' : ''
                          }`}
                          title={`True: ${trueDigit}, Pred: ${predDigit}, Count: ${count}`}
                        >
                          {count}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="p-8 text-center text-slate-500 text-xs italic">
            No confusion matrix data available for this model yet.
          </div>
        )}
      </div>

      {/* Comprehensive Academic Comparison Table */}
      <div className="glass-panel p-6 rounded-2xl space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              Summary Comparison of Learning Rules
            </h3>
            <p className="text-[11px] text-slate-400 mt-0.5">
              Populated using actual experimental results from the soft computing pipeline.
            </p>
          </div>
          <a
            href={getExportResultsUrl()}
            className="text-xs text-brand-cyan hover:underline flex items-center gap-1 font-semibold"
          >
            <Download className="w-3.5 h-3.5" />
            Download Table as CSV
          </a>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="border-b border-slate-800 bg-dark-900 text-slate-300 font-semibold">
                <th className="p-3">Learning Rule</th>
                <th className="p-3">Model</th>
                <th className="p-3">Update Mechanism</th>
                <th className="p-3">Error Objective</th>
                <th className="p-3">Convergence</th>
                <th className="p-3">Accuracy</th>
                <th className="p-3">Time</th>
                <th className="p-3">Advantages</th>
                <th className="p-3">Limitations</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {(data?.comparison_table || []).map((row, idx) => (
                <tr key={idx} className="hover:bg-dark-850/60 transition-colors">
                  <td className="p-3 font-bold text-white whitespace-nowrap">{row.learning_rule}</td>
                  <td className="p-3 text-slate-300 whitespace-nowrap">{row.model}</td>
                  <td className="p-3 font-mono text-[11px] text-slate-400 min-w-[200px]">{row.update_mechanism}</td>
                  <td className="p-3 text-slate-300 min-w-[140px]">{row.error_objective}</td>
                  <td className="p-3 text-slate-400 min-w-[160px]">{row.convergence_behavior}</td>
                  <td className="p-3 font-mono font-bold text-emerald-400 whitespace-nowrap">{row.accuracy}</td>
                  <td className="p-3 font-mono text-brand-cyan whitespace-nowrap">{row.training_time}</td>
                  <td className="p-3 text-slate-400 min-w-[180px]">{row.advantages}</td>
                  <td className="p-3 text-slate-400 min-w-[180px]">{row.limitations}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
