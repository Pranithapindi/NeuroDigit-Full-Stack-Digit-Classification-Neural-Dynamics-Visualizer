import React, { useState, useEffect, useRef } from 'react';
import { 
  Cpu, 
  Play, 
  CheckCircle, 
  AlertCircle, 
  Terminal, 
  Sliders, 
  Layers, 
  RotateCcw,
  Sparkles,
  Info
} from 'lucide-react';
import { triggerTraining, fetchTrainingStatus } from '../services/api';

const ALGO_DESCRIPTIONS = {
  perceptron: {
    name: "Perceptron Learning Rule",
    summary: "Ten One-Versus-Rest Rosenblatt Perceptrons with bipolar step threshold {-1, +1}. Updates weights only upon misclassification.",
    defaultEpochs: 20,
    defaultLr: 0.01,
    lrStep: 0.005,
    lrMin: 0.001,
    lrMax: 0.1
  },
  adaline: {
    name: "Delta Learning Rule (ADALINE)",
    summary: "Adaptive Linear Neuron using Widrow-Hoff LMS gradient descent to minimize continuous Mean Squared Error (MSE) before thresholding.",
    defaultEpochs: 25,
    defaultLr: 0.0005,
    lrStep: 0.0001,
    lrMin: 0.0001,
    lrMax: 0.005
  },
  hebb: {
    name: "Hebbian Learning Rule",
    summary: "Supervised correlation associator (w += η * x * t). Reinforces co-occurrence without error feedback or loss minimization.",
    defaultEpochs: 5,
    defaultLr: 0.01,
    lrStep: 0.005,
    lrMin: 0.001,
    lrMax: 0.05
  }
};

export default function TrainModels() {
  const [algorithm, setAlgorithm] = useState('perceptron');
  const [datasetSize, setDatasetSize] = useState(1200);
  const [epochs, setEpochs] = useState(20);
  const [learningRate, setLearningRate] = useState(0.01);
  const [randomSeed, setRandomSeed] = useState(42);

  const [activeJobId, setActiveJobId] = useState(null);
  const [jobState, setJobState] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);

  const logsEndRef = useRef(null);

  // Sync algorithm presets
  const handleAlgorithmChange = (algoKey) => {
    setAlgorithm(algoKey);
    const cfg = ALGO_DESCRIPTIONS[algoKey];
    setEpochs(cfg.defaultEpochs);
    setLearningRate(cfg.defaultLr);
  };

  // Poll job status if active
  useEffect(() => {
    if (!activeJobId || !isTraining) return;

    const interval = setInterval(async () => {
      try {
        const res = await fetchTrainingStatus(activeJobId);
        if (res.success && res.job) {
          setJobState(res.job);
          if (res.job.status === 'completed' || res.job.status === 'failed') {
            setIsTraining(false);
            if (res.job.status === 'failed') {
              setErrorMessage(res.job.error || "Training failed.");
            }
          }
        }
      } catch (err) {
        console.error("Error polling training status:", err);
      }
    }, 400);

    return () => clearInterval(interval);
  }, [activeJobId, isTraining]);

  // Auto-scroll logs
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [jobState?.logs]);

  const handleStartTraining = async () => {
    setErrorMessage(null);
    setIsTraining(true);
    setJobState(null);

    try {
      const res = await triggerTraining({
        algorithm,
        dataset_size: datasetSize,
        epochs,
        learning_rate: learningRate,
        random_seed: randomSeed
      });

      if (res.success && res.job_id) {
        setActiveJobId(res.job_id);
      } else {
        throw new Error(res.error || "Failed to initiate training.");
      }
    } catch (err) {
      setErrorMessage(err.message || "Failed to trigger training.");
      setIsTraining(false);
    }
  };

  const applyPreset = (size, ep) => {
    setDatasetSize(size);
    setEpochs(ep);
  };

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-12">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white tracking-tight flex items-center gap-2">
          <Cpu className="w-6 h-6 text-brand-violet" />
          Model Training Engine
        </h1>
        <p className="text-xs text-slate-400 mt-1">
          Train Perceptron, ADALINE, or Hebbian learning algorithms from scratch with customizable hyperparameters and live asynchronous progress tracking.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Hyperparameters Control Panel (5 Cols) */}
        <div className="lg:col-span-5 space-y-4">
          <div className="glass-panel p-6 rounded-2xl space-y-5">
            <h2 className="text-sm font-bold text-white flex items-center gap-2 border-b border-slate-800 pb-3">
              <Sliders className="w-4 h-4 text-brand-cyan" />
              Hyperparameter Settings
            </h2>

            {/* Algorithm Selector */}
            <div className="space-y-2">
              <label className="text-xs font-semibold text-slate-300 block">
                Learning Algorithm
              </label>
              <div className="grid grid-cols-3 gap-2">
                {Object.keys(ALGO_DESCRIPTIONS).map((algoKey) => (
                  <button
                    key={algoKey}
                    type="button"
                    onClick={() => handleAlgorithmChange(algoKey)}
                    className={`py-2 px-3 rounded-xl text-xs font-bold transition-all capitalize ${
                      algorithm === algoKey
                        ? 'bg-gradient-to-r from-brand-violet to-brand-cyan text-white shadow-md'
                        : 'bg-dark-800 text-slate-400 hover:text-white border border-slate-700/80'
                    }`}
                  >
                    {algoKey}
                  </button>
                ))}
              </div>
              <p className="text-[11px] text-slate-400 leading-relaxed mt-1">
                {ALGO_DESCRIPTIONS[algorithm].summary}
              </p>
            </div>

            {/* Quick Presets */}
            <div className="flex items-center gap-2 pt-1">
              <span className="text-[10px] text-slate-400 font-mono">Presets:</span>
              <button
                type="button"
                onClick={() => applyPreset(600, 10)}
                className="text-[11px] px-2 py-0.5 rounded bg-dark-800 hover:bg-dark-750 text-slate-300 border border-slate-700"
              >
                Fast (600)
              </button>
              <button
                type="button"
                onClick={() => applyPreset(1200, 20)}
                className="text-[11px] px-2 py-0.5 rounded bg-dark-800 hover:bg-dark-750 text-slate-300 border border-slate-700"
              >
                Standard (1,200)
              </button>
              <button
                type="button"
                onClick={() => applyPreset(3000, 30)}
                className="text-[11px] px-2 py-0.5 rounded bg-dark-800 hover:bg-dark-750 text-slate-300 border border-slate-700"
              >
                High (3,000)
              </button>
            </div>

            {/* Dataset Size Slider */}
            <div className="space-y-1.5">
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-300 font-medium">Training Samples:</span>
                <span className="text-brand-cyan font-mono font-bold">{datasetSize.toLocaleString()}</span>
              </div>
              <input
                type="range"
                min="200"
                max="5000"
                step="100"
                value={datasetSize}
                onChange={(e) => setDatasetSize(Number(e.target.value))}
                className="w-full accent-brand-cyan cursor-pointer"
              />
              <div className="flex justify-between text-[10px] text-slate-400 font-mono">
                <span>200</span>
                <span>5,000 max</span>
              </div>
            </div>

            {/* Epochs Slider */}
            <div className="space-y-1.5">
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-300 font-medium">Number of Epochs:</span>
                <span className="text-brand-violet font-mono font-bold">{epochs}</span>
              </div>
              <input
                type="range"
                min="1"
                max="50"
                step="1"
                value={epochs}
                onChange={(e) => setEpochs(Number(e.target.value))}
                className="w-full accent-brand-violet cursor-pointer"
              />
              <div className="flex justify-between text-[10px] text-slate-400 font-mono">
                <span>1</span>
                <span>50</span>
              </div>
            </div>

            {/* Learning Rate Input */}
            <div className="space-y-1.5">
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-300 font-medium">Learning Rate (η):</span>
                <span className="text-emerald-400 font-mono font-bold">{learningRate}</span>
              </div>
              <input
                type="range"
                min={ALGO_DESCRIPTIONS[algorithm].lrMin}
                max={ALGO_DESCRIPTIONS[algorithm].lrMax}
                step={ALGO_DESCRIPTIONS[algorithm].lrStep}
                value={learningRate}
                onChange={(e) => setLearningRate(Number(e.target.value))}
                className="w-full accent-emerald-400 cursor-pointer"
              />
              <div className="flex justify-between text-[10px] text-slate-400 font-mono">
                <span>{ALGO_DESCRIPTIONS[algorithm].lrMin}</span>
                <span>{ALGO_DESCRIPTIONS[algorithm].lrMax}</span>
              </div>
            </div>

            {/* Random Seed */}
            <div className="flex items-center justify-between pt-1">
              <span className="text-xs text-slate-300 font-medium">Random Seed:</span>
              <input
                type="number"
                value={randomSeed}
                onChange={(e) => setRandomSeed(Number(e.target.value))}
                className="w-20 px-2 py-1 rounded bg-dark-900 border border-slate-700 text-xs text-slate-200 font-mono text-center focus:outline-none focus:border-brand-cyan"
              />
            </div>

            {/* Trigger Button */}
            <button
              onClick={handleStartTraining}
              disabled={isTraining}
              className="w-full py-3 rounded-xl bg-gradient-to-r from-brand-violet to-brand-cyan hover:opacity-95 text-white font-bold text-xs shadow-lg shadow-brand-violet/25 flex items-center justify-center gap-2 transition-all disabled:opacity-50"
            >
              {isTraining ? (
                <>
                  <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                  <span>Training {algorithm.toUpperCase()} Model...</span>
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-current" />
                  <span>Start Background Training</span>
                </>
              )}
            </button>

            {errorMessage && (
              <div className="p-3 rounded-xl bg-rose-950/40 border border-rose-800 text-rose-300 text-xs flex items-center gap-2">
                <AlertCircle className="w-4 h-4 shrink-0" />
                <span>{errorMessage}</span>
              </div>
            )}
          </div>
        </div>

        {/* Right: Real-time Progress & Terminal Logs (7 Cols) */}
        <div className="lg:col-span-7 space-y-4">
          <div className="glass-panel p-6 rounded-2xl space-y-5">
            {/* Live Progress Banner */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div>
                  <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 block">
                    Training Progress
                  </span>
                  <div className="text-lg font-bold text-white flex items-center gap-2">
                    <span>
                      {jobState ? `Epoch ${jobState.current_epoch} of ${jobState.total_epochs}` : "Ready to Train"}
                    </span>
                    {isTraining && (
                      <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-mono bg-brand-cyan/20 text-brand-cyan animate-pulse">
                        LIVE
                      </span>
                    )}
                    {jobState?.status === 'completed' && (
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono bg-emerald-950 text-emerald-400 border border-emerald-800">
                        <CheckCircle className="w-3 h-3" /> Completed
                      </span>
                    )}
                  </div>
                </div>

                <div className="text-right">
                  <span className="text-3xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-brand-cyan to-brand-violet font-mono">
                    {jobState ? `${jobState.progress}%` : "0%"}
                  </span>
                </div>
              </div>

              {/* Progress Bar */}
              <div className="w-full h-3 bg-dark-900 rounded-full overflow-hidden border border-slate-800 p-0.5">
                <div
                  className="h-full bg-gradient-to-r from-brand-violet via-purple-500 to-brand-cyan rounded-full transition-all duration-300 shadow-sm"
                  style={{ width: `${jobState ? jobState.progress : 0}%` }}
                />
              </div>
            </div>

            {/* Live Metrics Row */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div className="p-3 rounded-xl bg-dark-900 border border-slate-800 text-center">
                <span className="text-[10px] uppercase font-semibold text-slate-400 block">Accuracy</span>
                <span className="text-base font-bold text-emerald-400 font-mono">
                  {jobState?.test_metrics ? `${(jobState.test_metrics.accuracy * 100).toFixed(1)}%` : 
                   jobState?.latest_metrics?.accuracy ? `${(jobState.latest_metrics.accuracy * 100).toFixed(1)}%` : '--'}
                </span>
              </div>

              <div className="p-3 rounded-xl bg-dark-900 border border-slate-800 text-center">
                <span className="text-[10px] uppercase font-semibold text-slate-400 block">Loss / Error</span>
                <span className="text-base font-bold text-rose-400 font-mono">
                  {jobState?.latest_metrics?.loss !== undefined ? jobState.latest_metrics.loss : '--'}
                </span>
              </div>

              <div className="p-3 rounded-xl bg-dark-900 border border-slate-800 text-center">
                <span className="text-[10px] uppercase font-semibold text-slate-400 block">Duration</span>
                <span className="text-base font-bold text-brand-cyan font-mono">
                  {jobState?.duration ? `${jobState.duration}s` : isTraining ? 'Running...' : '--'}
                </span>
              </div>

              <div className="p-3 rounded-xl bg-dark-900 border border-slate-800 text-center">
                <span className="text-[10px] uppercase font-semibold text-slate-400 block">Convergence</span>
                <span className="text-base font-bold text-purple-400 font-mono">
                  {jobState?.status === 'completed' ? 'Finished' : isTraining ? 'Optimizing' : 'Idle'}
                </span>
              </div>
            </div>

            {/* Terminal Log Console */}
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs text-slate-400 font-mono">
                <span className="flex items-center gap-1.5">
                  <Terminal className="w-3.5 h-3.5 text-brand-cyan" />
                  Execution Logs
                </span>
                <span>Job ID: {jobState?.job_id || 'None'}</span>
              </div>

              <div className="bg-dark-950 rounded-xl p-4 border border-slate-800 h-64 overflow-y-auto font-mono text-[11px] leading-relaxed text-slate-300 space-y-1 shadow-inner">
                {(!jobState?.logs || jobState.logs.length === 0) ? (
                  <div className="text-slate-500 italic">No active logs. Click 'Start Background Training' to begin.</div>
                ) : (
                  jobState.logs.map((log, index) => (
                    <div key={index} className="flex items-start gap-2">
                      <span className="text-slate-600 select-none">&gt;</span>
                      <span className={log.includes("completed") || log.includes("saved") ? "text-emerald-400" : log.includes("Error") ? "text-rose-400" : "text-slate-300"}>
                        {log}
                      </span>
                    </div>
                  ))
                )}
                <div ref={logsEndRef} />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
