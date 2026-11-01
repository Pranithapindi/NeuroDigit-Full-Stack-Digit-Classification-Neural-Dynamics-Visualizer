import React, { useRef, useState, useEffect } from 'react';
import { 
  PenTool, 
  RotateCcw, 
  Trash2, 
  Upload, 
  Play, 
  Sparkles, 
  AlertCircle, 
  Cpu, 
  HelpCircle,
  Sliders,
  Check
} from 'lucide-react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  Tooltip, 
  ResponsiveContainer, 
  Cell 
} from 'recharts';
import { predictDigitFromBase64, predictDigitFromFile, fetchModels } from '../services/api';

export default function DigitRecognition() {
  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [brushSize, setBrushSize] = useState(24);
  const [history, setHistory] = useState([]);
  const [selectedModel, setSelectedModel] = useState('perceptron_model');
  const [availableModels, setAvailableModels] = useState([]);
  const [isPredicting, setIsPredicting] = useState(false);
  const [predictionResult, setPredictionResult] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [inputMode, setInputMode] = useState('canvas'); // 'canvas' or 'upload'

  // Initialize canvas
  useEffect(() => {
    initCanvas();
  }, []);

  const initCanvas = () => {
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#000000';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      saveCanvasState();
    }
  };

  // Fetch available models
  useEffect(() => {
    async function loadModels() {
      try {
        const res = await fetchModels();
        if (res.success && res.models.length > 0) {
          setAvailableModels(res.models);
          // Set perceptron as default if present
          const hasPerceptron = res.models.some(m => m.model_id === 'perceptron_model');
          if (hasPerceptron) {
            setSelectedModel('perceptron_model');
          } else {
            setSelectedModel(res.models[0].model_id);
          }
        }
      } catch (err) {
        console.error("Failed to load models:", err);
      }
    }
    loadModels();
  }, []);

  const saveCanvasState = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    setHistory(prev => [...prev.slice(-10), imageData]); // keep last 10
  };

  // Convert viewport client coordinates to exact canvas pixel coordinates
  const getCanvasCoords = (clientX, clientY) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
      x: (clientX - rect.left) * scaleX,
      y: (clientY - rect.top) * scaleY
    };
  };

  const startDrawing = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const { x, y } = getCanvasCoords(e.clientX, e.clientY);
    
    const ctx = canvas.getContext('2d');
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = brushSize;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x, y);
    ctx.stroke();

    setIsDrawing(true);
    setErrorMessage(null);
  };

  const draw = (e) => {
    if (!isDrawing) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const { x, y } = getCanvasCoords(e.clientX, e.clientY);

    const ctx = canvas.getContext('2d');
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = brushSize;
    ctx.lineTo(x, y);
    ctx.stroke();
  };

  const stopDrawing = () => {
    if (isDrawing) {
      setIsDrawing(false);
      saveCanvasState();
    }
  };

  // Touch support for tablets/mobile
  const handleTouchStart = (e) => {
    e.preventDefault();
    const touch = e.touches[0];
    const canvas = canvasRef.current;
    if (!canvas) return;
    const { x, y } = getCanvasCoords(touch.clientX, touch.clientY);

    const ctx = canvas.getContext('2d');
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = brushSize;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x, y);
    ctx.stroke();

    setIsDrawing(true);
    setErrorMessage(null);
  };

  const handleTouchMove = (e) => {
    e.preventDefault();
    if (!isDrawing) return;
    const touch = e.touches[0];
    const canvas = canvasRef.current;
    if (!canvas) return;
    const { x, y } = getCanvasCoords(touch.clientX, touch.clientY);

    const ctx = canvas.getContext('2d');
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = brushSize;
    ctx.lineTo(x, y);
    ctx.stroke();
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    setHistory([]);
    saveCanvasState();
    setPredictionResult(null);
    setErrorMessage(null);
  };

  // Quick preset samples for instant testing
  const drawSamplePreset = (digit) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = 24;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    ctx.beginPath();
    if (digit === 0) {
      ctx.ellipse(140, 140, 48, 70, 0, 0, Math.PI * 2);
    } else if (digit === 1) {
      ctx.moveTo(115, 80);
      ctx.lineTo(140, 55);
      ctx.lineTo(140, 225);
      ctx.moveTo(105, 225);
      ctx.lineTo(175, 225);
    } else if (digit === 2) {
      // Natural handwritten 2 with rounded upper loop, diagonal neck, and horizontal base
      ctx.moveTo(95, 90);
      ctx.bezierCurveTo(95, 60, 185, 60, 185, 100);
      ctx.bezierCurveTo(185, 135, 140, 175, 100, 215);
      ctx.lineTo(190, 215);
    } else if (digit === 3) {
      ctx.arc(140, 105, 42, -Math.PI * 0.7, Math.PI * 0.45);
      ctx.arc(140, 175, 46, -Math.PI * 0.45, Math.PI * 0.75);
    } else if (digit === 4) {
      ctx.moveTo(165, 55);
      ctx.lineTo(95, 175);
      ctx.lineTo(185, 175);
      ctx.moveTo(165, 55);
      ctx.lineTo(165, 225);
    } else if (digit === 5) {
      // Natural handwritten 5 with wide top bar, left stem, and rounded lower belly
      ctx.moveTo(185, 65);
      ctx.lineTo(105, 65);
      ctx.lineTo(102, 135);
      ctx.bezierCurveTo(135, 120, 185, 135, 185, 175);
      ctx.bezierCurveTo(185, 215, 130, 225, 90, 210);
    } else if (digit === 6) {
      ctx.arc(140, 170, 44, 0, Math.PI * 2);
      ctx.moveTo(96, 170);
      ctx.quadraticCurveTo(100, 65, 160, 60);
    } else if (digit === 7) {
      // Clean MNIST 7: top horizontal bar and steep diagonal stroke
      ctx.moveTo(85, 65);
      ctx.lineTo(195, 65);
      ctx.lineTo(130, 225);
    } else if (digit === 8) {
      ctx.arc(140, 105, 36, 0, Math.PI * 2);
      ctx.moveTo(182, 175);
      ctx.arc(140, 175, 42, 0, Math.PI * 2);
    } else if (digit === 9) {
      ctx.arc(140, 110, 44, 0, Math.PI * 2);
      ctx.moveTo(184, 110);
      ctx.quadraticCurveTo(180, 215, 120, 225);
    }
    ctx.stroke();
    saveCanvasState();
    setPredictionResult(null);
    setErrorMessage(null);
  };

  const undoLastStroke = () => {
    const canvas = canvasRef.current;
    if (!canvas || history.length <= 1) return;
    const newHistory = [...history];
    newHistory.pop(); // remove current
    const prevState = newHistory[newHistory.length - 1];
    const ctx = canvas.getContext('2d');
    ctx.putImageData(prevState, 0, 0);
    setHistory(newHistory);
  };

  const handlePredict = async () => {
    setIsPredicting(true);
    setErrorMessage(null);
    try {
      let res;
      if (inputMode === 'canvas') {
        const canvas = canvasRef.current;
        const base64Data = canvas.toDataURL('image/png');
        res = await predictDigitFromBase64(selectedModel, base64Data);
      } else {
        if (!uploadedFile) {
          throw new Error("Please select an image file to upload first.");
        }
        res = await predictDigitFromFile(selectedModel, uploadedFile);
      }

      if (res.success) {
        setPredictionResult(res.result);
      } else {
        setErrorMessage(res.error || "Prediction failed.");
      }
    } catch (err) {
      setErrorMessage(err.message || "An unexpected error occurred during prediction.");
    } finally {
      setIsPredicting(false);
    }
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (!file.type.startsWith('image/')) {
        setErrorMessage("Please upload an image file (PNG, JPG, JPEG).");
        return;
      }
      setUploadedFile(file);
      setErrorMessage(null);
      setPredictionResult(null);
    }
  };

  // Format chart data for 0-9 decision scores
  const chartData = predictionResult ? predictionResult.normalized_scores.map((score, digit) => ({
    digit: `Digit ${digit}`,
    digitNum: digit,
    normalizedScore: Number((score * 100).toFixed(1)),
    rawScore: predictionResult.raw_scores[digit]
  })) : [];

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-12">
      {/* Header Info */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight flex items-center gap-2">
            <PenTool className="w-6 h-6 text-brand-cyan" />
            Handwritten Digit Recognition
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Draw a digit (0–9) or upload an image. The system centers, aspect-ratio resizes to 28 × 28, and classifies it.
          </p>
        </div>

        {/* Model Selector Bar */}
        <div className="flex items-center gap-3 bg-dark-900 border border-slate-800 p-2 rounded-xl">
          <Cpu className="w-4 h-4 text-brand-violet ml-2" />
          <span className="text-xs text-slate-300 font-medium">Model:</span>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="bg-dark-800 text-slate-200 text-xs font-semibold px-3 py-1.5 rounded-lg border border-slate-700 focus:outline-none focus:border-brand-violet"
          >
            {availableModels.map((m) => (
              <option key={m.model_id} value={m.model_id}>
                {m.display_name} {m.accuracy ? `(${Math.round(m.accuracy * 100)}% Acc)` : ''}
              </option>
            ))}
            {availableModels.length === 0 && (
              <>
                <option value="perceptron_model">Perceptron Classifier (OvR)</option>
                <option value="adaline_model">ADALINE Classifier (Delta LMS)</option>
                <option value="hebb_model">Hebbian Classifier</option>
              </>
            )}
          </select>
        </div>
      </div>

      {/* Main Grid: Input Canvas (Left) + Output Predictions (Right) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Drawing Canvas / Upload (7 Cols) */}
        <div className="lg:col-span-7 space-y-4">
          <div className="glass-panel p-6 rounded-2xl space-y-5">
            {/* Mode Switch Tabs */}
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div className="flex items-center gap-2">
                <button
                  onClick={() => { setInputMode('canvas'); setErrorMessage(null); }}
                  className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                    inputMode === 'canvas'
                      ? 'bg-brand-cyan/20 text-brand-cyan border border-brand-cyan/40'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-dark-800'
                  }`}
                >
                  Interactive Drawing Canvas
                </button>
                <button
                  onClick={() => { setInputMode('upload'); setErrorMessage(null); }}
                  className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                    inputMode === 'upload'
                      ? 'bg-brand-violet/20 text-brand-violet border border-brand-violet/40'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-dark-800'
                  }`}
                >
                  Upload Image File
                </button>
              </div>

              {inputMode === 'canvas' && (
                <div className="flex items-center gap-2">
                  <span className="text-[11px] text-slate-400 font-mono">Pen:</span>
                  <input
                    type="range"
                    min="16"
                    max="32"
                    value={brushSize}
                    onChange={(e) => setBrushSize(Number(e.target.value))}
                    className="w-20 accent-brand-cyan cursor-pointer"
                    title={`Brush size: ${brushSize}px`}
                  />
                  <span className="text-[11px] text-slate-300 font-mono w-6 text-right">{brushSize}px</span>
                </div>
              )}
            </div>

            {/* Quick Test Presets Bar */}
            {inputMode === 'canvas' && (
              <div className="bg-dark-900/70 border border-slate-800/80 rounded-xl px-3.5 py-2.5 flex items-center justify-between gap-2 flex-wrap">
                <span className="text-[11px] text-slate-400 font-medium flex items-center gap-1.5">
                  <Sparkles className="w-3.5 h-3.5 text-brand-cyan" />
                  Quick Presets:
                </span>
                <div className="flex items-center gap-1.5 flex-wrap">
                  {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map((digit) => (
                    <button
                      key={digit}
                      onClick={() => drawSamplePreset(digit)}
                      className="w-7 h-7 rounded-lg bg-dark-800 hover:bg-brand-violet/20 hover:text-brand-cyan hover:border-brand-cyan/40 border border-slate-700/60 text-slate-300 text-xs font-mono font-bold transition-all"
                      title={`Draw sample digit ${digit}`}
                    >
                      {digit}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Canvas Area (Kept permanently in DOM to prevent context reset) */}
            <div className={inputMode === 'canvas' ? 'flex flex-col items-center' : 'hidden'}>
              <div className="relative p-2 bg-dark-900 rounded-2xl border-2 border-slate-700 shadow-inner group">
                <canvas
                  ref={canvasRef}
                  width={280}
                  height={280}
                  onMouseDown={startDrawing}
                  onMouseMove={draw}
                  onMouseUp={stopDrawing}
                  onMouseLeave={stopDrawing}
                  onTouchStart={handleTouchStart}
                  onTouchMove={handleTouchMove}
                  onTouchEnd={stopDrawing}
                  className="rounded-xl cursor-crosshair touch-none bg-black"
                />
                <div className="absolute bottom-4 right-4 pointer-events-none opacity-40 group-hover:opacity-75 transition-opacity text-[10px] text-slate-400 font-mono bg-dark-950/80 px-2 py-0.5 rounded">
                  280 × 280
                </div>
              </div>

              <p className="text-[11px] text-slate-400 mt-2 text-center">
                Draw a digit (0–9) centered on the canvas, or click any <span className="text-brand-cyan font-semibold">Quick Preset</span> above to test instantly.
              </p>

              {/* Canvas Controls */}
              <div className="flex items-center gap-2 mt-4 w-full max-w-[340px]">
                <button
                  onClick={undoLastStroke}
                  disabled={history.length <= 1}
                  className="px-3.5 py-2 rounded-xl bg-dark-800 hover:bg-dark-750 disabled:opacity-40 text-slate-300 text-xs font-medium border border-slate-700/80 flex items-center gap-1.5 transition-colors"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                  Undo
                </button>
                <button
                  onClick={clearCanvas}
                  className="px-3.5 py-2 rounded-xl bg-rose-950/30 hover:bg-rose-900/40 text-rose-300 text-xs font-medium border border-rose-800/50 flex items-center gap-1.5 transition-colors"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  Clear
                </button>
                <button
                  onClick={handlePredict}
                  disabled={isPredicting}
                  className="ml-auto px-6 py-2 rounded-xl bg-gradient-to-r from-brand-violet to-brand-cyan text-white text-xs font-bold shadow-lg shadow-brand-violet/25 hover:opacity-95 transition-all flex items-center gap-2 disabled:opacity-50"
                >
                  {isPredicting ? (
                    <>
                      <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                      <span>Classifying...</span>
                    </>
                  ) : (
                    <>
                      <Play className="w-3.5 h-3.5 fill-current" />
                      <span>Classify Digit</span>
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* File Upload Area */}
            <div className={inputMode === 'upload' ? 'space-y-4' : 'hidden'}>
                <label className="border-2 border-dashed border-slate-700 hover:border-brand-violet/60 rounded-2xl p-8 flex flex-col items-center justify-center cursor-pointer bg-dark-900/50 hover:bg-dark-900 transition-all group">
                  <Upload className="w-10 h-10 text-slate-500 group-hover:text-brand-violet mb-3 transition-colors" />
                  <span className="text-sm font-semibold text-slate-200">
                    {uploadedFile ? uploadedFile.name : "Click to select or drag handwritten image"}
                  </span>
                  <span className="text-xs text-slate-400 mt-1">
                    Supports PNG, JPG, JPEG (Automatically inverted & centered)
                  </span>
                  <input
                    type="file"
                    accept="image/png, image/jpeg, image/jpg"
                    onChange={handleFileUpload}
                    className="hidden"
                  />
                </label>

                {uploadedFile && (
                  <div className="flex items-center justify-between pt-2">
                    <span className="text-xs text-slate-400 font-mono">
                      File: {uploadedFile.name} ({(uploadedFile.size / 1024).toFixed(1)} KB)
                    </span>
                    <button
                      onClick={handlePredict}
                      disabled={isPredicting}
                      className="px-6 py-2.5 rounded-xl bg-gradient-to-r from-brand-violet to-brand-cyan text-white text-xs font-bold shadow-lg shadow-brand-violet/25 hover:opacity-95 transition-all flex items-center gap-2"
                    >
                      {isPredicting ? (
                        <>
                          <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                          <span>Classifying...</span>
                        </>
                      ) : (
                        <>
                          <Play className="w-3.5 h-3.5 fill-current" />
                          <span>Classify Uploaded Image</span>
                        </>
                      )}
                    </button>
                  </div>
                )}
              </div>

            {/* Error Message Alert */}
            {errorMessage && (
              <div className="p-3.5 rounded-xl bg-rose-950/40 border border-rose-800/80 text-rose-300 text-xs flex items-center gap-2">
                <AlertCircle className="w-4 h-4 shrink-0" />
                <span>{errorMessage}</span>
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Prediction Results & Charts (5 Cols) */}
        <div className="lg:col-span-5 space-y-4">
          {predictionResult ? (
            <div className="glass-panel p-6 rounded-2xl space-y-5 animate-fadeIn">
              {/* Preprocessing Preview & Predicted Digit Badge */}
              <div className="flex items-center justify-between bg-dark-900/80 p-4 rounded-xl border border-slate-800">
                <div>
                  <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 block mb-1">
                    Predicted Class
                  </span>
                  <div className="flex items-baseline gap-2">
                    <span className="text-5xl font-black text-transparent bg-clip-text bg-gradient-to-r from-brand-cyan to-brand-violet font-mono">
                      {predictionResult.predicted_digit}
                    </span>
                    <span className="text-xs text-emerald-400 font-mono font-medium">
                      {Math.round(predictionResult.top_3[0].normalized_score * 100)}% norm score
                    </span>
                  </div>
                  <div className="text-[11px] text-slate-400 mt-1 font-mono">
                    Model: {predictionResult.algorithm}
                  </div>
                </div>

                {/* 28x28 Normalized Preprocessing Preview */}
                <div className="text-center">
                  <span className="text-[10px] text-slate-400 block mb-1 font-mono">
                    Preprocessed (28×28)
                  </span>
                  <div className="p-1 bg-black rounded-lg border border-slate-700 inline-block shadow-md">
                    <img
                      src={predictionResult.preview_image_b64}
                      alt="Preprocessed 28x28"
                      className="w-16 h-16 image-pixelated rounded"
                    />
                  </div>
                  <span className="text-[9px] text-slate-400 block mt-0.5">Centered & Scaled</span>
                </div>
              </div>

              {/* Top 3 Candidates Bar */}
              <div className="space-y-2">
                <span className="text-xs font-semibold text-slate-300 block">
                  Top 3 Candidate Digits
                </span>
                <div className="grid grid-cols-3 gap-2">
                  {predictionResult.top_3.map((cand, idx) => (
                    <div
                      key={cand.digit}
                      className={`p-3 rounded-xl border text-center ${
                        idx === 0
                          ? 'bg-brand-violet/15 border-brand-violet/40 text-white'
                          : 'bg-dark-900 border-slate-800 text-slate-300'
                      }`}
                    >
                      <div className="text-xs text-slate-400 font-mono">#{idx + 1} Candidate</div>
                      <div className="text-2xl font-bold font-mono my-0.5">Digit {cand.digit}</div>
                      <div className="text-[11px] font-mono text-brand-cyan">
                        {(cand.normalized_score * 100).toFixed(1)}%
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Class-wise Decision Scores Bar Chart (0–9) */}
              <div className="space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-semibold text-slate-300">
                    Decision Scores (Digits 0–9)
                  </span>
                  <span className="text-[10px] text-slate-400 font-mono">Normalized Softmax</span>
                </div>
                <div className="h-44 w-full bg-dark-900/60 rounded-xl p-2 border border-slate-800">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={chartData} margin={{ top: 10, right: 10, left: -25, bottom: 0 }}>
                      <XAxis dataKey="digitNum" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                      <YAxis tick={{ fill: '#94a3b8', fontSize: 10 }} domain={[0, 100]} />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '11px' }}
                        formatter={(val, name, item) => [`${val}% (Raw: ${item.payload.rawScore})`, 'Score']}
                      />
                      <Bar dataKey="normalizedScore" radius={[4, 4, 0, 0]}>
                        {chartData.map((entry, index) => (
                          <Cell
                            key={`cell-${index}`}
                            fill={entry.digitNum === predictionResult.predicted_digit ? '#06b6d4' : '#6366f1'}
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Educational Disclaimer */}
              <div className="p-3 rounded-lg bg-dark-900 border border-slate-800 text-[11px] text-slate-400 leading-relaxed flex items-start gap-2">
                <HelpCircle className="w-4 h-4 text-brand-violet shrink-0 mt-0.5" />
                <span>
                  <strong className="text-slate-300">Academic Note:</strong> Scores are linear decision outputs <code className="text-purple-300">z = x · W + b</code> normalized for comparative visualization. As single-layer linear models, they do not output true Bayesian probabilities.
                </span>
              </div>
            </div>
          ) : (
            /* Empty State */
            <div className="glass-panel p-8 rounded-2xl flex flex-col items-center justify-center text-center h-full min-h-[360px] border-dashed">
              <div className="w-14 h-14 rounded-2xl bg-dark-800 border border-slate-700/80 flex items-center justify-center text-slate-500 mb-4">
                <PenTool className="w-7 h-7" />
              </div>
              <h3 className="text-base font-bold text-slate-200 mb-1">
                Awaiting Handwritten Input
              </h3>
              <p className="text-xs text-slate-400 max-w-xs leading-relaxed">
                Draw a digit (0 to 9) on the canvas to the left or upload an image, then click <strong className="text-brand-cyan">Classify Digit</strong>.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
