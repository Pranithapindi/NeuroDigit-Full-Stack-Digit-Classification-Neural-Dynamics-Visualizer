import React, { useEffect, useState } from 'react';
import { Database, Image, BarChart2, Info, Hash, Layers } from 'lucide-react';
import { fetchDatasetInfo, fetchDatasetSamples } from '../services/api';

export default function DatasetExplorer() {
  const [info, setInfo] = useState(null);
  const [samples, setSamples] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedDigit, setSelectedDigit] = useState(null);

  useEffect(() => {
    async function load() {
      try {
        const [infoRes, samplesRes] = await Promise.all([
          fetchDatasetInfo(),
          fetchDatasetSamples(3),
        ]);
        if (infoRes.success) setInfo(infoRes.data);
        if (samplesRes.success) setSamples(samplesRes.samples);
      } catch (e) {
        console.error(e);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  const digits = Array.from({ length: 10 }, (_, i) => i);
  const filteredSamples = selectedDigit !== null
    ? samples.filter(s => s.digit === selectedDigit)
    : samples;

  const classDistItems = info
    ? Object.entries(info.class_distribution).map(([digit, count]) => ({
        digit: Number(digit),
        count,
        pct: ((count / info.train_size) * 100).toFixed(1)
      }))
    : [];

  const maxCount = classDistItems.length ? Math.max(...classDistItems.map(d => d.count)) : 1;

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-12">
      <div>
        <h1 className="text-2xl font-bold text-white tracking-tight flex items-center gap-2">
          <Database className="w-6 h-6 text-brand-cyan" />
          Dataset Explorer
        </h1>
        <p className="text-xs text-slate-400 mt-1">
          Inspect the handwritten digit training dataset: class distribution, image statistics, and sample previews.
        </p>
      </div>

      {/* Dataset Info Cards */}
      {info && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <div className="glass-panel p-4 rounded-xl">
            <span className="text-[11px] uppercase font-semibold text-slate-400 block mb-1">Training Samples</span>
            <span className="text-2xl font-bold text-white font-mono">{info.train_size.toLocaleString()}</span>
          </div>
          <div className="glass-panel p-4 rounded-xl">
            <span className="text-[11px] uppercase font-semibold text-slate-400 block mb-1">Test Samples</span>
            <span className="text-2xl font-bold text-brand-cyan font-mono">{info.test_size.toLocaleString()}</span>
          </div>
          <div className="glass-panel p-4 rounded-xl">
            <span className="text-[11px] uppercase font-semibold text-slate-400 block mb-1">Image Dimensions</span>
            <span className="text-2xl font-bold text-brand-violet font-mono">28 × 28</span>
          </div>
          <div className="glass-panel p-4 rounded-xl">
            <span className="text-[11px] uppercase font-semibold text-slate-400 block mb-1">Feature Dimensions</span>
            <span className="text-2xl font-bold text-emerald-400 font-mono">784</span>
          </div>
        </div>
      )}

      {/* Info Row */}
      {info && (
        <div className="glass-panel p-4 rounded-xl flex flex-wrap gap-6 text-xs">
          <div>
            <span className="text-slate-400 block">Pixel Range</span>
            <span className="text-slate-200 font-mono font-semibold">[{info.pixel_min.toFixed(2)}, {info.pixel_max.toFixed(2)}]</span>
          </div>
          <div>
            <span className="text-slate-400 block">Mean Intensity</span>
            <span className="text-slate-200 font-mono font-semibold">{info.pixel_mean}</span>
          </div>
          <div>
            <span className="text-slate-400 block">Classes</span>
            <span className="text-slate-200 font-mono font-semibold">{info.num_classes} (Digits 0–9)</span>
          </div>
          <div>
            <span className="text-slate-400 block">Normalization</span>
            <span className="text-slate-200 font-mono font-semibold">Min-Max → [0.0, 1.0]</span>
          </div>
          <div className="flex-1 min-w-[200px]">
            <span className="text-slate-400 block">Source</span>
            <span className="text-slate-200 font-semibold">{info.source}</span>
          </div>
        </div>
      )}

      {/* Class Distribution Bar Chart */}
      {classDistItems.length > 0 && (
        <div className="glass-panel p-6 rounded-2xl space-y-4">
          <h3 className="text-sm font-bold text-white flex items-center gap-2">
            <BarChart2 className="w-4 h-4 text-brand-cyan" />
            Training Set Class Distribution (Digits 0–9)
          </h3>
          <div className="space-y-2">
            {classDistItems.map(({ digit, count, pct }) => (
              <div key={digit} className="flex items-center gap-3 text-xs">
                <span className="w-14 font-mono text-slate-300 text-right shrink-0">Digit {digit}</span>
                <div className="flex-1 h-6 bg-dark-900 rounded-md overflow-hidden border border-slate-800">
                  <div
                    className="h-full bg-gradient-to-r from-brand-violet to-brand-cyan rounded-md transition-all duration-500"
                    style={{ width: `${(count / maxCount) * 100}%` }}
                  />
                </div>
                <span className="w-20 font-mono text-slate-300 shrink-0">
                  {count.toLocaleString()} <span className="text-slate-500">({pct}%)</span>
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Sample Image Gallery */}
      <div className="glass-panel p-6 rounded-2xl space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <h3 className="text-sm font-bold text-white flex items-center gap-2">
            <Image className="w-4 h-4 text-brand-violet" />
            Sample Digit Previews (28 × 28 Normalized)
          </h3>
          <div className="flex flex-wrap gap-1.5">
            <button
              onClick={() => setSelectedDigit(null)}
              className={`px-2.5 py-1 rounded-lg text-[11px] font-semibold transition-all ${
                selectedDigit === null
                  ? 'bg-brand-cyan/20 text-brand-cyan border border-brand-cyan/40'
                  : 'text-slate-400 hover:text-white bg-dark-800 border border-slate-700'
              }`}
            >
              All
            </button>
            {digits.map(d => (
              <button
                key={d}
                onClick={() => setSelectedDigit(d)}
                className={`px-2.5 py-1 rounded-lg text-[11px] font-semibold font-mono transition-all ${
                  selectedDigit === d
                    ? 'bg-brand-violet/20 text-brand-violet border border-brand-violet/40'
                    : 'text-slate-400 hover:text-white bg-dark-800 border border-slate-700'
                }`}
              >
                {d}
              </button>
            ))}
          </div>
        </div>

        {loading ? (
          <div className="grid grid-cols-5 sm:grid-cols-10 gap-3">
            {Array.from({ length: 20 }).map((_, i) => (
              <div key={i} className="aspect-square bg-dark-900 rounded-lg animate-pulse border border-slate-800" />
            ))}
          </div>
        ) : (
          <div className="grid grid-cols-5 sm:grid-cols-10 gap-3">
            {filteredSamples.map((sample, idx) => (
              <div
                key={idx}
                className="group flex flex-col items-center gap-1.5"
                title={`Digit ${sample.digit} | Mean: ${sample.mean_intensity} | Max: ${sample.max_intensity}`}
              >
                <div className="relative p-1 bg-black rounded-lg border border-slate-800 group-hover:border-brand-cyan/60 transition-colors shadow-sm">
                  <img
                    src={sample.image_b64}
                    alt={`Digit ${sample.digit}`}
                    className="w-10 h-10 image-pixelated"
                    style={{ imageRendering: 'pixelated' }}
                  />
                </div>
                <span className="text-[10px] font-bold font-mono text-slate-400 group-hover:text-brand-cyan transition-colors">
                  {sample.digit}
                </span>
              </div>
            ))}
          </div>
        )}

        <p className="text-[11px] text-slate-400 mt-2">
          <Info className="w-3 h-3 inline mr-1 text-brand-cyan" />
          Each preview is the actual 28 × 28 preprocessed grayscale image used for training. Pixel values normalized to [0.0, 1.0].
        </p>
      </div>
    </div>
  );
}
