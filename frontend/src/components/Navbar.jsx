import React from 'react';
import { Activity, Database, Sparkles, RefreshCw } from 'lucide-react';

export default function Navbar({ activeTitle, subtitle, backendStatus, onRefresh }) {
  return (
    <header className="h-16 border-b border-slate-800/80 bg-dark-900/90 backdrop-blur-md px-6 flex items-center justify-between z-20 shrink-0">
      <div>
        <h2 className="text-base font-semibold text-white tracking-tight flex items-center gap-2">
          {activeTitle}
        </h2>
        {subtitle && <p className="text-xs text-slate-400">{subtitle}</p>}
      </div>

      <div className="flex items-center gap-3">
        {/* Backend Health Badge */}
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-dark-850 border border-slate-800 text-xs">
          <div className="relative flex h-2 w-2">
            {backendStatus === 'online' ? (
              <>
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
              </>
            ) : (
              <span className="relative inline-flex rounded-full h-2 w-2 bg-rose-500"></span>
            )}
          </div>
          <span className="text-slate-300 font-medium">
            {backendStatus === 'online' ? 'Flask API Online' : 'Connecting API...'}
          </span>
          <span className="text-slate-400 font-mono text-[10px]">127.0.0.1:5000</span>
        </div>

        {/* Dataset Quick Tag */}
        <div className="hidden sm:flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-dark-850 border border-slate-800 text-xs text-slate-300">
          <Database className="w-3.5 h-3.5 text-brand-cyan" />
          <span>MNIST 28x28 Grayscale</span>
        </div>

        {/* Refresh Button */}
        {onRefresh && (
          <button
            onClick={onRefresh}
            title="Refresh status"
            className="p-1.5 rounded-lg bg-dark-800 hover:bg-dark-700 text-slate-300 hover:text-white border border-slate-700/60 transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        )}
      </div>
    </header>
  );
}
