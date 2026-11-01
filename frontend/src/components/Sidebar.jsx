import React from 'react';
import { 
  Home, 
  PenTool, 
  Cpu, 
  BarChart3, 
  BookOpen, 
  Database, 
  Info,
  ChevronRight,
  BrainCircuit
} from 'lucide-react';

const navItems = [
  { id: 'home', label: 'Home Dashboard', icon: Home, badge: null },
  { id: 'recognition', label: 'Digit Recognition', icon: PenTool, badge: 'Live Canvas' },
  { id: 'train', label: 'Train Models', icon: Cpu, badge: 'Async' },
  { id: 'compare', label: 'Model Comparison', icon: BarChart3, badge: 'Study' },
  { id: 'theory', label: 'Learning Theory', icon: BookOpen, badge: 'Sandbox' },
  { id: 'dataset', label: 'Dataset Explorer', icon: Database, badge: 'MNIST' },
  { id: 'about', label: 'About Project', icon: Info, badge: null }
];

export default function Sidebar({ currentTab, setCurrentTab }) {
  return (
    <aside className="w-64 bg-dark-900 border-r border-slate-800/80 flex flex-col justify-between select-none shrink-0 z-30">
      <div>
        {/* Brand Header */}
        <div className="p-5 border-b border-slate-800/80 flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-brand-violet to-brand-cyan flex items-center justify-center shadow-lg shadow-brand-violet/20">
            <BrainCircuit className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-base font-bold text-white tracking-tight leading-tight">
              NeuroDigit <span className="text-brand-cyan text-xs font-mono">28x28</span>
            </h1>
            <p className="text-[11px] text-slate-400 font-mono">Hebb · Perceptron · LMS</p>
          </div>
        </div>

        {/* Navigation Links */}
        <nav className="p-3 space-y-1">
          <div className="px-3 py-2 text-[10px] font-semibold text-slate-300 uppercase tracking-wider">
            Main Application
          </div>
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = currentTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setCurrentTab(item.id)}
                className={`w-full flex items-center justify-between px-3.5 py-2.5 rounded-lg text-sm font-medium transition-all group ${
                  isActive
                    ? 'bg-gradient-to-r from-brand-violet/20 to-brand-cyan/10 text-white border-l-4 border-brand-cyan shadow-sm'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-dark-800/60'
                }`}
              >
                <div className="flex items-center gap-3">
                  <Icon className={`w-4 h-4 transition-colors ${
                    isActive ? 'text-brand-cyan' : 'text-slate-400 group-hover:text-slate-300'
                  }`} />
                  <span>{item.label}</span>
                </div>
                {item.badge ? (
                  <span className={`text-[10px] px-1.5 py-0.5 rounded font-mono ${
                    isActive ? 'bg-brand-cyan/20 text-brand-cyan' : 'bg-dark-800 text-slate-400'
                  }`}>
                    {item.badge}
                  </span>
                ) : (
                  isActive && <ChevronRight className="w-3.5 h-3.5 text-brand-cyan" />
                )}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Footer Info */}
      <div className="p-4 border-t border-slate-800/80 bg-dark-950/40">
        <div className="p-3 rounded-lg bg-dark-850/80 border border-slate-800 text-xs">
          <div className="flex items-center justify-between mb-1">
            <span className="text-slate-300 font-medium">Soft Computing</span>
            <span className="text-[10px] font-mono text-emerald-400 bg-emerald-950/60 px-1.5 py-0.5 rounded border border-emerald-800/60">
              Academic
            </span>
          </div>
          <p className="text-[11px] text-slate-400 leading-relaxed">
            Perceptron, ADALINE & Comparative Learning Rules
          </p>
        </div>
      </div>
    </aside>
  );
}
