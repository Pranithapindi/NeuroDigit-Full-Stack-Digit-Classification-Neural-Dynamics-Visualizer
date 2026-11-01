import React, { useState } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { BookOpen, BrainCircuit, Layers, Calculator, CheckCircle2, Zap, Target, ChevronDown, ChevronUp, Play, RotateCcw } from "lucide-react";

const TABS = [
  { id: "overview",  label: "Neural Basics",    icon: BrainCircuit },
  { id: "rules",     label: "Learning Rules",   icon: Layers },
  { id: "sandbox",   label: "Live Sandbox",     icon: Calculator },
  { id: "compare",   label: "Comparison Table", icon: CheckCircle2 },
];

function simulateHistory(rule, x1, x2, w1Init, w2Init, bInit, target, lr, steps) {
  let w1 = w1Init, w2 = w2Init, b = bInit;
  const history = [{ step: 0, w1: +w1.toFixed(4), w2: +w2.toFixed(4), b: +b.toFixed(4), error: +(target - (x1*w1+x2*w2+b)).toFixed(4) }];
  for (let i = 1; i <= steps; i++) {
    const z = x1 * w1 + x2 * w2 + b;
    if (rule === "hebb") {
      w1 += lr * x1 * target;
      w2 += lr * x2 * target;
      b  += lr * target;
    } else if (rule === "perceptron") {
      const yhat = z >= 0 ? 1 : -1;
      const err = target - yhat;
      w1 += lr * err * x1;
      w2 += lr * err * x2;
      b  += lr * err;
    } else {
      const err = target - z;
      w1 += lr * err * x1;
      w2 += lr * err * x2;
      b  += lr * err;
    }
    const newZ = x1 * w1 + x2 * w2 + b;
    history.push({ step: i, w1: +w1.toFixed(4), w2: +w2.toFixed(4), b: +b.toFixed(4), error: +(target - newZ).toFixed(4) });
  }
  return history;
}

function NeuronDiagram({ x1, x2, w1, w2, bias, z, yhat }) {
  const cx = 240, cy = 110, r = 30;
  const inputs = [
    { lx: 20, ly: 55,  label: "x1="+x1, wLabel: "w1="+w1, color: "#06b6d4" },
    { lx: 20, ly: 110, label: "x2="+x2, wLabel: "w2="+w2, color: "#a855f7" },
    { lx: 20, ly: 165, label: "b="+bias, wLabel: "",        color: "#f59e0b" },
  ];
  return (
    <svg viewBox="0 0 400 220" className="w-full max-w-lg mx-auto">
      {inputs.map((inp, i) => (
        <g key={i}>
          <circle cx={inp.lx+18} cy={inp.ly} r={18} fill="#0d1117" stroke={inp.color} strokeWidth={2}/>
          <text x={inp.lx+18} y={inp.ly+4} textAnchor="middle" fill={inp.color} fontSize={8} fontFamily="monospace">{inp.label}</text>
          <line x1={inp.lx+36} y1={inp.ly} x2={cx-r} y2={cy} stroke={inp.color} strokeWidth={1.5} strokeOpacity={0.7}/>
          {inp.wLabel && (
            <text x={(inp.lx+36+cx-r)/2} y={(inp.ly+cy)/2-6} textAnchor="middle" fill={inp.color} fontSize={8} fontFamily="monospace" opacity={0.9}>{inp.wLabel}</text>
          )}
        </g>
      ))}
      <circle cx={cx} cy={cy} r={r} fill="#131822" stroke="#06b6d4" strokeWidth={2}/>
      <text x={cx} y={cy-5} textAnchor="middle" fill="white" fontSize={14} fontFamily="monospace">&#x3A3;</text>
      <text x={cx} y={cy+12} textAnchor="middle" fill="#94a3b8" fontSize={8} fontFamily="monospace">z={z}</text>
      <line x1={cx+r} y1={cy} x2={cx+r+45} y2={cy} stroke="#22d3ee" strokeWidth={2.5}/>
      <polygon points={`${cx+r+45},${cy-5} ${cx+r+58},${cy} ${cx+r+45},${cy+5}`} fill="#22d3ee"/>
      <circle cx={cx+r+80} cy={cy} r={22} fill="#07090e" stroke="#22d3ee" strokeWidth={2}/>
      <text x={cx+r+80} y={cy-4} textAnchor="middle" fill="#22d3ee" fontSize={9} fontFamily="monospace">sgn(z)</text>
      <text x={cx+r+80} y={cy+11} textAnchor="middle" fill={yhat>=0?"#34d399":"#f87171"} fontSize={11} fontWeight="bold" fontFamily="monospace">{yhat>=0?"+1":"-1"}</text>
    </svg>
  );
}

export default function LearningTheory() {
  const [activeTab, setActiveTab] = useState("overview");
  const [x1, setX1]     = useState(0.8);
  const [x2, setX2]     = useState(0.4);
  const [w1, setW1]     = useState(0.2);
  const [w2, setW2]     = useState(-0.3);
  const [bias, setBias] = useState(0.1);
  const [target, setTarget] = useState(1);
  const [lr, setLr]     = useState(0.1);
  const [steps, setSteps] = useState(15);
  const [rule, setRule] = useState("adaline");
  const [history, setHistory] = useState([]);
  const [hasRun, setHasRun] = useState(false);
  const [openFaq, setOpenFaq] = useState(null);

  const z         = +(x1*w1 + x2*w2 + bias).toFixed(4);
  const yhat_perc = z >= 0 ? 1 : -1;
  const perc_err  = target - yhat_perc;
  const adal_err  = +(target - z).toFixed(4);

  const runSimulation = () => {
    setHistory(simulateHistory(rule, x1, x2, w1, w2, bias, target, lr, steps));
    setHasRun(true);
  };

  const faqs = [
    { q: "Difference between Perceptron and ADALINE error signals?",
      a: "Perceptron computes error AFTER the hard step activation (discrete: -2, 0, or +2). ADALINE computes error BEFORE thresholding on the continuous linear output z. This makes ADALINE's gradient smoother and convergence more stable on non-separable data." },
    { q: "Why does Hebbian learning fail on digits without normalization?",
      a: "Hebb's rule grows weights unboundedly. Digits with dense pixel patterns (0, 3, 8) dominate the weight matrix. Without L2 normalization of each class weight vector, cosine similarity degrades to biased dot products favoring high-magnitude classes." },
    { q: "What does linearly separable mean?",
      a: "Two classes are linearly separable if a single hyperplane can perfectly separate them. Rosenblatt's Perceptron Convergence Theorem guarantees the perceptron finds this hyperplane in finite steps if such a hyperplane exists." },
    { q: "Why does ADALINE use MSE instead of Cross-Entropy?",
      a: "MSE on the linear output forms a convex quadratic loss bowl, making gradient descent guaranteed to converge to the global minimum. Cross-entropy assumes probabilistic softmax outputs, which ADALINE does not produce." },
  ];

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-16">
      <div className="glass-panel p-5 rounded-2xl flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-brand-violet to-brand-cyan flex items-center justify-center shadow-lg">
            <BookOpen className="w-5 h-5 text-white"/>
          </div>
          <div>
            <h1 className="text-xl font-bold text-white tracking-tight">Educational Learning Theory</h1>
            <p className="text-xs text-slate-400 mt-0.5">Hebb · Perceptron · Delta/LMS — Mathematical foundations with live sandbox</p>
          </div>
        </div>
        <div className="flex gap-2 flex-wrap">
          {["Hebb 1949","Rosenblatt 1958","Widrow-Hoff 1960"].map(b => (
            <span key={b} className="text-[10px] px-2 py-1 rounded-full bg-brand-violet/15 text-brand-violet border border-brand-violet/30 font-mono">{b}</span>
          ))}
        </div>
      </div>

      <div className="flex gap-2 flex-wrap">
        {TABS.map(tab => {
          const Icon = tab.icon;
          const active = activeTab === tab.id;
          return (
            <button key={tab.id} onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium transition-all ${
                active ? "bg-gradient-to-r from-brand-violet/30 to-brand-cyan/20 text-white border border-brand-cyan/40 shadow-sm"
                       : "bg-dark-900 text-slate-400 border border-slate-800 hover:text-slate-200 hover:border-slate-600"}`}>
              <Icon className={`w-4 h-4 ${active ? "text-brand-cyan" : "text-slate-500"}`}/>
              {tab.label}
            </button>
          );
        })}
      </div>

      {activeTab === "overview" && (
        <div className="space-y-6">
          <div className="glass-panel p-6 rounded-2xl">
            <h2 className="text-base font-bold text-white flex items-center gap-2 mb-4 pb-3 border-b border-slate-800">
              <BrainCircuit className="w-5 h-5 text-purple-400"/> Biological to Artificial Neuron
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-xs text-slate-300 leading-relaxed">
              <div className="p-4 rounded-xl bg-dark-900 border border-slate-800 space-y-2">
                <p className="text-[11px] text-brand-cyan font-semibold uppercase tracking-wider mb-2">Biological Neuron</p>
                {[["Dendrites","Receive incoming signals from other neurons"],
                  ["Soma","Integrates and sums electrical potentials"],
                  ["Axon","Transmits output signal to next neurons"],
                  ["Synapse","Controls signal strength between neurons"],
                  ["Action Potential","Fires when membrane threshold exceeded"]].map(([t,d]) => (
                  <div key={t} className="flex gap-2">
                    <span className="text-purple-300 font-semibold w-36 shrink-0">{t}:</span>
                    <span className="text-slate-400">{d}</span>
                  </div>
                ))}
              </div>
              <div className="space-y-3">
                <div className="p-4 rounded-xl bg-dark-900 border border-slate-800 space-y-2">
                  <p className="text-[11px] text-brand-cyan font-semibold uppercase tracking-wider mb-2">Artificial Neuron (McCulloch-Pitts, 1943)</p>
                  {[["Inputs xi","784 pixel intensities (28x28 MNIST image)"],
                    ["Weights wi","Adaptive synaptic strength — learned parameters"],
                    ["Bias b","Threshold offset — shifts the decision boundary"],
                    ["Summation z","z = sum(xi * wi) + b (dot product)"],
                    ["Activation f(z)","sgn(z) step | z linear | sigmoid"]].map(([t,d]) => (
                    <div key={t} className="flex gap-2">
                      <span className="text-brand-cyan font-semibold w-36 shrink-0">{t}:</span>
                      <span className="text-slate-400">{d}</span>
                    </div>
                  ))}
                </div>
                <div className="p-4 rounded-xl bg-dark-950 border border-brand-cyan/20 font-mono text-center space-y-2">
                  <div className="text-emerald-300 font-bold">z = x . w + b</div>
                  <div className="text-brand-cyan font-bold">y_hat = f(z)</div>
                </div>
              </div>
            </div>
          </div>

          <div className="glass-panel p-6 rounded-2xl">
            <h2 className="text-base font-bold text-white flex items-center gap-2 mb-3 pb-3 border-b border-slate-800">
              <Zap className="w-5 h-5 text-brand-cyan"/> Live Neuron Architecture Diagram
            </h2>
            <p className="text-xs text-slate-400 mb-3">Adjust inputs and weights in the <span className="text-brand-cyan font-semibold">Live Sandbox</span> tab — diagram values update automatically.</p>
            <NeuronDiagram x1={x1} x2={x2} w1={w1} w2={w2} bias={bias} z={z} yhat={yhat_perc}/>
            <div className="grid grid-cols-3 gap-3 mt-4 text-xs text-center font-mono">
              {[["Linear Output z", z, "text-brand-cyan"],
                ["Step Activation", yhat_perc>=0?"+1":"-1", yhat_perc>=0?"text-emerald-400":"text-rose-400"],
                ["Target t", target>=0?"+1":"-1","text-amber-400"]].map(([label,val,color]) => (
                <div key={label} className="p-3 rounded-xl bg-dark-900 border border-slate-800">
                  <div className="text-slate-400 text-[10px] mb-1">{label}</div>
                  <div className={`font-bold text-base ${color}`}>{val}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="glass-panel p-6 rounded-2xl space-y-3">
            <h2 className="text-base font-bold text-white flex items-center gap-2 mb-2 pb-3 border-b border-slate-800">
              <BookOpen className="w-5 h-5 text-amber-400"/> Key Concepts FAQ
            </h2>
            {faqs.map((faq, i) => (
              <div key={i} className="rounded-xl bg-dark-900 border border-slate-800 overflow-hidden">
                <button onClick={() => setOpenFaq(openFaq===i?null:i)}
                  className="w-full flex items-center justify-between px-4 py-3 text-left text-xs font-semibold text-slate-200 hover:text-white hover:bg-dark-800 transition-colors">
                  <span>{faq.q}</span>
                  {openFaq===i ? <ChevronUp className="w-4 h-4 text-brand-cyan shrink-0"/> : <ChevronDown className="w-4 h-4 text-slate-400 shrink-0"/>}
                </button>
                {openFaq===i && (
                  <div className="px-4 pb-3 text-xs text-slate-400 leading-relaxed border-t border-slate-800 pt-3">{faq.a}</div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {activeTab === "rules" && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
            {[
              { name:"A. Hebbian Learning", year:"1949", color:"text-purple-400", bg:"border-purple-900/40", hl:"bg-purple-950/30 border-purple-900/30 text-purple-200",
                formula:["Dw = eta * x * t","Db = eta * t"],
                pts:[["Principle",'"Neurons that fire together, wire together." (Donald Hebb, 1949)'],
                     ["Mechanism","Purely associative. Updates weights by correlating input x with target t, regardless of current output."],
                     ["Limitation","No error feedback. Weights grow unboundedly. Cross-talk occurs when digit patterns overlap."],
                     ["Application","Used as associative memory / correlation-based pattern storage in OvR digit recognition."]],
                insight:"Hebb updates always happen regardless of whether the prediction is correct. Correlation drives the update." },
              { name:"B. Perceptron Rule", year:"1958", color:"text-emerald-400", bg:"border-emerald-900/40", hl:"bg-emerald-950/30 border-emerald-900/30 text-emerald-200",
                formula:["Dw = eta * (t - y_hat) * x","y_hat = sgn(z)"],
                pts:[["Principle","Frank Rosenblatt error-correction rule. Only updates weights when prediction y_hat is wrong."],
                     ["Error Signal","Discrete: (t - y_hat) in {-2, 0, +2} after hard sign activation."],
                     ["Convergence","Guaranteed in finite steps if classes are linearly separable (Convergence Theorem)."],
                     ["Activation","Hard bipolar step function sgn(z): +1 if z >= 0, else -1."]],
                insight:"If prediction equals target, error = 0 and no update happens. Learning occurs only on mistakes." },
              { name:"C. Delta / LMS Rule", year:"1960", color:"text-brand-cyan", bg:"border-cyan-900/40", hl:"bg-cyan-950/30 border-cyan-900/30 text-cyan-200",
                formula:["Dw = eta * (t - z) * x","z = x*w + b (linear)"],
                pts:[["Principle","Widrow-Hoff LMS. Minimizes MSE on continuous linear output z before thresholding."],
                     ["Error Signal","Continuous: (t - z) in R. Smoother gradient than perceptron."],
                     ["Objective","J(w) = 0.5 * E[(t - z)^2] — convex bowl, single global minimum."],
                     ["Advantage","Converges even on non-separable data to the least-squares optimal weight vector."]],
                insight:"ADALINE computes error before the threshold — enabling gradient descent on a smooth, well-behaved loss surface." },
            ].map(alg => (
              <div key={alg.name} className={`glass-panel p-5 rounded-2xl space-y-4 border ${alg.bg}`}>
                <div className="flex items-center justify-between">
                  <span className={`font-bold text-sm ${alg.color}`}>{alg.name}</span>
                  <span className="text-[10px] px-2 py-0.5 rounded bg-dark-800 text-slate-400 border border-slate-700">{alg.year}</span>
                </div>
                <div className="p-3 rounded-lg bg-dark-950 font-mono text-center border border-slate-800 space-y-1">
                  {alg.formula.map(f => <div key={f} className={`text-xs font-bold ${alg.color}`}>{f}</div>)}
                </div>
                <div className="space-y-2 text-xs text-slate-300 leading-relaxed">
                  {alg.pts.map(([k,v]) => <p key={k}><strong className="text-white">{k}:</strong> {v}</p>)}
                </div>
                <div className={`p-3 rounded-lg border text-xs leading-relaxed ${alg.hl}`}>
                  <strong>Key insight:</strong> {alg.insight}
                </div>
              </div>
            ))}
          </div>

          <div className="glass-panel p-6 rounded-2xl">
            <h2 className="text-base font-bold text-white flex items-center gap-2 mb-4 pb-3 border-b border-slate-800">
              <Layers className="w-5 h-5 text-brand-cyan"/> Algorithm Execution Flow
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-5 text-xs font-mono">
              {[
                { name:"Hebbian", color:"text-purple-400", border:"border-purple-900/40",
                  steps:["1. Receive input x, target t","2. Compute z = x*w + b","3. (No error computed)","4. Dw = eta * x * t","5. Db = eta * t","6. w = w + Dw  (always)"] },
                { name:"Perceptron", color:"text-emerald-400", border:"border-emerald-900/40",
                  steps:["1. Receive input x, target t","2. Compute z = x*w + b","3. y_hat = sgn(z) in {+1,-1}","4. e = t - y_hat in {-2,0,+2}","5. Dw = eta * e * x","6. Update only if e != 0"] },
                { name:"ADALINE", color:"text-brand-cyan", border:"border-cyan-900/40",
                  steps:["1. Receive input x, target t","2. Compute z = x*w + b","3. e = t - z in R (real)","4. Dw = eta * e * x","5. Db = eta * e","6. w = w + Dw  (always)"] },
              ].map(alg => (
                <div key={alg.name} className={`p-4 rounded-xl bg-dark-900 border ${alg.border} space-y-2`}>
                  <div className={`font-bold ${alg.color} mb-3`}>{alg.name}</div>
                  {alg.steps.map((step,i) => (
                    <div key={i} className="flex gap-2 text-[11px]">
                      <span className={`${alg.color} opacity-60 shrink-0`}>{step.split(" ")[0]}</span>
                      <span className="text-slate-300">{step.split(" ").slice(1).join(" ")}</span>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {activeTab === "sandbox" && (
        <div className="space-y-6">
          <div className="glass-panel p-6 rounded-2xl space-y-5">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-800">
              <div>
                <h2 className="text-base font-bold text-white flex items-center gap-2">
                  <Calculator className="w-5 h-5 text-brand-violet"/> Live Weight Update Sandbox
                </h2>
                <p className="text-xs text-slate-400 mt-0.5">Set parameters, choose a rule, then click Run Simulation to see the weight trajectory chart.</p>
              </div>
              <div className="flex gap-2">
                <button onClick={runSimulation}
                  className="flex items-center gap-2 px-4 py-2 rounded-xl bg-gradient-to-r from-brand-violet to-brand-cyan text-white text-xs font-semibold hover:opacity-90 transition-opacity shadow-lg shadow-brand-violet/20">
                  <Play className="w-3.5 h-3.5"/> Run Simulation
                </button>
                <button onClick={() => { setHistory([]); setHasRun(false); }}
                  className="flex items-center gap-2 px-3 py-2 rounded-xl bg-dark-800 text-slate-300 text-xs border border-slate-700 hover:border-slate-500 hover:text-white transition-all">
                  <RotateCcw className="w-3.5 h-3.5"/> Reset
                </button>
              </div>
            </div>

            <div className="flex gap-2 flex-wrap">
              {[{id:"hebb",label:"Hebbian",ac:"bg-purple-900/50 text-purple-300 border-purple-500"},
                {id:"perceptron",label:"Perceptron",ac:"bg-emerald-900/50 text-emerald-300 border-emerald-500"},
                {id:"adaline",label:"ADALINE",ac:"bg-cyan-900/50 text-cyan-300 border-cyan-500"}].map(r => (
                <button key={r.id} onClick={() => { setRule(r.id); setHasRun(false); }}
                  className={`px-4 py-2 rounded-lg text-xs font-semibold border transition-all ${
                    rule===r.id ? r.ac : "bg-dark-900 text-slate-400 border-slate-700 hover:border-slate-500"}`}>
                  {r.label}
                </button>
              ))}
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-3">
              {[{label:"Input x1",val:x1,set:setX1,step:0.1},
                {label:"Input x2",val:x2,set:setX2,step:0.1},
                {label:"Weight w1",val:w1,set:setW1,step:0.1},
                {label:"Weight w2",val:w2,set:setW2,step:0.1},
                {label:"Bias b",val:bias,set:setBias,step:0.1},
                {label:"Rate eta",val:lr,set:setLr,step:0.01,min:0.001,max:1},
                {label:"Steps",val:steps,set:setSteps,step:1,min:1,max:50}].map(({label,val,set,step,min,max}) => (
                <div key={label} className="p-3 rounded-xl bg-dark-900 border border-slate-800">
                  <label className="text-[10px] uppercase font-semibold text-slate-400 block mb-1">{label}</label>
                  <input type="number" step={step} min={min} max={max} value={val}
                    onChange={(e) => { set(Number(e.target.value)); setHasRun(false); }}
                    className="w-full bg-dark-950 border border-slate-700 rounded px-2 py-1 text-xs text-slate-200 font-mono text-center focus:outline-none focus:border-brand-cyan/60"/>
                </div>
              ))}
            </div>

            <div className="flex items-center gap-3 text-xs">
              <span className="text-slate-400 font-semibold">Target t:</span>
              {[1,-1].map(v => (
                <button key={v} onClick={() => { setTarget(v); setHasRun(false); }}
                  className={`px-4 py-1.5 rounded-lg font-mono font-bold border transition-all ${
                    target===v ? (v===1?"bg-emerald-500/20 text-emerald-300 border-emerald-500/50"
                                      :"bg-rose-500/20 text-rose-300 border-rose-500/50")
                               : "bg-dark-900 text-slate-400 border-slate-700 hover:border-slate-500"}`}>
                  {v===1?"+1":"-1"}
                </button>
              ))}
            </div>
          </div>

          <div className="glass-panel p-5 rounded-2xl">
            <h3 className="text-sm font-bold text-white mb-3">Current Step Calculations (Single Update Preview)</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
              <div className="p-4 rounded-xl bg-dark-900 border border-purple-900/40 space-y-2 font-mono">
                <div className="text-purple-400 font-bold mb-2">Hebbian</div>
                <div className="text-slate-300">Dw1 = eta*x1*t = <span className="text-purple-300 font-bold">{+(lr*x1*target).toFixed(4)}</span></div>
                <div className="text-slate-300">Dw2 = eta*x2*t = <span className="text-purple-300 font-bold">{+(lr*x2*target).toFixed(4)}</span></div>
                <div className="text-slate-300">Db  = eta*t   = <span className="text-purple-300 font-bold">{+(lr*target).toFixed(4)}</span></div>
                <div className="pt-2 border-t border-slate-800">
                  <div className="text-slate-300">w1: {w1} to <strong className="text-purple-400">{+(w1+lr*x1*target).toFixed(4)}</strong></div>
                  <div className="text-slate-300">w2: {w2} to <strong className="text-purple-400">{+(w2+lr*x2*target).toFixed(4)}</strong></div>
                </div>
              </div>
              <div className="p-4 rounded-xl bg-dark-900 border border-emerald-900/40 space-y-2 font-mono">
                <div className="text-emerald-400 font-bold mb-2">Perceptron</div>
                <div className="text-slate-300">z={z} | y_hat=<span className={yhat_perc>=0?"text-emerald-400 font-bold":"text-rose-400 font-bold"}>{yhat_perc>=0?"+1":"-1"}</span></div>
                <div className="text-slate-300">error = t-y_hat = <span className={perc_err===0?"text-emerald-400 font-bold":"text-rose-400 font-bold"}>{perc_err}</span></div>
                <div className="text-slate-300">Dw1 = <span className="text-emerald-300 font-bold">{+(lr*perc_err*x1).toFixed(4)}</span></div>
                <div className="pt-2 border-t border-slate-800">
                  <div className="text-slate-300">w1: {w1} to <strong className="text-emerald-400">{+(w1+lr*perc_err*x1).toFixed(4)}</strong></div>
                  <div className="text-slate-300">w2: {w2} to <strong className="text-emerald-400">{+(w2+lr*perc_err*x2).toFixed(4)}</strong></div>
                </div>
              </div>
              <div className="p-4 rounded-xl bg-dark-900 border border-cyan-900/40 space-y-2 font-mono">
                <div className="text-brand-cyan font-bold mb-2">ADALINE</div>
                <div className="text-slate-300">z = {z} (continuous)</div>
                <div className="text-slate-300">error = t-z = <span className="text-brand-cyan font-bold">{adal_err}</span></div>
                <div className="text-slate-300">Dw1 = <span className="text-cyan-300 font-bold">{+(lr*adal_err*x1).toFixed(4)}</span></div>
                <div className="pt-2 border-t border-slate-800">
                  <div className="text-slate-300">w1: {w1} to <strong className="text-brand-cyan">{+(w1+lr*adal_err*x1).toFixed(4)}</strong></div>
                  <div className="text-slate-300">w2: {w2} to <strong className="text-brand-cyan">{+(w2+lr*adal_err*x2).toFixed(4)}</strong></div>
                </div>
              </div>
            </div>
          </div>

          {hasRun && history.length > 0 ? (
            <div className="glass-panel p-6 rounded-2xl space-y-4">
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                <Target className="w-4 h-4 text-brand-cyan"/>
                Weight Trajectory: {rule.charAt(0).toUpperCase()+rule.slice(1)} Rule over {steps} steps
              </h3>
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={history} margin={{ top:5, right:20, left:0, bottom:5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b"/>
                  <XAxis dataKey="step" stroke="#475569" tick={{ fill:"#94a3b8", fontSize:11 }}/>
                  <YAxis stroke="#475569" tick={{ fill:"#94a3b8", fontSize:11 }}/>
                  <Tooltip contentStyle={{ background:"#0d1117", border:"1px solid #1e293b", borderRadius:"8px", fontSize:"11px" }} labelStyle={{ color:"#94a3b8" }}/>
                  <Legend wrapperStyle={{ fontSize:"11px", paddingTop:"8px" }}/>
                  <Line type="monotone" dataKey="w1" stroke="#06b6d4" strokeWidth={2} dot={false} name="w1"/>
                  <Line type="monotone" dataKey="w2" stroke="#a855f7" strokeWidth={2} dot={false} name="w2"/>
                  <Line type="monotone" dataKey="b"  stroke="#f59e0b" strokeWidth={1.5} dot={false} name="bias" strokeDasharray="5 3"/>
                  <Line type="monotone" dataKey="error" stroke="#f87171" strokeWidth={2} dot={false} name="Error"/>
                </LineChart>
              </ResponsiveContainer>
              <div className="grid grid-cols-4 gap-3 text-xs text-center font-mono">
                {[["Final w1",history[history.length-1].w1,"text-brand-cyan"],
                  ["Final w2",history[history.length-1].w2,"text-purple-400"],
                  ["Final b", history[history.length-1].b, "text-amber-400"],
                  ["Final Error",history[history.length-1].error,"text-rose-400"]].map(([label,val,color]) => (
                  <div key={label} className="p-3 rounded-xl bg-dark-900 border border-slate-800">
                    <div className="text-slate-400 text-[10px] mb-1">{label}</div>
                    <div className={`font-bold text-sm ${color}`}>{val}</div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="glass-panel p-10 rounded-2xl flex flex-col items-center justify-center text-center gap-3">
              <div className="w-14 h-14 rounded-2xl bg-brand-violet/10 border border-brand-violet/20 flex items-center justify-center">
                <Play className="w-6 h-6 text-brand-violet"/>
              </div>
              <p className="text-sm font-semibold text-slate-300">Set parameters and click <span className="text-brand-cyan font-bold">Run Simulation</span></p>
              <p className="text-xs text-slate-500 max-w-sm">The weight trajectory chart appears here, showing how w1, w2, bias, and error evolve over each training step for the selected learning rule.</p>
            </div>
          )}
        </div>
      )}

      {activeTab === "compare" && (
        <div className="space-y-6">
          <div className="glass-panel p-6 rounded-2xl">
            <h2 className="text-base font-bold text-white flex items-center gap-2 mb-4 pb-3 border-b border-slate-800">
              <CheckCircle2 className="w-5 h-5 text-emerald-400"/> Full Comparative Analysis: Hebb vs Perceptron vs ADALINE
            </h2>
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="border-b border-slate-700 bg-dark-900">
                    <th className="p-3 text-slate-400 font-semibold">Property</th>
                    <th className="p-3 text-purple-400 font-semibold">Hebb (1949)</th>
                    <th className="p-3 text-emerald-400 font-semibold">Perceptron (1958)</th>
                    <th className="p-3 text-brand-cyan font-semibold">ADALINE / LMS (1960)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 text-slate-300">
                  {[["Update Rule","Dw = eta*x*t","Dw = eta*(t-yhat)*x","Dw = eta*(t-z)*x"],
                    ["Error Signal","None (associative)","Discrete: {-2,0,+2}","Continuous: real-valued"],
                    ["Training Activation","N/A","Hard Step sgn(z)","Linear identity z"],
                    ["Training Objective","Correlation max","Perceptron Criterion","MSE: 0.5*E[(t-z)^2]"],
                    ["Optimization","None (pure Hebb)","Error-correction step","Gradient Descent (LMS)"],
                    ["Convergence (Separable)","Not guaranteed","Guaranteed (finite)","Guaranteed to global min"],
                    ["Non-Separable Data","Weights diverge","Oscillates forever","Converges to least-squares"],
                    ["Requires Error Feedback","No","Yes (discrete)","Yes (continuous)"],
                    ["Weight Growth","Unbounded","Bounded if separable","Bounded by gradient"],
                    ["MNIST Accuracy (approx)","~72-78%","~82-87%","~84-89%"]
                  ].map(([prop,hebb,perc,adal],i) => (
                    <tr key={i} className={i%2===0?"":"bg-dark-900/30"}>
                      <td className="p-3 font-semibold text-white">{prop}</td>
                      <td className="p-3 text-purple-300 font-mono text-[11px]">{hebb}</td>
                      <td className="p-3 text-emerald-300 font-mono text-[11px]">{perc}</td>
                      <td className="p-3 text-cyan-300 font-mono text-[11px]">{adal}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
            {[{label:"Best for Speed",winner:"Hebbian",color:"text-purple-400",bg:"border-purple-900/40",
               reason:"O(1) per sample. No loss computation or backpropagation. Pure input-target correlation update."},
              {label:"Best for Accuracy",winner:"ADALINE",color:"text-brand-cyan",bg:"border-cyan-900/40",
               reason:"Continuous gradient minimizes MSE to the true global optimum. Handles non-separable classes gracefully."},
              {label:"Best Guarantee",winner:"Perceptron",color:"text-emerald-400",bg:"border-emerald-900/40",
               reason:"Convergence Theorem guarantees perfect classification in finite steps for linearly separable data."}
            ].map(card => (
              <div key={card.label} className={`glass-panel p-5 rounded-2xl border ${card.bg} space-y-2`}>
                <div className="text-[10px] text-slate-400 uppercase tracking-wider">{card.label}</div>
                <div className={`text-lg font-bold ${card.color}`}>{card.winner}</div>
                <p className="text-xs text-slate-400 leading-relaxed">{card.reason}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
