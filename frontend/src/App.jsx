import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import DigitRecognition from './pages/DigitRecognition';
import TrainModels from './pages/TrainModels';
import ModelComparison from './pages/ModelComparison';
import LearningTheory from './pages/LearningTheory';
import DatasetExplorer from './pages/DatasetExplorer';
import AboutProject from './pages/AboutProject';
import { fetchHealth } from './services/api';

const PAGE_CONFIG = {
  home: {
    title: 'Home Dashboard',
    subtitle: 'Project overview, model metrics, and quick navigation',
  },
  recognition: {
    title: 'Handwritten Digit Recognition',
    subtitle: 'Draw a digit or upload an image — preprocessing and live inference',
  },
  train: {
    title: 'Model Training Engine',
    subtitle: 'Configure hyperparameters and train Hebb / Perceptron / ADALINE models',
  },
  compare: {
    title: 'Comparative Study Dashboard',
    subtitle: 'Side-by-side metrics, confusion matrices, and learning curve analysis',
  },
  theory: {
    title: 'Educational Learning Theory',
    subtitle: 'Mathematical principles, weight update sandbox, and algorithm comparisons',
  },
  dataset: {
    title: 'Dataset Explorer',
    subtitle: 'Inspect the 28×28 handwritten digit dataset class distribution and samples',
  },
  about: {
    title: 'About This Project',
    subtitle: 'Problem statement, methodology, architecture, and future enhancements',
  },
};

export default function App() {
  const [currentTab, setCurrentTab] = useState('home');
  const [backendStatus, setBackendStatus] = useState('checking');

  // Poll backend health every 10 seconds
  useEffect(() => {
    async function checkHealth() {
      try {
        const res = await fetchHealth();
        setBackendStatus(res.status === 'healthy' ? 'online' : 'offline');
      } catch {
        setBackendStatus('offline');
      }
    }
    checkHealth();
    const interval = setInterval(checkHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  const page = PAGE_CONFIG[currentTab];

  const renderPage = () => {
    switch (currentTab) {
      case 'home': return <Home onNavigate={setCurrentTab} />;
      case 'recognition': return <DigitRecognition />;
      case 'train': return <TrainModels />;
      case 'compare': return <ModelComparison />;
      case 'theory': return <LearningTheory />;
      case 'dataset': return <DatasetExplorer />;
      case 'about': return <AboutProject />;
      default: return <Home onNavigate={setCurrentTab} />;
    }
  };

  return (
    <div className="flex h-screen bg-dark-950 text-slate-100 overflow-hidden">
      {/* Sidebar Navigation */}
      <Sidebar currentTab={currentTab} setCurrentTab={setCurrentTab} />

      {/* Main Content Area */}
      <div className="flex flex-col flex-1 overflow-hidden">
        <Navbar
          activeTitle={page.title}
          subtitle={page.subtitle}
          backendStatus={backendStatus}
          onRefresh={() => setBackendStatus('checking')}
        />

        {/* Scrollable Page Content */}
        <main className="flex-1 overflow-y-auto p-6">
          {renderPage()}
        </main>
      </div>
    </div>
  );
}
