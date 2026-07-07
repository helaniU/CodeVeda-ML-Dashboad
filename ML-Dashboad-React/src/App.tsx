import React, { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import DecisionTree from './components/DecisionTree';
import KMeansClustering from './components/KMeansClustering';
import NeuralNetwork from './components/NeuralNetwork';
import RandomForest from './components/RandomForest';
import SvmSentiment from './components/SvmSentiment';
import { Brain, Network, Sliders, Layers, BarChart, Database, Cpu, HelpCircle, GraduationCap } from 'lucide-react';

type TabId = 'decision_tree' | 'kmeans' | 'neural_network' | 'random_forest' | 'svm';

interface TabItem {
  id: TabId;
  label: string;
  level: string;
  type: string;
  icon: React.ComponentType<{ className?: string; size?: number }>;
  color: string;
  desc: string;
  activeClass: string;
  inactiveClass: string;
}

const TABS: TabItem[] = [
  {
    id: 'decision_tree',
    label: 'Decision Tree Classifier',
    level: 'Level 02',
    type: 'Classification',
    icon: Layers,
    color: 'text-sky-600 border-sky-200 bg-sky-50',
    activeClass: 'bg-gradient-to-br from-sky-100 via-sky-50 to-blue-50/50 border-sky-300 text-slate-900 shadow-md shadow-sky-100/50 ring-1 ring-sky-200/50',
    inactiveClass: 'bg-white border-slate-200 text-slate-600 hover:from-sky-50/40 hover:to-blue-50/20 hover:border-sky-200 hover:shadow-sm',
    desc: 'Interactive visual CART tree training on the Fisher Iris dataset with real-time predictions.'
  },
  {
    id: 'kmeans',
    label: 'K-Means & DBSCAN',
    level: 'Level 02',
    type: 'Clustering',
    icon: Database,
    color: 'text-purple-600 border-purple-200 bg-purple-50',
    activeClass: 'bg-gradient-to-br from-violet-100 via-purple-50 to-indigo-50/50 border-purple-300 text-slate-900 shadow-md shadow-purple-100/50 ring-1 ring-purple-200/50',
    inactiveClass: 'bg-white border-slate-200 text-slate-600 hover:from-purple-50/40 hover:to-indigo-50/20 hover:border-purple-200 hover:shadow-sm',
    desc: 'Unsupervised spatial clustering, PCA projection, Silhouette scores, and Elbow curve plots.'
  },
  {
    id: 'random_forest',
    label: 'Random Forest Regressor',
    level: 'Level 03',
    type: 'Regression',
    icon: Sliders,
    color: 'text-emerald-600 border-emerald-200 bg-emerald-50',
    activeClass: 'bg-gradient-to-br from-emerald-100 via-emerald-50 to-teal-50/50 border-emerald-300 text-slate-900 shadow-md shadow-emerald-100/50 ring-1 ring-emerald-200/50',
    inactiveClass: 'bg-white border-slate-200 text-slate-600 hover:from-emerald-50/40 hover:to-teal-50/20 hover:border-emerald-200 hover:shadow-sm',
    desc: 'Multiple CART-based regressor forest with IQR outlier cleaning, cross-validation, and error residuals.'
  },
  {
    id: 'neural_network',
    label: 'Deep Neural Network',
    level: 'Level 03',
    type: 'Deep Learning',
    icon: Brain,
    color: 'text-fuchsia-600 border-fuchsia-200 bg-fuchsia-50',
    activeClass: 'bg-gradient-to-br from-fuchsia-100 via-fuchsia-50 to-pink-50/50 border-fuchsia-300 text-slate-900 shadow-md shadow-fuchsia-100/50 ring-1 ring-fuchsia-200/50',
    inactiveClass: 'bg-white border-slate-200 text-slate-600 hover:from-fuchsia-50/40 hover:to-pink-50/20 hover:border-fuchsia-200 hover:shadow-sm',
    desc: 'Dense feed-forward TFJS text classification simulation with live training convergence logs.'
  },
  {
    id: 'svm',
    label: 'Support Vector Machine',
    level: 'Level 03',
    type: 'Classification',
    icon: Cpu,
    color: 'text-blue-600 border-blue-200 bg-blue-50',
    activeClass: 'bg-gradient-to-br from-blue-100 via-blue-50 to-cyan-50/50 border-blue-300 text-slate-900 shadow-md shadow-blue-100/50 ring-1 ring-blue-200/50',
    inactiveClass: 'bg-white border-slate-200 text-slate-600 hover:from-blue-50/40 hover:to-cyan-50/20 hover:border-blue-200 hover:shadow-sm',
    desc: 'Multiclass SGD hyperplane training with linear vs. RBF kernel confusion matrix comparisons.'
  }
];

export default function App() {
  const [activeTab, setActiveTab] = useState<TabId>('decision_tree');

  const renderActiveComponent = () => {
    switch (activeTab) {
      case 'decision_tree':
        return <DecisionTree />;
      case 'kmeans':
        return <KMeansClustering />;
      case 'neural_network':
        return <NeuralNetwork />;
      case 'random_forest':
        return <RandomForest />;
      case 'svm':
        return <SvmSentiment />;
    }
  };

  const activeTabItem = TABS.find(t => t.id === activeTab)!;

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 font-sans flex flex-col selection:bg-purple-100 selection:text-purple-950">
      
      {/* Header Area */}
      <header className="border-b border-slate-200 bg-white/95 backdrop-blur sticky top-0 z-50 py-4 px-6 flex flex-col md:flex-row md:items-center justify-between gap-4 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-purple-600 to-sky-500 flex items-center justify-center shadow-md shadow-purple-500/10">
            <Network className="text-white" size={20} />
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight text-slate-900 flex items-center gap-2 font-display">
              CodeVeda ML Dashboard
              <span className="text-xs bg-purple-100 border border-purple-200 text-purple-700 font-semibold px-2.5 py-0.5 rounded-full font-mono uppercase tracking-wider">
                Interactive Studio
              </span>
            </h1>
            <p className="text-sm text-slate-500 mt-0.5">Explore, train, and test classic Level 02 & Level 03 Machine Learning models in real-time.</p>
          </div>
        </div>

        {/* Level Legend */}
        <div className="flex items-center gap-3 text-xs font-mono text-slate-600 bg-white border border-slate-200 shadow-sm py-1.5 px-3 rounded-lg w-fit">
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-sky-500"></span> Level 02 (Basic)
          </span>
          <span className="text-slate-300">|</span>
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-purple-500"></span> Level 03 (Advanced)
          </span>
        </div>
      </header>

      {/* Main Grid Content - Widened to fit width perfectly */}
      <div className="flex-grow max-w-full px-4 md:px-6 lg:px-8 py-6 space-y-6 mx-auto w-full flex flex-col">
        
        {/* Navigation Cards at the Top */}
        <div className="space-y-3">
          <h2 className="text-xs font-extrabold uppercase tracking-widest text-slate-400 px-1">Select ML Algorithm</h2>
          
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
            {TABS.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              
              return (
                <button
                  key={tab.id}
                  id={`nav-tab-${tab.id}`}
                  onClick={() => setActiveTab(tab.id)}
                  className={`text-left p-4 rounded-xl border transition-all relative overflow-hidden group flex flex-col justify-between gap-3 min-h-[120px] cursor-pointer
                    ${isActive ? tab.activeClass : tab.inactiveClass}`}
                >
                  {/* Subtle active state gradient strip running along the top */}
                  {isActive && (
                    <motion.div 
                      layoutId="activeTabIndicator"
                      className="absolute left-0 right-0 top-0 h-1 bg-gradient-to-r from-purple-500 to-sky-500"
                    />
                  )}

                  <div className="flex items-start justify-between w-full gap-2">
                    <div className="flex items-center gap-2">
                      <div className={`p-1.5 rounded-lg border ${tab.color}`}>
                        <Icon size={15} />
                      </div>
                      <span className="text-sm font-bold tracking-tight text-slate-800">{tab.label}</span>
                    </div>
                    
                    <span className={`text-[9px] font-mono font-bold px-1.5 py-0.5 rounded border shrink-0 ${tab.level === 'Level 02' ? 'bg-sky-50 border-sky-100 text-sky-600' : 'bg-purple-50 border-purple-100 text-purple-600'}`}>
                      {tab.level}
                    </span>
                  </div>

                  <p className="text-xs text-slate-500 leading-relaxed line-clamp-2 group-hover:text-slate-700 transition-colors">
                    {tab.desc}
                  </p>
                </button>
              );
            })}
          </div>
        </div>

        {/* Algorithm Stage and Helper Banner */}
        <div className="flex flex-col gap-6">
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -12 }}
              transition={{ duration: 0.25, ease: 'easeInOut' }}
              className="space-y-6 animate-fadeIn"
            >
              <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                {/* Algorithm Title Card */}
                <div className="lg:col-span-3 bg-white border border-slate-200 p-6 rounded-xl flex flex-col justify-center shadow-sm">
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-mono bg-purple-50 border border-purple-100 text-purple-700 font-bold px-2.5 py-0.5 rounded">
                        {activeTabItem.level}
                      </span>
                      <span className="text-xs font-mono bg-sky-50 border border-sky-100 text-sky-700 font-bold px-2.5 py-0.5 rounded">
                        {activeTabItem.type}
                      </span>
                    </div>
                    <h2 className="text-2xl font-extrabold font-display text-slate-900 mt-2 flex items-center gap-2">
                      {activeTabItem.label}
                    </h2>
                    <p className="text-sm text-slate-600 mt-1.5 leading-relaxed">
                      {activeTabItem.desc}
                    </p>
                  </div>
                </div>

                {/* Quick Informational card */}
                <div className="lg:col-span-1 bg-white border border-slate-200 p-6 rounded-xl flex flex-col justify-center shadow-sm">
                  <div className="space-y-2">
                    <h4 className="text-xs uppercase font-extrabold text-slate-700 tracking-wider flex items-center gap-1.5">
                      <GraduationCap size={14} className="text-purple-600" /> Client-Side Compute
                    </h4>
                    <p className="text-xs text-slate-600 leading-relaxed">
                      Every classifier, cluster finder, neural network, and regression tree runs completely in your web browser. No remote servers are invoked. Adjust parameters on the fly to see live recalculations.
                    </p>
                  </div>
                </div>
              </div>

              {/* Main component render */}
              <div className="w-full">
                {renderActiveComponent()}
              </div>
            </motion.div>
          </AnimatePresence>
        </div>

      </div>

      {/* Footer bar */}
      <footer className="border-t border-slate-200 py-4 px-6 text-center text-xs text-slate-500 mt-auto bg-white">
        CodeVeda Interactive Machine Learning Dashboard • Built with React, Vite, Tailwind CSS, and Recharts
      </footer>

    </div>
  );
}
