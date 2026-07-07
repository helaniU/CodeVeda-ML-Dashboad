import React, { useState, useMemo } from 'react';
import Papa from 'papaparse';
import irisCsv from '../../datasets/1) iris.csv?raw';
import { IrisData } from '../types';
import { Play, RotateCcw, AlertCircle, Info, ChevronRight, Activity } from 'lucide-react';

// Seeding random split for stability
function seededShuffle<T>(array: T[], seed: number): T[] {
  let m = array.length, t, i;
  let s = seed;
  const rand = () => {
    let x = Math.sin(s++) * 10000;
    return x - Math.floor(x);
  };
  const copy = [...array];
  while (m) {
    i = Math.floor(rand() * m--);
    t = copy[m];
    copy[m] = copy[i];
    copy[i] = t;
  }
  return copy;
}

// Tree Node interface
interface TreeNode {
  featureIndex?: number;
  featureName?: string;
  threshold?: number;
  left?: TreeNode;
  right?: TreeNode;
  isLeaf: boolean;
  value?: string;
  distribution: Record<string, number>;
  impurity: number;
  samples: number;
  id: string;
}

export default function DecisionTree() {
  const [maxDepth, setMaxDepth] = useState<number>(3);
  const [criterion, setCriterion] = useState<'gini' | 'entropy'>('gini');
  const [minSamplesSplit, setMinSamplesSplit] = useState<number>(2);

  // Live prediction inputs (defaults set to means)
  const [sepalLength, setSepalLength] = useState<number>(5.8);
  const [sepalWidth, setSepalWidth] = useState<number>(3.0);
  const [petalLength, setPetalLength] = useState<number>(3.8);
  const [petalWidth, setPetalWidth] = useState<number>(1.2);

  // Parse Data
  const { allData, X, y, features, classNames } = useMemo(() => {
    const parsed = Papa.parse<any>(irisCsv, { header: true, dynamicTyping: true });
    const cleanData: IrisData[] = parsed.data
      .filter((row: any) => row.species && row.sepal_length !== undefined)
      .map((row: any) => ({
        sepal_length: Number(row.sepal_length),
        sepal_width: Number(row.sepal_width),
        petal_length: Number(row.petal_length),
        petal_width: Number(row.petal_width),
        species: String(row.species).trim().toLowerCase()
      }));

    const featuresList = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'];
    const uniqueClasses = Array.from(new Set(cleanData.map(d => d.species))).sort();

    return {
      allData: cleanData,
      X: cleanData.map(d => [d.sepal_length, d.sepal_width, d.petal_length, d.petal_width]),
      y: cleanData.map(d => d.species),
      features: featuresList,
      classNames: uniqueClasses
    };
  }, []);

  // Split Train/Test deterministically
  const { trainData, testData } = useMemo(() => {
    const indices = Array.from({ length: allData.length }, (_, i) => i);
    const shuffledIndices = seededShuffle(indices, 42); // Scikit-learn random_state=42 seed
    
    const splitIndex = Math.floor(allData.length * 0.8);
    const trainIndices = shuffledIndices.slice(0, splitIndex);
    const testIndices = shuffledIndices.slice(splitIndex);

    return {
      trainData: trainIndices.map(i => allData[i]),
      testData: testIndices.map(i => allData[i])
    };
  }, [allData]);

  // Build Decision Tree model
  const treeModel = useMemo(() => {
    let nodeCount = 0;

    const getDistribution = (subset: IrisData[]): Record<string, number> => {
      const dist: Record<string, number> = {};
      classNames.forEach(c => { dist[c] = 0; });
      subset.forEach(d => {
        dist[d.species] = (dist[d.species] || 0) + 1;
      });
      return dist;
    };

    const calculateImpurity = (subset: IrisData[], crit: 'gini' | 'entropy'): number => {
      if (subset.length === 0) return 0;
      const dist = getDistribution(subset);
      const len = subset.length;

      if (crit === 'gini') {
        let sumSq = 0;
        for (const c in dist) {
          const p = dist[c] / len;
          sumSq += p * p;
        }
        return 1 - sumSq;
      } else {
        let sumEnt = 0;
        for (const c in dist) {
          const p = dist[c] / len;
          if (p > 0) {
            sumEnt += p * Math.log2(p);
          }
        }
        return -sumEnt;
      }
    };

    const buildTree = (subset: IrisData[], depth: number): TreeNode => {
      const id = `node-${nodeCount++}`;
      const samples = subset.length;
      const impurity = calculateImpurity(subset, criterion);
      const distribution = getDistribution(subset);

      // Find majority class
      let bestClass = classNames[0];
      let maxCount = -1;
      for (const c in distribution) {
        if (distribution[c] > maxCount) {
          maxCount = distribution[c];
          bestClass = c;
        }
      }

      // Base cases: Leaf conditions
      if (
        depth >= maxDepth ||
        samples < minSamplesSplit ||
        impurity === 0
      ) {
        return {
          isLeaf: true,
          value: bestClass,
          distribution,
          impurity,
          samples,
          id
        };
      }

      // Search for best split
      let bestGain = -1;
      let bestFeatureIndex = -1;
      let bestThreshold = -1;
      let bestLeftSubset: IrisData[] = [];
      let bestRightSubset: IrisData[] = [];

      for (let fIdx = 0; fIdx < features.length; fIdx++) {
        const featureKey = features[fIdx] as keyof IrisData;
        // Unique feature values in this subset
        const values = Array.from(new Set(subset.map(d => d[featureKey] as number))).sort((a, b) => a - b);
        
        // Test split thresholds at midpoints
        for (let i = 0; i < values.length - 1; i++) {
          const threshold = (values[i] + values[i + 1]) / 2;
          const left = subset.filter(d => (d[featureKey] as number) <= threshold);
          const right = subset.filter(d => (d[featureKey] as number) > threshold);

          if (left.length === 0 || right.length === 0) continue;

          const leftImpurity = calculateImpurity(left, criterion);
          const rightImpurity = calculateImpurity(right, criterion);

          const leftWeight = left.length / samples;
          const rightWeight = right.length / samples;
          const splitImpurity = leftWeight * leftImpurity + rightWeight * rightImpurity;
          const gain = impurity - splitImpurity;

          if (gain > bestGain) {
            bestGain = gain;
            bestFeatureIndex = fIdx;
            bestThreshold = threshold;
            bestLeftSubset = left;
            bestRightSubset = right;
          }
        }
      }

      // If no split yields gain, make leaf node
      if (bestFeatureIndex === -1) {
        return {
          isLeaf: true,
          value: bestClass,
          distribution,
          impurity,
          samples,
          id
        };
      }

      // Recurse left and right
      return {
        isLeaf: false,
        featureIndex: bestFeatureIndex,
        featureName: features[bestFeatureIndex],
        threshold: bestThreshold,
        left: buildTree(bestLeftSubset, depth + 1),
        right: buildTree(bestRightSubset, depth + 1),
        distribution,
        impurity,
        samples,
        id
      };
    };

    const rootNode = buildTree(trainData, 0);

    const predictRow = (node: TreeNode, row: number[]): { label: string; proba: Record<string, number> } => {
      if (node.isLeaf) {
        const total = Object.values(node.distribution).reduce((a, b) => a + b, 0);
        const proba: Record<string, number> = {};
        classNames.forEach(c => {
          proba[c] = total > 0 ? (node.distribution[c] || 0) / total : 0;
        });
        return { label: node.value!, proba };
      }

      const val = row[node.featureIndex!];
      if (val <= node.threshold!) {
        return predictRow(node.left!, row);
      } else {
        return predictRow(node.right!, row);
      }
    };

    return { rootNode, predictRow };
  }, [trainData, criterion, maxDepth, minSamplesSplit, classNames, features]);

  // Evaluate model on test data
  const evaluation = useMemo(() => {
    const predictions = testData.map(row => 
      treeModel.predictRow(treeModel.rootNode, [row.sepal_length, row.sepal_width, row.petal_length, row.petal_width])
    );

    let correct = 0;
    predictions.forEach((p, i) => {
      if (p.label === testData[i].species) correct++;
    });

    const accuracy = correct / testData.length;

    // Create Confusion Matrix
    // Columns: actual species, Rows: predicted species
    const confMat: Record<string, Record<string, number>> = {};
    classNames.forEach(actual => {
      confMat[actual] = {};
      classNames.forEach(pred => {
        confMat[actual][pred] = 0;
      });
    });

    testData.forEach((row, i) => {
      const pred = predictions[i].label;
      const actual = row.species;
      if (confMat[actual] && confMat[actual][pred] !== undefined) {
        confMat[actual][pred]++;
      }
    });

    // Classification Report Calculations
    const report: Record<string, { precision: number; recall: number; f1: number; support: number }> = {};
    classNames.forEach(cls => {
      let tp = 0, fp = 0, fn = 0;
      testData.forEach((row, i) => {
        const actual = row.species;
        const pred = predictions[i].label;
        if (actual === cls && pred === cls) tp++;
        else if (actual !== cls && pred === cls) fp++;
        else if (actual === cls && pred !== cls) fn++;
      });

      const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
      const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
      const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
      const support = testData.filter(d => d.species === cls).length;

      report[cls] = { precision, recall, f1, support };
    });

    return { accuracy, confMat, report };
  }, [treeModel, testData, classNames]);

  // Live prediction calculations
  const livePrediction = useMemo(() => {
    const inputRow = [sepalLength, sepalWidth, petalLength, petalWidth];
    return treeModel.predictRow(treeModel.rootNode, inputRow);
  }, [treeModel, sepalLength, sepalWidth, petalLength, petalWidth]);

  // Simple clean visual node mapping for visual tree
  const renderTreeHtml = (node: TreeNode, depth: number = 0): React.ReactNode => {
    const featureLabels: Record<string, string> = {
      sepal_length: 'Sepal L',
      sepal_width: 'Sepal W',
      petal_length: 'Petal L',
      petal_width: 'Petal W'
    };

    // Find out if live inputs would pass through this node
    const val = [sepalLength, sepalWidth, petalLength, petalWidth][node.featureIndex ?? 0];
    const isHighlightNode = true;

    return (
      <div key={node.id} className="flex flex-col items-center">
        {/* Node Box */}
        <div className={`p-3.5 rounded-xl border text-center transition-all duration-300 shadow-sm max-w-[210px] min-w-[155px]
          ${node.isLeaf 
            ? 'bg-emerald-50 border-emerald-200 text-emerald-800' 
            : 'bg-white border-slate-200 text-slate-800 shadow-sm'}`}>
          
          <div className={`text-xs font-mono uppercase tracking-wide font-semibold ${node.isLeaf ? 'text-emerald-600' : 'text-slate-400'}`}>
            {node.isLeaf ? '🍃 Leaf Node' : '🌿 Split Node'}
          </div>

          <div className="text-sm font-bold mt-1 text-slate-900">
            {node.isLeaf 
              ? `Class: ${node.value}` 
              : `${featureLabels[node.featureName!] || node.featureName} ≤ ${node.threshold!.toFixed(2)}`}
          </div>

          <div className="grid grid-cols-2 gap-1 mt-2 text-xs border-t border-slate-100 pt-1.5">
            <span className="text-slate-500">Samples:</span>
            <span className="font-bold text-right text-slate-800">{node.samples}</span>
            <span className="text-slate-500">{criterion === 'gini' ? 'Gini:' : 'Entropy:'}</span>
            <span className="font-bold text-right text-slate-800">{node.impurity.toFixed(3)}</span>
          </div>

          {/* Value distribution pie indicator */}
          <div className="flex gap-1 justify-center mt-2 pt-2 border-t border-slate-100">
            {Object.entries(node.distribution).map(([species, count]) => {
              const bg = species === 'setosa' ? 'bg-rose-500' : species === 'versicolor' ? 'bg-sky-500' : 'bg-emerald-500';
              return (
                <div 
                  key={species} 
                  title={`${species}: ${count}`}
                  className={`h-2.5 rounded-full ${bg} transition-all`}
                  style={{ width: `${node.samples > 0 ? (count / node.samples) * 100 : 0}%` }}
                />
              );
            })}
          </div>
        </div>

        {/* Children Rows */}
        {!node.isLeaf && (
          <div className="flex gap-8 mt-6 relative">
            {/* Connectors left and right */}
            <div className="absolute top-[-24px] left-1/2 w-0.5 h-6 bg-slate-200"></div>
            
            <div className="flex flex-col items-center">
              <span className="text-xs font-mono font-bold text-emerald-600 bg-slate-50 px-1.5 py-0.5 relative top-[-10px] border border-slate-200 rounded">Yes</span>
              {renderTreeHtml(node.left!, depth + 1)}
            </div>

            <div className="flex flex-col items-center">
              <span className="text-xs font-mono font-bold text-rose-600 bg-slate-50 px-1.5 py-0.5 relative top-[-10px] border border-slate-200 rounded">No</span>
              {renderTreeHtml(node.right!, depth + 1)}
            </div>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
      
      {/* Parameters Panel */}
      <div className="lg:col-span-1 bg-white border border-slate-200 p-5 rounded-xl space-y-6 h-fit shadow-sm">
        <div>
          <h3 className="text-base font-bold font-display uppercase tracking-wider text-emerald-600 flex items-center gap-2">
            <Activity size={16} /> Model Setup
          </h3>
          <p className="text-xs text-slate-500 mt-1">Configure hyper-parameters for Iris Classifier CART algorithm.</p>
        </div>

        {/* Criterion Select */}
        <div className="space-y-2">
          <label className="block text-xs font-semibold text-slate-700">Criterion</label>
          <div className="grid grid-cols-2 gap-2">
            <button 
              onClick={() => setCriterion('gini')}
              className={`py-2 px-3 rounded-lg text-xs font-bold border transition-all ${criterion === 'gini' ? 'bg-emerald-50 border-emerald-500 text-emerald-700' : 'bg-slate-50 border-slate-200 text-slate-600 hover:border-slate-300'}`}>
              Gini Impurity
            </button>
            <button 
              onClick={() => setCriterion('entropy')}
              className={`py-2 px-3 rounded-lg text-xs font-bold border transition-all ${criterion === 'entropy' ? 'bg-emerald-50 border-emerald-500 text-emerald-700' : 'bg-slate-50 border-slate-200 text-slate-600 hover:border-slate-300'}`}>
              Entropy
            </button>
          </div>
        </div>

        {/* Max Depth Slider */}
        <div className="space-y-2">
          <div className="flex justify-between text-xs font-medium">
            <span className="text-slate-700">Max Depth</span>
            <span className="font-mono font-bold text-emerald-600">{maxDepth}</span>
          </div>
          <input 
            type="range" 
            min="1" 
            max="10" 
            value={maxDepth} 
            onChange={(e) => setMaxDepth(Number(e.target.value))}
            className="w-full accent-emerald-600 cursor-pointer h-1.5 bg-slate-100 rounded-lg"
          />
        </div>

        {/* Min Samples Split */}
        <div className="space-y-2">
          <div className="flex justify-between text-xs font-medium">
            <span className="text-slate-700">Min Samples Split</span>
            <span className="font-mono font-bold text-emerald-600">{minSamplesSplit}</span>
          </div>
          <input 
            type="range" 
            min="2" 
            max="10" 
            value={minSamplesSplit} 
            onChange={(e) => setMinSamplesSplit(Number(e.target.value))}
            className="w-full accent-emerald-600 cursor-pointer h-1.5 bg-slate-100 rounded-lg"
          />
        </div>

        {/* Live Predict Panel inside Settings (Simulating Sidebar) */}
        <div className="pt-5 border-t border-slate-100 space-y-4">
          <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider">🌸 Live Prediction Input</h4>
          
          <div className="space-y-3">
            {/* Sepal Length */}
            <div className="space-y-1">
              <div className="flex justify-between text-xs font-medium text-slate-500">
                <span>Sepal Length</span>
                <span className="font-mono font-bold text-emerald-600">{sepalLength.toFixed(1)} cm</span>
              </div>
              <input 
                type="range" 
                min="4.3" 
                max="7.9" 
                step="0.1"
                value={sepalLength} 
                onChange={(e) => setSepalLength(Number(e.target.value))}
                className="w-full accent-emerald-600 cursor-pointer h-1.5 bg-slate-100 rounded-lg"
              />
            </div>

            {/* Sepal Width */}
            <div className="space-y-1">
              <div className="flex justify-between text-xs font-medium text-slate-500">
                <span>Sepal Width</span>
                <span className="font-mono font-bold text-emerald-600">{sepalWidth.toFixed(1)} cm</span>
              </div>
              <input 
                type="range" 
                min="2.0" 
                max="4.4" 
                step="0.1"
                value={sepalWidth} 
                onChange={(e) => setSepalWidth(Number(e.target.value))}
                className="w-full accent-emerald-600 cursor-pointer h-1.5 bg-slate-100 rounded-lg"
              />
            </div>

            {/* Petal Length */}
            <div className="space-y-1">
              <div className="flex justify-between text-xs font-medium text-slate-500">
                <span>Petal Length</span>
                <span className="font-mono font-bold text-emerald-600">{petalLength.toFixed(1)} cm</span>
              </div>
              <input 
                type="range" 
                min="1.0" 
                max="6.9" 
                step="0.1"
                value={petalLength} 
                onChange={(e) => setPetalLength(Number(e.target.value))}
                className="w-full accent-emerald-600 cursor-pointer h-1.5 bg-slate-100 rounded-lg"
              />
            </div>

            {/* Petal Width */}
            <div className="space-y-1">
              <div className="flex justify-between text-xs font-medium text-slate-500">
                <span>Petal Width</span>
                <span className="font-mono font-bold text-emerald-600">{petalWidth.toFixed(1)} cm</span>
              </div>
              <input 
                type="range" 
                min="0.1" 
                max="2.5" 
                step="0.1"
                value={petalWidth} 
                onChange={(e) => setPetalWidth(Number(e.target.value))}
                className="w-full accent-emerald-600 cursor-pointer h-1.5 bg-slate-100 rounded-lg"
              />
            </div>
          </div>
        </div>
      </div>

      {/* Main Analysis Panels */}
      <div className="lg:col-span-3 space-y-6">
        
        {/* Top metrics and dataset info */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-white border border-slate-200 p-4 rounded-xl flex flex-col items-center justify-center shadow-sm">
            <span className="text-xs uppercase font-bold tracking-wider text-slate-500">Total Samples</span>
            <span className="text-3xl font-mono font-extrabold mt-1 text-emerald-600">{allData.length}</span>
            <span className="text-xs text-slate-400 mt-1 font-medium">Split: 120 Train / 30 Test</span>
          </div>

          <div className="bg-white border border-slate-200 p-4 rounded-xl flex flex-col items-center justify-center shadow-sm">
            <span className="text-xs uppercase font-bold tracking-wider text-slate-500">Criterion Impurity</span>
            <span className="text-3xl font-mono font-extrabold mt-1 text-emerald-600 capitalize">{criterion}</span>
            <span className="text-xs text-slate-400 mt-1 font-medium">Split splitting metric</span>
          </div>

          <div className="bg-emerald-50 border border-emerald-100 p-4 rounded-xl flex flex-col items-center justify-center relative overflow-hidden group shadow-sm">
            <span className="text-xs uppercase font-bold tracking-wider text-emerald-700 z-10">Test Accuracy</span>
            <span className="text-3xl font-mono font-extrabold mt-1 text-emerald-600 z-10">
              {(evaluation.accuracy * 100).toFixed(1)}%
            </span>
            <span className="text-xs text-emerald-700 mt-1 z-10 font-medium">Correct: {evaluation.accuracy * testData.length} / {testData.length}</span>
          </div>
        </div>

        {/* Prediction Results & Confusion Matrix Side-by-Side */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          
          {/* Prediction Outcome Card */}
          <div className="bg-white border border-slate-200 p-5 rounded-xl space-y-4 shadow-sm">
            <h3 className="text-sm font-bold font-display text-slate-800 uppercase tracking-wider border-b border-slate-100 pb-2">
              🔮 Live Prediction Result
            </h3>

            <div className="bg-emerald-50 border border-emerald-100 p-4 rounded-xl text-center">
              <span className="text-xs text-emerald-600 uppercase tracking-wide block font-bold">Predicted Class</span>
              <span className="text-2xl font-extrabold text-emerald-700 capitalize mt-1 block font-display">
                🌸 {livePrediction.label}
              </span>
            </div>

            <div className="space-y-2 pt-2">
              <span className="text-xs text-slate-500 uppercase tracking-wide block font-semibold">Class Probabilities</span>
              {classNames.map((species) => {
                const prob = livePrediction.proba[species] || 0;
                const pct = prob * 100;
                const color = species === 'setosa' ? 'bg-rose-500' : species === 'versicolor' ? 'bg-sky-500' : 'bg-emerald-500';
                const textCol = species === 'setosa' ? 'text-rose-600' : species === 'versicolor' ? 'text-sky-600' : 'text-emerald-600';
                return (
                  <div key={species} className="space-y-1">
                    <div className="flex justify-between text-xs font-semibold">
                      <span className="capitalize font-medium text-slate-700">{species}</span>
                      <span className={`font-mono font-bold ${textCol}`}>{pct.toFixed(0)}%</span>
                    </div>
                    <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                      <div 
                        className={`h-full ${color} rounded-full transition-all duration-500`} 
                        style={{ width: `${pct}%` }}
                      ></div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Confusion Matrix Card */}
          <div className="bg-white border border-slate-200 p-5 rounded-xl flex flex-col justify-between shadow-sm">
            <div>
              <h3 className="text-sm font-bold font-display text-slate-800 uppercase tracking-wider border-b border-slate-100 pb-2 mb-4">
                📉 Confusion Matrix (Test Set)
              </h3>
              
              <div className="grid grid-cols-4 gap-2 text-center text-xs">
                {/* Headers */}
                <div className="text-slate-400 text-[10px] flex items-center justify-center font-mono uppercase">Pred → Actual ↓</div>
                {classNames.map(c => (
                  <div key={c} className="font-bold text-slate-600 capitalize text-xs truncate">{c}</div>
                ))}

                {/* Rows */}
                {classNames.map(actual => (
                  <React.Fragment key={actual}>
                    <div className="font-bold text-slate-600 capitalize text-xs flex items-center justify-start text-left pl-1 truncate">
                      {actual}
                    </div>
                    {classNames.map(pred => {
                      const count = evaluation.confMat[actual][pred] || 0;
                      const isDiagonal = actual === pred;
                      return (
                        <div 
                          key={pred} 
                          className={`p-3 rounded-lg font-mono font-bold text-sm flex items-center justify-center transition-all
                            ${isDiagonal 
                              ? count > 0 ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' : 'bg-slate-100 text-slate-500'
                              : count > 0 ? 'bg-rose-50 text-rose-600 border border-rose-200' : 'bg-slate-50 text-slate-300'}`}>
                          {count}
                        </div>
                      );
                    })}
                  </React.Fragment>
                ))}
              </div>
            </div>

            <div className="bg-slate-50 border border-slate-100 p-3 rounded-lg text-xs text-slate-600 mt-4 flex items-start gap-2">
              <Info size={14} className="text-emerald-600 flex-shrink-0 mt-0.5" />
              <span>
                Diagonals show correct predictions (True Positives). Off-diagonals show classification errors.
              </span>
            </div>
          </div>
        </div>

        {/* Classification Report Table */}
        <div className="bg-white border border-slate-200 p-5 rounded-xl shadow-sm">
          <h3 className="text-sm font-bold font-display text-slate-800 uppercase tracking-wider border-b border-slate-100 pb-2 mb-4">
            📄 Classification Report (Detailed Metrics)
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-slate-100 text-slate-500 uppercase text-xs tracking-wider font-semibold">
                  <th className="py-2">Class Name</th>
                  <th className="py-2 text-right">Precision</th>
                  <th className="py-2 text-right">Recall</th>
                  <th className="py-2 text-right">F1-Score</th>
                  <th className="py-2 text-right">Support</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {classNames.map(cls => {
                  const m = evaluation.report[cls] || { precision: 0, recall: 0, f1: 0, support: 0 };
                  return (
                    <tr key={cls} className="hover:bg-slate-50/50">
                      <td className="py-3 font-semibold capitalize text-slate-800">{cls}</td>
                      <td className="py-3 text-right font-mono text-emerald-600 font-bold">{(m.precision * 100).toFixed(0)}%</td>
                      <td className="py-3 text-right font-mono text-emerald-600 font-bold">{(m.recall * 100).toFixed(0)}%</td>
                      <td className="py-3 text-right font-mono text-blue-600 font-bold">{(m.f1 * 100).toFixed(0)}%</td>
                      <td className="py-3 text-right font-mono text-slate-500">{m.support}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* Scrollable Tree Diagram Container */}
        <div className="bg-white border border-slate-200 p-6 rounded-xl overflow-x-auto shadow-sm">
          <h3 className="text-sm font-bold font-display text-slate-800 uppercase tracking-wider border-b border-slate-100 pb-2 mb-6">
            🌳 Interactive Decision Tree Visualization
          </h3>
          
          <div className="flex justify-center min-w-[700px] py-4">
            {renderTreeHtml(treeModel.rootNode)}
          </div>
        </div>

      </div>
    </div>
  );
}
