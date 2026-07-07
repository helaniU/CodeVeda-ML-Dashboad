import React, { useState, useMemo } from 'react';
import Papa from 'papaparse';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, LineChart, Line, BarChart, Bar, ReferenceLine } from 'recharts';
import irisCsv from '../../datasets/1) iris.csv?raw';
import { IrisData } from '../types';
import { Activity, Download, Info, Settings } from 'lucide-react';

// Standardize features helper
function standardize(data: number[][]): number[][] {
  if (!data || data.length === 0 || !data[0]) return [];
  const numCols = data[0].length;
  const means = Array(numCols).fill(0);
  const stds = Array(numCols).fill(0);

  // Mean
  let validCount = 0;
  for (let i = 0; i < data.length; i++) {
    if (!data[i]) continue;
    validCount++;
    for (let j = 0; j < numCols; j++) {
      const val = data[i][j];
      means[j] += isNaN(val) || val === null || val === undefined ? 0 : val;
    }
  }
  if (validCount > 0) {
    for (let j = 0; j < numCols; j++) means[j] /= validCount;
  }

  // Std Dev
  for (let i = 0; i < data.length; i++) {
    if (!data[i]) continue;
    for (let j = 0; j < numCols; j++) {
      const val = data[i][j];
      const cleanVal = isNaN(val) || val === null || val === undefined ? 0 : val;
      stds[j] += Math.pow(cleanVal - means[j], 2);
    }
  }
  if (validCount > 0) {
    for (let j = 0; j < numCols; j++) stds[j] = Math.sqrt(stds[j] / validCount) || 1;
  } else {
    stds.fill(1);
  }

  // Standardize
  return data.map(row => {
    if (!row) return Array(numCols).fill(0);
    return row.map((val, colIdx) => {
      const cleanVal = isNaN(val) || val === null || val === undefined ? 0 : val;
      return (cleanVal - means[colIdx]) / (stds[colIdx] || 1);
    });
  });
}

// 2D PCA implementation (SVD / Power Iteration method for dimensional reduction)
function runPCA2D(data: number[][]): number[][] {
  if (!data || data.length === 0 || !data[0]) return [];
  const n = data.length;
  const m = data[0].length;
  
  // Covariance matrix
  const cov = Array(m).fill(0).map(() => Array(m).fill(0));
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < m; j++) {
      let sum = 0;
      let count = 0;
      for (let k = 0; k < n; k++) {
        if (data[k] && data[k][i] !== undefined && data[k][j] !== undefined && !isNaN(data[k][i]) && !isNaN(data[k][j])) {
          sum += data[k][i] * data[k][j];
          count++;
        }
      }
      cov[i][j] = count > 1 ? sum / (count - 1) : 0;
    }
  }

  // Find top 2 eigenvectors via simple Power Iteration
  const getTopEigenvector = (matrix: number[][], numIters = 50): number[] => {
    let vec = Array(m).fill(0).map(() => Math.random() - 0.5);
    for (let iter = 0; iter < numIters; iter++) {
      let nextVec = Array(m).fill(0);
      for (let i = 0; i < m; i++) {
        for (let j = 0; j < m; j++) {
          if (matrix[i] && matrix[i][j] !== undefined && vec[j] !== undefined) {
            nextVec[i] += matrix[i][j] * vec[j];
          }
        }
      }
      const len = Math.sqrt(nextVec.reduce((sum, v) => sum + v * v, 0)) || 1;
      vec = nextVec.map(v => v / len);
    }
    return vec;
  };

  // Find PC1
  const pc1 = getTopEigenvector(cov);

  // Deflate covariance matrix to find PC2
  const cov2 = cov.map((row, i) => 
    row.map((val, j) => val - (pc1[i] || 0) * (pc1[j] || 0) * (cov[i] ? cov[i][j] || 0 : 0))
  );
  const pc2 = getTopEigenvector(cov2);

  // Project data onto PC1 and PC2
  return data.map(row => {
    if (!row) return [0, 0];
    const x = row.reduce((sum, val, idx) => sum + (isNaN(val) ? 0 : val) * (pc1[idx] || 0), 0);
    const y = row.reduce((sum, val, idx) => sum + (isNaN(val) ? 0 : val) * (pc2[idx] || 0), 0);
    return [x, y];
  });
}

// KMeans algorithm in pure TS
function runKMeans(data: number[][], k: number, maxIters = 50, seed = 42): {
  clusters: number[];
  centroids: number[][];
  wcss: number;
} {
  if (!data || data.length === 0 || !data[0]) {
    return { clusters: [], centroids: [], wcss: 0 };
  }
  const n = data.length;
  const m = data[0].length;
  
  // Seeded random centroid initialization
  let s = seed;
  const rand = () => {
    let x = Math.sin(s++) * 10000;
    return x - Math.floor(x);
  };

  const centroids: number[][] = [];
  const chosenIdxs = new Set<number>();
  const maxAttempts = n * 2;
  let attempts = 0;
  while (centroids.length < k && attempts < maxAttempts) {
    attempts++;
    const idx = Math.floor(rand() * n);
    if (!chosenIdxs.has(idx) && data[idx]) {
      chosenIdxs.add(idx);
      const cleanCentroid = [...data[idx]].map(v => isNaN(v) || v === null || v === undefined ? 0 : v);
      centroids.push(cleanCentroid);
    }
  }
  // Fill rest with zeros if we couldn't find enough centroids
  while (centroids.length < k) {
    centroids.push(Array(m).fill(0));
  }

  let clusters = Array(n).fill(-1);
  let wcss = 0;

  for (let iter = 0; iter < maxIters; iter++) {
    const nextClusters = Array(n).fill(-1);
    
    // Assign clusters
    for (let i = 0; i < n; i++) {
      let minDist = Infinity;
      let minCluster = -1;
      if (!data[i]) continue;
      for (let c = 0; c < k; c++) {
        let dist = 0;
        for (let j = 0; j < m; j++) {
          const val = data[i][j];
          const cleanVal = isNaN(val) || val === null || val === undefined ? 0 : val;
          const centVal = centroids[c] ? centroids[c][j] || 0 : 0;
          dist += Math.pow(cleanVal - centVal, 2);
        }
        if (!isNaN(dist) && dist < minDist) {
          minDist = dist;
          minCluster = c;
        }
      }
      nextClusters[i] = minCluster >= 0 ? minCluster : 0; // Default to cluster 0 if everything failed
    }

    // Check convergence
    let changed = false;
    for (let i = 0; i < n; i++) {
      if (nextClusters[i] !== clusters[i]) {
        changed = true;
        break;
      }
    }
    clusters = nextClusters;
    if (!changed) break;

    // Recalculate centroids
    const counts = Array(k).fill(0);
    const sums = Array(k).fill(0).map(() => Array(m).fill(0));
    for (let i = 0; i < n; i++) {
      const c = clusters[i];
      if (c >= 0 && c < k) {
        counts[c]++;
        for (let j = 0; j < m; j++) {
          if (data[i]) {
            const val = data[i][j];
            sums[c][j] += isNaN(val) || val === null || val === undefined ? 0 : val;
          }
        }
      }
    }

    for (let c = 0; c < k; c++) {
      if (counts[c] > 0) {
        for (let j = 0; j < m; j++) {
          if (centroids[c] && sums[c]) {
            centroids[c][j] = sums[c][j] / counts[c];
          }
        }
      }
    }
  }

  // Calculate WCSS (inertia)
  wcss = 0;
  for (let i = 0; i < n; i++) {
    const c = clusters[i];
    if (c >= 0 && c < k && centroids[c]) {
      let dist = 0;
      for (let j = 0; j < m; j++) {
        if (data[i]) {
          const val = data[i][j];
          const cleanVal = isNaN(val) || val === null || val === undefined ? 0 : val;
          dist += Math.pow(cleanVal - (centroids[c][j] || 0), 2);
        }
      }
      wcss += dist;
    }
  }

  return { clusters, centroids, wcss };
}

// DBSCAN algorithm in pure TS
function runDBSCAN(data: number[][], eps: number, minSamples: number): number[] {
  if (!data || data.length === 0) return [];
  const n = data.length;
  const labels = Array(n).fill(-2); // -2: unvisited, -1: noise
  let clusterId = 0;

  const getNeighbors = (pointIdx: number): number[] => {
    const neighbors: number[] = [];
    const p1 = data[pointIdx];
    if (!p1) return [];
    for (let i = 0; i < n; i++) {
      const p2 = data[i];
      if (!p2) continue;
      let distSq = 0;
      const len = Math.min(p1.length, p2.length);
      for (let j = 0; j < len; j++) {
        const v1 = isNaN(p1[j]) ? 0 : p1[j];
        const v2 = isNaN(p2[j]) ? 0 : p2[j];
        distSq += Math.pow(v1 - v2, 2);
      }
      if (Math.sqrt(distSq) <= eps) neighbors.push(i);
    }
    return neighbors;
  };

  const expandCluster = (pointIdx: number, neighbors: number[], cid: number) => {
    labels[pointIdx] = cid;
    let i = 0;
    while (i < neighbors.length) {
      const neighborIdx = neighbors[i];
      if (labels[neighborIdx] === -1) {
        // Noise becomes border point
        labels[neighborIdx] = cid;
      } else if (labels[neighborIdx] === -2) {
        labels[neighborIdx] = cid;
        const subNeighbors = getNeighbors(neighborIdx);
        if (subNeighbors.length >= minSamples) {
          neighbors.push(...subNeighbors.filter(idx => !neighbors.includes(idx)));
        }
      }
      i++;
    }
  };

  for (let i = 0; i < n; i++) {
    if (labels[i] !== -2) continue;
    const neighbors = getNeighbors(i);
    if (neighbors.length < minSamples) {
      labels[i] = -1; // noise
    } else {
      expandCluster(i, neighbors, clusterId++);
    }
  }

  return labels;
}

// Silhouette Score calculator
function calculateSilhouetteScore(data: number[][], labels: number[]): number {
  if (!data || data.length < 2 || !labels || labels.length === 0) return 0;
  const n = data.length;
  if (!data[0]) return 0;
  const m = data[0].length;
  const uniqueClusters = Array.from(new Set(labels)).filter(l => l !== -1 && l !== undefined && l !== null); // ignore noise and invalid values in dbscan
  if (uniqueClusters.length < 2) return 0;

  let totalScore = 0;
  let validPoints = 0;

  const distance = (a: number[], b: number[]) => {
    if (!a || !b) return 0;
    let sum = 0;
    const len = Math.min(a.length, b.length);
    for (let i = 0; i < len; i++) {
      if (a[i] !== undefined && b[i] !== undefined) {
        sum += Math.pow(a[i] - b[i], 2);
      }
    }
    return Math.sqrt(sum);
  };

  for (let i = 0; i < n; i++) {
    const labelI = labels[i];
    if (labelI === -1 || labelI === undefined || labelI === null) continue; // skip noise and invalid labels

    // Calculate a(i): average distance to other points in same cluster
    let sameClusterSum = 0;
    let sameClusterCount = 0;
    for (let j = 0; j < n; j++) {
      if (i !== j && labels[j] === labelI && data[i] && data[j]) {
        sameClusterSum += distance(data[i], data[j]);
        sameClusterCount++;
      }
    }
    const ai = sameClusterCount > 0 ? sameClusterSum / sameClusterCount : 0;

    // Calculate b(i): average distance to points in nearest other cluster
    let minBi = Infinity;
    uniqueClusters.forEach(otherLabel => {
      if (otherLabel === labelI) return;
      let otherClusterSum = 0;
      let otherClusterCount = 0;
      for (let j = 0; j < n; j++) {
        if (labels[j] === otherLabel && data[i] && data[j]) {
          otherClusterSum += distance(data[i], data[j]);
          otherClusterCount++;
        }
      }
      const avgDist = otherClusterCount > 0 ? otherClusterSum / otherClusterCount : 0;
      if (avgDist < minBi) minBi = avgDist;
    });

    const bi = minBi;
    const maxVal = Math.max(ai, bi);
    if (maxVal > 0) {
      totalScore += (bi - ai) / maxVal;
      validPoints++;
    }
  }

  return validPoints > 0 ? totalScore / validPoints : 0;
}

export default function KMeansClustering() {
  const [algo, setAlgo] = useState<'KMeans' | 'DBSCAN'>('KMeans');
  const [k, setK] = useState<number>(3);
  const [eps, setEps] = useState<number>(0.8);
  const [minSamples, setMinSamples] = useState<number>(5);
  const [selectedCluster, setSelectedCluster] = useState<number>(0);

  // Color Palette for Clusters (up to 10 clusters)
  const CLUSTER_COLORS = [
    '#3b82f6', // blue
    '#10b981', // emerald
    '#eab308', // yellow
    '#ec4899', // pink
    '#8b5cf6', // purple
    '#f97316', // orange
    '#14b8a6', // teal
    '#ef4444', // red
    '#6366f1', // indigo
    '#06b6d4', // cyan
  ];

  // Parse Raw Iris CSV
  const { originalData, X, features } = useMemo(() => {
    const parsed = Papa.parse<any>(irisCsv, { header: true, dynamicTyping: true });
    const cleanData: IrisData[] = parsed.data
      .filter((row: any) => row.sepal_length !== undefined)
      .map((row: any) => ({
        sepal_length: Number(row.sepal_length),
        sepal_width: Number(row.sepal_width),
        petal_length: Number(row.petal_length),
        petal_width: Number(row.petal_width),
        species: String(row.species).trim().toLowerCase()
      }));

    const XMatrix = cleanData.map(d => [d.sepal_length, d.sepal_width, d.petal_length, d.petal_width]);
    return {
      originalData: cleanData,
      X: XMatrix,
      features: ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    };
  }, []);

  // Standardized Matrix
  const X_scaled = useMemo(() => standardize(X), [X]);

  // PCA reduced 2D projections
  const X_pca = useMemo(() => runPCA2D(X_scaled), [X_scaled]);

  // Model Selection metrics (Inertia & Sil scores vs K for K=2 to 10)
  const modelSelectionMetrics = useMemo(() => {
    const metrics: { k: number; inertia: number; silhouette: number }[] = [];
    for (let i = 2; i <= 10; i++) {
      const { clusters, wcss } = runKMeans(X_scaled, i);
      const sil = calculateSilhouetteScore(X_scaled, clusters);
      metrics.push({ k: i, inertia: wcss, silhouette: sil });
    }
    return metrics;
  }, [X_scaled]);

  // Active Clustering Calculation
  const activeClustering = useMemo(() => {
    if (algo === 'KMeans') {
      const { clusters, centroids, wcss } = runKMeans(X_scaled, k);
      const silhouette = calculateSilhouetteScore(X_scaled, clusters);
      return { clusters, centroids, score: silhouette, wcss };
    } else {
      const clusters = runDBSCAN(X_scaled, eps, minSamples);
      const silhouette = calculateSilhouetteScore(X_scaled, clusters);
      return { clusters, score: silhouette, wcss: 0 };
    }
  }, [algo, k, eps, minSamples, X_scaled]);

  // Clustered DataFrame structure
  const clusteredData = useMemo(() => {
    return originalData.map((d, i) => ({
      ...d,
      pc1: X_pca[i] ? X_pca[i][0] : 0,
      pc2: X_pca[i] ? X_pca[i][1] : 0,
      cluster: activeClustering.clusters && activeClustering.clusters[i] !== undefined ? activeClustering.clusters[i] : -1
    }));
  }, [originalData, X_pca, activeClustering]);

  // Cluster counts and mean values
  const clusterAnalysis = useMemo(() => {
    const counts: Record<number, number> = {};
    const sums: Record<number, Record<string, number>> = {};
    
    clusteredData.forEach(row => {
      const c = row.cluster;
      counts[c] = (counts[c] || 0) + 1;
      
      if (!sums[c]) {
        sums[c] = { sepal_length: 0, sepal_width: 0, petal_length: 0, petal_width: 0 };
      }
      sums[c].sepal_length += row.sepal_length;
      sums[c].sepal_width += row.sepal_width;
      sums[c].petal_length += row.petal_length;
      sums[c].petal_width += row.petal_width;
    });

    const stats = Object.keys(counts).map(cStr => {
      const c = Number(cStr);
      const n = counts[c];
      return {
        cluster: c,
        count: n,
        sepal_length: sums[c].sepal_length / n,
        sepal_width: sums[c].sepal_width / n,
        petal_length: sums[c].petal_length / n,
        petal_width: sums[c].petal_width / n,
      };
    }).sort((a, b) => a.cluster - b.cluster);

    return { counts, stats };
  }, [clusteredData]);

  // PCA Scatter plot dataset
  const scatterDataset = useMemo(() => {
    return clusteredData.map(d => ({
      x: d.pc1,
      y: d.pc2,
      cluster: d.cluster,
      species: d.species,
      sepal_length: d.sepal_length,
      sepal_width: d.sepal_width,
      petal_length: d.petal_length,
      petal_width: d.petal_width,
    }));
  }, [clusteredData]);

  // Centroids coordinates projected onto 2D PCA space
  const centroidPcaCoordinates = useMemo(() => {
    if (algo !== 'KMeans' || !activeClustering.centroids) return [];
    // PCA projections on centroids
    const n = X_scaled.length;
    const m = X_scaled[0].length;
    
    // Recalculate covariance eigenvectors again to project centers properly
    const cov = Array(m).fill(0).map(() => Array(m).fill(0));
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < m; j++) {
        let sum = 0;
        for (let k = 0; k < n; k++) sum += X_scaled[k][i] * X_scaled[k][j];
        cov[i][j] = sum / (n - 1);
      }
    }

    const getTopEigenvector = (matrix: number[][], numIters = 50): number[] => {
      let vec = Array(m).fill(0).map(() => Math.random() - 0.5);
      for (let iter = 0; iter < numIters; iter++) {
        let nextVec = Array(m).fill(0);
        for (let i = 0; i < m; i++) {
          for (let j = 0; j < m; j++) nextVec[i] += matrix[i][j] * vec[j];
        }
        const len = Math.sqrt(nextVec.reduce((sum, v) => sum + v * v, 0)) || 1;
        vec = nextVec.map(v => v / len);
      }
      return vec;
    };

    const pc1 = getTopEigenvector(cov);
    const cov2 = cov.map((row, i) => row.map((val, j) => val - pc1[i] * pc1[j] * cov[i][j]));
    const pc2 = getTopEigenvector(cov2);

    return activeClustering.centroids.map((centroid, idx) => {
      const x = centroid.reduce((sum, val, cIdx) => sum + val * pc1[cIdx], 0);
      const y = centroid.reduce((sum, val, cIdx) => sum + val * pc2[cIdx], 0);
      return { x, y, cluster: idx };
    });
  }, [algo, activeClustering, X_scaled]);

  // Feature Influence / Separation (Variance of means of features across clusters)
  const featureInfluence = useMemo(() => {
    const stats = clusterAnalysis.stats.filter(s => s.cluster !== -1); // exclude noise
    if (stats.length < 2) return [];

    const result = features.map(feat => {
      const values = stats.map(s => s[feat as keyof typeof s] as number);
      // calculate variance
      const mean = values.reduce((a, b) => a + b, 0) / values.length;
      const variance = values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / values.length;
      return { feature: feat.replace('_', ' '), importance: variance };
    });

    return result.sort((a, b) => b.importance - a.importance);
  }, [clusterAnalysis, features]);

  // Cluster explorer details
  const explorerData = useMemo(() => {
    return clusteredData.filter(d => d.cluster === selectedCluster);
  }, [clusteredData, selectedCluster]);

  const explorerStats = useMemo(() => {
    if (explorerData.length === 0) return null;
    const len = explorerData.length;
    const sums = { sepal_length: 0, sepal_width: 0, petal_length: 0, petal_width: 0 };
    const mins = { sepal_length: Infinity, sepal_width: Infinity, petal_length: Infinity, petal_width: Infinity };
    const maxs = { sepal_length: -Infinity, sepal_width: -Infinity, petal_length: -Infinity, petal_width: -Infinity };

    explorerData.forEach(d => {
      features.forEach(f => {
        const val = d[f as keyof typeof d] as number;
        sums[f as keyof typeof sums] += val;
        if (val < mins[f as keyof typeof mins]) mins[f as keyof typeof mins] = val;
        if (val > maxs[f as keyof typeof maxs]) maxs[f as keyof typeof maxs] = val;
      });
    });

    return features.map(f => ({
      feature: f.replace('_', ' '),
      mean: sums[f as keyof typeof sums] / len,
      min: mins[f as keyof typeof mins],
      max: maxs[f as keyof typeof maxs]
    }));
  }, [explorerData, features]);

  // Download Clustered Data CSV
  const downloadCsv = () => {
    const csvContent = Papa.unparse(clusteredData);
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `${algo.toLowerCase()}_clusters.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
      
      {/* Settings Side Panel */}
      <div className="lg:col-span-1 bg-white border border-slate-200 p-5 rounded-xl space-y-6 h-fit shadow-sm">
        <div>
          <h3 className="text-sm font-bold font-display uppercase tracking-wider text-amber-600 flex items-center gap-2">
            <Settings size={16} /> Clustering Controls
          </h3>
          <p className="text-xs text-slate-500 mt-1">Select algorithm and tune parameters in real-time.</p>
        </div>

        {/* Algorithm Selection */}
        <div className="space-y-2">
          <label className="block text-xs font-semibold text-slate-700">Choose Algorithm</label>
          <div className="grid grid-cols-2 gap-2">
            <button 
              onClick={() => setAlgo('KMeans')}
              className={`py-2 px-3 rounded-lg text-xs font-bold border transition-all ${algo === 'KMeans' ? 'bg-amber-50 border-amber-500 text-amber-700' : 'bg-slate-50 border-slate-200 text-slate-600 hover:border-slate-300'}`}>
              K-Means
            </button>
            <button 
              onClick={() => setAlgo('DBSCAN')}
              className={`py-2 px-3 rounded-lg text-xs font-bold border transition-all ${algo === 'DBSCAN' ? 'bg-amber-50 border-amber-500 text-amber-700' : 'bg-slate-50 border-slate-200 text-slate-600 hover:border-slate-300'}`}>
              DBSCAN
            </button>
          </div>
        </div>

        {/* Dynamic Controls based on selected algorithm */}
        {algo === 'KMeans' ? (
          <div className="space-y-4 pt-4 border-t border-slate-100">
            <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider">KMeans Settings</h4>
            {/* Number of Clusters slider */}
            <div className="space-y-2">
              <div className="flex justify-between text-xs font-medium">
                <span className="text-slate-600">Clusters (K)</span>
                <span className="font-mono text-amber-600 font-bold">{k}</span>
              </div>
              <input 
                type="range" 
                min="2" 
                max="10" 
                value={k} 
                onChange={(e) => {
                  const val = Number(e.target.value);
                  setK(val);
                  if (selectedCluster >= val) setSelectedCluster(0);
                }}
                className="w-full accent-amber-500 cursor-pointer h-1.5 bg-slate-100 rounded-lg"
              />
            </div>
          </div>
        ) : (
          <div className="space-y-4 pt-4 border-t border-slate-100">
            <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider">DBSCAN Settings</h4>
            {/* Epsilon slider */}
            <div className="space-y-2">
              <div className="flex justify-between text-xs font-medium">
                <span className="text-slate-600">Epsilon (eps)</span>
                <span className="font-mono text-amber-600 font-bold">{eps.toFixed(2)}</span>
              </div>
              <input 
                type="range" 
                min="0.1" 
                max="5.0" 
                step="0.1"
                value={eps} 
                onChange={(e) => setEps(Number(e.target.value))}
                className="w-full accent-amber-500 cursor-pointer h-1.5 bg-slate-100 rounded-lg"
              />
            </div>

            {/* Min Samples slider */}
            <div className="space-y-2">
              <div className="flex justify-between text-xs font-medium">
                <span className="text-slate-600">Min Samples</span>
                <span className="font-mono text-amber-600 font-bold">{minSamples}</span>
              </div>
              <input 
                type="range" 
                min="2" 
                max="10" 
                value={minSamples} 
                onChange={(e) => setMinSamples(Number(e.target.value))}
                className="w-full accent-amber-500 cursor-pointer h-1.5 bg-slate-100 rounded-lg"
              />
            </div>
          </div>
        )}

        {/* Export clustered data */}
        <button 
          onClick={downloadCsv}
          className="w-full py-2.5 px-4 rounded-lg bg-amber-500 hover:bg-amber-600 text-white font-bold text-xs flex items-center justify-center gap-2 transition-all shadow-sm">
          <Download size={14} /> Download Clustered Data
        </button>
      </div>

      {/* Analysis Main View */}
      <div className="lg:col-span-3 space-y-6">

        {/* Global Stats bar */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-white border border-slate-200 p-4 rounded-xl flex flex-col items-center justify-center shadow-sm">
            <span className="text-xs uppercase font-bold tracking-wider text-slate-500">Total Records</span>
            <span className="text-3xl font-mono font-extrabold mt-1 text-amber-600">{originalData.length}</span>
            <span className="text-xs text-slate-400 mt-1 font-medium">4 Features (standardized before clustering)</span>
          </div>

          <div className="bg-amber-50 border border-amber-100 p-4 rounded-xl flex flex-col items-center justify-center shadow-sm">
            <span className="text-xs uppercase font-bold tracking-wider text-amber-800">Silhouette Score</span>
            <span className="text-3xl font-mono font-extrabold mt-1 text-amber-700">
              {activeClustering.score.toFixed(3)}
            </span>
            <span className="text-xs text-amber-800 mt-1 font-medium">
              {algo === 'KMeans' ? `For Current K = ${k}` : `DBSCAN (Density-based, noise-ignored)`}
            </span>
          </div>
        </div>

        {/* KMeans Selection curves (Elbow & Silhouette) */}
        {algo === 'KMeans' && (
          <div className="bg-white border border-slate-200 p-5 rounded-xl shadow-sm">
            <h3 className="text-sm font-bold font-display text-slate-800 uppercase tracking-wider border-b border-slate-100 pb-2 mb-4">
              📉 Model Selection Analysis (KMeans Parameter Tuning)
            </h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Elbow Plot (WCSS) */}
              <div className="space-y-2">
                <span className="text-xs font-semibold text-slate-500 block text-center">Elbow Method (Inertia/WCSS)</span>
                <div className="h-48">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={modelSelectionMetrics} margin={{ top: 10, right: 10, bottom: 5, left: -10 }}>
                      <CartesianGrid stroke="#f1f5f9" strokeDasharray="3 3" />
                      <XAxis dataKey="k" stroke="#94a3b8" tick={{ fontSize: 10 }} />
                      <YAxis stroke="#94a3b8" tick={{ fontSize: 10 }} />
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#ffffff', borderColor: '#e2e8f0', color: '#1e293b' }}
                        labelFormatter={(label) => `K = ${label}`}
                      />
                      <Line type="monotone" dataKey="inertia" stroke="#3b82f6" strokeWidth={2} activeDot={{ r: 6 }} dot={{ r: 4 }} />
                      <ReferenceLine x={k} stroke="#d97706" strokeDasharray="3 3" strokeWidth={1.5} label={{ value: `Selected`, fill: '#d97706', fontSize: 10, position: 'top', fontWeight: 'bold' }} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Silhouette Plot */}
              <div className="space-y-2">
                <span className="text-xs font-semibold text-slate-500 block text-center">Silhouette Score (Higher is Better)</span>
                <div className="h-48">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={modelSelectionMetrics} margin={{ top: 10, right: 10, bottom: 5, left: -10 }}>
                      <CartesianGrid stroke="#f1f5f9" strokeDasharray="3 3" />
                      <XAxis dataKey="k" stroke="#94a3b8" tick={{ fontSize: 10 }} />
                      <YAxis stroke="#94a3b8" tick={{ fontSize: 10 }} domain={[0, 1]} />
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#ffffff', borderColor: '#e2e8f0', color: '#1e293b' }}
                        labelFormatter={(label) => `K = ${label}`}
                      />
                      <Line type="monotone" dataKey="silhouette" stroke="#d97706" strokeWidth={2} activeDot={{ r: 6 }} dot={{ r: 4 }} />
                      <ReferenceLine x={k} stroke="#d97706" strokeDasharray="3 3" strokeWidth={1.5} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* PCA Cluster Scatter Visualization */}
        <div className="bg-white border border-slate-200 p-5 rounded-xl shadow-sm">
          <h3 className="text-sm font-bold font-display text-slate-800 uppercase tracking-wider border-b border-slate-100 pb-2 mb-4">
            📌 PCA Cluster View (Dimensionality Reduced to 2D)
          </h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 20, right: 20, bottom: 10, left: -10 }}>
                <CartesianGrid stroke="#f1f5f9" />
                <XAxis type="number" dataKey="x" name="PC1" stroke="#94a3b8" tick={{ fontSize: 10 }} label={{ value: 'Principal Component 1', fill: '#64748b', position: 'bottom', offset: 0, fontSize: 10, fontWeight: 'medium' }} />
                <YAxis type="number" dataKey="y" name="PC2" stroke="#94a3b8" tick={{ fontSize: 10 }} label={{ value: 'Principal Component 2', fill: '#64748b', angle: -90, position: 'left', offset: 5, fontSize: 10, fontWeight: 'medium' }} />
                <Tooltip 
                  cursor={{ strokeDasharray: '3 3' }} 
                  contentStyle={{ backgroundColor: '#ffffff', borderColor: '#e2e8f0' }}
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const data = payload[0].payload;
                      const clusterLabel = data.cluster === -1 ? 'Noise' : `Cluster ${data.cluster}`;
                      return (
                        <div className="p-3 bg-white border border-slate-200 rounded-lg text-xs space-y-1 shadow-lg">
                          <p className="font-bold text-amber-600 uppercase tracking-wider">{clusterLabel}</p>
                          <p className="capitalize text-slate-700">Species: {data.species}</p>
                          <p className="text-slate-500 text-[11px]">Sepal L: {data.sepal_length.toFixed(1)} | Sepal W: {data.sepal_width.toFixed(1)}</p>
                          <p className="text-slate-500 text-[11px]">Petal L: {data.petal_length.toFixed(1)} | Petal W: {data.petal_width.toFixed(1)}</p>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Scatter name="Points" data={scatterDataset} shape="circle">
                  {scatterDataset.map((entry, index) => {
                    const color = entry.cluster === -1 ? '#94a3b8' : CLUSTER_COLORS[entry.cluster % CLUSTER_COLORS.length];
                    return <Cell key={`cell-${index}`} fill={color} fillOpacity={0.8} stroke={color} strokeWidth={0.5} r={4.5} />;
                  })}
                </Scatter>
                
                {/* Render Centroids for KMeans */}
                {algo === 'KMeans' && (
                  <Scatter name="Centroids" data={centroidPcaCoordinates} shape="cross">
                    {centroidPcaCoordinates.map((entry, index) => (
                      <Cell key={`centroid-${index}`} fill="#ef4444" stroke="#ef4444" strokeWidth={3} r={10} />
                    ))}
                  </Scatter>
                )}
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          <div className="flex flex-wrap gap-4 justify-center text-xs mt-3">
            {clusterAnalysis.stats.map(s => {
              const color = s.cluster === -1 ? '#94a3b8' : CLUSTER_COLORS[s.cluster % CLUSTER_COLORS.length];
              return (
                <div key={s.cluster} className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: color }}></div>
                  <span className="font-semibold text-slate-700">
                    {s.cluster === -1 ? 'Noise' : `Cluster ${s.cluster}`} ({s.count} pts)
                  </span>
                </div>
              );
            })}
            {algo === 'KMeans' && (
              <div className="flex items-center gap-2">
                <div className="text-red-500 font-bold font-mono">❌</div>
                <span className="font-semibold text-red-500">Centroids</span>
              </div>
            )}
          </div>
        </div>

        {/* Cluster Stats and Feature Influence Side-by-Side */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          
          {/* Cluster Means values Table */}
          <div className="bg-white border border-slate-200 p-5 rounded-xl shadow-sm">
            <h3 className="text-sm font-bold font-display text-slate-800 uppercase tracking-wider border-b border-slate-100 pb-2 mb-4">
              🧠 Cluster Profiles (Average Feature Values)
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-slate-100 text-slate-500 uppercase text-xs font-semibold tracking-wider">
                    <th className="py-2">Cluster</th>
                    <th className="py-2 text-right">Sepal L</th>
                    <th className="py-2 text-right">Sepal W</th>
                    <th className="py-2 text-right">Petal L</th>
                    <th className="py-2 text-right">Petal W</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {clusterAnalysis.stats.map(s => {
                    const hexCol = s.cluster === -1 ? '#94a3b8' : CLUSTER_COLORS[s.cluster % CLUSTER_COLORS.length];
                    return (
                      <tr key={s.cluster} className="hover:bg-slate-50/50">
                        <td className="py-3 font-semibold flex items-center gap-1.5" style={{ color: hexCol }}>
                          <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: hexCol }}></div>
                          {s.cluster === -1 ? 'Noise' : `C${s.cluster}`}
                        </td>
                        <td className="py-3 text-right font-mono text-slate-600">{s.sepal_length.toFixed(2)} cm</td>
                        <td className="py-3 text-right font-mono text-slate-600">{s.sepal_width.toFixed(2)} cm</td>
                        <td className="py-3 text-right font-mono text-slate-600">{s.petal_length.toFixed(2)} cm</td>
                        <td className="py-3 text-right font-mono text-slate-600">{s.petal_width.toFixed(2)} cm</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Feature Influence Bar Chart */}
          <div className="bg-white border border-slate-200 p-5 rounded-xl flex flex-col justify-between shadow-sm">
            <div>
              <h3 className="text-sm font-bold font-display text-slate-800 uppercase tracking-wider border-b border-slate-100 pb-2 mb-4">
                📊 Feature Separation Strength
              </h3>
              <div className="h-44">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={featureInfluence} margin={{ top: 10, right: 10, bottom: 5, left: -15 }}>
                    <CartesianGrid stroke="#f1f5f9" strokeDasharray="3 3" />
                    <XAxis dataKey="feature" stroke="#94a3b8" tick={{ fontSize: 10 }} className="capitalize" />
                    <YAxis stroke="#94a3b8" tick={{ fontSize: 10 }} />
                    <Tooltip contentStyle={{ backgroundColor: '#ffffff', borderColor: '#e2e8f0', color: '#1e293b' }} />
                    <Bar dataKey="importance" fill="#d97706">
                      {featureInfluence.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={CLUSTER_COLORS[index % CLUSTER_COLORS.length]} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
            <div className="bg-slate-50 border border-slate-100 p-3 rounded-lg text-xs text-slate-600 mt-2 flex items-start gap-1.5 shadow-inner">
              <Info size={14} className="text-amber-500 flex-shrink-0 mt-0.5" />
              <span>
                Feature Separation indicates variance of cluster means. Features with higher values contribute more to separating the clusters.
              </span>
            </div>
          </div>
        </div>

        {/* Cluster Explorer Module */}
        <div className="bg-white border border-slate-200 p-5 rounded-xl shadow-sm">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-100 pb-3 mb-4">
            <h3 className="text-sm font-bold font-display text-slate-800 uppercase tracking-wider">
              🔍 Cluster Explorer Mode
            </h3>
            
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-500 font-medium">Select Cluster:</span>
              <select 
                value={selectedCluster} 
                onChange={(e) => setSelectedCluster(Number(e.target.value))}
                className="bg-white border border-slate-200 text-amber-600 rounded px-2.5 py-1 text-xs font-bold focus:outline-none focus:ring-2 focus:ring-amber-500/20 focus:border-amber-500">
                {Object.keys(clusterAnalysis.counts).map(cStr => {
                  const c = Number(cStr);
                  return (
                    <option key={c} value={c}>
                      {c === -1 ? 'Noise Points' : `Cluster ${c}`}
                    </option>
                  );
                })}
              </select>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            
            {/* Cluster stats cards */}
            <div className="md:col-span-1 space-y-4">
              <div className="bg-slate-50 border border-slate-200 p-4 rounded-xl text-center shadow-inner">
                <span className="text-xs uppercase text-slate-500 font-semibold">Cluster Size</span>
                <span className="text-3xl font-bold font-mono text-amber-600 block mt-1">
                  {explorerData.length}
                </span>
                <span className="text-xs text-slate-400 block mt-1 font-medium">
                  {((explorerData.length / originalData.length) * 100).toFixed(1)}% of total dataset
                </span>
              </div>

              <div className="bg-slate-50 border border-slate-200 p-4 rounded-xl shadow-inner">
                <span className="text-xs uppercase text-slate-500 block text-center border-b border-slate-200 pb-1.5 mb-2 font-semibold">Species Composition</span>
                <div className="space-y-1.5 text-xs">
                  {['setosa', 'versicolor', 'virginica'].map(sp => {
                    const count = explorerData.filter(d => d.species === sp).length;
                    const pct = explorerData.length > 0 ? (count / explorerData.length) * 100 : 0;
                    return (
                      <div key={sp} className="flex justify-between items-center text-xs">
                        <span className="capitalize text-slate-600 font-medium">{sp}:</span>
                        <span className="font-mono font-bold text-amber-600">{count} ({pct.toFixed(0)}%)</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Feature Statistics table for explorer */}
            <div className="md:col-span-2">
              {explorerStats ? (
                <div className="border border-slate-100 rounded-xl overflow-hidden shadow-sm">
                  <table className="w-full text-left text-xs">
                    <thead>
                      <tr className="bg-slate-50/50 border-b border-slate-100 text-slate-500 uppercase text-xs font-semibold tracking-wider">
                        <th className="py-2.5 px-3">Feature</th>
                        <th className="py-2.5 text-right">Minimum</th>
                        <th className="py-2.5 text-right">Average Mean</th>
                        <th className="py-2.5 px-3 text-right">Maximum</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {explorerStats.map(stat => (
                        <tr key={stat.feature} className="hover:bg-slate-50/50">
                          <td className="py-3 px-3 font-semibold capitalize text-slate-800">{stat.feature}</td>
                          <td className="py-3 text-right font-mono text-slate-500">{stat.min.toFixed(2)} cm</td>
                          <td className="py-3 text-right font-mono text-amber-600 font-bold">{stat.mean.toFixed(2)} cm</td>
                          <td className="py-3 px-3 text-right font-mono text-slate-500">{stat.max.toFixed(2)} cm</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="flex items-center justify-center h-full text-slate-400 text-xs font-medium">
                  No data in selected cluster.
                </div>
              )}
            </div>

          </div>
        </div>

      </div>
    </div>
  );
}
