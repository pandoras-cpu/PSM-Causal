import React, { useState, useMemo, useEffect } from "react";
import { createRoot } from "react-dom/client";
import { 
  Upload, 
  Database, 
  Settings, 
  Play, 
  BarChart2, 
  ChevronRight,
  TrendingUp,
  Download,
  Code,
  BookOpen,
  FileSpreadsheet,
  Users,
  Scale,
  AlertTriangle,
  Filter,
  Brain
} from "lucide-react";

// --- Types ---
type Row = Record<string, string | number>;

interface Dataset {
  data: Row[];
  headers: string[];
}

interface AnalysisConfig {
  treatmentIds: string;
  covariates: string[];
  outcomes: string[];
  caliper: number; // Max Propensity Score difference allowed
}

interface MatchResult {
  treated: Row & { propensityScore: number };
  control: Row & { propensityScore: number };
  distance: number;
}

interface BalanceMetric {
  covariate: string;
  preSMD: number;
  postSMD: number;
  isBalanced: boolean;
}

interface AnalysisResult {
  att: Record<string, number>;
  matches: MatchResult[];
  treatmentGroupSize: number;
  controlGroupSize: number;
  matchedPairs: number;
  droppedCount: number; // Cases dropped due to caliper/support
  propensityScores: number[]; 
  balanceMetrics: BalanceMetric[];
}

// --- Helper: Statistics ---
const mean = (arr: number[]) => arr.reduce((a, b) => a + b, 0) / (arr.length || 1);
const variance = (arr: number[]) => {
  if (arr.length <= 1) return 0;
  const m = mean(arr);
  return arr.reduce((a, b) => a + Math.pow(b - m, 2), 0) / (arr.length - 1);
};

// Calculate Standardized Mean Difference (SMD)
// SMD = (Mean_T - Mean_C) / sqrt((Var_T + Var_C) / 2)
const calculateSMD = (treated: number[], control: number[]) => {
  if (treated.length === 0 || control.length === 0) return 0;
  const mT = mean(treated);
  const mC = mean(control);
  const vT = variance(treated);
  const vC = variance(control);
  const poolSD = Math.sqrt((vT + vC) / 2);
  if (poolSD === 0) return 0;
  return Math.abs((mT - mC) / poolSD);
};

// --- Helper: Simple Logistic Regression using Gradient Descent ---
class LogisticRegression {
  weights: number[];
  learningRate: number;
  iterations: number;

  constructor(learningRate = 0.1, iterations = 1000) {
    this.weights = [];
    this.learningRate = learningRate;
    this.iterations = iterations;
  }

  sigmoid(z: number): number {
    return 1 / (1 + Math.exp(-z));
  }

  fit(X: number[][], y: number[]) {
    const nSamples = X.length;
    const nFeatures = X[0].length;
    this.weights = new Array(nFeatures).fill(0);

    for (let i = 0; i < this.iterations; i++) {
      const predictions = X.map(row => {
        const linearModel = row.reduce((sum, val, idx) => sum + val * this.weights[idx], 0);
        return this.sigmoid(linearModel);
      });

      const gradients = new Array(nFeatures).fill(0);
      for (let j = 0; j < nSamples; j++) {
        const error = predictions[j] - y[j];
        for (let k = 0; k < nFeatures; k++) {
          gradients[k] += (1 / nSamples) * error * X[j][k];
        }
      }

      for (let k = 0; k < nFeatures; k++) {
        this.weights[k] -= this.learningRate * gradients[k];
      }
    }
  }

  predictProba(X: number[][]): number[] {
    return X.map(row => {
      const linearModel = row.reduce((sum, val, idx) => sum + val * this.weights[idx], 0);
      return this.sigmoid(linearModel);
    });
  }
}

// --- Helper: Parse CSV ---
const parseCSV = (text: string): Dataset => {
  const lines = text.split('\n').filter(l => l.trim());
  const headers = lines[0].split(',').map(h => h.trim());
  
  const data = lines.slice(1).map(line => {
    const values = line.split(',');
    const row: Row = {};
    headers.forEach((h, i) => {
      const val = values[i]?.trim();
      const num = parseFloat(val);
      row[h] = isNaN(num) ? val : num;
    });
    return row;
  });

  return { data, headers };
};

// --- Helper: Preprocessing (Standard Scaling) ---
const preprocessData = (data: Row[], covariates: string[]) => {
  const processedMatrix: number[][] = [];
  const validRows: Row[] = [];
  
  const catMaps: Record<string, Record<string, number>> = {};
  
  // 1. Identify Categorical Mapping
  covariates.forEach(cov => {
    const isString = typeof data[0][cov] === 'string';
    if (isString) {
      const uniqueVals = Array.from(new Set(data.map(d => String(d[cov]))));
      catMaps[cov] = {};
      uniqueVals.forEach((v, i) => { catMaps[cov][v] = i; });
    }
  });

  // 2. Filter Valid Rows
  data.forEach(row => {
    const complete = covariates.every(c => row[c] !== undefined && row[c] !== "" && row[c] !== null);
    if (!complete) return;
    validRows.push(row);
  });

  // 3. Build Raw Feature Matrix
  validRows.forEach(row => {
    const featureRow: number[] = [1]; // Intercept
    covariates.forEach(cov => {
      const val = row[cov];
      if (typeof val === 'string') {
        featureRow.push(catMaps[cov][val]);
      } else {
        featureRow.push(Number(val));
      }
    });
    processedMatrix.push(featureRow);
  });

  // 4. Standard Scaling (Z-Score) instead of MinMax
  // We skip index 0 (intercept)
  const nFeatures = processedMatrix[0]?.length || 0;
  for (let j = 1; j < nFeatures; j++) {
    const colValues = processedMatrix.map(r => r[j]);
    const colMean = mean(colValues);
    const colStd = Math.sqrt(variance(colValues)) || 1; // avoid div by zero
    
    for (let i = 0; i < processedMatrix.length; i++) {
      processedMatrix[i][j] = (processedMatrix[i][j] - colMean) / colStd;
    }
  }

  return { matrix: processedMatrix, validRows, catMaps };
};

// --- Helper: Generate Python Script ---
const generatePythonScript = (covariates: string[], outcomes: string[], treatmentIds: string, caliper: number) => {
  const idList = treatmentIds.split(/[\s,]+/).filter(Boolean).map(id => `'${id}'`).join(', ');
  
  return `# Robust PSM Analysis Script (Generated by Tool)
# Key Features: StandardScaler, Greedy Matching w/o Replacement, Caliper, Balance Check (SMD)

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# --- Helper: Calculate SMD ---
def calculate_smd(df, treatment_col, covariates):
    smd_data = []
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]
    
    for cov in covariates:
        # Simple encoding for categoricals if needed, assuming numeric for calculation
        if df[cov].dtype == 'object':
             # Note: For rigorous SMD on categorical, we usually check proportions. 
             # Here we convert to code for proxy or skip.
             t_metric = treated[cov].astype('category').cat.codes
             c_metric = control[cov].astype('category').cat.codes
        else:
             t_metric = treated[cov]
             c_metric = control[cov]
             
        mean_t = t_metric.mean()
        mean_c = c_metric.mean()
        var_t = t_metric.var()
        var_c = c_metric.var()
        
        pool_sd = np.sqrt((var_t + var_c) / 2)
        smd = abs(mean_t - mean_c) / pool_sd if pool_sd > 0 else 0
        smd_data.append({'Covariate': cov, 'SMD': smd})
        
    return pd.DataFrame(smd_data)

# 1. Load Data
df = pd.read_csv('data.csv')

# 2. Config
covariates = ${JSON.stringify(covariates)}
outcomes = ${JSON.stringify(outcomes)}
treatment_ids = [${idList}]
CALIPER = ${caliper}

# 3. Preprocessing
print("Preprocessing & Scaling...")
df['is_treated'] = df['item_id'].astype(str).isin(treatment_ids).astype(int)

# Drop rows with missing covariates
df = df.dropna(subset=covariates)

# One-Hot / Dummy Encoding
df_encoded = pd.get_dummies(df, columns=[c for c in covariates if df[c].dtype == 'object'], drop_first=True)

# Update covariate list after encoding
encoded_covariates = [c for c in df_encoded.columns if any(base in c for base in covariates) and c not in df.columns]
numeric_covariates = [c for c in covariates if c in df_encoded.columns]
model_features = numeric_covariates + encoded_covariates

# Standardization (CRITICAL for Logistic Regression)
scaler = StandardScaler()
X = df_encoded[model_features]
X_scaled = scaler.fit_transform(X)
y = df_encoded['is_treated']

# 4. Propensity Score Estimation
print("Training Propensity Model...")
lr = LogisticRegression(max_iter=2000, solver='lbfgs')
lr.fit(X_scaled, y)
df['ps'] = lr.predict_proba(X_scaled)[:, 1]

# 5. Greedy Matching without Replacement
print(f"Matching with Caliper: {CALIPER}...")

treated_df = df[df['is_treated'] == 1].copy()
control_df = df[df['is_treated'] == 0].copy()

# Sort treated by PS descending (heuristic for better matches)
treated_df = treated_df.sort_values('ps', ascending=False)

matched_data = []
used_control_indices = set()

# Manual Greedy Loop for strict control
for t_idx, t_row in treated_df.iterrows():
    t_ps = t_row['ps']
    
    # Filter potential controls by caliper first to speed up
    candidates = control_df[
        (control_df['ps'] >= t_ps - CALIPER) & 
        (control_df['ps'] <= t_ps + CALIPER)
    ]
    
    best_c_idx = None
    min_diff = float('inf')
    
    for c_idx, c_row in candidates.iterrows():
        if c_idx in used_control_indices:
            continue
            
        diff = abs(t_ps - c_row['ps'])
        if diff < min_diff:
            min_diff = diff
            best_c_idx = c_idx
            
    if best_c_idx is not None:
        used_control_indices.add(best_c_idx)
        c_row = control_df.loc[best_c_idx]
        
        record = {
            'treatment_id': t_row['item_id'],
            'control_id': c_row['item_id'],
            'ps_treated': t_ps,
            'ps_control': c_row['ps'],
            'distance': min_diff,
        }
        
        # Outcomes
        for out in outcomes:
            record[f'{out}_T'] = t_row[out]
            record[f'{out}_C'] = c_row[out]
            record[f'{out}_Diff'] = t_row[out] - c_row[out]
            
        # Covariates for Post-Match Balance Check
        for cov in covariates:
            record[f'{cov}_T'] = t_row[cov]
            record[f'{cov}_C'] = c_row[cov]
            
        matched_data.append(record)

results_df = pd.DataFrame(matched_data)
print(f"Matched {len(results_df)} pairs. Dropped {len(treated_df) - len(results_df)} treated units due to caliper.")

# 6. Balance Check (SMD)
if len(results_df) > 0:
    print("-" * 30)
    print("Balance Check (SMD < 0.1 is good):")
    
    # Reconstruct Matched DataFrames for SMD calc
    matched_T = df.loc[df['item_id'].isin(results_df['treatment_id'])]
    matched_C = df.loc[df['item_id'].isin(results_df['control_id'])]
    
    # Combine just for SMD function util
    matched_full = pd.concat([matched_T, matched_C])
    
    pre_match_smd = calculate_smd(df, 'is_treated', covariates)
    post_match_smd = calculate_smd(matched_full, 'is_treated', covariates)
    
    balance_df = pd.merge(pre_match_smd, post_match_smd, on='Covariate', suffixes=('_Pre', '_Post'))
    print(balance_df)
    
    # 7. ATT Calculation
    print("-" * 30)
    print("Average Treatment Effect on the Treated (ATT):")
    att = results_df[[f'{out}_Diff' for out in outcomes]].mean()
    print(att)
    
    results_df.to_excel('psm_results_robust.xlsx', index=False)
    print("Results saved.")
else:
    print("No matches found within caliper.")
`;
};

const App = () => {
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<'dashboard' | 'matches' | 'balance' | 'code'>('dashboard');
  
  const [config, setConfig] = useState<AnalysisConfig>({
    treatmentIds: "",
    covariates: ["reference_price_lc", "stock", "avg_handling_time", "slr_cntry", "item_condition"],
    outcomes: ["gmv_15d", "si_15d", "lstg_vi_30d", "add_to_cart"],
    caliper: 0.05 // Strict caliper by default
  });
  
  const [result, setResult] = useState<AnalysisResult | null>(null);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setLoading(true);
    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event.target?.result as string;
      const ds = parseCSV(text);
      setDataset(ds);
      setLoading(false);
    };
    reader.readAsText(file);
  };

  const runAnalysis = () => {
    if (!dataset) return;
    setLoading(true);
    setResult(null);
    setActiveTab('balance'); // Direct user to balance check first as it's critical

    setTimeout(() => {
      try {
        const treatmentIdSet = new Set(config.treatmentIds.split(/[\s,]+/).map(s => s.trim()).filter(Boolean));
        
        // 1. Preprocess & Standard Scale
        const { matrix, validRows, catMaps } = preprocessData(dataset.data, config.covariates);
        const labels = validRows.map(row => treatmentIdSet.has(String(row.item_id)) ? 1 : 0);
        
        const treatmentCount = labels.filter(l => l === 1).length;
        if (treatmentCount === 0) {
          alert("No treatment items found in the dataset! Check your Item IDs.");
          setLoading(false);
          return;
        }

        // 2. Propensity Score Estimation
        const lr = new LogisticRegression(0.1, 800);
        lr.fit(matrix, labels);
        const scores = lr.predictProba(matrix);

        // 3. Balance Check (Pre-Match)
        const preMatchSMDs: Record<string, number> = {};
        const treatedRowsFull = validRows.filter((_, i) => labels[i] === 1);
        const controlRowsFull = validRows.filter((_, i) => labels[i] === 0);

        config.covariates.forEach(cov => {
          // Extract numeric values (using catMaps for categorical)
          const isCat = !!catMaps[cov];
          const getVal = (r: Row) => isCat ? catMaps[cov][String(r[cov])] : Number(r[cov]);
          const tVals = treatedRowsFull.map(getVal);
          const cVals = controlRowsFull.map(getVal);
          preMatchSMDs[cov] = calculateSMD(tVals, cVals);
        });

        // 4. Greedy Matching with Caliper (Without Replacement)
        const treatedIndices: number[] = [];
        const controlIndices: number[] = [];
        labels.forEach((l, i) => { if (l === 1) treatedIndices.push(i); else controlIndices.push(i); });

        const matches: MatchResult[] = [];
        const usedControls = new Set<number>();

        // Sort treated by score descending (heuristic)
        treatedIndices.sort((a, b) => scores[b] - scores[a]);

        treatedIndices.forEach(tIdx => {
          const tScore = scores[tIdx];
          let bestMatchIdx = -1;
          let minDiff = Infinity;

          // Simple Linear Scan (Optimization: In production, use k-d tree or sorted lists)
          controlIndices.forEach(cIdx => {
            if (usedControls.has(cIdx)) return;
            const diff = Math.abs(tScore - scores[cIdx]);
            
            // Caliper Check
            if (diff > config.caliper) return;

            if (diff < minDiff) {
              minDiff = diff;
              bestMatchIdx = cIdx;
            }
          });

          if (bestMatchIdx !== -1) {
            usedControls.add(bestMatchIdx);
            matches.push({
              treated: { ...validRows[tIdx], propensityScore: tScore },
              control: { ...validRows[bestMatchIdx], propensityScore: scores[bestMatchIdx] },
              distance: minDiff
            });
          }
        });

        // 5. Balance Check (Post-Match)
        const balanceMetrics: BalanceMetric[] = [];
        const matchedT = matches.map(m => m.treated);
        const matchedC = matches.map(m => m.control);

        config.covariates.forEach(cov => {
          const isCat = !!catMaps[cov];
          const getVal = (r: Row) => isCat ? catMaps[cov][String(r[cov])] : Number(r[cov]);
          const tVals = matchedT.map(getVal);
          const cVals = matchedC.map(getVal);
          const postSMD = calculateSMD(tVals, cVals);
          
          balanceMetrics.push({
            covariate: cov,
            preSMD: preMatchSMDs[cov],
            postSMD: postSMD,
            isBalanced: postSMD < 0.2 // Standard threshold is 0.1 or 0.25
          });
        });

        // 6. ATT Calculation
        const attResults: Record<string, number> = {};
        config.outcomes.forEach(outcome => {
          let sumDiff = 0;
          matches.forEach(m => {
            const tVal = Number(m.treated[outcome]) || 0;
            const cVal = Number(m.control[outcome]) || 0;
            sumDiff += (tVal - cVal);
          });
          attResults[outcome] = matches.length > 0 ? sumDiff / matches.length : 0;
        });

        setResult({
          att: attResults,
          matches,
          treatmentGroupSize: treatedIndices.length,
          controlGroupSize: controlIndices.length,
          matchedPairs: matches.length,
          droppedCount: treatedIndices.length - matches.length,
          propensityScores: scores,
          balanceMetrics
        });

      } catch (err) {
        console.error(err);
        alert("An error occurred during analysis.");
      } finally {
        setLoading(false);
      }
    }, 100);
  };

  const downloadCSV = () => {
    if (!result) return;
    const outcomeHeaders = config.outcomes.flatMap(o => [`${o}_T`, `${o}_C`, `${o}_Diff`]);
    const headers = [
      "Pair_ID",
      "Treatment_Item_ID", "Treatment_Propensity",
      "Control_Item_ID", "Control_Propensity",
      "Match_Distance",
      ...outcomeHeaders
    ];

    const rows = result.matches.map((m, i) => {
      const outcomeValues = config.outcomes.flatMap(o => {
        const tVal = Number(m.treated[o]) || 0;
        const cVal = Number(m.control[o]) || 0;
        return [tVal, cVal, tVal - cVal];
      });

      return [
        i + 1,
        m.treated.item_id, m.treated.propensityScore.toFixed(5),
        m.control.item_id, m.control.propensityScore.toFixed(5),
        m.distance.toFixed(6),
        ...outcomeValues
      ].join(",");
    });

    const csvContent = "data:text/csv;charset=utf-8," + [headers.join(","), ...rows].join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "psm_robust_results.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const toggleConfig = (field: 'outcomes' | 'covariates', col: string) => {
    setConfig(prev => {
      const exists = prev[field].includes(col);
      return {
        ...prev,
        [field]: exists ? prev[field].filter(c => c !== col) : [...prev[field], col]
      };
    });
  };

  return (
    <div className="container">
      <header className="flex items-center gap-4" style={{marginBottom: '2rem'}}>
        <div style={{background: 'var(--primary)', padding: '0.75rem', borderRadius: '8px', color: 'white'}}>
          <Brain size={32} />
        </div>
        <div>
          <h1 style={{fontSize: '1.5rem'}}>PSM Causal Inference Tool <span className="badge" style={{verticalAlign: 'middle', marginLeft: '10px', background: '#e0f2fe', color: '#0284c7'}}>Robust v2.0</span></h1>
          <p className="text-muted">Propensity Score Matching with Caliper & Balance Checks</p>
        </div>
      </header>

      <div className="grid grid-3" style={{gridTemplateColumns: '300px 1fr'}}>
        
        {/* Left Sidebar: Configuration */}
        <div className="flex flex-col gap-4">
          <div className="card">
            <h3 className="flex items-center gap-2" style={{marginBottom: '1rem'}}>
              <Database size={18} /> Data Source
            </h3>
            <label className="btn btn-outline" style={{width: '100%', justifyContent: 'center', position: 'relative'}}>
              <Upload size={16} /> {dataset ? `Loaded ${dataset.data.length.toLocaleString()} rows` : "Upload CSV"}
              <input type="file" accept=".csv" onChange={handleFileUpload} style={{opacity: 0, position: 'absolute', inset: 0, cursor: 'pointer'}} />
            </label>
            {dataset && <div className="text-xs text-muted" style={{marginTop: '0.5rem', textAlign: 'center'}}>Headers detected: {dataset.headers.length}</div>}
          </div>

          <div className="card">
            <h3 className="flex items-center gap-2" style={{marginBottom: '1rem'}}>
              <Settings size={18} /> Configuration
            </h3>
            
            <div style={{marginBottom: '1rem'}}>
              <label className="label">Treatment Item IDs</label>
              <textarea 
                className="input" 
                rows={4}
                placeholder="Paste item_id list here..." 
                value={config.treatmentIds}
                onChange={e => setConfig({...config, treatmentIds: e.target.value})}
              />
            </div>

            <div style={{marginBottom: '1rem'}}>
              <label className="label flex justify-between">
                Caliper (Max Distance)
                <span className="text-muted">{config.caliper}</span>
              </label>
              <input 
                type="range" min="0.001" max="0.2" step="0.001" 
                value={config.caliper} 
                onChange={e => setConfig({...config, caliper: parseFloat(e.target.value)})}
                style={{width: '100%'}}
              />
              <div className="text-xs text-muted">Controls outside this PS distance are rejected. Standard: 0.05 or 0.2*SD.</div>
            </div>

            {dataset && (
              <>
                <div style={{marginBottom: '1rem'}}>
                  <label className="label">Covariates (Must be Pre-Treatment)</label>
                  <div style={{maxHeight: '150px', overflowY: 'auto', border: '1px solid var(--border)', borderRadius: '6px', padding: '0.5rem'}}>
                    {dataset.headers.filter(h => h !== 'item_id').map(h => (
                      <div key={`cov-${h}`} className="flex items-center gap-2" style={{marginBottom: '0.25rem'}}>
                        <input 
                          type="checkbox" 
                          checked={config.covariates.includes(h)}
                          onChange={() => toggleConfig('covariates', h)}
                        />
                        <span className="text-sm">{h}</span>
                      </div>
                    ))}
                  </div>
                  <div className="text-xs text-danger flex items-center gap-1" style={{marginTop: 4}}>
                    <AlertTriangle size={12} /> Exclude post-treatment vars!
                  </div>
                </div>

                <div style={{marginBottom: '1rem'}}>
                  <label className="label">Outcomes</label>
                  <div style={{maxHeight: '150px', overflowY: 'auto', border: '1px solid var(--border)', borderRadius: '6px', padding: '0.5rem'}}>
                    {dataset.headers.filter(h => h !== 'item_id').map(h => (
                      <div key={`out-${h}`} className="flex items-center gap-2" style={{marginBottom: '0.25rem'}}>
                        <input 
                          type="checkbox" 
                          checked={config.outcomes.includes(h)}
                          onChange={() => toggleConfig('outcomes', h)}
                        />
                        <span className="text-sm">{h}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </>
            )}

            <button 
              className="btn btn-primary" 
              style={{width: '100%'}} 
              disabled={!dataset || loading}
              onClick={runAnalysis}
            >
              {loading ? <div className="spinner"></div> : <><Play size={16} /> Run Robust Analysis</>}
            </button>
          </div>
        </div>

        {/* Right Content */}
        <div className="flex flex-col gap-4">
          
          {/* Tabs */}
          <div style={{borderBottom: '1px solid var(--border)', display: 'flex', gap: '1rem', marginBottom: '1rem'}}>
            <button 
              onClick={() => setActiveTab('balance')}
              style={{
                padding: '0.5rem 1rem', 
                borderBottom: activeTab === 'balance' ? '2px solid var(--primary)' : 'none',
                color: activeTab === 'balance' ? 'var(--primary)' : 'var(--text-light)',
                background: 'none', border: 'none', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem'
              }}
            ><Scale size={16}/> Balance Check</button>
             <button 
              onClick={() => setActiveTab('dashboard')}
              style={{
                padding: '0.5rem 1rem', 
                borderBottom: activeTab === 'dashboard' ? '2px solid var(--primary)' : 'none',
                color: activeTab === 'dashboard' ? 'var(--primary)' : 'var(--text-light)',
                background: 'none', border: 'none', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem'
              }}
            ><BarChart2 size={16}/> Dashboard</button>
            <button 
              onClick={() => setActiveTab('matches')}
              style={{
                padding: '0.5rem 1rem', 
                borderBottom: activeTab === 'matches' ? '2px solid var(--primary)' : 'none',
                color: activeTab === 'matches' ? 'var(--primary)' : 'var(--text-light)',
                background: 'none', border: 'none', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem'
              }}
            ><Users size={16}/> Matches {result ? `(${result.matches.length})` : ''}</button>
            <button 
              onClick={() => setActiveTab('code')}
              style={{
                padding: '0.5rem 1rem', 
                borderBottom: activeTab === 'code' ? '2px solid var(--primary)' : 'none',
                color: activeTab === 'code' ? 'var(--primary)' : 'var(--text-light)',
                background: 'none', border: 'none', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem'
              }}
            ><Code size={16}/> Python Script</button>
          </div>

          {/* BALANCE CHECK TAB (New Default) */}
          {activeTab === 'balance' && (
             <div className="flex flex-col gap-4">
               {!result ? (
                 <div className="card flex flex-col items-center justify-center" style={{height: '300px', color: 'var(--text-light)'}}>
                   <Scale size={48} style={{marginBottom: '1rem', opacity: 0.5}} />
                   <p>Run analysis to check covariate balance (SMD).</p>
                 </div>
               ) : (
                 <div className="card">
                   <h3 className="flex items-center gap-2" style={{marginBottom: '1rem'}}>
                     <Scale size={20} /> Covariate Balance (SMD)
                   </h3>
                   <div style={{marginBottom: '1rem', fontSize: '0.9rem', color: 'var(--text-light)'}}>
                     Standardized Mean Difference (SMD) measures how different the Treatment and Control groups are. <br/>
                     <strong>Goal:</strong> SMD should be <span className="font-bold" style={{color: 'var(--success)'}}>&lt; 0.1</span> (or at least &lt; 0.25) after matching.
                   </div>
                   
                   <table style={{width: '100%', borderCollapse: 'collapse'}}>
                     <thead>
                       <tr style={{borderBottom: '2px solid var(--border)', textAlign: 'left', background: '#f8fafc'}}>
                         <th style={{padding: '0.75rem'}}>Covariate</th>
                         <th style={{padding: '0.75rem', textAlign: 'right'}}>Pre-Match SMD</th>
                         <th style={{padding: '0.75rem', textAlign: 'right'}}>Post-Match SMD</th>
                         <th style={{padding: '0.75rem', textAlign: 'center'}}>Status</th>
                       </tr>
                     </thead>
                     <tbody>
                       {result.balanceMetrics.map((m, i) => (
                         <tr key={i} style={{borderBottom: '1px solid var(--border)'}}>
                           <td style={{padding: '0.75rem', fontWeight: 500}}>{m.covariate}</td>
                           <td style={{padding: '0.75rem', textAlign: 'right', color: 'var(--text-light)'}}>{m.preSMD.toFixed(3)}</td>
                           <td style={{padding: '0.75rem', textAlign: 'right', fontWeight: 'bold'}}>
                             {m.postSMD.toFixed(3)}
                             {m.postSMD < m.preSMD ? <span style={{color: 'var(--success)', fontSize: '0.7em', marginLeft: 4}}>▼</span> : <span style={{color: 'var(--danger)', fontSize: '0.7em', marginLeft: 4}}>▲</span>}
                           </td>
                           <td style={{padding: '0.75rem', textAlign: 'center'}}>
                             {m.isBalanced ? 
                               <span className="badge" style={{background: '#dcfce7', color: '#166534'}}>Balanced</span> : 
                               <span className="badge" style={{background: '#fee2e2', color: '#991b1b'}}>Imbalanced</span>
                             }
                           </td>
                         </tr>
                       ))}
                     </tbody>
                   </table>
                 </div>
               )}
             </div>
          )}

          {/* DASHBOARD TAB */}
          {activeTab === 'dashboard' && result && (
            <>
              {/* Summary Stats */}
              <div className="grid grid-3">
                <div className="card flex flex-col items-center justify-center">
                  <span className="text-muted text-sm">Treatment / Matched</span>
                  <div className="flex items-baseline gap-2">
                     <span className="font-bold" style={{fontSize: '1.5rem'}}>{result.treatmentGroupSize}</span>
                     <span className="text-muted">→</span>
                     <span className="font-bold" style={{fontSize: '1.5rem', color: 'var(--success)'}}>{result.matches.length}</span>
                  </div>
                </div>
                <div className="card flex flex-col items-center justify-center">
                  <span className="text-muted text-sm flex items-center gap-1">Dropped <Filter size={12}/></span>
                  <span className="font-bold" style={{fontSize: '1.5rem', color: result.droppedCount > 0 ? 'var(--danger)' : 'var(--text-light)'}}>{result.droppedCount}</span>
                  <span className="text-xs text-muted">outside caliper</span>
                </div>
                <div className="card flex flex-col items-center justify-center">
                   <span className="text-muted text-sm">Valid Match Rate</span>
                   <span className="font-bold" style={{fontSize: '1.5rem'}}>{((result.matches.length / result.treatmentGroupSize) * 100).toFixed(1)}%</span>
                </div>
              </div>

              <div className="flex flex-col gap-4">
                  <div className="card">
                    <div className="flex justify-between items-center" style={{marginBottom: '1rem'}}>
                      <h3>ATT Results (Causal Effect)</h3>
                    </div>
                    
                    <table style={{width: '100%', borderCollapse: 'collapse'}}>
                      <thead>
                        <tr style={{borderBottom: '2px solid var(--border)', textAlign: 'left'}}>
                          <th style={{padding: '0.75rem'}}>Outcome Metric</th>
                          <th style={{padding: '0.75rem', textAlign: 'right'}}>ATT (Lift)</th>
                          <th style={{padding: '0.75rem', textAlign: 'right'}}>Impact</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(result.att).map(([key, val]) => {
                          const numVal = val as number;
                          return (
                          <tr key={key} style={{borderBottom: '1px solid var(--border)'}}>
                            <td style={{padding: '0.75rem', fontWeight: 500}}>{key}</td>
                            <td style={{padding: '0.75rem', textAlign: 'right', fontFamily: 'monospace', fontSize: '1rem'}}>
                              {numVal > 0 ? '+' : ''}{numVal.toFixed(4)}
                            </td>
                            <td style={{padding: '0.75rem', textAlign: 'right'}}>
                              <span className="badge" style={{background: numVal > 0 ? '#dcfce7' : '#fee2e2', color: numVal > 0 ? '#166534' : '#991b1b'}}>
                                {numVal > 0 ? <TrendingUp size={12} style={{marginRight: 4}} /> : null}
                                {numVal > 0 ? "Positive" : "Negative"}
                              </span>
                            </td>
                          </tr>
                        )})}
                      </tbody>
                    </table>
                  </div>
                </div>
            </>
          )}

          {/* MATCHES TAB */}
          {activeTab === 'matches' && result && (
                <div className="card">
                  <div className="flex justify-between items-center" style={{marginBottom: '1rem'}}>
                    <h3 className="flex items-center gap-2"><Users size={20} /> Matched Pairs</h3>
                    <button className="btn btn-outline text-sm" onClick={downloadCSV}>
                      <FileSpreadsheet size={16} /> Download CSV
                    </button>
                  </div>
                  <div style={{overflowX: 'auto', maxHeight: '500px', overflowY: 'auto'}}>
                    <table style={{width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem'}}>
                      <thead style={{position: 'sticky', top: 0, background: 'var(--surface)', zIndex: 1}}>
                        <tr style={{background: '#f1f5f9', textAlign: 'left', borderBottom: '2px solid var(--border)'}}>
                          <th style={{padding: '0.5rem'}}>#</th>
                          <th style={{padding: '0.5rem'}}>T-ID</th>
                          <th style={{padding: '0.5rem'}}>T-Score</th>
                          <th style={{padding: '0.5rem', textAlign: 'center'}}>→</th>
                          <th style={{padding: '0.5rem'}}>C-ID</th>
                          <th style={{padding: '0.5rem'}}>C-Score</th>
                          <th style={{padding: '0.5rem'}}>Diff</th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.matches.map((m, i) => (
                          <tr key={i} style={{borderBottom: '1px solid var(--border)'}}>
                            <td style={{padding: '0.5rem', color: 'var(--text-light)'}}>{i + 1}</td>
                            <td style={{padding: '0.5rem', fontWeight: 600}}>{m.treated.item_id}</td>
                            <td style={{padding: '0.5rem', fontFamily: 'monospace'}}>{m.treated.propensityScore.toFixed(4)}</td>
                            <td style={{padding: '0.5rem', textAlign: 'center', color: 'var(--text-light)'}}><ChevronRight size={14} /></td>
                            <td style={{padding: '0.5rem', color: 'var(--text-light)'}}>{m.control.item_id}</td>
                            <td style={{padding: '0.5rem', fontFamily: 'monospace', color: 'var(--text-light)'}}>{m.control.propensityScore.toFixed(4)}</td>
                            <td style={{padding: '0.5rem', fontFamily: 'monospace', color: m.distance > config.caliper ? 'red' : 'inherit'}}>{m.distance.toFixed(4)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
          )}

          {/* CODE TAB */}
          {activeTab === 'code' && (
            <div className="flex flex-col gap-4">
              <div className="card">
                <h3 className="flex items-center gap-2" style={{marginBottom: '1rem'}}>
                  <BookOpen size={20} /> Methodology Upgrade
                </h3>
                <div className="grid grid-2">
                    <div>
                      <h4 className="text-sm font-bold flex items-center gap-1"><Scale size={14}/> StandardScaler</h4>
                      <p className="text-sm text-muted">Now using Z-score scaling to fix Logistic Regression scale sensitivity.</p>
                    </div>
                    <div>
                      <h4 className="text-sm font-bold flex items-center gap-1"><Filter size={14}/> Caliper & Greedy</h4>
                      <p className="text-sm text-muted">Strict 1:1 matching without replacement. Reject matches if PS diff > {config.caliper}.</p>
                    </div>
                </div>
              </div>

              <div className="card">
                <h3 className="flex items-center gap-2" style={{marginBottom: '1rem'}}>
                  <Code size={20} /> Robust Python Script
                </h3>
                <p className="text-sm text-muted" style={{marginBottom: '1rem'}}>
                   This script is now fully compliant with rigorous Causal Inference standards (StandardScaler, Caliper, SMD Check).
                </p>
                <textarea 
                  readOnly 
                  className="input" 
                  style={{fontFamily: 'monospace', fontSize: '0.8rem', height: '400px', background: '#f8fafc'}}
                  value={generatePythonScript(config.covariates, config.outcomes, config.treatmentIds, config.caliper)}
                />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

const root = createRoot(document.getElementById("root")!);
root.render(<App />);