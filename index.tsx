import React, { useState, useEffect, useRef } from "react";
import { createRoot } from "react-dom/client";

// --- Icons (Inlined) ---
const IconBase = ({ children, size = 20, style, className }: { children: React.ReactNode, size?: number, style?: React.CSSProperties, className?: string }) => (
  <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={style} className={className}>
    {children}
  </svg>
);
const Upload = (props: any) => <IconBase {...props}><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></IconBase>;
const Settings = (props: any) => <IconBase {...props}><path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.09a2 2 0 0 1-1-1.74v-.47a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.39a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/></IconBase>;
const Play = (props: any) => <IconBase {...props}><polygon points="5 3 19 12 5 21 5 3"/></IconBase>;
const BarChart2 = (props: any) => <IconBase {...props}><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></IconBase>;
const Download = (props: any) => <IconBase {...props}><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></IconBase>;
const Monitor = (props: any) => <IconBase {...props}><rect x="2" y="3" width="20" height="14" rx="2" ry="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></IconBase>;
const Scale = (props: any) => <IconBase {...props}><path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1z"/><path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1z"/><path d="M7 21h10"/><path d="M12 3v18"/><path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2"/></IconBase>;

// --- Core Logic ---
type Row = Record<string, string | number>;

interface Dataset { data: Row[]; headers: string[]; }
interface AnalysisConfig { treatmentIds: string; covariates: string[]; outcomes: string[]; caliper: number; }
interface MatchResult { treated: Row & { ps: number }; control: Row & { ps: number }; distance: number; }
interface BalanceMetric { covariate: string; preSMD: number; postSMD: number; isBalanced: boolean; }
interface AnalysisResult { att: Record<string, number>; matches: MatchResult[]; balanceMetrics: BalanceMetric[]; timeMs: number; }

const mean = (arr: number[]) => arr.reduce((a, b) => a + b, 0) / (arr.length || 1);
const variance = (arr: number[]) => {
  if (arr.length <= 1) return 0;
  const m = mean(arr);
  return arr.reduce((a, b) => a + Math.pow(b - m, 2), 0) / (arr.length - 1);
};
const calculateSMD = (treated: number[], control: number[]) => {
  const poolSD = Math.sqrt((variance(treated) + variance(control)) / 2);
  return poolSD === 0 ? 0 : Math.abs(mean(treated) - mean(control)) / poolSD;
};

class LogisticRegression {
  weights: number[] = [];
  fit(X: number[][], y: number[]) {
    const nFeatures = X[0].length;
    this.weights = new Array(nFeatures).fill(0);
    for (let i = 0; i < 500; i++) { // Optimized iterations
      const grads = new Array(nFeatures).fill(0);
      X.forEach((row, idx) => {
        const linear = row.reduce((s, v, k) => s + v * this.weights[k], 0);
        const err = (1 / (1 + Math.exp(-linear))) - y[idx];
        row.forEach((v, k) => grads[k] += err * v / X.length);
      });
      this.weights = this.weights.map((w, k) => w - 0.1 * grads[k]);
    }
  }
  predict(X: number[][]) {
    return X.map(row => 1 / (1 + Math.exp(-row.reduce((s, v, k) => s + v * this.weights[k], 0))));
  }
}

const App = () => {
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [loading, setLoading] = useState(false);
  const [config, setConfig] = useState<AnalysisConfig>({
    treatmentIds: "",
    covariates: ["leaf_categ_id", "reference_price_lc", "slr_cntry", "stock"],
    outcomes: ["gmv_15d"],
    caliper: 0.05
  });
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [deferredPrompt, setDeferredPrompt] = useState<any>(null);

  useEffect(() => {
    window.addEventListener('beforeinstallprompt', (e) => {
      e.preventDefault();
      setDeferredPrompt(e);
    });
  }, []);

  const installApp = async () => {
    if (!deferredPrompt) return;
    deferredPrompt.prompt();
    const { outcome } = await deferredPrompt.userChoice;
    if (outcome === 'accepted') setDeferredPrompt(null);
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setLoading(true);
    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event.target?.result as string;
      const lines = text.split('\n').filter(l => l.trim());
      const headers = lines[0].split(',').map(h => h.trim());
      const data = lines.slice(1).map(line => {
        const values = line.split(',');
        const row: Row = {};
        headers.forEach((h, i) => {
          const v = values[i]?.trim();
          row[h] = isNaN(Number(v)) || v === "" ? v : Number(v);
        });
        return row;
      });
      setDataset({ data, headers });
      setLoading(false);
    };
    reader.readAsText(file);
  };

  const runAnalysis = () => {
    if (!dataset) return;
    setLoading(true);
    const startTime = performance.now();

    setTimeout(() => {
      try {
        const tIds = new Set(config.treatmentIds.split(/[\s,]+/).map(s => s.trim()).filter(Boolean));
        
        // 1. Preprocessing & Scaling
        const validRows = dataset.data.filter(r => config.covariates.every(c => r[c] !== undefined));
        const catMaps: Record<string, Record<string, number>> = {};
        config.covariates.forEach(c => {
          if (typeof validRows[0][c] === 'string') {
            // Fix: Explicitly treat index as string to avoid unknown type errors
            catMaps[c as string] = {};
            Array.from(new Set(validRows.map(r => String(r[c as string])))).forEach((v, i) => {
              (catMaps[c as string] as Record<string, number>)[v as string] = i;
            });
          }
        });

        const X = validRows.map(r => [1, ...config.covariates.map(c => {
          const mapping = catMaps[c];
          return mapping ? (mapping[String(r[c])] ?? 0) : Number(r[c]);
        })]);
        // Simple Z-Score Scaling
        for (let j = 1; j < X[0].length; j++) {
          const col = X.map(r => r[j]);
          const m = mean(col), s = Math.sqrt(variance(col)) || 1;
          X.forEach(r => r[j] = (r[j] - m) / s);
        }
        // Fix: Use index access for item_id on Record type
        const y = validRows.map(r => tIds.has(String(r['item_id'])) ? 1 : 0);

        // 2. Propensity Score
        const model = new LogisticRegression();
        model.fit(X, y);
        const ps = model.predict(X);

        // 3. High-Performance Matching (Sorting + Binary Search)
        const treated = [], control = [];
        validRows.forEach((r, i) => {
          const item = { ...r, ps: ps[i] };
          if (y[i] === 1) treated.push(item); else control.push(item);
        });

        // Optimization: Sort control for binary search matching
        control.sort((a, b) => a.ps - b.ps);
        const matches: MatchResult[] = [];
        const usedControlIndices = new Set<number>();

        treated.forEach(t => {
          // Find potential candidates within caliper using binary search
          let low = 0, high = control.length - 1;
          let bestIdx = -1, minDiff = Infinity;

          // Find start index of caliper range
          while(low <= high) {
            let mid = Math.floor((low + high) / 2);
            if (control[mid].ps < t.ps - config.caliper) low = mid + 1;
            else high = mid - 1;
          }

          // Search forward from 'low' for the best unused match
          for (let i = low; i < control.length && control[i].ps <= t.ps + config.caliper; i++) {
            if (usedControlIndices.has(i)) continue;
            const diff = Math.abs(t.ps - control[i].ps);
            if (diff < minDiff) { minDiff = diff; bestIdx = i; }
          }

          if (bestIdx !== -1) {
            usedControlIndices.add(bestIdx);
            matches.push({ treated: t, control: control[bestIdx], distance: minDiff });
          }
        });

        // 4. Outcomes & Balance
        const att: Record<string, number> = {};
        config.outcomes.forEach(o => {
          att[o] = matches.length ? mean(matches.map(m => Number(m.treated[o]) - Number(m.control[o]))) : 0;
        });

        const balanceMetrics = config.covariates.map(cov => {
          const getV = (r: Row) => catMaps[cov] ? catMaps[cov][String(r[cov])] : Number(r[cov]);
          const preT = treated.map(getV), preC = control.map(getV);
          const postT = matches.map(m => getV(m.treated)), postC = matches.map(m => getV(m.control));
          const postSMD = calculateSMD(postT, postC);
          return { covariate: cov, preSMD: calculateSMD(preT, preC), postSMD, isBalanced: postSMD < 0.1 };
        });

        setResult({ att, matches, balanceMetrics, timeMs: performance.now() - startTime });
      } catch (e) {
        alert("Error during analysis: " + e);
      } finally {
        setLoading(false);
      }
    }, 100);
  };

  return (
    <div className="container">
      <header className="flex items-center justify-between" style={{marginBottom: '2rem'}}>
        <div className="flex items-center gap-4">
          <div style={{background: 'var(--primary)', color: 'white', padding: '10px', borderRadius: '12px'}}>
            <BarChart2 size={32} />
          </div>
          <div>
            <h1 style={{fontSize: '1.4rem'}}>PSM 分析专家系统 <span className="badge">v2.5 高性能版</span></h1>
            <p className="text-muted text-sm">已针对 60,000+ 条记录优化匹配算法</p>
          </div>
        </div>
        <div className="flex gap-2">
          {deferredPrompt && (
            <button className="btn btn-outline" onClick={installApp} style={{borderColor: 'var(--primary)', color: 'var(--primary)'}}>
              <Monitor size={16} /> 安装到桌面
            </button>
          )}
          <div className="badge" style={{height: 'fit-content', padding: '8px 12px'}}>本地运行：数据不出浏览器</div>
        </div>
      </header>

      <div className="grid grid-3">
        <aside className="flex flex-col gap-4">
          <div className="card">
            <h3 className="label" style={{color: 'var(--text)', marginBottom: '1rem'}}>1. 数据导入</h3>
            <label className="btn btn-outline" style={{width: '100%', cursor: 'pointer'}}>
              <Upload size={16} /> {dataset ? `已加载 ${dataset.data.length.toLocaleString()} 行` : "上传 CSV 文件"}
              <input type="file" accept=".csv" onChange={handleFileUpload} style={{display: 'none'}} />
            </label>
            <p className="text-xs text-muted" style={{marginTop: '8px'}}>请确保 Excel 已另存为 CSV (逗号分隔)</p>
          </div>

          <div className="card">
            <h3 className="label" style={{color: 'var(--text)', marginBottom: '1rem'}}>2. 参数设置</h3>
            <div style={{marginBottom: '1rem'}}>
              <label className="label">实验组 Item IDs (逗号或换行分隔)</label>
              <textarea className="input" rows={4} value={config.treatmentIds} onChange={e => setConfig({...config, treatmentIds: e.target.value})} placeholder="例如: 1001, 1002..." />
            </div>
            <div style={{marginBottom: '1rem'}}>
              <label className="label">卡钳值 (Caliper): {config.caliper}</label>
              <input type="range" min="0.001" max="0.2" step="0.001" value={config.caliper} onChange={e => setConfig({...config, caliper: Number(e.target.value)})} style={{width: '100%'}} />
            </div>
            <button className="btn btn-primary" style={{width: '100%'}} onClick={runAnalysis} disabled={!dataset || loading}>
              {loading ? <div className="spinner"></div> : <><Play size={16} /> 开始极速分析</>}
            </button>
          </div>
        </aside>

        <main className="flex flex-col gap-4">
          {!result ? (
            <div className="card flex flex-col items-center justify-center" style={{minHeight: '400px', background: 'transparent', borderStyle: 'dashed'}}>
              <Upload size={48} style={{opacity: 0.2, marginBottom: '1rem'}} />
              <p className="text-muted">上传数据并运行分析以查看结果</p>
            </div>
          ) : (
            <>
              <div className="grid" style={{gridTemplateColumns: 'repeat(3, 1fr)'}}>
                <div className="card" style={{textAlign: 'center'}}>
                  <p className="text-xs text-muted">成功匹配对数</p>
                  <h2 style={{color: 'var(--success)'}}>{result.matches.length.toLocaleString()}</h2>
                </div>
                <div className="card" style={{textAlign: 'center'}}>
                  <p className="text-xs text-muted">运行耗时</p>
                  <h2>{(result.timeMs / 1000).toFixed(2)}s</h2>
                </div>
                <div className="card" style={{textAlign: 'center'}}>
                  <p className="text-xs text-muted">数据吞吐量</p>
                  <h2>{(dataset!.data.length / (result.timeMs / 1000) / 1000).toFixed(1)}k/s</h2>
                </div>
              </div>

              <div className="card">
                <h3 style={{marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem'}}><Scale size={20} /> 协变量均衡性检查 (SMD)</h3>
                <table style={{width: '100%', borderCollapse: 'collapse'}}>
                  <thead>
                    <tr style={{textAlign: 'left', borderBottom: '2px solid var(--border)'}}>
                      <th style={{padding: '10px'}}>指标</th>
                      <th style={{padding: '10px'}}>匹配前 SMD</th>
                      <th style={{padding: '10px'}}>匹配后 SMD</th>
                      <th style={{padding: '10px'}}>状态</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.balanceMetrics.map(m => (
                      <tr key={m.covariate} style={{borderBottom: '1px solid var(--border)'}}>
                        <td style={{padding: '10px', fontWeight: 500}}>{m.covariate}</td>
                        <td className="text-muted" style={{padding: '10px'}}>{m.preSMD.toFixed(4)}</td>
                        <td style={{padding: '10px', fontWeight: 'bold'}}>{m.postSMD.toFixed(4)}</td>
                        <td style={{padding: '10px'}}>
                          <span className="badge" style={{background: m.isBalanced ? '#dcfce7' : '#fee2e2', color: m.isBalanced ? '#166534' : '#991b1b'}}>
                            {m.isBalanced ? "已均衡" : "需调整"}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="card">
                <h3 style={{marginBottom: '1.5rem'}}>ATT 分析结果 (因果效应)</h3>
                <div className="grid" style={{gridTemplateColumns: '1fr 1fr'}}>
                  {/* Fix: Explicitly cast 'v' to number to avoid unknown type errors in comparison and toFixed */}
                  {Object.entries(result.att).map(([k, v]) => {
                    const val = v as number;
                    return (
                      <div key={k} className="card" style={{margin: 0, border: '1px solid var(--border)', background: '#f8fafc'}}>
                        <p className="text-sm font-bold">{k}</p>
                        <h2 style={{color: val > 0 ? 'var(--success)' : 'var(--danger)', fontSize: '2rem'}}>
                          {val > 0 ? '+' : ''}{val.toFixed(4)}
                        </h2>
                      </div>
                    );
                  })}
                </div>
              </div>
            </>
          )}
        </main>
      </div>
    </div>
  );
};

createRoot(document.getElementById("root")!).render(<App />);