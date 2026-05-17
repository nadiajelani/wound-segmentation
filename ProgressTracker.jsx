/**
 * Improvement 4 – Progress Tracker UI Component
 * -----------------------------------------------
 * Drop into WoundAnalyzer_v3.jsx or use as a standalone page.
 * Shows side-by-side Before/After, delta metrics, and a
 * simple healing trend bar.
 *
 * Usage in parent:
 *   import ProgressTracker from "./ProgressTracker";
 *   <ProgressTracker apiUrl={window.location.origin} />
 */

import { useState, useRef, useCallback } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

const css = `
.pt { font-family:'Geist',sans-serif; }
.pt-grid { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
@media(max-width:700px){ .pt-grid{ grid-template-columns:1fr; } }
.pt-upload { border:1.5px dashed rgba(255,255,255,.15); border-radius:12px; padding:28px 20px;
  text-align:center; cursor:pointer; background:rgba(255,255,255,.03);
  transition:border-color .18s; font-size:13px; color:#9ca3af; }
.pt-upload:hover{ border-color:#00e5a0; }
.pt-upload.has-file{ padding:0; border-style:solid; border-color:rgba(255,255,255,.12); }
.pt-preview{ width:100%; border-radius:10px; display:block; max-height:220px; object-fit:cover; }
.pt-label{ font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:1px;
  color:#6b7280; margin-bottom:8px; }
.pt-date{ width:100%; padding:8px 12px; border-radius:7px; border:1px solid rgba(255,255,255,.12);
  background:rgba(255,255,255,.05); color:#e8eaf0; font-size:13px; margin-top:8px; }
.pt-date:focus{ outline:none; border-color:#00e5a0; }
.delta-grid{ display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:10px; margin:18px 0; }
.delta-tile{ background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.08);
  border-radius:10px; padding:14px; }
.delta-lbl{ font-size:10px; text-transform:uppercase; letter-spacing:1px; color:#6b7280; }
.delta-val{ font-size:22px; font-family:'Instrument Serif',serif; margin:4px 0 2px; }
.delta-unit{ font-size:11px; color:#6b7280; font-family:'Geist Mono',monospace; }
.improving{ color:#00e5a0; } .deteriorating{ color:#ff4d4d; } .stable{ color:#f5c518; }
.trend-bar-wrap{ height:6px; background:rgba(255,255,255,.08); border-radius:3px; overflow:hidden; margin-top:6px; }
.trend-bar{ height:100%; border-radius:3px; transition:width .6s ease; }
.scan-hdr{ font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:1px; 
  color:#6b7280; padding:10px 14px; border-bottom:1px solid rgba(255,255,255,.07); }
.scan-img{ width:100%; display:block; border-radius:0; }
.scan-metric-row{ display:flex; justify-content:space-between; padding:7px 14px;
  border-top:1px solid rgba(255,255,255,.06); font-size:12px; }
.smr-key{ color:#6b7280; } .smr-val{ color:#e8eaf0; font-family:'Geist Mono',monospace; }
.sev-mild{ color:#00e5a0; } .sev-moderate{ color:#f5c518; } .sev-severe{ color:#ff4d4d; }
.pt-btn{ display:inline-flex; align-items:center; gap:7px; padding:10px 22px; border-radius:8px;
  border:none; cursor:pointer; background:linear-gradient(135deg,#00e5a0,#00b8ff);
  color:#000; font-size:13px; font-weight:600; transition:opacity .18s; margin-top:14px; }
.pt-btn:disabled{ opacity:.3; cursor:not-allowed; }
.pt-err{ background:rgba(255,77,77,.08); border:1px solid rgba(255,77,77,.22);
  border-radius:10px; padding:14px; font-size:13px; color:#ff4d4d; margin-top:12px; }
`;

function UploadSlot({ label, file, preview, date, onFile, onDate, inputRef }) {
  return (
    <div>
      <div className="pt-label">{label}</div>
      <div
        className={`pt-upload ${file ? "has-file" : ""}`}
        onClick={() => !file && inputRef.current.click()}
      >
        {!file ? (
          <>
            <div style={{ fontSize: 28, marginBottom: 8 }}>🖼</div>
            <div>Drop image or click to browse</div>
            <div style={{ fontSize: 11, marginTop: 4 }}>JPEG / PNG · max 8 MB</div>
          </>
        ) : (
          <img className="pt-preview" src={preview} alt={label} />
        )}
      </div>
      <input ref={inputRef} type="file" accept="image/*" style={{ display: "none" }}
        onChange={e => onFile(e.target.files[0])} />
      <input className="pt-date" type="date" value={date}
        onChange={e => onDate(e.target.value)}
        placeholder="Date (optional)" />
    </div>
  );
}

function DeltaValue({ label, value, unit, invert = false }) {
  if (value === null || value === undefined) return null;
  const num = parseFloat(value);
  const good = invert ? num > 0 : num < 0;
  const cls  = Math.abs(num) < 0.05 ? "stable" : good ? "improving" : "deteriorating";
  const sign = num > 0 ? "+" : "";
  return (
    <div className="delta-tile">
      <div className="delta-lbl">{label}</div>
      <div className={`delta-val ${cls}`}>{sign}{typeof value === "number" ? value.toFixed(2) : value}</div>
      <div className="delta-unit">{unit}</div>
    </div>
  );
}

function ScanSummary({ label, data }) {
  if (!data) return null;
  const m = data.metrics || {};
  const h = data.healing_stage || {};
  const sevCls = (s) => ({ mild: "sev-mild", moderate: "sev-moderate" }[s?.toLowerCase()] ?? "sev-severe");
  const rows = [
    ["Area",       `${m.area_percentage?.toFixed(2) ?? "—"}%`],
    ["Perimeter",  `${Math.round(m.perimeter ?? 0)} px`],
    ["Circularity",`${m.circularity ?? "—"}`],
    ["Severity",   <span className={sevCls(m.severity)}>{m.severity ?? "—"}</span>],
    ["Healing stage", h.stage ?? "—"],
  ];
  return (
    <div style={{ border: "1px solid rgba(255,255,255,.08)", borderRadius: 12, overflow: "hidden" }}>
      <div className="scan-hdr">{label}</div>
      <img className="scan-img" src={`data:image/png;base64,${data.overlay_image}`} alt={label} />
      {rows.map(([k, v]) => (
        <div key={k} className="scan-metric-row">
          <span className="smr-key">{k}</span>
          <span className="smr-val">{v}</span>
        </div>
      ))}
    </div>
  );
}

function TrendChart({ delta }) {
  if (!delta) return null;
  // Synthetic trend line from linear extrapolation
  const rate = delta.healing_rate_pct_per_day || 0;
  const start = (delta.area_pct_change < 0
    ? (delta.area_pct_change * -1 / rate) * (-rate) + Math.abs(delta.area_pct_change)
    : 5) || 5;
  const data = Array.from({ length: 8 }, (_, i) => ({
    day: i === 0 ? "Before" : i === 1 ? "Now" : `Day +${(i - 1) * 7}`,
    area: Math.max(0, parseFloat((start + rate * (i === 0 ? -delta.days_elapsed : (i - 1) * 7)).toFixed(2))),
  }));

  return (
    <div style={{ marginTop: 20 }}>
      <div className="pt-label">Projected healing trajectory</div>
      <div style={{ height: 160, marginTop: 8 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 4, right: 16, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,.06)" />
            <XAxis dataKey="day" tick={{ fill: "#6b7280", fontSize: 10 }} />
            <YAxis tick={{ fill: "#6b7280", fontSize: 10 }} unit="%" />
            <Tooltip
              contentStyle={{ background: "#111620", border: "1px solid rgba(255,255,255,.12)", borderRadius: 8, fontSize: 12 }}
              labelStyle={{ color: "#9ca3af" }}
            />
            <Line type="monotone" dataKey="area" stroke="#00e5a0" strokeWidth={2}
              dot={{ r: 3, fill: "#00e5a0" }} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      {delta.projected_closure_days && (
        <div style={{ fontSize: 12, color: "#9ca3af", marginTop: 6, fontFamily: "'Geist Mono',monospace" }}>
          Estimated closure at current rate: <span style={{ color: "#00e5a0" }}>{delta.projected_closure_days} days</span>
        </div>
      )}
    </div>
  );
}

export default function ProgressTracker({ apiUrl = "" }) {
  const [fileA,    setFileA]    = useState(null);
  const [fileB,    setFileB]    = useState(null);
  const [prevA,    setPrevA]    = useState(null);
  const [prevB,    setPrevB]    = useState(null);
  const [dateA,    setDateA]    = useState("");
  const [dateB,    setDateB]    = useState("");
  const [loading,  setLoading]  = useState(false);
  const [result,   setResult]   = useState(null);
  const [error,    setError]    = useState(null);
  const refA = useRef(), refB = useRef();

  const handleFile = useCallback((f, which) => {
    if (!f) return;
    if (which === "a") { setFileA(f); setPrevA(URL.createObjectURL(f)); }
    else               { setFileB(f); setPrevB(URL.createObjectURL(f)); }
  }, []);

  const compare = useCallback(async () => {
    if (!fileA || !fileB) return;
    setLoading(true); setError(null); setResult(null);
    try {
      const fd = new FormData();
      fd.append("image_a", fileA);
      fd.append("image_b", fileB);
      if (dateA) fd.append("date_a", dateA);
      if (dateB) fd.append("date_b", dateB);
      const res = await fetch(`${apiUrl}/compare`, { method: "POST", body: fd });
      if (!res.ok) throw new Error(`Server ${res.status}`);
      const data = await res.json();
      if (!data.success) throw new Error(data.error || "Comparison failed");
      setResult(data);
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  }, [fileA, fileB, dateA, dateB, apiUrl]);

  const d = result?.delta;
  const trendCls = d?.trend === "improving" ? "improving"
                 : d?.trend === "deteriorating" ? "deteriorating" : "stable";

  return (
    <>
      <style>{css}</style>
      <div className="pt">
        {/* Upload row */}
        {!result && (
          <div className="pt-grid" style={{ marginBottom: 16 }}>
            <UploadSlot label="Before (older scan)" file={fileA} preview={prevA}
              date={dateA} onFile={f => handleFile(f, "a")} onDate={setDateA} inputRef={refA} />
            <UploadSlot label="After (recent scan)"  file={fileB} preview={prevB}
              date={dateB} onFile={f => handleFile(f, "b")} onDate={setDateB} inputRef={refB} />
          </div>
        )}

        {!result && (
          <button className="pt-btn" disabled={!fileA || !fileB || loading} onClick={compare}>
            {loading ? "Comparing…" : "Compare Progress →"}
          </button>
        )}

        {error && <div className="pt-err">⚠️ {error}</div>}

        {/* Results */}
        {result && (
          <>
            {/* Trend badge */}
            <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 18 }}>
              <div style={{ fontSize: 22, fontFamily: "'Instrument Serif',serif", fontStyle: "italic" }}>
                Progress Report
              </div>
              <span className={`${trendCls}`} style={{ fontSize: 13, fontWeight: 600 }}>
                {d?.trend === "improving" ? "↓ Improving" :
                 d?.trend === "deteriorating" ? "↑ Deteriorating" : "→ Stable"}
              </span>
              <button className="pt-btn" style={{ marginTop: 0, padding: "7px 14px", fontSize: 12 }}
                onClick={() => setResult(null)}>← New Comparison</button>
            </div>

            {/* Side by side */}
            <div className="pt-grid">
              <ScanSummary label="Before" data={result.scan_a} />
              <ScanSummary label="After"  data={result.scan_b} />
            </div>

            {/* Delta metrics */}
            <div style={{ marginTop: 18 }}>
              <div className="pt-label">Changes between scans · {d?.days_elapsed} day(s)</div>
              <div className="delta-grid">
                <DeltaValue label="Area change"       value={d?.area_pct_change}    unit="% of image" />
                <DeltaValue label="Perimeter"         value={d?.perimeter_change}   unit="px" />
                <DeltaValue label="Circularity"       value={d?.circularity_change} unit="0-1" invert />
                <DeltaValue label="Healing rate"      value={d?.healing_rate_pct_per_day} unit="% per day" />
                {d?.pct_improvement !== null && (
                  <div className="delta-tile">
                    <div className="delta-lbl">Overall improvement</div>
                    <div className={`delta-val ${d.pct_improvement > 0 ? "improving" : "deteriorating"}`}>
                      {d.pct_improvement > 0 ? "+" : ""}{d.pct_improvement?.toFixed(1)}%
                    </div>
                    <div className="delta-unit">vs baseline</div>
                  </div>
                )}
              </div>
            </div>

            <TrendChart delta={d} />
          </>
        )}
      </div>
    </>
  );
}
