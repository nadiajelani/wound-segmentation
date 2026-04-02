import { useState, useRef, useCallback } from "react";

const API_URL = "https://wound-api-929636759806.us-central1.run.app"; 

// ── Palette & tokens ──────────────────────────────────────────────────────────
const css = `
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:         #0a0c10;
    --surface:    #111318;
    --surface2:   #181c24;
    --border:     rgba(255,255,255,0.07);
    --accent:     #00e5a0;
    --accent2:    #0099ff;
    --warn:       #ff6b3d;
    --mild:       #00e5a0;
    --moderate:   #f5c518;
    --severe:     #ff4444;
    --text:       #e8eaf0;
    --muted:      #5a6070;
    --card-r:     16px;
    --transition: 0.22s cubic-bezier(0.4,0,0.2,1);
  }

  body { background: var(--bg); color: var(--text); font-family: 'Syne', sans-serif; min-height: 100vh; }

  /* ── Layout ── */
  .app { display: flex; flex-direction: column; min-height: 100vh; }

  .header {
    padding: 20px 40px;
    display: flex; align-items: center; justify-content: space-between;
    border-bottom: 1px solid var(--border);
    background: rgba(10,12,16,0.85);
    backdrop-filter: blur(12px);
    position: sticky; top: 0; z-index: 100;
  }

  .logo { display: flex; align-items: center; gap: 12px; }
  .logo-mark {
    width: 36px; height: 36px; border-radius: 10px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
  }
  .logo-text { font-family: 'DM Serif Display', serif; font-size: 20px; letter-spacing: -0.3px; }
  .logo-text span { color: var(--accent); }

  .status-pill {
    display: flex; align-items: center; gap: 8px;
    padding: 6px 14px; border-radius: 999px;
    border: 1px solid var(--border);
    font-family: 'DM Mono', monospace; font-size: 12px; color: var(--muted);
  }
  .status-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--muted);
    transition: background var(--transition);
  }
  .status-dot.online { background: var(--accent); box-shadow: 0 0 8px var(--accent); }
  .status-dot.checking { background: var(--moderate); animation: pulse 1s infinite; }

  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }

  .main { flex: 1; padding: 48px 40px; max-width: 1280px; margin: 0 auto; width: 100%; }

  /* ── Hero ── */
  .hero { text-align: center; margin-bottom: 56px; }
  .hero-eyebrow {
    display: inline-block; font-family: 'DM Mono', monospace;
    font-size: 11px; letter-spacing: 2px; text-transform: uppercase;
    color: var(--accent); margin-bottom: 16px;
  }
  .hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(36px, 5vw, 64px);
    line-height: 1.1; letter-spacing: -1px;
    margin-bottom: 16px;
  }
  .hero h1 em { color: var(--accent); font-style: italic; }
  .hero p { color: var(--muted); font-size: 16px; max-width: 480px; margin: 0 auto; line-height: 1.6; }

  /* ── Upload zone ── */
  .upload-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--card-r);
    padding: 48px;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    gap: 20px; cursor: pointer;
    transition: border-color var(--transition), background var(--transition);
    position: relative; overflow: hidden; min-height: 260px;
  }
  .upload-card::before {
    content: ''; position: absolute; inset: 0;
    background: radial-gradient(ellipse at 50% 0%, rgba(0,229,160,0.04) 0%, transparent 70%);
    pointer-events: none;
  }
  .upload-card:hover, .upload-card.drag-over {
    border-color: var(--accent);
    background: rgba(0,229,160,0.03);
  }
  .upload-card.has-file { border-color: rgba(0,229,160,0.3); }

  .upload-icon { font-size: 48px; opacity: 0.5; }
  .upload-title { font-size: 18px; font-weight: 700; }
  .upload-sub { color: var(--muted); font-size: 13px; }
  .upload-btn {
    padding: 10px 24px; border-radius: 8px;
    background: transparent; border: 1px solid var(--accent); color: var(--accent);
    font-family: 'Syne', sans-serif; font-size: 14px; font-weight: 600;
    cursor: pointer; transition: all var(--transition);
  }
  .upload-btn:hover { background: var(--accent); color: var(--bg); }

  /* ── Preview ── */
  .preview-wrap { position: relative; width: 100%; max-width: 340px; margin: 0 auto; }
  .preview-img { width: 100%; border-radius: 12px; display: block; }
  .preview-clear {
    position: absolute; top: -10px; right: -10px;
    width: 28px; height: 28px; border-radius: 50%;
    background: var(--warn); border: 2px solid var(--bg);
    display: flex; align-items: center; justify-content: center;
    cursor: pointer; font-size: 14px; color: white;
    transition: transform var(--transition);
  }
  .preview-clear:hover { transform: scale(1.15); }

  /* ── Analyze button ── */
  .analyze-btn {
    width: 100%; padding: 16px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    border: none; border-radius: 10px;
    font-family: 'Syne', sans-serif; font-size: 16px; font-weight: 700;
    color: var(--bg); cursor: pointer;
    transition: opacity var(--transition), transform var(--transition);
    margin-top: 8px;
  }
  .analyze-btn:hover:not(:disabled) { opacity: 0.9; transform: translateY(-1px); }
  .analyze-btn:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }

  /* ── Loading ── */
  .loading-wrap {
    display: flex; flex-direction: column; align-items: center; gap: 20px;
    padding: 64px 0;
  }
  .spinner {
    width: 52px; height: 52px; border-radius: 50%;
    border: 3px solid var(--border);
    border-top-color: var(--accent);
    animation: spin 0.8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .loading-text { font-family: 'DM Mono', monospace; font-size: 13px; color: var(--muted); }
  .loading-steps { display: flex; flex-direction: column; gap: 8px; width: 100%; max-width: 240px; }
  .loading-step {
    display: flex; align-items: center; gap: 10px;
    font-family: 'DM Mono', monospace; font-size: 12px; color: var(--muted);
    transition: color var(--transition);
  }
  .loading-step.active { color: var(--accent); }
  .loading-step.done { color: var(--text); }
  .step-icon { width: 16px; text-align: center; }

  /* ── Results layout ── */
  .results-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-top: 40px; }
  @media (max-width: 900px) { .results-grid { grid-template-columns: 1fr; } }

  .card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--card-r); overflow: hidden;
  }
  .card-header {
    padding: 16px 20px; border-bottom: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between;
  }
  .card-title { font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); }
  .card-body { padding: 20px; }

  /* ── Image grid ── */
  .image-tabs { display: flex; gap: 4px; padding: 16px 20px 0; }
  .tab-btn {
    padding: 6px 14px; border-radius: 6px; border: none;
    background: transparent; color: var(--muted);
    font-family: 'Syne', sans-serif; font-size: 12px; font-weight: 600;
    cursor: pointer; transition: all var(--transition);
  }
  .tab-btn.active { background: var(--surface2); color: var(--text); }
  .tab-btn:hover:not(.active) { color: var(--text); }

  .result-img { width: 100%; display: block; border-radius: 8px; }

  /* ── Metrics ── */
  .metrics-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .metric-card {
    background: var(--surface2); border-radius: 10px; padding: 16px;
    display: flex; flex-direction: column; gap: 4px;
  }
  .metric-label { font-family: 'DM Mono', monospace; font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; }
  .metric-value { font-family: 'DM Serif Display', serif; font-size: 28px; color: var(--text); }
  .metric-unit { font-size: 13px; color: var(--muted); font-family: 'DM Mono', monospace; }

  .severity-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 12px; border-radius: 999px;
    font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;
  }
  .severity-badge.mild   { background: rgba(0,229,160,0.15); color: var(--mild); }
  .severity-badge.moderate { background: rgba(245,197,24,0.15); color: var(--moderate); }
  .severity-badge.severe { background: rgba(255,68,68,0.15); color: var(--severe); }

  /* ── Healing stage ── */
  .stage-header { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; margin-bottom: 16px; }
  .stage-name { font-family: 'DM Serif Display', serif; font-size: 22px; }
  .confidence-ring { position: relative; width: 64px; height: 64px; flex-shrink: 0; }
  .confidence-ring svg { transform: rotate(-90deg); }
  .confidence-ring .ring-bg { fill: none; stroke: var(--surface2); stroke-width: 6; }
  .confidence-ring .ring-fill { fill: none; stroke: var(--accent); stroke-width: 6; stroke-linecap: round; transition: stroke-dashoffset 1s ease; }
  .confidence-ring .ring-text {
    position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
    font-family: 'DM Mono', monospace; font-size: 12px; font-weight: 500;
  }
  .stage-desc { color: var(--muted); font-size: 14px; line-height: 1.6; margin-bottom: 16px; }
  .recs-label { font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); margin-bottom: 10px; }
  .rec-item { display: flex; gap: 10px; padding: 8px 0; border-top: 1px solid var(--border); font-size: 13px; color: var(--muted); line-height: 1.4; }
  .rec-dot { color: var(--accent); flex-shrink: 0; margin-top: 1px; }

  /* ── Skin tone ── */
  .skin-row { display: flex; align-items: center; gap: 16px; }
  .skin-swatch { width: 48px; height: 48px; border-radius: 50%; flex-shrink: 0; border: 2px solid var(--border); }
  .skin-info { flex: 1; }
  .skin-type { font-size: 16px; font-weight: 700; }
  .skin-sub { font-family: 'DM Mono', monospace; font-size: 12px; color: var(--muted); }
  .luminance-bar { margin-top: 12px; }
  .luminance-track { height: 6px; border-radius: 3px; background: linear-gradient(to right, #1a1a1a, #fff); position: relative; margin-top: 6px; }
  .luminance-thumb { position: absolute; top: 50%; width: 12px; height: 12px; border-radius: 50%; background: var(--accent); border: 2px solid var(--bg); transform: translate(-50%, -50%); transition: left 1s ease; }
  .luminance-label { font-family: 'DM Mono', monospace; font-size: 11px; color: var(--muted); }

  /* ── Report ── */
  .report-id { font-family: 'DM Mono', monospace; font-size: 11px; color: var(--muted); margin-bottom: 16px; }
  .report-section { margin-bottom: 16px; }
  .report-section-title { font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: var(--accent); margin-bottom: 10px; }
  .report-finding { display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-top: 1px solid var(--border); font-size: 13px; }
  .report-finding-label { color: var(--muted); }
  .report-finding-value { font-family: 'DM Mono', monospace; font-size: 12px; }
  .report-followup { font-size: 13px; color: var(--muted); line-height: 1.6; background: var(--surface2); padding: 14px; border-radius: 8px; }

  /* ── Download buttons ── */
  .dl-row { display: flex; gap: 10px; flex-wrap: wrap; }
  .dl-btn {
    flex: 1; min-width: 120px; padding: 10px 16px;
    border: 1px solid var(--border); border-radius: 8px;
    background: transparent; color: var(--text);
    font-family: 'Syne', sans-serif; font-size: 13px; font-weight: 600;
    cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 8px;
    transition: all var(--transition);
  }
  .dl-btn:hover { border-color: var(--accent); color: var(--accent); }

  /* ── Error ── */
  .error-card {
    background: rgba(255,107,61,0.08); border: 1px solid rgba(255,107,61,0.25);
    border-radius: var(--card-r); padding: 24px;
    display: flex; gap: 16px; align-items: flex-start;
  }
  .error-icon { font-size: 24px; }
  .error-title { font-weight: 700; margin-bottom: 6px; }
  .error-msg { color: var(--muted); font-size: 14px; font-family: 'DM Mono', monospace; }

  /* ── Footer ── */
  .footer { text-align: center; padding: 32px; border-top: 1px solid var(--border); color: var(--muted); font-size: 12px; font-family: 'DM Mono', monospace; }

  /* ── Animations ── */
  @keyframes fadeUp { from { opacity: 0; transform: translateY(16px); } to { opacity: 1; transform: translateY(0); } }
  .fade-up { animation: fadeUp 0.5s ease forwards; }
`;

// ── Helpers ──────────────────────────────────────────────────────────────────
function downloadBase64(b64, filename) {
  const a = document.createElement("a");
  a.href = b64;
  a.download = filename;
  a.click();
}

function getSeverityClass(s) {
  if (!s) return "mild";
  const l = s.toLowerCase();
  if (l.includes("severe")) return "severe";
  if (l.includes("moderate")) return "moderate";
  return "mild";
}

// ── Sub-components ───────────────────────────────────────────────────────────
function ConfidenceRing({ value }) {
  const r = 26, circ = 2 * Math.PI * r;
  const offset = circ - (value / 100) * circ;
  return (
    <div className="confidence-ring">
      <svg width="64" height="64" viewBox="0 0 64 64">
        <circle className="ring-bg" cx="32" cy="32" r={r} />
        <circle
          className="ring-fill"
          cx="32" cy="32" r={r}
          strokeDasharray={circ}
          strokeDashoffset={offset}
        />
      </svg>
      <div className="ring-text">{value}%</div>
    </div>
  );
}

function MetricsPanel({ metrics }) {
  const sev = getSeverityClass(metrics.severity);
  return (
    <div className="card fade-up">
      <div className="card-header">
        <span className="card-title">Wound Metrics</span>
        <span className={`severity-badge ${sev}`}>
          ● {metrics.severity || "—"}
        </span>
      </div>
      <div className="card-body">
        <div className="metrics-grid">
          <div className="metric-card">
            <span className="metric-label">Area</span>
            <span className="metric-value">
              {metrics.area_pixels?.toLocaleString() ?? "—"}
            </span>
            <span className="metric-unit">pixels</span>
          </div>
          <div className="metric-card">
            <span className="metric-label">Coverage</span>
            <span className="metric-value">
              {typeof metrics.area_percentage === "number"
                ? metrics.area_percentage.toFixed(2)
                : "—"}
            </span>
            <span className="metric-unit">% of image</span>
          </div>
          <div className="metric-card">
            <span className="metric-label">Perimeter</span>
            <span className="metric-value">
              {metrics.perimeter?.toLocaleString() ?? "—"}
            </span>
            <span className="metric-unit">px</span>
          </div>
          <div className="metric-card">
            <span className="metric-label">Severity</span>
            <span className="metric-value" style={{ fontSize: 18, marginTop: 6 }}>
              <span className={`severity-badge ${sev}`}>{metrics.severity || "—"}</span>
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

function HealingPanel({ healing }) {
  const conf = Math.round((healing.confidence ?? 0) * 100);
  return (
    <div className="card fade-up" style={{ animationDelay: "0.1s" }}>
      <div className="card-header">
        <span className="card-title">Healing Stage</span>
      </div>
      <div className="card-body">
        <div className="stage-header">
          <span className="stage-name">{healing.stage ?? "Unknown"}</span>
          <ConfidenceRing value={conf} />
        </div>
        <p className="stage-desc">{healing.description ?? ""}</p>
        {healing.recommendations?.length > 0 && (
          <>
            <div className="recs-label">Recommendations</div>
            {healing.recommendations.map((r, i) => (
              <div key={i} className="rec-item">
                <span className="rec-dot">▸</span>
                <span>{r}</span>
              </div>
            ))}
          </>
        )}
      </div>
    </div>
  );
}

function SkinPanel({ skin }) {
  const lum = skin.luminance ?? 128;
  const pct = (lum / 255) * 100;
  return (
    <div className="card fade-up" style={{ animationDelay: "0.15s" }}>
      <div className="card-header">
        <span className="card-title">Skin Analysis</span>
      </div>
      <div className="card-body">
        <div className="skin-row">
          <div
            className="skin-swatch"
            style={{
              background: skin.rgb_avg
                ? `rgb(${skin.rgb_avg[0]},${skin.rgb_avg[1]},${skin.rgb_avg[2]})`
                : "#888",
            }}
          />
          <div className="skin-info">
            <div className="skin-type">{skin.skin_type ?? "—"}</div>
            <div className="skin-sub">Fitzpatrick {skin.fitzpatrick ?? "—"}</div>
          </div>
        </div>
        <div className="luminance-bar">
          <div className="luminance-label">Luminance · {lum.toFixed(0)}</div>
          <div className="luminance-track">
            <div className="luminance-thumb" style={{ left: `${pct}%` }} />
          </div>
        </div>
      </div>
    </div>
  );
}

function ReportPanel({ report }) {
  const findings = report.clinical_findings ?? {};
  return (
    <div className="card fade-up" style={{ animationDelay: "0.2s" }}>
      <div className="card-header">
        <span className="card-title">Clinical Report</span>
      </div>
      <div className="card-body">
        <div className="report-id">ID: {report.report_id ?? "—"} · {report.generated_at ? new Date(report.generated_at).toLocaleString() : ""}</div>
        {Object.keys(findings).length > 0 && (
          <div className="report-section">
            <div className="report-section-title">Clinical Findings</div>
            {Object.entries(findings).map(([k, v]) => (
              <div key={k} className="report-finding">
                <span className="report-finding-label">{k.replace(/_/g, " ")}</span>
                <span className="report-finding-value">{String(v)}</span>
              </div>
            ))}
          </div>
        )}
        {report.follow_up_care && (
          <div className="report-section">
            <div className="report-section-title">Follow-up Care</div>
            <div className="report-followup">{report.follow_up_care}</div>
          </div>
        )}
      </div>
    </div>
  );
}

function ImageViewer({ original, mask, heatmap, overlay, timestamp }) {
  const [tab, setTab] = useState("overlay");
  const tabs = [
    { id: "original", label: "Original", src: original },
    { id: "mask",     label: "Mask",     src: mask },
    { id: "heatmap",  label: "Heatmap",  src: heatmap },
    { id: "overlay",  label: "Overlay",  src: overlay },
  ];
  const current = tabs.find((t) => t.id === tab);

  return (
    <div className="card fade-up" style={{ animationDelay: "0.05s" }}>
      <div className="card-header">
        <span className="card-title">Segmentation Output</span>
      </div>
      <div className="image-tabs">
        {tabs.map((t) => (
          <button
            key={t.id}
            className={`tab-btn ${tab === t.id ? "active" : ""}`}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </div>
      <div className="card-body">
        {current?.src && (
          <img className="result-img" src={current.src} alt={tab} />
        )}
        <div className="dl-row" style={{ marginTop: 16 }}>
          {tabs.slice(1).map((t) => (
            <button
              key={t.id}
              className="dl-btn"
              onClick={() => downloadBase64(t.src, `wound_${t.id}_${timestamp}.png`)}
            >
              ↓ {t.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── Loading animation ─────────────────────────────────────────────────────────
const STEPS = [
  "Uploading image…",
  "Preprocessing…",
  "Running U-Net inference…",
  "Computing metrics…",
  "Generating heatmap…",
  "Building clinical report…",
];

function LoadingView({ step }) {
  return (
    <div className="loading-wrap">
      <div className="spinner" />
      <div className="loading-steps">
        {STEPS.map((s, i) => (
          <div
            key={i}
            className={`loading-step ${i < step ? "done" : i === step ? "active" : ""}`}
          >
            <span className="step-icon">{i < step ? "✓" : i === step ? "›" : "·"}</span>
            {s}
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Main App ─────────────────────────────────────────────────────────────────
export default function WoundAnalyzer() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [loadStep, setLoadStep] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [apiStatus, setApiStatus] = useState("checking"); // checking | online | offline
  const fileRef = useRef();
  const stepTimerRef = useRef();

  // Check API health on mount
  useState(() => {
    fetch(`${API_URL}/health`)
      .then((r) => r.ok ? setApiStatus("online") : setApiStatus("offline"))
      .catch(() => setApiStatus("offline"));
  });

  const handleFile = useCallback((f) => {
    if (!f || !f.type.startsWith("image/")) return;
    setFile(f);
    setResult(null);
    setError(null);
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target.result);
    reader.readAsDataURL(f);
  }, []);

  const onDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    handleFile(f);
  };

  const analyze = async () => {
    if (!file) return;
    setLoading(true);
    setLoadStep(0);
    setResult(null);
    setError(null);

    // Animate steps
    let s = 0;
    stepTimerRef.current = setInterval(() => {
      s++;
      if (s < STEPS.length - 1) setLoadStep(s);
    }, 900);

    try {
      const fd = new FormData();
      fd.append("image", file);
      const res = await fetch(`${API_URL}/analyze`, { method: "POST", body: fd });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if (!data.success) throw new Error(data.error || "Analysis failed");
      clearInterval(stepTimerRef.current);
      setLoadStep(STEPS.length);
      setResult({ ...data, _ts: Date.now() });
    } catch (e) {
      setError(e.message);
    } finally {
      clearInterval(stepTimerRef.current);
      setLoading(false);
    }
  };

  return (
    <>
      <style>{css}</style>
      <div className="app">
        {/* ── Header ── */}
        <header className="header">
          <div className="logo">
            <div className="logo-mark">🩹</div>
            <div className="logo-text">Wound<span>AI</span></div>
          </div>
          <div className="status-pill">
            <div className={`status-dot ${apiStatus}`} />
            {apiStatus === "online" ? "API Online" : apiStatus === "checking" ? "Connecting…" : "API Offline"}
          </div>
        </header>

        {/* ── Main ── */}
        <main className="main">
          {/* Hero */}
          {!result && !loading && (
            <div className="hero">
              <span className="hero-eyebrow">AI-Powered Clinical Analysis</span>
              <h1>Wound <em>Segmentation</em><br />& Assessment</h1>
              <p>Upload a wound image for instant AI analysis — segmentation mask, heatmap, severity scoring, and a full clinical report.</p>
            </div>
          )}

          {/* Upload */}
          {!result && !loading && (
            <>
              <div
                className={`upload-card ${dragging ? "drag-over" : ""} ${file ? "has-file" : ""}`}
                onClick={() => !file && fileRef.current.click()}
                onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
                onDragLeave={() => setDragging(false)}
                onDrop={onDrop}
              >
                {!file ? (
                  <>
                    <div className="upload-icon">🖼</div>
                    <div className="upload-title">Drop wound image here</div>
                    <div className="upload-sub">JPEG or PNG · Max 8 MB</div>
                    <button className="upload-btn" onClick={(e) => { e.stopPropagation(); fileRef.current.click(); }}>
                      Browse files
                    </button>
                  </>
                ) : (
                  <div className="preview-wrap">
                    <img className="preview-img" src={preview} alt="preview" />
                    <div className="preview-clear" onClick={(e) => { e.stopPropagation(); setFile(null); setPreview(null); }}>✕</div>
                  </div>
                )}
              </div>
              <input
                ref={fileRef} type="file" accept="image/*"
                style={{ display: "none" }}
                onChange={(e) => handleFile(e.target.files[0])}
              />
              <button
                className="analyze-btn"
                disabled={!file}
                onClick={analyze}
              >
                {file ? "🔍  Analyze Wound" : "Select an image to begin"}
              </button>
            </>
          )}

          {/* Loading */}
          {loading && <LoadingView step={loadStep} />}

          {/* Error */}
          {error && !loading && (
            <div className="error-card" style={{ marginTop: 24 }}>
              <div className="error-icon">⚠️</div>
              <div>
                <div className="error-title">Analysis Failed</div>
                <div className="error-msg">{error}</div>
              </div>
            </div>
          )}

          {/* Results */}
          {result && !loading && (
            <>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8 }}>
                <div style={{ fontFamily: "'DM Serif Display', serif", fontSize: 24 }}>Analysis Complete</div>
                <button
                  className="upload-btn"
                  onClick={() => { setResult(null); setFile(null); setPreview(null); }}
                >
                  ← New Analysis
                </button>
              </div>

              <div className="results-grid">
                {/* Left column */}
                <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
                  <ImageViewer
                    original={preview}
                    mask={result.mask_image}
                    heatmap={result.heatmap_image}
                    overlay={result.overlay_image}
                    timestamp={result._ts}
                  />
                  {result.metrics && <MetricsPanel metrics={result.metrics} />}
                </div>

                {/* Right column */}
                <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
                  {result.healing_stage && <HealingPanel healing={result.healing_stage} />}
                  {result.skin_analysis && <SkinPanel skin={result.skin_analysis} />}
                  {result.doctor_report && <ReportPanel report={result.doctor_report} />}
                </div>
              </div>
            </>
          )}
        </main>

        <footer className="footer">
          WoundAI · SimCLR + U-Net · v2.0 · For research use only
        </footer>
      </div>
    </>
  );
}
