import { useState, useRef, useCallback, useEffect } from "react";

// ─────────────────────────────────────────────────────────────────────────────
// 🔧 CONFIG — set your Railway URL and API key here
// ─────────────────────────────────────────────────────────────────────────────
const API_URL = "https://your-app.up.railway.app"; // ← your Railway URL
const STORAGE_KEY = "woundai_api_key";

// ── CSS ───────────────────────────────────────────────────────────────────────
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
  .app { display: flex; flex-direction: column; min-height: 100vh; }

  /* ── Header ── */
  .header {
    padding: 16px 40px;
    display: flex; align-items: center; justify-content: space-between;
    border-bottom: 1px solid var(--border);
    background: rgba(10,12,16,0.9);
    backdrop-filter: blur(12px);
    position: sticky; top: 0; z-index: 100;
    gap: 16px;
  }
  .logo { display: flex; align-items: center; gap: 12px; flex-shrink: 0; }
  .logo-mark {
    width: 34px; height: 34px; border-radius: 9px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    display: flex; align-items: center; justify-content: center; font-size: 17px;
  }
  .logo-text { font-family: 'DM Serif Display', serif; font-size: 19px; }
  .logo-text span { color: var(--accent); }

  .header-right { display: flex; align-items: center; gap: 12px; }

  /* API key input */
  .key-wrap { display: flex; align-items: center; gap: 8px; }
  .key-input {
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 8px; padding: 7px 12px;
    color: var(--text); font-family: 'DM Mono', monospace; font-size: 12px;
    width: 220px; outline: none; transition: border-color var(--transition);
  }
  .key-input:focus { border-color: var(--accent); }
  .key-input::placeholder { color: var(--muted); }
  .key-save-btn {
    padding: 7px 14px; border-radius: 8px;
    background: transparent; border: 1px solid var(--accent); color: var(--accent);
    font-family: 'Syne', sans-serif; font-size: 12px; font-weight: 700;
    cursor: pointer; transition: all var(--transition); white-space: nowrap;
  }
  .key-save-btn:hover { background: var(--accent); color: var(--bg); }
  .key-saved { font-family: 'DM Mono', monospace; font-size: 11px; color: var(--accent); white-space: nowrap; }

  /* Status pill */
  .status-pill {
    display: flex; align-items: center; gap: 8px;
    padding: 6px 14px; border-radius: 999px;
    border: 1px solid var(--border);
    font-family: 'DM Mono', monospace; font-size: 12px; color: var(--muted);
    white-space: nowrap;
  }
  .status-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--muted); transition: background var(--transition); }
  .status-dot.online { background: var(--accent); box-shadow: 0 0 8px var(--accent); }
  .status-dot.checking { background: var(--moderate); animation: pulse 1s infinite; }
  .status-dot.offline { background: var(--severe); }

  /* Rate limit bar */
  .rl-bar {
    display: flex; align-items: center; gap: 10px;
    font-family: 'DM Mono', monospace; font-size: 11px; color: var(--muted);
    padding: 6px 12px; border-radius: 8px; background: var(--surface2);
    border: 1px solid var(--border);
  }
  .rl-track { width: 60px; height: 4px; background: var(--surface); border-radius: 2px; overflow: hidden; }
  .rl-fill { height: 100%; border-radius: 2px; background: var(--accent); transition: width 0.4s ease; }
  .rl-fill.warn { background: var(--moderate); }
  .rl-fill.danger { background: var(--severe); }

  @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.3; } }

  /* ── Main ── */
  .main { flex: 1; padding: 48px 40px; max-width: 1280px; margin: 0 auto; width: 100%; }

  /* ── Hero ── */
  .hero { text-align: center; margin-bottom: 56px; }
  .hero-eyebrow {
    display: inline-block; font-family: 'DM Mono', monospace;
    font-size: 11px; letter-spacing: 2px; text-transform: uppercase;
    color: var(--accent); margin-bottom: 16px;
  }
  .hero h1 { font-family: 'DM Serif Display', serif; font-size: clamp(36px, 5vw, 60px); line-height: 1.1; letter-spacing: -1px; margin-bottom: 16px; }
  .hero h1 em { color: var(--accent); font-style: italic; }
  .hero p { color: var(--muted); font-size: 15px; max-width: 460px; margin: 0 auto; line-height: 1.6; }

  /* ── Upload ── */
  .upload-card {
    background: var(--surface); border: 1px solid var(--border); border-radius: var(--card-r);
    padding: 48px; display: flex; flex-direction: column; align-items: center;
    justify-content: center; gap: 20px; cursor: pointer;
    transition: border-color var(--transition), background var(--transition);
    position: relative; overflow: hidden; min-height: 240px;
  }
  .upload-card::before {
    content: ''; position: absolute; inset: 0;
    background: radial-gradient(ellipse at 50% 0%, rgba(0,229,160,0.04) 0%, transparent 70%);
    pointer-events: none;
  }
  .upload-card:hover, .upload-card.drag-over { border-color: var(--accent); background: rgba(0,229,160,0.03); }
  .upload-card.has-file { border-color: rgba(0,229,160,0.3); }
  .upload-icon { font-size: 44px; opacity: 0.5; }
  .upload-title { font-size: 17px; font-weight: 700; }
  .upload-sub { color: var(--muted); font-size: 13px; }
  .upload-btn {
    padding: 9px 22px; border-radius: 8px;
    background: transparent; border: 1px solid var(--accent); color: var(--accent);
    font-family: 'Syne', sans-serif; font-size: 13px; font-weight: 600;
    cursor: pointer; transition: all var(--transition);
  }
  .upload-btn:hover { background: var(--accent); color: var(--bg); }

  .preview-wrap { position: relative; width: 100%; max-width: 320px; margin: 0 auto; }
  .preview-img { width: 100%; border-radius: 12px; display: block; }
  .preview-clear {
    position: absolute; top: -10px; right: -10px; width: 26px; height: 26px;
    border-radius: 50%; background: var(--warn); border: 2px solid var(--bg);
    display: flex; align-items: center; justify-content: center;
    cursor: pointer; font-size: 13px; color: white; transition: transform var(--transition);
  }
  .preview-clear:hover { transform: scale(1.15); }

  .analyze-btn {
    width: 100%; padding: 15px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    border: none; border-radius: 10px;
    font-family: 'Syne', sans-serif; font-size: 15px; font-weight: 700;
    color: var(--bg); cursor: pointer;
    transition: opacity var(--transition), transform var(--transition);
    margin-top: 8px;
  }
  .analyze-btn:hover:not(:disabled) { opacity: 0.9; transform: translateY(-1px); }
  .analyze-btn:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }

  /* ── Loading ── */
  .loading-wrap { display: flex; flex-direction: column; align-items: center; gap: 20px; padding: 64px 0; }
  .spinner { width: 48px; height: 48px; border-radius: 50%; border: 3px solid var(--border); border-top-color: var(--accent); animation: spin 0.8s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .loading-steps { display: flex; flex-direction: column; gap: 8px; width: 100%; max-width: 240px; }
  .loading-step { display: flex; align-items: center; gap: 10px; font-family: 'DM Mono', monospace; font-size: 12px; color: var(--muted); transition: color var(--transition); }
  .loading-step.active { color: var(--accent); }
  .loading-step.done { color: var(--text); }
  .step-icon { width: 16px; text-align: center; }

  /* ── Results ── */
  .results-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-top: 40px; }
  @media (max-width: 900px) { .results-grid { grid-template-columns: 1fr; } }

  .card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--card-r); overflow: hidden; }
  .card-header { padding: 16px 20px; border-bottom: 1px solid var(--border); display: flex; align-items: center; justify-content: space-between; }
  .card-title { font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); }
  .card-body { padding: 20px; }

  .image-tabs { display: flex; gap: 4px; padding: 14px 20px 0; }
  .tab-btn { padding: 6px 13px; border-radius: 6px; border: none; background: transparent; color: var(--muted); font-family: 'Syne', sans-serif; font-size: 12px; font-weight: 600; cursor: pointer; transition: all var(--transition); }
  .tab-btn.active { background: var(--surface2); color: var(--text); }
  .tab-btn:hover:not(.active) { color: var(--text); }
  .result-img { width: 100%; display: block; border-radius: 8px; }

  .metrics-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .metric-card { background: var(--surface2); border-radius: 10px; padding: 16px; display: flex; flex-direction: column; gap: 4px; }
  .metric-label { font-family: 'DM Mono', monospace; font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; }
  .metric-value { font-family: 'DM Serif Display', serif; font-size: 26px; color: var(--text); }
  .metric-unit { font-size: 12px; color: var(--muted); font-family: 'DM Mono', monospace; }

  .severity-badge { display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; border-radius: 999px; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; }
  .severity-badge.mild     { background: rgba(0,229,160,0.15); color: var(--mild); }
  .severity-badge.moderate { background: rgba(245,197,24,0.15); color: var(--moderate); }
  .severity-badge.severe   { background: rgba(255,68,68,0.15); color: var(--severe); }

  .stage-header { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; margin-bottom: 14px; }
  .stage-name { font-family: 'DM Serif Display', serif; font-size: 21px; }
  .confidence-ring { position: relative; width: 60px; height: 60px; flex-shrink: 0; }
  .confidence-ring svg { transform: rotate(-90deg); }
  .confidence-ring .ring-bg { fill: none; stroke: var(--surface2); stroke-width: 6; }
  .confidence-ring .ring-fill { fill: none; stroke: var(--accent); stroke-width: 6; stroke-linecap: round; }
  .confidence-ring .ring-text { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; font-family: 'DM Mono', monospace; font-size: 11px; font-weight: 500; }
  .stage-desc { color: var(--muted); font-size: 13px; line-height: 1.6; margin-bottom: 14px; }
  .recs-label { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); margin-bottom: 10px; }
  .rec-item { display: flex; gap: 10px; padding: 8px 0; border-top: 1px solid var(--border); font-size: 13px; color: var(--muted); line-height: 1.4; }
  .rec-dot { color: var(--accent); flex-shrink: 0; margin-top: 1px; }

  .skin-row { display: flex; align-items: center; gap: 16px; }
  .skin-swatch { width: 46px; height: 46px; border-radius: 50%; flex-shrink: 0; border: 2px solid var(--border); }
  .skin-type { font-size: 15px; font-weight: 700; }
  .skin-sub { font-family: 'DM Mono', monospace; font-size: 12px; color: var(--muted); }
  .luminance-bar { margin-top: 12px; }
  .luminance-track { height: 6px; border-radius: 3px; background: linear-gradient(to right, #1a1a1a, #fff); position: relative; margin-top: 6px; }
  .luminance-thumb { position: absolute; top: 50%; width: 12px; height: 12px; border-radius: 50%; background: var(--accent); border: 2px solid var(--bg); transform: translate(-50%, -50%); }
  .luminance-label { font-family: 'DM Mono', monospace; font-size: 11px; color: var(--muted); }

  .report-id { font-family: 'DM Mono', monospace; font-size: 11px; color: var(--muted); margin-bottom: 14px; }
  .report-section { margin-bottom: 14px; }
  .report-section-title { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: var(--accent); margin-bottom: 10px; }
  .report-finding { display: flex; justify-content: space-between; align-items: center; padding: 7px 0; border-top: 1px solid var(--border); font-size: 13px; }
  .report-finding-label { color: var(--muted); }
  .report-finding-value { font-family: 'DM Mono', monospace; font-size: 12px; }
  .report-followup { font-size: 13px; color: var(--muted); line-height: 1.6; background: var(--surface2); padding: 12px; border-radius: 8px; }

  .dl-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 14px; }
  .dl-btn { flex: 1; min-width: 100px; padding: 9px 14px; border: 1px solid var(--border); border-radius: 8px; background: transparent; color: var(--text); font-family: 'Syne', sans-serif; font-size: 12px; font-weight: 600; cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 6px; transition: all var(--transition); }
  .dl-btn:hover { border-color: var(--accent); color: var(--accent); }

  /* ── Error ── */
  .error-card { background: rgba(255,107,61,0.08); border: 1px solid rgba(255,107,61,0.25); border-radius: var(--card-r); padding: 22px; display: flex; gap: 14px; align-items: flex-start; margin-top: 20px; }
  .error-icon { font-size: 22px; }
  .error-title { font-weight: 700; margin-bottom: 5px; }
  .error-msg { color: var(--muted); font-size: 13px; font-family: 'DM Mono', monospace; }
  .error-hint { color: var(--moderate); font-size: 12px; margin-top: 8px; font-family: 'DM Mono', monospace; }

  /* ── Auth error ── */
  .auth-error { background: rgba(255,68,68,0.07); border: 1px solid rgba(255,68,68,0.2); border-radius: var(--card-r); padding: 18px 22px; display: flex; align-items: center; gap: 12px; margin-top: 12px; }
  .auth-error-text { font-size: 13px; }
  .auth-error-text strong { color: var(--severe); }

  /* ── Rate limit warning ── */
  .rate-warn { background: rgba(245,197,24,0.07); border: 1px solid rgba(245,197,24,0.2); border-radius: var(--card-r); padding: 14px 18px; display: flex; align-items: center; gap: 10px; margin-top: 8px; font-size: 13px; color: var(--moderate); font-family: 'DM Mono', monospace; }

  /* ── Animations ── */
  @keyframes fadeUp { from { opacity:0; transform:translateY(16px); } to { opacity:1; transform:translateY(0); } }
  .fade-up { animation: fadeUp 0.5s ease forwards; }

  .footer { text-align: center; padding: 28px; border-top: 1px solid var(--border); color: var(--muted); font-size: 11px; font-family: 'DM Mono', monospace; }
`;

// ── Helpers ───────────────────────────────────────────────────────────────────
function downloadBase64(b64, filename) {
  const a = document.createElement("a");
  a.href = b64;
  a.download = filename;
  a.click();
}
function getSeverityClass(s = "") {
  const l = s.toLowerCase();
  if (l.includes("severe")) return "severe";
  if (l.includes("moderate")) return "moderate";
  return "mild";
}

// ── Sub-components ────────────────────────────────────────────────────────────
function ConfidenceRing({ value }) {
  const r = 24, circ = 2 * Math.PI * r;
  return (
    <div className="confidence-ring">
      <svg width="60" height="60" viewBox="0 0 60 60">
        <circle className="ring-bg" cx="30" cy="30" r={r} />
        <circle className="ring-fill" cx="30" cy="30" r={r}
          strokeDasharray={circ}
          strokeDashoffset={circ - (value / 100) * circ} />
      </svg>
      <div className="ring-text">{value}%</div>
    </div>
  );
}

function RateLimitBar({ info }) {
  if (!info) return null;
  const pct = (info.remaining / info.limit) * 100;
  const cls = pct < 20 ? "danger" : pct < 50 ? "warn" : "";
  return (
    <div className="rl-bar" title={`${info.remaining}/${info.limit} requests remaining`}>
      <div className="rl-track">
        <div className="rl-fill" style={{ width: `${pct}%` }} />
      </div>
      {info.remaining}/{info.limit} req/min
    </div>
  );
}

function ApiKeyInput({ apiKey, onSave }) {
  const [draft, setDraft] = useState(apiKey || "");
  const [saved, setSaved] = useState(!!apiKey);

  const save = () => {
    onSave(draft.trim());
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  return (
    <div className="key-wrap">
      <input
        className="key-input"
        type="password"
        placeholder="API key (wai_…)"
        value={draft}
        onChange={(e) => { setDraft(e.target.value); setSaved(false); }}
        onKeyDown={(e) => e.key === "Enter" && save()}
      />
      {saved
        ? <span className="key-saved">✓ saved</span>
        : <button className="key-save-btn" onClick={save}>Save key</button>
      }
    </div>
  );
}

function MetricsPanel({ metrics }) {
  const sev = getSeverityClass(metrics.severity);
  return (
    <div className="card fade-up">
      <div className="card-header">
        <span className="card-title">Wound Metrics</span>
        <span className={`severity-badge ${sev}`}>● {metrics.severity || "—"}</span>
      </div>
      <div className="card-body">
        <div className="metrics-grid">
          {[
            { label: "Area", value: metrics.area_pixels?.toLocaleString() ?? "—", unit: "pixels" },
            { label: "Coverage", value: typeof metrics.area_percentage === "number" ? metrics.area_percentage.toFixed(2) : "—", unit: "% of image" },
            { label: "Perimeter", value: metrics.perimeter?.toLocaleString() ?? "—", unit: "px" },
          ].map((m) => (
            <div key={m.label} className="metric-card">
              <span className="metric-label">{m.label}</span>
              <span className="metric-value">{m.value}</span>
              <span className="metric-unit">{m.unit}</span>
            </div>
          ))}
          <div className="metric-card">
            <span className="metric-label">Severity</span>
            <div style={{ marginTop: 8 }}>
              <span className={`severity-badge ${sev}`}>{metrics.severity || "—"}</span>
            </div>
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
      <div className="card-header"><span className="card-title">Healing Stage</span></div>
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
              <div key={i} className="rec-item"><span className="rec-dot">▸</span><span>{r}</span></div>
            ))}
          </>
        )}
      </div>
    </div>
  );
}

function SkinPanel({ skin }) {
  const lum = skin.luminance ?? 128;
  return (
    <div className="card fade-up" style={{ animationDelay: "0.15s" }}>
      <div className="card-header"><span className="card-title">Skin Analysis</span></div>
      <div className="card-body">
        <div className="skin-row">
          <div className="skin-swatch" style={{ background: skin.rgb_avg ? `rgb(${skin.rgb_avg[0]},${skin.rgb_avg[1]},${skin.rgb_avg[2]})` : "#888" }} />
          <div>
            <div className="skin-type">{skin.skin_type ?? "—"}</div>
            <div className="skin-sub">Fitzpatrick {skin.fitzpatrick ?? "—"}</div>
          </div>
        </div>
        <div className="luminance-bar">
          <div className="luminance-label">Luminance · {lum.toFixed(0)}</div>
          <div className="luminance-track">
            <div className="luminance-thumb" style={{ left: `${(lum / 255) * 100}%` }} />
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
      <div className="card-header"><span className="card-title">Clinical Report</span></div>
      <div className="card-body">
        <div className="report-id">
          ID: {report.report_id ?? "—"} · {report.generated_at ? new Date(report.generated_at).toLocaleString() : ""}
        </div>
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
    <div className="card fade-up">
      <div className="card-header"><span className="card-title">Segmentation Output</span></div>
      <div className="image-tabs">
        {tabs.map((t) => (
          <button key={t.id} className={`tab-btn ${tab === t.id ? "active" : ""}`} onClick={() => setTab(t.id)}>
            {t.label}
          </button>
        ))}
      </div>
      <div className="card-body">
        {current?.src && <img className="result-img" src={current.src} alt={tab} />}
        <div className="dl-row">
          {tabs.slice(1).map((t) => (
            <button key={t.id} className="dl-btn" onClick={() => downloadBase64(t.src, `wound_${t.id}_${timestamp}.png`)}>
              ↓ {t.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

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
          <div key={i} className={`loading-step ${i < step ? "done" : i === step ? "active" : ""}`}>
            <span className="step-icon">{i < step ? "✓" : i === step ? "›" : "·"}</span>
            {s}
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function WoundAnalyzer() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [loadStep, setLoadStep] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [errorType, setErrorType] = useState(null); // "auth" | "rate" | "server"
  const [apiStatus, setApiStatus] = useState("checking");
  const [apiKey, setApiKey] = useState(() => localStorage.getItem(STORAGE_KEY) || "");
  const [rateLimitInfo, setRateLimitInfo] = useState(null);
  const [retryAfter, setRetryAfter] = useState(null);
  const fileRef = useRef();
  const stepTimer = useRef();

  // Health check
  useEffect(() => {
    fetch(`${API_URL}/health`)
      .then((r) => setApiStatus(r.ok ? "online" : "offline"))
      .catch(() => setApiStatus("offline"));
  }, []);

  const saveKey = useCallback((k) => {
    setApiKey(k);
    localStorage.setItem(STORAGE_KEY, k);
  }, []);

  const handleFile = useCallback((f) => {
    if (!f || !f.type.startsWith("image/")) return;
    setFile(f);
    setResult(null);
    setError(null);
    setErrorType(null);
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target.result);
    reader.readAsDataURL(f);
  }, []);

  const onDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    handleFile(e.dataTransfer.files[0]);
  };

  const analyze = async () => {
    if (!file) return;
    setLoading(true);
    setLoadStep(0);
    setResult(null);
    setError(null);
    setErrorType(null);
    setRetryAfter(null);

    let s = 0;
    stepTimer.current = setInterval(() => {
      s++;
      if (s < STEPS.length - 1) setLoadStep(s);
    }, 900);

    try {
      const fd = new FormData();
      fd.append("image", file);

      const headers = {};
      if (apiKey) headers["X-API-Key"] = apiKey;

      const res = await fetch(`${API_URL}/analyze`, { method: "POST", headers, body: fd });

      // Parse rate limit headers
      const rl = {
        limit: parseInt(res.headers.get("X-RateLimit-Limit") || "10"),
        remaining: parseInt(res.headers.get("X-RateLimit-Remaining") || "10"),
        reset_after: parseInt(res.headers.get("X-RateLimit-Reset") || "60"),
      };
      setRateLimitInfo(rl);

      if (res.status === 401 || res.status === 403) {
        setErrorType("auth");
        setError(res.status === 401 ? "Missing API key — add your key above." : "Invalid API key — check and try again.");
        return;
      }
      if (res.status === 429) {
        const data = await res.json().catch(() => ({}));
        setErrorType("rate");
        setRetryAfter(data.retry_after || rl.reset_after || 60);
        setError(`Rate limit reached. Try again in ${data.retry_after || 60}s.`);
        return;
      }
      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const data = await res.json();
      if (!data.success) throw new Error(data.error || "Analysis failed");

      clearInterval(stepTimer.current);
      setLoadStep(STEPS.length);
      setResult({ ...data, _ts: Date.now() });
    } catch (e) {
      if (!errorType) {
        setErrorType("server");
        setError(e.message);
      }
    } finally {
      clearInterval(stepTimer.current);
      setLoading(false);
    }
  };

  return (
    <>
      <style>{css}</style>
      <div className="app">
        {/* Header */}
        <header className="header">
          <div className="logo">
            <div className="logo-mark">🩹</div>
            <div className="logo-text">Wound<span>AI</span></div>
          </div>
          <div className="header-right">
            <ApiKeyInput apiKey={apiKey} onSave={saveKey} />
            <RateLimitBar info={rateLimitInfo} />
            <div className="status-pill">
              <div className={`status-dot ${apiStatus}`} />
              {apiStatus === "online" ? "API Online" : apiStatus === "checking" ? "Connecting…" : "Offline"}
            </div>
          </div>
        </header>

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
              {!apiKey && (
                <div className="auth-error" style={{ marginBottom: 16 }}>
                  <span style={{ fontSize: 20 }}>🔑</span>
                  <div className="auth-error-text">
                    <strong>API key required</strong> — paste your key in the header field above, then click <em>Save key</em>.
                  </div>
                </div>
              )}
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
                    <button className="upload-btn" onClick={(e) => { e.stopPropagation(); fileRef.current.click(); }}>Browse files</button>
                  </>
                ) : (
                  <div className="preview-wrap">
                    <img className="preview-img" src={preview} alt="preview" />
                    <div className="preview-clear" onClick={(e) => { e.stopPropagation(); setFile(null); setPreview(null); }}>✕</div>
                  </div>
                )}
              </div>
              <input ref={fileRef} type="file" accept="image/*" style={{ display: "none" }} onChange={(e) => handleFile(e.target.files[0])} />
              <button className="analyze-btn" disabled={!file || !apiKey} onClick={analyze}>
                {!apiKey ? "🔑 Set your API key to continue" : file ? "🔍  Analyze Wound" : "Select an image to begin"}
              </button>
            </>
          )}

          {/* Loading */}
          {loading && <LoadingView step={loadStep} />}

          {/* Auth error */}
          {errorType === "auth" && !loading && (
            <div className="auth-error" style={{ marginTop: 16 }}>
              <span style={{ fontSize: 20 }}>🔒</span>
              <div>
                <strong style={{ color: "var(--severe)" }}>Authentication failed</strong>
                <div style={{ color: "var(--muted)", fontSize: 13, marginTop: 4 }}>{error}</div>
              </div>
            </div>
          )}

          {/* Rate limit error */}
          {errorType === "rate" && !loading && (
            <div className="rate-warn" style={{ marginTop: 16 }}>
              ⏱ {error} Your limit resets in {retryAfter}s.
            </div>
          )}

          {/* Server error */}
          {errorType === "server" && !loading && (
            <div className="error-card">
              <div className="error-icon">⚠️</div>
              <div>
                <div className="error-title">Analysis Failed</div>
                <div className="error-msg">{error}</div>
                <div className="error-hint">Check your Railway logs for details.</div>
              </div>
            </div>
          )}

          {/* Results */}
          {result && !loading && (
            <>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8 }}>
                <div style={{ fontFamily: "'DM Serif Display', serif", fontSize: 22 }}>Analysis Complete</div>
                <button className="upload-btn" onClick={() => { setResult(null); setFile(null); setPreview(null); }}>← New Analysis</button>
              </div>
              <div className="results-grid">
                <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
                  <ImageViewer original={preview} mask={result.mask_image} heatmap={result.heatmap_image} overlay={result.overlay_image} timestamp={result._ts} />
                  {result.metrics && <MetricsPanel metrics={result.metrics} />}
                </div>
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
          WoundAI · SimCLR + U-Net · v2.0 · API key secured · For research use only
        </footer>
      </div>
    </>
  );
}
