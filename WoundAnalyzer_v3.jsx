import { useState, useRef, useCallback, useEffect } from "react";

/* ═══════════════════════════ CSS ═══════════════════════════ */
const css = `
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Geist+Mono:wght@300;400;500&family=Geist:wght@300;400;500;600&display=swap');
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0b0e13;--surface:#111620;--s2:#181e2a;--s3:#1e2535;--s4:#232c3d;
  --border:rgba(255,255,255,0.07);--b2:rgba(255,255,255,0.12);--b3:rgba(255,255,255,0.18);
  --text:#e8eaf0;--muted:#6b7280;--muted2:#9ca3af;
  --accent:#00e5a0;--accent2:#00b8ff;--accent3:#7c6dff;
  --severe:#ff4d4d;--moderate:#f5c518;--mild:#00e5a0;
  --gran:#e74c3c;--slou:#f39c12;--necr:#2c3e50;--epth:#27ae60;
  --r:14px;--tr:0.18s ease;
}
html,body,#root{height:100%;background:var(--bg);color:var(--text);font-family:'Geist',sans-serif}
body::before{content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
  background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.03'/%3E%3C/svg%3E");
  background-size:128px 128px}

/* layout */
.app{position:relative;z-index:1;min-height:100vh;display:flex;flex-direction:column}
.topbar{display:flex;align-items:center;justify-content:space-between;padding:0 28px;height:56px;
  border-bottom:1px solid var(--border);background:rgba(11,14,19,.88);backdrop-filter:blur(16px);
  position:sticky;top:0;z-index:50}
.logo{display:flex;align-items:center;gap:9px}
.logo-mark{width:28px;height:28px;border-radius:7px;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:700;color:#000}
.logo-text{font-family:'Instrument Serif',serif;font-size:17px}
.topbar-right{display:flex;align-items:center;gap:12px}
.api-pill{display:flex;align-items:center;gap:6px;padding:4px 11px;border-radius:999px;
  border:1px solid var(--border);font-family:'Geist Mono',monospace;font-size:11px;color:var(--muted2)}
.api-dot{width:6px;height:6px;border-radius:50%;background:var(--accent);box-shadow:0 0 7px var(--accent)}
.main{flex:1;max-width:1240px;width:100%;margin:0 auto;padding:40px 24px 80px}

/* buttons */
.btn-p{display:inline-flex;align-items:center;gap:7px;padding:10px 22px;border-radius:8px;border:none;cursor:pointer;
  background:linear-gradient(135deg,var(--accent),var(--accent2));color:#000;font-family:'Geist',sans-serif;
  font-size:13px;font-weight:600;transition:opacity var(--tr),transform var(--tr)}
.btn-p:hover:not(:disabled){opacity:.86;transform:translateY(-1px)}
.btn-p:disabled{opacity:.3;cursor:not-allowed}
.btn-g{display:inline-flex;align-items:center;gap:6px;padding:8px 15px;border-radius:7px;cursor:pointer;
  background:transparent;color:var(--muted2);font-family:'Geist',sans-serif;font-size:12px;font-weight:500;
  border:1px solid var(--b2);transition:all var(--tr)}
.btn-g:hover{background:var(--s2);color:var(--text)}

/* upload */
.upload-zone{border:1.5px dashed var(--b2);border-radius:var(--r);padding:56px 40px;text-align:center;
  cursor:pointer;transition:all var(--tr);background:var(--surface);position:relative;overflow:hidden}
.upload-zone::before{content:'';position:absolute;inset:0;pointer-events:none;
  background:radial-gradient(ellipse at 50% 0%,rgba(0,229,160,.06),transparent 70%)}
.upload-zone:hover,.upload-zone.drag{border-color:var(--accent);background:var(--s2)}
.upload-zone.has-file{padding:0;cursor:default;border-style:solid;border-color:var(--b2)}
.upload-icon{font-size:34px;margin-bottom:14px}
.upload-title{font-family:'Instrument Serif',serif;font-size:21px;margin-bottom:7px}
.upload-sub{font-size:13px;color:var(--muted);margin-bottom:18px}
.preview-wrap{position:relative}
.preview-img{width:100%;max-height:320px;object-fit:cover;display:block;border-radius:12px}
.preview-clear{position:absolute;top:10px;right:10px;width:27px;height:27px;border-radius:50%;
  background:rgba(0,0,0,.72);color:#fff;display:flex;align-items:center;justify-content:center;
  font-size:12px;cursor:pointer;border:1px solid var(--b2);transition:background var(--tr)}
.preview-clear:hover{background:var(--severe)}
.btn-row{display:flex;align-items:center;gap:9px;margin-top:14px;flex-wrap:wrap}

/* intro */
.intro{text-align:center;margin-bottom:36px}
.intro-title{font-family:'Instrument Serif',serif;font-size:38px;font-style:italic;margin-bottom:11px;line-height:1.1}
.intro-sub{font-size:13px;color:var(--muted2);max-width:420px;margin:0 auto;line-height:1.7}
.intro-badge{display:inline-flex;align-items:center;gap:6px;padding:5px 13px;border-radius:999px;
  font-size:11px;font-family:'Geist Mono',monospace;border:1px solid var(--b2);color:var(--muted2);margin-bottom:18px}

/* loading */
.loading-wrap{display:flex;flex-direction:column;align-items:center;gap:22px;padding:72px 0}
.spin{width:52px;height:52px;border-radius:50%;border:2px solid var(--b2);border-top-color:var(--accent);animation:spin .9s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.load-steps{display:flex;flex-direction:column;gap:9px}
.lstep{display:flex;align-items:center;gap:9px;font-family:'Geist Mono',monospace;font-size:12px;color:var(--muted);transition:color .3s}
.lstep.active{color:var(--accent)}.lstep.done{color:var(--muted2)}
.ldot{width:5px;height:5px;border-radius:50%;background:currentColor;flex-shrink:0}

/* result header */
.res-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:26px;
  padding-bottom:18px;border-bottom:1px solid var(--border)}
.res-title{font-family:'Instrument Serif',serif;font-size:26px}
.res-sub{font-size:12px;color:var(--muted);margin-top:3px;font-family:'Geist Mono',monospace}

/* grid */
.results-grid{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(0,.9fr);gap:18px;align-items:start}
@media(max-width:860px){.results-grid{grid-template-columns:1fr}}
.col{display:flex;flex-direction:column;gap:15px}

/* cards */
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);overflow:hidden;
  animation:fadeUp .32s ease both}
@keyframes fadeUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}
.card:nth-child(1){animation-delay:0s}.card:nth-child(2){animation-delay:.06s}
.card:nth-child(3){animation-delay:.12s}.card:nth-child(4){animation-delay:.18s}
.card:nth-child(5){animation-delay:.24s}.card:nth-child(6){animation-delay:.30s}
.card-hdr{padding:13px 18px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between}
.card-lbl{font-family:'Geist Mono',monospace;font-size:10px;font-weight:500;text-transform:uppercase;letter-spacing:1.4px;color:var(--muted)}
.card-body{padding:18px}

/* image viewer */
.img-tabs{display:flex;gap:3px;padding:11px 14px 0;flex-wrap:wrap}
.img-tab{padding:5px 11px;border-radius:6px;border:none;cursor:pointer;font-family:'Geist',sans-serif;
  font-size:12px;font-weight:500;color:var(--muted);background:transparent;transition:all var(--tr)}
.img-tab.active{background:var(--s3);color:var(--text)}
.img-frame{padding:14px;position:relative}
.view-img{width:100%;border-radius:7px;display:block;aspect-ratio:1;object-fit:cover}
.hm-legend{display:flex;align-items:center;gap:9px;padding:9px 14px;border-top:1px solid var(--border);
  font-family:'Geist Mono',monospace;font-size:10px;color:var(--muted)}
.hm-grad{flex:1;height:5px;border-radius:3px;background:linear-gradient(to right,#0000ff,#00ffff,#00ff00,#ffff00,#ff0000)}
.dl-strip{display:flex;border-top:1px solid var(--border)}
.dl-btn{flex:1;padding:10px 0;font-size:11px;font-family:'Geist',sans-serif;font-weight:500;
  color:var(--muted2);background:transparent;border:none;border-right:1px solid var(--border);
  cursor:pointer;transition:all var(--tr);display:flex;align-items:center;justify-content:center;gap:5px}
.dl-btn:last-child{border-right:none}
.dl-btn:hover{background:var(--s2);color:var(--text)}

/* metrics */
.m2{display:grid;grid-template-columns:1fr 1fr;gap:9px}
.m-tile{background:var(--s2);border-radius:9px;padding:14px 13px;display:flex;flex-direction:column;gap:4px;
  border:1px solid var(--border);transition:border-color var(--tr)}
.m-tile:hover{border-color:var(--b2)}
.m-lbl{font-family:'Geist Mono',monospace;font-size:9px;text-transform:uppercase;letter-spacing:1px;color:var(--muted)}
.m-val{font-family:'Instrument Serif',serif;font-size:26px;color:var(--text);line-height:1}
.m-unit{font-family:'Geist Mono',monospace;font-size:10px;color:var(--muted)}
.m-full{grid-column:1/-1;flex-direction:row;align-items:center;justify-content:space-between}

/* uncertainty bar */
.unc-row{display:flex;align-items:center;gap:10px;margin-top:10px}
.unc-bar-wrap{flex:1;height:3px;background:var(--s3);border-radius:2px;overflow:hidden}
.unc-bar{height:100%;border-radius:2px;background:var(--accent3)}
.unc-lbl{font-family:'Geist Mono',monospace;font-size:11px;color:var(--muted2);min-width:36px;text-align:right}

/* severity */
.sev{display:inline-flex;align-items:center;gap:5px;padding:4px 12px;border-radius:999px;font-size:11px;font-weight:600;letter-spacing:.4px}
.sev.mild    {background:rgba(0,229,160,.1);color:var(--mild);border:1px solid rgba(0,229,160,.22)}
.sev.moderate{background:rgba(245,197,24,.1);color:var(--moderate);border:1px solid rgba(245,197,24,.22)}
.sev.severe  {background:rgba(255,77,77,.1);color:var(--severe);border:1px solid rgba(255,77,77,.22)}
.sev-dot{width:5px;height:5px;border-radius:50%;background:currentColor}

/* confidence bar */
.conf-row{display:flex;align-items:center;gap:10px;margin-top:10px}
.conf-wrap{flex:1;height:4px;background:var(--s3);border-radius:2px;overflow:hidden}
.conf-fill{height:100%;border-radius:2px;background:linear-gradient(90deg,var(--accent2),var(--accent));transition:width .6s ease}
.conf-pct{font-family:'Geist Mono',monospace;font-size:12px;color:var(--accent);min-width:34px;text-align:right}

/* healing */
.stage-name{font-family:'Instrument Serif',serif;font-size:22px;font-style:italic}
.stage-desc{font-size:13px;color:var(--muted2);line-height:1.65;margin:9px 0 14px}
.recs{display:flex;flex-direction:column}
.rec{display:flex;align-items:flex-start;gap:9px;padding:8px 0;border-top:1px solid var(--border);font-size:12px;color:var(--muted2);line-height:1.5}
.rec-arr{color:var(--accent);flex-shrink:0;font-size:10px;margin-top:2px}

/* tissue donut */
.tissue-wrap{display:flex;align-items:center;gap:18px}
.donut-svg{flex-shrink:0}
.tissue-legend{display:flex;flex-direction:column;gap:7px;flex:1}
.tleg-row{display:flex;align-items:center;gap:8px;font-size:12px}
.tleg-dot{width:8px;height:8px;border-radius:2px;flex-shrink:0}
.tleg-name{color:var(--muted2);flex:1}
.tleg-pct{font-family:'Geist Mono',monospace;font-size:11px;color:var(--text);min-width:36px;text-align:right}

/* skin */
.skin-row{display:flex;align-items:center;gap:13px;margin-bottom:14px}
.skin-swatch{width:42px;height:42px;border-radius:50%;border:2px solid var(--b2);flex-shrink:0}
.skin-name{font-family:'Instrument Serif',serif;font-size:17px}
.skin-sub{font-family:'Geist Mono',monospace;font-size:10px;color:var(--muted);margin-top:2px}
.lum-hdr{display:flex;justify-content:space-between;margin-bottom:5px;font-family:'Geist Mono',monospace;font-size:10px;color:var(--muted)}
.lum-track{width:100%;height:5px;border-radius:3px;background:linear-gradient(to right,#1a1a1a,#fff);position:relative}
.lum-pip{width:11px;height:11px;border-radius:50%;background:var(--text);border:2px solid var(--bg);
  position:absolute;top:50%;transform:translate(-50%,-50%);pointer-events:none}
.ita-badge{display:inline-flex;align-items:center;gap:5px;padding:3px 9px;border-radius:999px;
  font-family:'Geist Mono',monospace;font-size:10px;color:var(--muted2);border:1px solid var(--border);margin-top:10px}
.rgb-row{display:flex;gap:7px;margin-top:11px}
.rgb-chip{flex:1;padding:7px 5px;border-radius:7px;text-align:center;background:var(--s2);border:1px solid var(--border)}
.rgb-lbl{font-family:'Geist Mono',monospace;font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:1px}
.rgb-val{font-family:'Geist Mono',monospace;font-size:13px;color:var(--text);margin-top:2px}

/* report */
.rpt-id{font-family:'Geist Mono',monospace;font-size:10px;color:var(--muted);margin-bottom:14px}
.rpt-sec{font-size:9px;font-weight:600;text-transform:uppercase;letter-spacing:1.4px;color:var(--accent);margin:14px 0 8px;font-family:'Geist Mono',monospace}
.rpt-row{display:flex;justify-content:space-between;align-items:flex-start;padding:7px 0;border-top:1px solid var(--border);font-size:12px;gap:14px}
.rpt-key{color:var(--muted)}.rpt-val{color:var(--text);font-weight:500;text-align:right;font-size:12px}
.follow-box{background:var(--s2);border-radius:7px;padding:12px 14px;font-size:12px;color:var(--muted2);line-height:1.7;border:1px solid var(--border);margin-top:3px}
.meta{font-family:'Geist Mono',monospace;font-size:10px;color:var(--muted);line-height:1.9}
.disclaimer{font-size:10px;color:var(--muted);border-top:1px solid var(--border);padding-top:10px;margin-top:10px;line-height:1.6;font-style:italic}

/* error */
.err-card{background:rgba(255,77,77,.07);border:1px solid rgba(255,77,77,.22);border-radius:var(--r);
  padding:18px 22px;display:flex;gap:12px;align-items:flex-start;margin-top:18px}
.err-title{font-weight:600;color:var(--severe)}.err-msg{font-size:12px;color:var(--muted2);margin-top:3px}

/* footer */
.footer{text-align:center;padding:20px;font-family:'Geist Mono',monospace;font-size:10px;color:var(--muted);
  border-top:1px solid var(--border);display:flex;gap:14px;justify-content:center;flex-wrap:wrap}
.footer span{color:var(--b2)}

/* print */
@media print{
  .topbar,.btn-row,.upload-zone,.loading-wrap,.dl-strip,.img-tabs,.btn-g,.btn-p{display:none!important}
  .app{background:#fff;color:#000}
  .card{border:1px solid #ccc;break-inside:avoid}
  .results-grid{grid-template-columns:1fr}
}
`;

/* ═══════════════════════════ HELPERS ═══════════════════════════ */
const sevClass = s => ({ mild:"mild", moderate:"moderate" }[s?.toLowerCase()] ?? "severe");

function dlBase64(b64, name) {
  const a = document.createElement("a");
  a.href = `data:image/png;base64,${b64}`;
  a.download = name; a.click();
}

/* ═══════════════════════════ TISSUE DONUT ═══════════════════════════ */
function TissueDonut({ tissue }) {
  const items = [
    { key: "granulation", label: "Granulation", color: "#e74c3c" },
    { key: "slough",      label: "Slough",       color: "#f39c12" },
    { key: "necrotic",    label: "Necrotic",     color: "#5d6d7e" },
    { key: "epithelial",  label: "Epithelial",   color: "#27ae60" },
    { key: "other",       label: "Other",        color: "#34495e" },
  ].map(i => ({ ...i, pct: tissue?.[i.key] ?? 0 })).filter(i => i.pct > 0);

  const R = 36, cx = 44, cy = 44, stroke = 14;
  const circ = 2 * Math.PI * R;
  let offset = 0;
  const segments = items.map(i => {
    const dash = (i.pct / 100) * circ;
    const seg  = { ...i, dash, offset };
    offset += dash;
    return seg;
  });

  return (
    <div className="tissue-wrap">
      <svg className="donut-svg" width="88" height="88" viewBox="0 0 88 88">
        <circle cx={cx} cy={cy} r={R} fill="none" stroke="var(--s3)" strokeWidth={stroke} />
        {segments.map(s => (
          <circle key={s.key} cx={cx} cy={cy} r={R} fill="none"
            stroke={s.color} strokeWidth={stroke}
            strokeDasharray={`${s.dash} ${circ - s.dash}`}
            strokeDashoffset={-s.offset}
            transform={`rotate(-90 ${cx} ${cy})`} />
        ))}
      </svg>
      <div className="tissue-legend">
        {items.map(i => (
          <div key={i.key} className="tleg-row">
            <div className="tleg-dot" style={{ background: i.color }} />
            <span className="tleg-name">{i.label}</span>
            <span className="tleg-pct">{i.pct.toFixed(1)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ═══════════════════════════ IMAGE VIEWER ═══════════════════════════ */
function ImageViewer({ original, mask, heatmap, uncertainty, overlay, contour }) {
  const [tab, setTab] = useState("overlay");
  const tabs = [
    { id:"original",    label:"Original"    },
    { id:"mask",        label:"Mask"        },
    { id:"heatmap",     label:"Heatmap"     },
    { id:"uncertainty", label:"Uncertainty" },
    { id:"contour",     label:"Contour"     },
    { id:"overlay",     label:"Overlay"     },
  ].filter(t => t.id === "original" || !!{ mask, heatmap, uncertainty, overlay, contour }[t.id]);

  const b64src = b64 => `data:image/png;base64,${b64}`;
  const src = {
    original:    original,
    mask:        mask        ? b64src(mask)        : original,
    heatmap:     heatmap     ? b64src(heatmap)     : original,
    uncertainty: uncertainty ? b64src(uncertainty) : original,
    overlay:     overlay     ? b64src(overlay)     : original,
    contour:     contour     ? b64src(contour)     : original,
  }[tab];

  const showLegend = tab === "heatmap" || tab === "uncertainty";
  const legendLabel = tab === "heatmap"
    ? { left: "Low confidence", right: "High confidence" }
    : { left: "Certain",        right: "Uncertain" };

  const dlMap = { mask, heatmap, uncertainty, overlay, contour };

  return (
    <div className="card">
      <div className="img-tabs">
        {tabs.map(t => (
          <button key={t.id} className={`img-tab ${tab === t.id ? "active" : ""}`} onClick={() => setTab(t.id)}>
            {t.label}
          </button>
        ))}
      </div>
      <div className="img-frame">
        <img className="view-img" src={src} alt={tab} />
      </div>
      {showLegend && (
        <div className="hm-legend">
          <span>{legendLabel.left}</span>
          <div className="hm-grad" />
          <span>{legendLabel.right}</span>
        </div>
      )}
      <div className="dl-strip">
        {["mask","heatmap","uncertainty","overlay","contour"].filter(k => dlMap[k]).map(k => (
          <button key={k} className="dl-btn" onClick={() => dlBase64(dlMap[k], `${k}.png`)}>
            ↓ {k.charAt(0).toUpperCase() + k.slice(1)}
          </button>
        ))}
      </div>
    </div>
  );
}

/* ═══════════════════════════ METRICS ═══════════════════════════ */
function MetricsCard({ metrics }) {
  const sev  = sevClass(metrics.severity);
  const conf = Math.round((metrics.mean_confidence ?? 0) * 100);
  const unc  = Math.round((metrics.mean_uncertainty ?? 0) * 100);
  return (
    <div className="card">
      <div className="card-hdr">
        <span className="card-lbl">Wound Metrics</span>
        <span className={`sev ${sev}`}><span className="sev-dot" /> {metrics.severity || "—"}</span>
      </div>
      <div className="card-body">
        <div className="m2">
          <div className="m-tile">
            <span className="m-lbl">Area</span>
            <span className="m-val">{metrics.area_pixels?.toLocaleString() ?? "—"}</span>
            <span className="m-unit">pixels</span>
          </div>
          <div className="m-tile">
            <span className="m-lbl">Coverage</span>
            <span className="m-val">{typeof metrics.area_percentage === "number" ? metrics.area_percentage.toFixed(2) : "—"}</span>
            <span className="m-unit">% of image</span>
          </div>
          <div className="m-tile">
            <span className="m-lbl">Perimeter</span>
            <span className="m-val">{metrics.perimeter ? Math.round(metrics.perimeter).toLocaleString() : "—"}</span>
            <span className="m-unit">px</span>
          </div>
          <div className="m-tile">
            <span className="m-lbl">Circularity</span>
            <span className="m-val">{metrics.circularity ?? "—"}</span>
            <span className="m-unit">0–1 (round)</span>
          </div>
          <div className="m-tile">
            <span className="m-lbl">Convexity</span>
            <span className="m-val">{metrics.convexity ?? "—"}</span>
            <span className="m-unit">0–1</span>
          </div>
          <div className="m-tile">
            <span className="m-lbl">Eccentricity</span>
            <span className="m-val">{metrics.eccentricity ?? "—"}</span>
            <span className="m-unit">0=circle</span>
          </div>
        </div>

        {/* Confidence + Uncertainty */}
        <div style={{ marginTop: 14, padding: "12px 0 0", borderTop: "1px solid var(--border)" }}>
          <div style={{ display:"flex", justifyContent:"space-between", marginBottom:6 }}>
            <span className="m-lbl">Model confidence</span>
            <span className="m-lbl">Uncertainty (±)</span>
          </div>
          <div className="conf-row">
            <div className="conf-wrap"><div className="conf-fill" style={{ width:`${conf}%` }} /></div>
            <span className="conf-pct">{conf}%</span>
          </div>
          <div className="unc-row">
            <div className="unc-bar-wrap"><div className="unc-bar" style={{ width:`${Math.min(unc*4,100)}%` }} /></div>
            <span className="unc-lbl">±{unc}%</span>
          </div>
        </div>

        <div style={{ marginTop:10, fontSize:11, color:"var(--muted)", fontFamily:"'Geist Mono',monospace" }}>
          Shape: {metrics.shape_description ?? "—"}
        </div>
      </div>
    </div>
  );
}

/* ═══════════════════════════ TISSUE ═══════════════════════════ */
function TissueCard({ tissue }) {
  return (
    <div className="card">
      <div className="card-hdr">
        <span className="card-lbl">Tissue Composition</span>
        <span style={{ fontSize:10, color:"var(--muted)", fontFamily:"'Geist Mono',monospace" }}>
          {tissue?.method?.includes("heuristic") ? "HSV heuristic" : "Classifier"}
        </span>
      </div>
      <div className="card-body">
        <TissueDonut tissue={tissue} />
        <div style={{ marginTop:12, fontSize:11, color:"var(--muted)", fontFamily:"'Geist Mono',monospace", lineHeight:1.7 }}>
          Granulation tissue drives healing velocity. Necrotic &gt;30% warrants debridement review.
        </div>
      </div>
    </div>
  );
}

/* ═══════════════════════════ HEALING ═══════════════════════════ */
function HealingCard({ healing }) {
  const conf = Math.round((healing.confidence ?? 0) * 100);
  return (
    <div className="card">
      <div className="card-hdr"><span className="card-lbl">Healing Stage</span></div>
      <div className="card-body">
        <div className="stage-name">{healing.stage ?? "Unknown"}</div>
        <div className="conf-row">
          <div className="conf-wrap"><div className="conf-fill" style={{ width:`${conf}%` }} /></div>
          <span className="conf-pct">{conf}%</span>
        </div>
        <p className="stage-desc" style={{ marginTop:12 }}>{healing.description ?? ""}</p>
        {healing.recommendations?.length > 0 && (
          <div className="recs">
            {healing.recommendations.map((r,i) => (
              <div key={i} className="rec"><span className="rec-arr">▸</span><span>{r}</span></div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

/* ═══════════════════════════ SKIN ═══════════════════════════ */
function SkinCard({ skin }) {
  const lum = skin.luminance ?? 128;
  const rgb = skin.rgb_average ?? null;
  const swatch = rgb ? `rgb(${Math.round(rgb[0])},${Math.round(rgb[1])},${Math.round(rgb[2])})` : "#888";
  return (
    <div className="card">
      <div className="card-hdr"><span className="card-lbl">Skin Analysis · ITA Method</span></div>
      <div className="card-body">
        <div className="skin-row">
          <div className="skin-swatch" style={{ background: swatch }} />
          <div>
            <div className="skin-name">{skin.skin_type ?? "—"}</div>
            <div className="skin-sub">Fitzpatrick Type {skin.fitzpatrick ?? "—"} · {skin.fitz_desc ?? ""}</div>
          </div>
        </div>
        <div className="lum-hdr"><span>Luminance</span><span>{lum.toFixed(0)} / 255</span></div>
        <div className="lum-track"><div className="lum-pip" style={{ left:`${(lum/255)*100}%` }} /></div>
        <div className="ita-badge">
          ITA angle: {skin.ita_angle?.toFixed(1) ?? "—"}° · {skin.method ?? "ITA"}
        </div>
        {rgb && (
          <div className="rgb-row">
            {[["R",rgb[0],"#ff6b6b"],["G",rgb[1],"#51cf66"],["B",rgb[2],"#74c0fc"]].map(([ch,v,col]) => (
              <div key={ch} className="rgb-chip">
                <div className="rgb-lbl" style={{ color:col }}>{ch}</div>
                <div className="rgb-val">{Math.round(v)}</div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

/* ═══════════════════════════ REPORT ═══════════════════════════ */
function ReportCard({ report }) {
  const findings  = report?.clinical_findings ?? {};
  const followUp  = report?.follow_up_care ?? null;
  const tissue    = report?.tissue_composition ?? null;
  return (
    <div className="card">
      <div className="card-hdr"><span className="card-lbl">Clinical Report</span></div>
      <div className="card-body">
        <div className="rpt-id">ID: {report?.report_id ?? "—"} · {report?.generated_at ?? "—"}</div>

        <div className="rpt-sec">Clinical Findings</div>
        {Object.entries(findings).map(([k,v]) => (
          <div key={k} className="rpt-row">
            <span className="rpt-key">{k}</span>
            <span className="rpt-val">{String(v)}</span>
          </div>
        ))}

        {followUp && (
          <>
            <div className="rpt-sec">Follow-up Care</div>
            <div className="follow-box">{followUp}</div>
          </>
        )}

        <div className="rpt-sec">Analysis Info</div>
        <div className="meta">
          Model: SimCLR + U-Net · v{report?.version ?? "3"}<br />
          Input resolution: 128×128 px<br />
          Uncertainty method: MC-Dropout ({`${8} passes`})<br />
          Skin classification: ITA (CIE L*a*b*)<br />
          Threshold: Otsu adaptive
        </div>

        <div className="disclaimer">
          {report?.disclaimer ?? "AI research tool only. Not a substitute for clinical diagnosis."}
        </div>
      </div>
    </div>
  );
}

/* ═══════════════════════════ LOADING ═══════════════════════════ */
function LoadingView({ step }) {
  const steps = [
    "Uploading image…",
    "Running MC-Dropout inference…",
    "Computing uncertainty map…",
    "Classifying tissue types…",
    "Building clinical report…",
  ];
  return (
    <div className="loading-wrap">
      <div className="spin" />
      <div className="load-steps">
        {steps.map((s,i) => (
          <div key={i} className={`lstep ${i===step?"active":i<step?"done":""}`}>
            <span className="ldot" />{s}
          </div>
        ))}
      </div>
    </div>
  );
}

/* ═══════════════════════════ MAIN APP ═══════════════════════════ */
export default function WoundAI() {
  const [file,    setFile]    = useState(null);
  const [preview, setPreview] = useState(null);
  const [dragging,setDragging]= useState(false);
  const [loading, setLoading] = useState(false);
  const [loadStep,setLoadStep]= useState(0);
  const [result,  setResult]  = useState(null);
  const [error,   setError]   = useState(null);
  const fileRef = useRef();

  const API_URL = typeof window !== "undefined" ? window.location.origin : "";

  const handleFile = useCallback(f => {
    if (!f || !f.type.startsWith("image/")) return;
    if (f.size > 8*1024*1024) { setError("File too large. Max 8 MB."); return; }
    setFile(f); setPreview(URL.createObjectURL(f));
    setResult(null); setError(null);
  }, []);

  const onDrop = useCallback(e => {
    e.preventDefault(); setDragging(false); handleFile(e.dataTransfer.files[0]);
  }, [handleFile]);

  const analyze = useCallback(async () => {
    if (!file) return;
    setLoading(true); setError(null); setResult(null);
    const t = setInterval(() => setLoadStep(s => Math.min(s+1,4)), 900);
    try {
      const fd = new FormData(); fd.append("image", file);
      const res  = await fetch(`${API_URL}/analyze`, { method:"POST", body:fd });
      clearInterval(t);
      if (!res.ok) throw new Error(`Server ${res.status}`);
      const data = await res.json();
      if (!data.success) throw new Error(data.error || "Analysis failed");
      setResult(data);
    } catch(e) { clearInterval(t); setError(e.message); }
    finally { setLoading(false); setLoadStep(0); }
  }, [file, API_URL]);

  const reset = () => { setFile(null); setPreview(null); setResult(null); setError(null); };

  return (
    <>
      <style>{css}</style>
      <div className="app">
        {/* topbar */}
        <header className="topbar">
          <div className="logo">
            <div className="logo-mark">W</div>
            <span className="logo-text">WoundAI</span>
          </div>
          <div className="topbar-right">
            {result && <button className="btn-g" onClick={() => window.print()}>↗ Export PDF</button>}
            {result && <button className="btn-g" onClick={reset}>← New Analysis</button>}
            <div className="api-pill"><div className="api-dot" /><span>API Online</span></div>
          </div>
        </header>

        <main className="main">
          {/* upload */}
          {!result && !loading && (
            <>
              <div className="intro">
                <div className="intro-badge"><span style={{width:6,height:6,borderRadius:"50%",background:"var(--accent)",boxShadow:"0 0 6px var(--accent)",display:"inline-block"}} /> SimCLR + U-Net · MC-Dropout Uncertainty · For research use only</div>
                <div className="intro-title">Wound Segmentation<br />& Clinical Analysis</div>
                <p className="intro-sub">Upload a wound image to receive automated segmentation, uncertainty quantification, tissue composition, and a structured clinical report — across all Fitzpatrick skin types using ITA classification.</p>
              </div>
              <div
                className={`upload-zone ${dragging?"drag":""} ${file?"has-file":""}`}
                onClick={() => !file && fileRef.current.click()}
                onDragOver={e => { e.preventDefault(); setDragging(true); }}
                onDragLeave={() => setDragging(false)}
                onDrop={onDrop}
              >
                {!file ? (
                  <>
                    <div className="upload-icon">🖼</div>
                    <div className="upload-title">Drop wound image here</div>
                    <div className="upload-sub">JPEG or PNG · Max 8 MB</div>
                  </>
                ) : (
                  <div className="preview-wrap">
                    <img className="preview-img" src={preview} alt="preview" />
                    <div className="preview-clear" onClick={e => { e.stopPropagation(); reset(); }}>✕</div>
                  </div>
                )}
              </div>
              <input ref={fileRef} type="file" accept="image/*" style={{display:"none"}} onChange={e => handleFile(e.target.files[0])} />
              {error && (
                <div className="err-card">
                  <span style={{fontSize:18}}>⚠️</span>
                  <div><div className="err-title">Error</div><div className="err-msg">{error}</div></div>
                </div>
              )}
              {file && (
                <div className="btn-row">
                  <button className="btn-p" onClick={analyze}>Analyse Wound →</button>
                  <button className="btn-g" onClick={reset}>Clear</button>
                </div>
              )}
            </>
          )}

          {/* loading */}
          {loading && <LoadingView step={loadStep} />}

          {/* results */}
          {result && !loading && (
            <>
              <div className="res-header">
                <div>
                  <div className="res-title">Analysis Complete</div>
                  <div className="res-sub">{result.timestamp} · {result.processing_ms}ms · v{result.version}</div>
                </div>
              </div>
              <div className="results-grid">
                <div className="col">
                  <ImageViewer
                    original={preview}
                    mask={result.mask_image}
                    heatmap={result.heatmap_image}
                    uncertainty={result.uncertainty_image}
                    overlay={result.overlay_image}
                    contour={result.contour_image}
                  />
                  {result.metrics && <MetricsCard metrics={result.metrics} />}
                  {result.tissue_composition && <TissueCard tissue={result.tissue_composition} />}
                </div>
                <div className="col">
                  {result.healing_stage    && <HealingCard healing={result.healing_stage} />}
                  {result.skin_analysis    && <SkinCard    skin={result.skin_analysis} />}
                  {result.doctor_report    && <ReportCard  report={result.doctor_report} />}
                </div>
              </div>
            </>
          )}
        </main>

        <footer className="footer">
          <span>WoundAI</span><span>·</span>
          <span>SimCLR + U-Net</span><span>·</span>
          <span>MC-Dropout Uncertainty</span><span>·</span>
          <span>ITA Skin Classification</span><span>·</span>
          <span>Google Cloud Run</span><span>·</span>
          <span>For research use only</span>
        </footer>
      </div>
    </>
  );
}
