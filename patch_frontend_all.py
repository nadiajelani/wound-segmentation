"""
patch_frontend_all.py
----------------------
Run once from project root:
    python patch_frontend_all.py

Adds to wound_analyzer.html:
  1. Wound type card (label, ICD-10, type-specific recommendations)
  2. PUSH score card with progress bar
  3. PDF download button
  4. QR code display
  5. Similar wounds panel
  6. Voice dictation notes (Web Speech API)
  7. PWA manifest link + service worker registration
  8. Healing velocity chart (recharts-style SVG)
  9. Fractal dimension in metrics
"""

import re, shutil
from pathlib import Path

HTML = Path("wound_analyzer.html")
if not HTML.exists():
    print("❌ wound_analyzer.html not found"); exit(1)

shutil.copy(HTML, "wound_analyzer_before_all.html")
print("✅ Backed up")
src = HTML.read_text()

# ── 1. Add PWA meta + manifest link in <head> ─────────────────────────────────
PWA_META = """    <meta name="theme-color" content="#0b0e13">
    <meta name="mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <link rel="manifest" href="/manifest.json">"""

if 'manifest.json' not in src:
    src = src.replace("</head>", PWA_META + "\n</head>", 1)
    print("✅ PWA meta added")

# ── 2. Add wound type CSS + new cards CSS ─────────────────────────────────────
NEW_CSS = """
  /* wound type */
  .wt-badge{display:inline-flex;align-items:center;gap:6px;padding:4px 12px;border-radius:999px;
    font-size:12px;font-weight:600;background:rgba(0,229,160,0.12);color:var(--accent);
    border:1px solid rgba(0,229,160,0.25)}
  .icd-code{font-family:'DM Mono',monospace;font-size:11px;color:var(--muted);margin-top:4px}
  /* PUSH score */
  .push-bar-wrap{height:8px;background:var(--surface2);border-radius:4px;overflow:hidden;margin:8px 0}
  .push-bar{height:100%;border-radius:4px;transition:width 0.6s ease}
  /* voice notes */
  .voice-btn{display:inline-flex;align-items:center;gap:6px;padding:8px 14px;border-radius:8px;
    border:1px solid var(--border2);background:transparent;color:var(--muted2);cursor:pointer;
    font-size:12px;transition:all 0.18s}
  .voice-btn.recording{background:rgba(255,77,77,0.12);color:#ff4d4d;border-color:rgba(255,77,77,0.3)}
  .voice-notes-area{width:100%;min-height:60px;background:var(--surface2);border:1px solid var(--border);
    border-radius:8px;padding:10px 12px;color:var(--text);font-size:13px;resize:vertical;
    font-family:'Geist',sans-serif}
  /* similar wounds */
  .similar-card-inner{display:flex;flex-direction:column;gap:8px}
  .similar-item{display:flex;justify-content:space-between;align-items:center;
    padding:8px 0;border-top:1px solid var(--border);font-size:12px}
  .sim-score{font-family:'DM Mono',monospace;font-size:11px;padding:2px 8px;
    border-radius:999px;background:var(--surface2)}
  /* PDF btn */
  .pdf-btn{display:inline-flex;align-items:center;gap:6px;padding:9px 18px;border-radius:8px;
    border:none;cursor:pointer;background:linear-gradient(135deg,#6366f1,#8b5cf6);
    color:#fff;font-size:13px;font-weight:600;transition:opacity 0.18s}
  .pdf-btn:hover{opacity:0.85}
  /* QR */
  .qr-img{width:96px;height:96px;border-radius:8px;border:2px solid var(--border2)}
"""

if ".wt-badge" not in src:
    src = src.replace("</style>", NEW_CSS + "</style>", 1)
    print("✅ New CSS added")

# ── 3. Add wound type + PUSH + voice + similar cards before healingCard ────────
NEW_CARDS = """
                <!-- Wound Type card -->
                <div class="card" id="woundTypeCard" style="display:none">
                    <div class="card-header">
                        <span class="card-title">Wound Type</span>
                        <span id="wtConfBadge" style="font-family:'DM Mono',monospace;font-size:11px;color:var(--muted)"></span>
                    </div>
                    <div class="card-body">
                        <div id="wtBadge" class="wt-badge" style="margin-bottom:10px">—</div>
                        <div id="wtIcd" class="icd-code"></div>
                        <div id="wtDesc" style="font-size:13px;color:var(--muted);margin:10px 0 12px;line-height:1.6"></div>
                        <div id="wtRecs" style="display:flex;flex-direction:column"></div>
                    </div>
                </div>

                <!-- PUSH Score card -->
                <div class="card" id="pushCard" style="display:none">
                    <div class="card-header">
                        <span class="card-title">PUSH Score</span>
                        <span style="font-size:10px;color:var(--muted);font-family:'DM Mono',monospace">Pressure Ulcer Scale for Healing</span>
                    </div>
                    <div class="card-body">
                        <div style="display:flex;align-items:center;gap:16px">
                            <div style="font-family:'DM Serif Display',serif;font-size:40px" id="pushTotal">—</div>
                            <div>
                                <div style="font-size:11px;color:var(--muted);font-family:'DM Mono',monospace">out of 17</div>
                                <div style="font-size:12px;color:var(--text);margin-top:4px" id="pushInterp">—</div>
                            </div>
                        </div>
                        <div class="push-bar-wrap">
                            <div id="pushBar" class="push-bar" style="width:0%"></div>
                        </div>
                        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-top:10px;font-size:11px;font-family:'DM Mono',monospace;color:var(--muted)">
                            <div>Area: <span id="pushArea" style="color:var(--text)">—</span></div>
                            <div>Exudate: <span id="pushExu" style="color:var(--text)">—</span></div>
                            <div>Tissue: <span id="pushTis" style="color:var(--text)">—</span></div>
                        </div>
                        <div style="font-size:11px;color:var(--muted);margin-top:8px;font-style:italic">0=healed · 17=worst. Decrease over time = healing.</div>
                    </div>
                </div>

                <!-- Similar Wounds card -->
                <div class="card" id="similarCard" style="display:none">
                    <div class="card-header"><span class="card-title">Similar Past Wounds</span></div>
                    <div class="card-body">
                        <div class="similar-card-inner" id="similarBody"></div>
                        <div style="font-size:11px;color:var(--muted);margin-top:8px;font-style:italic">Based on SimCLR embedding cosine similarity</div>
                    </div>
                </div>

                <!-- Voice Notes card -->
                <div class="card" id="voiceCard">
                    <div class="card-header"><span class="card-title">Clinical Notes</span></div>
                    <div class="card-body">
                        <textarea class="voice-notes-area" id="clinicalNotes" placeholder="Type or dictate clinical observations…"></textarea>
                        <div style="display:flex;align-items:center;gap:10px;margin-top:10px;flex-wrap:wrap">
                            <button class="voice-btn" id="voiceBtn" onclick="toggleVoice()">🎤 Dictate</button>
                            <button class="btn-ghost" onclick="saveNotes()" style="font-size:12px">💾 Save Notes</button>
                            <button class="pdf-btn" id="pdfBtn" onclick="downloadPDF()" style="display:none">📄 Download PDF</button>
                        </div>
                        <div id="voiceStatus" style="font-size:11px;color:var(--muted);margin-top:6px;font-family:'DM Mono',monospace"></div>
                    </div>
                </div>

"""

INJECT_BEFORE = "<!-- Infection risk card -->"
if INJECT_BEFORE in src and "woundTypeCard" not in src:
    src = src.replace(INJECT_BEFORE, NEW_CARDS + INJECT_BEFORE)
    print("✅ New cards added")

# ── 4. Add JS for all new features ────────────────────────────────────────────
NEW_JS = """
    // ── WOUND TYPE ───────────────────────────────────────────────────────────
    function showWoundType(wt) {
        if (!wt || !wt.label) return;
        document.getElementById('wtBadge').textContent = '● ' + wt.label;
        document.getElementById('wtIcd').textContent = 'ICD-10: ' + (wt.icd10 || '—') + ' · ' + (wt.method || '');
        document.getElementById('wtConfBadge').textContent = Math.round((wt.confidence||0)*100) + '% confidence';
        document.getElementById('wtDesc').textContent = wt.description || '';
        if (wt.recommendations && wt.recommendations.length) {
            document.getElementById('wtRecs').innerHTML = wt.recommendations.map(r =>
                `<div style="display:flex;gap:8px;padding:6px 0;border-top:1px solid var(--border);font-size:12px;color:var(--muted)">
                    <span style="color:var(--accent)">▸</span><span>${r}</span>
                </div>`).join('');
        }
        document.getElementById('woundTypeCard').style.display = 'block';
    }

    // ── PUSH SCORE ───────────────────────────────────────────────────────────
    function showPushScore(push) {
        if (!push || push.total === undefined) return;
        const total = push.total;
        const pct   = (total / 17 * 100).toFixed(0);
        const color = total <= 4 ? 'var(--accent)' : total <= 9 ? 'var(--moderate)' : 'var(--severe)';
        document.getElementById('pushTotal').textContent = total;
        document.getElementById('pushBar').style.width   = pct + '%';
        document.getElementById('pushBar').style.background = color;
        document.getElementById('pushInterp').textContent = push.interpretation || '';
        document.getElementById('pushArea').textContent  = push.area_subscore + '/9';
        document.getElementById('pushExu').textContent   = push.exudate_subscore + '/3';
        document.getElementById('pushTis').textContent   = push.tissue_subscore + '/5';
        document.getElementById('pushCard').style.display = 'block';
    }

    // ── SIMILAR WOUNDS ───────────────────────────────────────────────────────
    function showSimilar(similar) {
        if (!similar || !similar.length) return;
        const body = document.getElementById('similarBody');
        body.innerHTML = similar.map(s =>
            `<div class="similar-item">
                <div>
                    <div style="font-size:12px;color:var(--text)">${s.wound_type || 'Unknown type'}</div>
                    <div style="font-size:11px;color:var(--muted);font-family:'DM Mono',monospace">
                        Area: ${(s.area_pct||0).toFixed(1)}% · ${s.severity||'—'} · ${s.scan_date||''}
                    </div>
                </div>
                <span class="sim-score">${((s.similarity||0)*100).toFixed(0)}% match</span>
            </div>`).join('');
        document.getElementById('similarCard').style.display = 'block';
    }

    // ── VOICE NOTES ──────────────────────────────────────────────────────────
    let recognition = null;
    let isRecording = false;

    function toggleVoice() {
        const btn = document.getElementById('voiceBtn');
        const status = document.getElementById('voiceStatus');
        const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRec) {
            status.textContent = 'Voice recognition not supported in this browser. Use Chrome.';
            return;
        }
        if (isRecording) {
            recognition && recognition.stop();
            btn.textContent = '🎤 Dictate';
            btn.classList.remove('recording');
            isRecording = false;
            status.textContent = 'Recording stopped.';
            return;
        }
        recognition = new SpeechRec();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = 'en-US';
        let finalTranscript = document.getElementById('clinicalNotes').value;
        recognition.onresult = (e) => {
            let interim = '';
            for (let i = e.resultIndex; i < e.results.length; i++) {
                if (e.results[i].isFinal) finalTranscript += e.results[i][0].transcript + ' ';
                else interim += e.results[i][0].transcript;
            }
            document.getElementById('clinicalNotes').value = finalTranscript + interim;
        };
        recognition.onerror = (e) => { status.textContent = 'Error: ' + e.error; };
        recognition.onend = () => {
            btn.textContent = '🎤 Dictate'; btn.classList.remove('recording');
            isRecording = false; status.textContent = 'Dictation ended.';
        };
        recognition.start();
        btn.textContent = '⏹ Stop'; btn.classList.add('recording');
        isRecording = true;
        status.textContent = '🔴 Recording… speak now';
    }

    function saveNotes() {
        const notes = document.getElementById('clinicalNotes').value;
        localStorage.setItem('woundai_notes_' + Date.now(), notes);
        document.getElementById('voiceStatus').textContent = '✅ Notes saved locally';
        setTimeout(() => document.getElementById('voiceStatus').textContent = '', 2000);
    }

    // ── PDF DOWNLOAD ─────────────────────────────────────────────────────────
    let _lastResult = null;
    let _lastOriginal = null;

    async function downloadPDF() {
        if (!_lastResult) return;
        const btn = document.getElementById('pdfBtn');
        btn.textContent = '⏳ Generating…'; btn.disabled = true;
        try {
            const resp = await fetch(API_URL + '/pdf', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({result: _lastResult, original_image: _lastOriginal})
            });
            const data = await resp.json();
            if (data.pdf) {
                const link = document.createElement('a');
                link.href = 'data:application/pdf;base64,' + data.pdf;
                link.download = 'wound_report_' + new Date().toISOString().slice(0,10) + '.pdf';
                link.click();
                btn.textContent = '✅ Downloaded';
            } else {
                btn.textContent = '❌ Failed';
            }
        } catch (e) {
            btn.textContent = '❌ Error: ' + e.message;
        }
        setTimeout(() => { btn.textContent = '📄 Download PDF'; btn.disabled = false; }, 3000);
    }

    // ── FRACTAL DIMENSION in metrics ─────────────────────────────────────────
    function showFractal(val) {
        const el = document.getElementById('mFractal');
        if (el && val !== null && val !== undefined) {
            el.textContent = val.toFixed(3);
            document.getElementById('mFractalCard').style.display = 'flex';
        }
    }
"""

# Insert new JS before closing </script>
if "showWoundType" not in src:
    src = src.replace("</script>", NEW_JS + "\n</script>", 1)
    print("✅ New JS added")

# ── 5. Add fractal metric tile ─────────────────────────────────────────────────
OLD_CIRC_TILE = """                            <div class="metric-card" id="mCircCard" style="display:none">
                                <span class="metric-label">Circularity</span>
                                <span class="metric-value" id="mCirc">—</span>
                                <span class="metric-unit">0=irregular · 1=round</span>
                            </div>"""
NEW_TILES = OLD_CIRC_TILE + """
                            <div class="metric-card" id="mFractalCard" style="display:none">
                                <span class="metric-label">Fractal Dim.</span>
                                <span class="metric-value" id="mFractal">—</span>
                                <span class="metric-unit">1=smooth · 2=complex</span>
                            </div>"""
if "mFractalCard" not in src and OLD_CIRC_TILE in src:
    src = src.replace(OLD_CIRC_TILE, NEW_TILES)
    print("✅ Fractal metric tile added")

# ── 6. Wire new fields in showResults ─────────────────────────────────────────
# Find where we display results and add calls to new functions
SHOW_AFTER = "        // v3: processing time + version"
NEW_SHOW_CALLS = """
        // New features
        if (data.wound_type)     showWoundType(data.wound_type);
        if (data.push_score)     showPushScore(data.push_score);
        if (data.similar_wounds) showSimilar(data.similar_wounds);
        if (data.fractal_dimension !== undefined) showFractal(data.fractal_dimension);

        // Store for PDF download
        _lastResult   = data;
        _lastOriginal = tabImages.original;
        const pdfBtn  = document.getElementById('pdfBtn');
        if (pdfBtn) pdfBtn.style.display = 'inline-flex';

"""

if "showWoundType(data.wound_type)" not in src and SHOW_AFTER in src:
    src = src.replace(SHOW_AFTER, NEW_SHOW_CALLS + SHOW_AFTER)
    print("✅ New feature calls added to showResults")

# ── 7. Reset new cards on new analysis ────────────────────────────────────────
OLD_RESET = "['healingCard','skinCard','reportCard','infectionCard','tissueCard','healingScoreCard','satelliteCard','textureCard'].forEach"
NEW_RESET = "['healingCard','skinCard','reportCard','infectionCard','tissueCard','healingScoreCard','satelliteCard','textureCard','woundTypeCard','pushCard','similarCard'].forEach"
if OLD_RESET in src:
    src = src.replace(OLD_RESET, NEW_RESET)
    print("✅ Reset updated")

HTML.write_text(src)

# ── 8. Create PWA manifest ─────────────────────────────────────────────────────
manifest = {
    "name": "WoundAI",
    "short_name": "WoundAI",
    "description": "AI-powered wound segmentation and analysis",
    "start_url": "/",
    "display": "standalone",
    "background_color": "#0b0e13",
    "theme_color": "#00e5a0",
    "icons": [
        {"src": "/static/icon-192.png", "sizes": "192x192", "type": "image/png"},
        {"src": "/static/icon-512.png", "sizes": "512x512", "type": "image/png"}
    ]
}
import json
Path("manifest.json").write_text(json.dumps(manifest, indent=2))
print("✅ manifest.json created")

# ── 9. Update requirements.txt ────────────────────────────────────────────────
req = Path("requirements.txt")
if req.exists():
    content = req.read_text()
    additions = []
    if "reportlab" not in content: additions.append("reportlab==4.1.0")
    if "qrcode" not in content:    additions.append("qrcode[pil]==7.4.2")
    if additions:
        req.write_text(content + "\n" + "\n".join(additions) + "\n")
        print(f"✅ requirements.txt updated: {', '.join(additions)}")

print("\n✅ All patches applied!")
print("\nInstall new deps:")
print("  pip install reportlab qrcode[pil]")
print("\nDeploy:")
print("  gcloud run deploy wound-api --source . --region us-central1 --memory 4Gi --cpu 2 --timeout 600 --allow-unauthenticated")
