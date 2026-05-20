"""
patch_frontend_features.py
---------------------------
Run once from project root:
    python patch_frontend_features.py

Adds to wound_analyzer.html:
  • Grad-CAM tab in image viewer (+ Grad-CAM Overlay tab)
  • Healing Score ring (0-100 with grade)
  • Satellite lesion count badge
  • Edge sharpness + undermining risk in metrics
  • Texture feature mini-table
  • Wound orientation hint
"""

import re, shutil
from pathlib import Path

HTML = Path("wound_analyzer.html")
if not HTML.exists():
    print("❌ wound_analyzer.html not found"); exit(1)

shutil.copy(HTML, "wound_analyzer_before_features.html")
print("✅ Backed up")

src = HTML.read_text()

# ── 1. Add Grad-CAM tabs ──────────────────────────────────────────────────────
OLD_TABS = """<button class="tab" id="tabUncertainty" onclick="showTab('uncertainty',event)" style="display:none">Uncertainty</button>
                        <button class="tab" id="tabContour" onclick="showTab('contour',event)" style="display:none">Contour</button>"""
NEW_TABS = """<button class="tab" id="tabUncertainty" onclick="showTab('uncertainty',event)" style="display:none">Uncertainty</button>
                        <button class="tab" id="tabContour" onclick="showTab('contour',event)" style="display:none">Contour</button>
                        <button class="tab" id="tabGradcam" onclick="showTab('gradcam',event)" style="display:none">Grad-CAM</button>
                        <button class="tab" id="tabGradcamOv" onclick="showTab('gradcam_overlay',event)" style="display:none">Grad-CAM Overlay</button>"""

if OLD_TABS in src:
    src = src.replace(OLD_TABS, NEW_TABS)
    print("✅ Grad-CAM tabs added")
else:
    print("⚠️  Tab insertion point not found — tabs may already exist")

# ── 2. Add Grad-CAM download buttons ─────────────────────────────────────────
OLD_DL = """<button class="dl-btn" id="dlContour" onclick="download('contour')" style="display:none">↓ Contour</button>"""
NEW_DL = """<button class="dl-btn" id="dlContour" onclick="download('contour')" style="display:none">↓ Contour</button>
                            <button class="dl-btn" id="dlGradcam" onclick="download('gradcam')" style="display:none">↓ Grad-CAM</button>
                            <button class="dl-btn" id="dlGradcamOv" onclick="download('gradcam_overlay')" style="display:none">↓ Grad-CAM Ov.</button>"""

if OLD_DL in src:
    src = src.replace(OLD_DL, NEW_DL)
    print("✅ Grad-CAM download buttons added")

# ── 3. Add Healing Score + Satellite cards to right column ────────────────────
HEALING_CARD_MARKER = "<!-- Infection risk card -->"
NEW_CARDS = """<!-- Healing score card -->
                <div class="card" id="healingScoreCard" style="display:none">
                    <div class="card-header">
                        <span class="card-title">Healing Score</span>
                        <span id="healingGradeBadge" class="badge">—</span>
                    </div>
                    <div class="card-body" style="display:flex;align-items:center;gap:20px">
                        <svg width="80" height="80" viewBox="0 0 80 80">
                            <circle cx="40" cy="40" r="32" fill="none" stroke="var(--surface2)" stroke-width="10"/>
                            <circle id="healingScoreRing" cx="40" cy="40" r="32" fill="none"
                                stroke="var(--accent)" stroke-width="10"
                                stroke-dasharray="201" stroke-dashoffset="201"
                                stroke-linecap="round"
                                transform="rotate(-90 40 40)"
                                style="transition:stroke-dashoffset 0.8s ease"/>
                            <text x="40" y="45" text-anchor="middle"
                                font-family="DM Serif Display,serif" font-size="18"
                                fill="var(--text)" id="healingScoreNum">—</text>
                        </svg>
                        <div>
                            <div style="font-size:13px;color:var(--muted);margin-bottom:8px">Composite score from confidence, size, circularity, edge quality and undermining risk.</div>
                            <div id="healingFactors" style="font-family:'DM Mono',monospace;font-size:11px;color:var(--muted);line-height:1.9"></div>
                        </div>
                    </div>
                </div>

                <!-- Satellite lesions card -->
                <div class="card" id="satelliteCard" style="display:none">
                    <div class="card-header">
                        <span class="card-title">Lesion Detection</span>
                        <span id="satelliteBadge" class="badge mild">—</span>
                    </div>
                    <div class="card-body">
                        <div id="satelliteBody" style="font-size:13px;color:var(--muted);line-height:1.7"></div>
                    </div>
                </div>

                <!-- Texture + orientation card -->
                <div class="card" id="textureCard" style="display:none">
                    <div class="card-header"><span class="card-title">Texture & Shape Analysis</span></div>
                    <div class="card-body">
                        <div id="textureBody"></div>
                    </div>
                </div>

"""

if HEALING_CARD_MARKER in src:
    src = src.replace(HEALING_CARD_MARKER, NEW_CARDS + HEALING_CARD_MARKER)
    print("✅ Healing score + satellite + texture cards added")
else:
    print("⚠️  Card insertion point not found")

# ── 4. Add extra metric tiles for undermining + edge sharpness ────────────────
OLD_CIRC_CARD = """                            <div class="metric-card" id="mCircCard" style="display:none">
                                <span class="metric-label">Circularity</span>
                                <span class="metric-value" id="mCirc">—</span>
                                <span class="metric-unit">0=irregular · 1=round</span>
                            </div>"""
NEW_CIRC_CARD = """                            <div class="metric-card" id="mCircCard" style="display:none">
                                <span class="metric-label">Circularity</span>
                                <span class="metric-value" id="mCirc">—</span>
                                <span class="metric-unit">0=irregular · 1=round</span>
                            </div>
                            <div class="metric-card" id="mEdgeCard" style="display:none">
                                <span class="metric-label">Edge sharpness</span>
                                <span class="metric-value" id="mEdge">—</span>
                                <span class="metric-unit" id="mEdgeConf">boundary confidence</span>
                            </div>
                            <div class="metric-card" id="mUmineCard" style="display:none">
                                <span class="metric-label">Undermining risk</span>
                                <span class="metric-value" id="mUmine" style="font-size:18px;margin-top:4px">—</span>
                                <span class="metric-unit" id="mDefects">convexity defects</span>
                            </div>"""

if OLD_CIRC_CARD in src:
    src = src.replace(OLD_CIRC_CARD, NEW_CIRC_CARD)
    print("✅ Edge sharpness + undermining metric tiles added")

# ── 5. Wire new fields in showResults JS ─────────────────────────────────────
OLD_RESET = "        ['healingCard','skinCard','reportCard','infectionCard','tissueCard'].forEach(id =>"
NEW_RESET = "        ['healingCard','skinCard','reportCard','infectionCard','tissueCard','healingScoreCard','satelliteCard','textureCard'].forEach(id =>"
src = src.replace(OLD_RESET, NEW_RESET)

OLD_TAB_IMAGES = "    const tabImages = { original: null, mask: null, heatmap: null, overlay: null, uncertainty: null, contour: null };"
NEW_TAB_IMAGES = "    const tabImages = { original: null, mask: null, heatmap: null, overlay: null, uncertainty: null, contour: null, gradcam: null, gradcam_overlay: null };"
src = src.replace(OLD_TAB_IMAGES, NEW_TAB_IMAGES)

# Add new images to tabImages assignment
OLD_STORE = """            tabImages.uncertainty = data.uncertainty_image || null;
            tabImages.contour     = data.contour_image     || null;"""
NEW_STORE = """            tabImages.uncertainty     = data.uncertainty_image || null;
            tabImages.contour         = data.contour_image     || null;
            tabImages.gradcam         = data.gradcam_image     || null;
            tabImages.gradcam_overlay = data.gradcam_overlay   || null;"""
src = src.replace(OLD_STORE, NEW_STORE)

# Show Grad-CAM tabs/buttons when present
OLD_CONTOUR_SHOW = """        if (data.contour_image) {
            tabImages.contour = data.contour_image;
            document.getElementById('tabContour').style.display = 'inline-flex';
            document.getElementById('dlContour').style.display = 'inline-flex';
        }"""
NEW_CONTOUR_SHOW = """        if (data.contour_image) {
            tabImages.contour = data.contour_image;
            document.getElementById('tabContour').style.display = 'inline-flex';
            document.getElementById('dlContour').style.display = 'inline-flex';
        }
        if (data.gradcam_image) {
            tabImages.gradcam = data.gradcam_image;
            document.getElementById('tabGradcam').style.display = 'inline-flex';
            document.getElementById('dlGradcam').style.display = 'inline-flex';
        }
        if (data.gradcam_overlay) {
            tabImages.gradcam_overlay = data.gradcam_overlay;
            document.getElementById('tabGradcamOv').style.display = 'inline-flex';
            document.getElementById('dlGradcamOv').style.display = 'inline-flex';
        }

        // Healing score
        if (data.healing_score && data.healing_score.score !== undefined) {
            const hs = data.healing_score;
            const circumference = 2 * Math.PI * 32;
            const offset = circumference * (1 - hs.score / 100);
            document.getElementById('healingScoreRing').style.strokeDashoffset = offset;
            document.getElementById('healingScoreNum').textContent = hs.score;
            const gb = document.getElementById('healingGradeBadge');
            gb.textContent = '● ' + hs.grade;
            const gradeClass = {Excellent:'mild', Good:'mild', Fair:'moderate', Poor:'severe', Unknown:'moderate'}[hs.grade] || 'moderate';
            gb.className = 'badge ' + gradeClass;
            if (hs.factors) {
                document.getElementById('healingFactors').innerHTML = Object.entries(hs.factors)
                    .map(([k,v]) => `<div>${k.replace(/_/g,' ')}: <b>${(v*100).toFixed(0)}%</b></div>`).join('');
            }
            document.getElementById('healingScoreCard').style.display = 'block';
        }

        // Satellite lesions
        if (data.satellite_lesions && data.satellite_lesions.count !== undefined) {
            const sl = data.satellite_lesions;
            const sb = document.getElementById('satelliteBadge');
            sb.textContent = sl.count + ' region' + (sl.count !== 1 ? 's' : '');
            sb.className = 'badge ' + (sl.count > 1 ? 'moderate' : 'mild');
            let html = sl.count > 1
                ? `<div style="color:var(--moderate);margin-bottom:8px">⚠ ${sl.count} separate wound regions detected</div>`
                : `<div style="color:var(--mild);margin-bottom:8px">✓ Single wound region</div>`;
            if (sl.regions && sl.regions.length) {
                html += sl.regions.map((r,i) =>
                    `<div style="display:flex;justify-content:space-between;padding:5px 0;border-top:1px solid var(--border);font-size:12px">
                        <span style="color:var(--muted)">${i===0?'Primary':'Satellite '+(i)} · ${r.area_percentage.toFixed(2)}%</span>
                        <span class="badge ${r.severity.toLowerCase()}" style="font-size:10px">● ${r.severity}</span>
                    </div>`
                ).join('');
            }
            document.getElementById('satelliteBody').innerHTML = html;
            document.getElementById('satelliteCard').style.display = 'block';
        }

        // Texture + orientation
        const texRows = [];
        if (data.texture_features && Object.keys(data.texture_features).length) {
            const tf = data.texture_features;
            texRows.push('<div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:var(--accent);margin-bottom:8px">Texture (GLCM)</div>');
            texRows.push(Object.entries(tf).map(([k,v]) =>
                `<div style="display:flex;justify-content:space-between;padding:5px 0;border-top:1px solid var(--border);font-size:12px">
                    <span style="color:var(--muted)">${k}</span>
                    <span style="font-family:'DM Mono',monospace">${typeof v === 'number' ? v.toFixed(3) : v}</span>
                </div>`).join(''));
        }
        if (data.metrics && data.metrics.wound_type_hint) {
            const m = data.metrics;
            texRows.push('<div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:var(--accent);margin:12px 0 8px">Shape</div>');
            if (m.wound_type_hint) texRows.push(`<div style="font-size:13px;color:var(--text);margin-bottom:6px">Type: <b>${m.wound_type_hint}</b></div>`);
            if (m.aspect_ratio)     texRows.push(`<div style="font-size:12px;color:var(--muted)">Aspect ratio: ${m.aspect_ratio}</div>`);
            if (m.orientation_deg !== undefined) texRows.push(`<div style="font-size:12px;color:var(--muted)">Orientation: ${m.orientation_deg}°</div>`);
        }
        if (texRows.length) {
            document.getElementById('textureBody').innerHTML = texRows.join('');
            document.getElementById('textureCard').style.display = 'block';
        }

        // Edge sharpness + undermining in metrics
        if (data.metrics) {
            const m = data.metrics;
            if (m.edge_sharpness !== null && m.edge_sharpness !== undefined) {
                document.getElementById('mEdge').textContent = (m.edge_sharpness * 100).toFixed(0) + '%';
                document.getElementById('mEdgeConf').textContent = m.boundary_confidence + ' confidence';
                document.getElementById('mEdgeCard').style.display = 'flex';
            }
            if (m.undermining_risk) {
                const riskMap = {none:'✓ None', low:'Low', moderate:'Moderate', high:'High'};
                document.getElementById('mUmine').textContent = riskMap[m.undermining_risk] || m.undermining_risk;
                const defCnt = m.defect_count !== null ? m.defect_count + ' defects' : '';
                document.getElementById('mDefects').textContent = defCnt;
                document.getElementById('mUmineCard').style.display = 'flex';
            }
        }"""
src = src.replace(OLD_CONTOUR_SHOW, NEW_CONTOUR_SHOW)
print("✅ All new feature JS wired into showResults")

HTML.write_text(src)
print("\n✅ wound_analyzer.html updated successfully!")
print("\nNow deploy:")
print("  gcloud run deploy wound-api --source . --region us-central1 --memory 4Gi --cpu 2 --timeout 600 --allow-unauthenticated")
