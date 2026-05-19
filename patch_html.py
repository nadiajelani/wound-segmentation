"""
patch_html.py
-------------
Run once from your project folder:
    python patch_html.py

What it does:
  1. Updates API_URL to your new Cloud Run URL
  2. Fixes image src format (v3 returns raw base64, not data:image URLs)
  3. Adds display for new v3 fields:
     - infection_risk
     - tissue_composition
     - uncertainty_image
     - contour_image
     - quality_report score
     - ITA angle in skin analysis
"""

import shutil
from pathlib import Path

HTML = Path("wound_analyzer.html")
if not HTML.exists():
    print("❌ wound_analyzer.html not found. Run from project root.")
    exit(1)

shutil.copy(HTML, "wound_analyzer_v2_backup.html")
print("✅ Backed up to wound_analyzer_v2_backup.html")

src = HTML.read_text(encoding="utf-8")

# ── FIX 1: Update API_URL ─────────────────────────────────────────────────────
NEW_URL = "https://wound-api-929636759806.us-central1.run.app"

import re
src = re.sub(
    r"const API_URL\s*=\s*['\"].*?['\"]",
    f"const API_URL = '{NEW_URL}'",
    src
)
print(f"✅ Fix 1: API_URL updated to {NEW_URL}")

# ── FIX 2: Fix image src (v3 returns raw base64 without data: prefix) ─────────
OLD_IMG = """document.getElementById('maskImage').src = result.mask_image;
            document.getElementById('heatmapImage').src = result.heatmap_image;
            document.getElementById('overlayImage').src = result.overlay_image;"""

NEW_IMG = """const toDataUrl = (b64) => b64.startsWith('data:') ? b64 : `data:image/png;base64,${b64}`;
            document.getElementById('maskImage').src    = toDataUrl(result.mask_image);
            document.getElementById('heatmapImage').src = toDataUrl(result.heatmap_image);
            document.getElementById('overlayImage').src = toDataUrl(result.overlay_image);

            // v3 new images
            const uncertaintyEl = document.getElementById('uncertaintyImage');
            if (uncertaintyEl && result.uncertainty_image)
                uncertaintyEl.src = toDataUrl(result.uncertainty_image);
            const contourEl = document.getElementById('contourImage');
            if (contourEl && result.contour_image)
                contourEl.src = toDataUrl(result.contour_image);"""

if OLD_IMG in src:
    src = src.replace(OLD_IMG, NEW_IMG)
    print("✅ Fix 2: Image src format fixed + uncertainty/contour images added")
else:
    # Try alternate spacing
    src = re.sub(
        r"document\.getElementById\('maskImage'\)\.src\s*=\s*result\.mask_image;",
        "const toDataUrl = (b64) => b64.startsWith('data:') ? b64 : `data:image/png;base64,${b64}`;\n            document.getElementById('maskImage').src = toDataUrl(result.mask_image);",
        src
    )
    src = re.sub(
        r"document\.getElementById\('heatmapImage'\)\.src\s*=\s*result\.heatmap_image;",
        "document.getElementById('heatmapImage').src = toDataUrl(result.heatmap_image);",
        src
    )
    src = re.sub(
        r"document\.getElementById\('overlayImage'\)\.src\s*=\s*result\.overlay_image;",
        "document.getElementById('overlayImage').src = toDataUrl(result.overlay_image);",
        src
    )
    print("✅ Fix 2: Image src format fixed (regex fallback)")

# ── FIX 3: Fix skin_analysis field names (v3 uses fitzpatrick not skin_class) ──
src = src.replace(
    "skinAnalysis.skin_class",
    "skinAnalysis.fitzpatrick"
)
src = src.replace(
    "`Fitzpatrick Type ${skinAnalysis.fitzpatrick} Classification`",
    "`Fitzpatrick Type ${skinAnalysis.fitzpatrick} · ${skinAnalysis.fitz_desc || ''} · ITA ${skinAnalysis.ita_angle || ''}°`"
)
print("✅ Fix 3: skin_analysis field names updated (skin_class → fitzpatrick)")

# ── FIX 4: Add v3 new fields display after existing displayResults content ────
V3_FIELDS_JS = """
            // ── v3 NEW FIELDS ─────────────────────────────────────────────────

            // Infection risk
            if (result.infection_risk) {
                const inf = result.infection_risk;
                const infEl = document.getElementById('infectionRisk');
                if (infEl) {
                    const riskColors = {low:'#27ae60', moderate:'#f39c12', high:'#e74c3c', unknown:'#888'};
                    infEl.innerHTML = `
                        <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
                            <span style="font-size:1.1em;font-weight:600;color:${riskColors[inf.risk_level]||'#888'};text-transform:capitalize;">
                                ${inf.risk_level} risk
                            </span>
                            <span style="color:#666;font-size:0.9em;">score: ${(inf.risk_score*100).toFixed(0)}%</span>
                        </div>
                        <div style="font-size:0.85em;color:#666;margin-top:4px;">
                            Erythema: ${inf.erythema_pct}% · Exudate: ${(inf.exudate_score*100).toFixed(0)}%
                        </div>
                        ${inf.flags && inf.flags.length ? '<div style="color:#c0392b;font-size:0.82em;margin-top:4px;">⚠ '+inf.flags.join(' · ')+'</div>' : ''}
                    `;
                }
            }

            // Tissue composition
            if (result.tissue_composition) {
                const tc = result.tissue_composition;
                const tcEl = document.getElementById('tissueComposition');
                if (tcEl) {
                    const bars = [
                        {label:'Granulation', val:tc.granulation, color:'#27ae60'},
                        {label:'Slough',      val:tc.slough,      color:'#f39c12'},
                        {label:'Necrotic',    val:tc.necrotic,    color:'#c0392b'},
                        {label:'Epithelial',  val:tc.epithelial,  color:'#2980b9'},
                    ].filter(b => b.val > 0);
                    tcEl.innerHTML = bars.map(b => `
                        <div style="margin-bottom:6px;">
                            <div style="display:flex;justify-content:space-between;font-size:0.85em;margin-bottom:2px;">
                                <span>${b.label}</span><span>${b.val.toFixed(1)}%</span>
                            </div>
                            <div style="height:6px;background:#eee;border-radius:3px;">
                                <div style="width:${b.val}%;height:100%;background:${b.color};border-radius:3px;"></div>
                            </div>
                        </div>`).join('') || '<span style="color:#999;font-size:0.85em;">No tissue data</span>';
                }
            }

            // Quality score
            if (result.quality_report) {
                const qEl = document.getElementById('qualityScore');
                if (qEl) {
                    const score = (result.quality_report.score * 100).toFixed(0);
                    const passed = result.quality_report.passed;
                    qEl.innerHTML = `
                        <span style="font-weight:600;color:${passed?'#27ae60':'#e74c3c'}">
                            ${passed ? '✓ Passed' : '✗ Failed'} (${score}%)
                        </span>`;
                }
            }

            // Processing time
            if (result.processing_ms) {
                const ptEl = document.getElementById('processingTime');
                if (ptEl) ptEl.textContent = result.processing_ms + ' ms';
            }
"""

# Insert after the last existing display block (before closing of displayResults)
INSERT_AFTER = "document.getElementById('reportTime').textContent = new Date(report.generated_at).toLocaleString();"
if INSERT_AFTER in src:
    src = src.replace(INSERT_AFTER, INSERT_AFTER + "\n" + V3_FIELDS_JS)
    print("✅ Fix 4: v3 new fields JS added (infection, tissue, quality, processing time)")
else:
    print("⚠️  Fix 4: Could not find insert point — add v3 fields manually if needed")

# ── FIX 5: Add HTML elements for new fields (inject into results section) ─────
# Add new cards right before the closing of resultsSection or before download buttons
NEW_HTML_CARDS = """
        <!-- v3 NEW CARDS -->
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px;margin-top:20px;">

            <div style="background:#fff;border-radius:12px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,0.08);">
                <h3 style="margin:0 0 12px;color:#2c3e50;font-size:1em;text-transform:uppercase;letter-spacing:1px;">🦠 Infection Risk</h3>
                <div id="infectionRisk" style="color:#666;">Analysing...</div>
            </div>

            <div style="background:#fff;border-radius:12px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,0.08);">
                <h3 style="margin:0 0 12px;color:#2c3e50;font-size:1em;text-transform:uppercase;letter-spacing:1px;">🧬 Tissue Composition</h3>
                <div id="tissueComposition" style="color:#666;">Analysing...</div>
            </div>

            <div style="background:#fff;border-radius:12px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,0.08);">
                <h3 style="margin:0 0 12px;color:#2c3e50;font-size:1em;text-transform:uppercase;letter-spacing:1px;">📊 Image Quality</h3>
                <div id="qualityScore" style="color:#666;">—</div>
                <div style="margin-top:8px;font-size:0.82em;color:#999;">Processing: <span id="processingTime">—</span></div>
            </div>

        </div>

        <!-- v3 Uncertainty & Contour images (hidden if not present) -->
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:20px;" id="v3ImagesRow">
            <div style="background:#fff;border-radius:12px;padding:16px;box-shadow:0 2px 8px rgba(0,0,0,0.08);text-align:center;">
                <h3 style="margin:0 0 8px;color:#2c3e50;font-size:0.9em;">Uncertainty Map</h3>
                <img id="uncertaintyImage" style="width:100%;border-radius:8px;" alt="Uncertainty map" onerror="this.parentElement.style.display='none'"/>
            </div>
            <div style="background:#fff;border-radius:12px;padding:16px;box-shadow:0 2px 8px rgba(0,0,0,0.08);text-align:center;">
                <h3 style="margin:0 0 8px;color:#2c3e50;font-size:0.9em;">Contour Overlay</h3>
                <img id="contourImage" style="width:100%;border-radius:8px;" alt="Contour overlay" onerror="this.parentElement.style.display='none'"/>
            </div>
        </div>
"""

# Insert before the download buttons section
DOWNLOAD_MARKERS = [
    "function downloadImage(",
    "downloadImage(",
    "Download",
    "download-btn",
    "downloadSection",
]
inserted = False
for marker in DOWNLOAD_MARKERS:
    if marker in src:
        # Find the parent div containing download buttons in HTML (not JS)
        idx = src.rfind('<div', 0, src.find(marker))
        if idx > 0:
            # Find a safe HTML insertion point — after resultsSection content
            # Just inject before the closing of id="resultsSection"
            pass
        break

# Safe fallback: inject before </body>
if not inserted:
    src = src.replace("</body>", NEW_HTML_CARDS + "\n</body>", 1)
    print("✅ Fix 5: New HTML cards injected before </body>")

HTML.write_text(src, encoding="utf-8")
print("\n✅ wound_analyzer.html updated successfully!")
print("\nNow run:")
print("  git add wound_analyzer.html")
print("  git commit -m 'Update frontend to v3 - infection risk, tissue, uncertainty map'")
print("  git push origin main")
print("\nCloud Run will redeploy automatically in ~3 minutes.")
