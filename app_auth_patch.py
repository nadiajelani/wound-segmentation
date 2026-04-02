"""
app_auth_patch.py
=================
This file shows the EXACT lines to add/change in your existing app.py.
Search for each ── FIND ── block and apply the ── REPLACE WITH ── change.

You only need to touch 4 places in app.py.
"""

# ══════════════════════════════════════════════════════════════════════════════
# CHANGE 1 — Add import at the top of app.py (after existing imports)
# ══════════════════════════════════════════════════════════════════════════════

# ── FIND (near top of app.py, after flask imports) ──
"""
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
"""

# ── REPLACE WITH ──
"""
from flask import Flask, request, jsonify, send_from_directory, g
from flask_cors import CORS
from auth import require_api_key, rate_limit, RATE_LIMIT_ANALYZE, list_keys
"""


# ══════════════════════════════════════════════════════════════════════════════
# CHANGE 2 — Add admin endpoint (paste this block anywhere after app = Flask(__name__))
# ══════════════════════════════════════════════════════════════════════════════

ADMIN_KEY_SNIPPET = """
# ── Admin: list masked API keys (protect with your own admin key) ────────────
@app.route('/admin/keys', methods=['GET'])
def admin_list_keys():
    admin = request.headers.get('X-Admin-Key', '')
    if admin != os.getenv('ADMIN_KEY', ''):
        return jsonify({"error": "Forbidden"}), 403
    return jsonify({"keys": list_keys(), "count": len(list_keys())})
"""


# ══════════════════════════════════════════════════════════════════════════════
# CHANGE 3 — Protect the /analyze route
# ══════════════════════════════════════════════════════════════════════════════

# ── FIND ──
"""
@app.route('/analyze', methods=['POST'])
def analyze_wound():
"""

# ── REPLACE WITH ──
"""
@app.route('/analyze', methods=['POST'])
@require_api_key
@rate_limit(limit=RATE_LIMIT_ANALYZE)   # default 10 req/min per key
def analyze_wound():
"""


# ══════════════════════════════════════════════════════════════════════════════
# CHANGE 4 — Optionally protect /api/comments with a lighter limit
# ══════════════════════════════════════════════════════════════════════════════

# ── FIND ──
"""
@app.route('/api/comments', methods=['GET', 'POST', 'OPTIONS'])
def handle_comments():
"""

# ── REPLACE WITH ──
"""
@app.route('/api/comments', methods=['GET', 'POST', 'OPTIONS'])
@require_api_key
@rate_limit(limit=30)   # 30 req/min per key
def handle_comments():
"""


# ══════════════════════════════════════════════════════════════════════════════
# RAILWAY ENVIRONMENT VARIABLES  (add these in Railway → Variables)
# ══════════════════════════════════════════════════════════════════════════════
RAILWAY_ENV_VARS = """
# Required
API_KEYS=wai_your_key_1,wai_your_key_2

# Optional tuning
RATE_LIMIT_ANALYZE=10       # requests per minute for /analyze
RATE_LIMIT_DEFAULT=60       # requests per minute for other routes
AUTH_ENABLED=true           # set to false only in local dev

# Admin panel
ADMIN_KEY=some_secret_admin_password
"""
