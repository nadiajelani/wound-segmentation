"""
auth.py — API Key Authentication + Rate Limiting
Drop this file next to app.py and follow the integration steps below.

Environment variables to set in Railway:
  API_KEYS          Comma-separated list of valid keys
                    e.g. wai_abc123,wai_xyz789
  RATE_LIMIT_ANALYZE   Requests per minute for /analyze  (default 10)
  RATE_LIMIT_DEFAULT   Requests per minute for other routes (default 60)
  AUTH_ENABLED         Set to "false" to disable auth (dev mode, default true)
"""

import os
import time
import hashlib
import secrets
import logging
from functools import wraps
from collections import defaultdict
from threading import Lock
from flask import request, jsonify, g

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

def _load_keys() -> set:
    """Load valid API keys from environment variable."""
    raw = os.getenv("API_KEYS", "").strip()
    if not raw:
        # Auto-generate a default key if none are set (shown in startup logs)
        default = f"wai_{secrets.token_hex(16)}"
        logger.warning(
            f"⚠️  API_KEYS not set. Using auto-generated key (set it in Railway → Variables):\n"
            f"    API_KEYS={default}"
        )
        return {default}
    keys = {k.strip() for k in raw.split(",") if k.strip()}
    logger.info(f"✅ Loaded {len(keys)} API key(s)")
    return keys

VALID_KEYS: set = _load_keys()
AUTH_ENABLED: bool = os.getenv("AUTH_ENABLED", "true").lower() != "false"

RATE_LIMIT_ANALYZE: int = int(os.getenv("RATE_LIMIT_ANALYZE", "10"))   # per minute
RATE_LIMIT_DEFAULT: int = int(os.getenv("RATE_LIMIT_DEFAULT", "60"))   # per minute

# ─────────────────────────────────────────────────────────────────────────────
# In-memory rate limiter  (sliding window per API key)
# ─────────────────────────────────────────────────────────────────────────────

class SlidingWindowRateLimiter:
    """
    Thread-safe sliding-window rate limiter.
    Stores request timestamps per (key, endpoint) bucket.
    """
    def __init__(self):
        self._store: dict[str, list] = defaultdict(list)
        self._lock = Lock()

    def is_allowed(self, bucket: str, limit: int, window: int = 60) -> tuple[bool, dict]:
        """
        Returns (allowed, info_dict).
        info_dict contains: limit, remaining, reset_after (seconds).
        """
        now = time.time()
        cutoff = now - window

        with self._lock:
            # Purge expired timestamps
            self._store[bucket] = [t for t in self._store[bucket] if t > cutoff]
            count = len(self._store[bucket])

            if count >= limit:
                oldest = self._store[bucket][0]
                reset_after = int(oldest + window - now) + 1
                return False, {
                    "limit": limit,
                    "remaining": 0,
                    "reset_after": reset_after,
                }

            self._store[bucket].append(now)
            return True, {
                "limit": limit,
                "remaining": limit - count - 1,
                "reset_after": window,
            }

_limiter = SlidingWindowRateLimiter()

# ─────────────────────────────────────────────────────────────────────────────
# Key hashing (for safe log output)
# ─────────────────────────────────────────────────────────────────────────────

def _mask_key(key: str) -> str:
    """Show first 8 chars only — safe for logs."""
    return key[:8] + "…" if len(key) > 8 else "****"

# ─────────────────────────────────────────────────────────────────────────────
# Decorators
# ─────────────────────────────────────────────────────────────────────────────

def require_api_key(f):
    """
    Decorator: validates X-API-Key header.
    Skipped when AUTH_ENABLED=false (dev mode).
    Attaches g.api_key for downstream use.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        if not AUTH_ENABLED:
            g.api_key = "dev"
            return f(*args, **kwargs)

        key = (
            request.headers.get("X-API-Key")
            or request.args.get("api_key")  # allow ?api_key= in query string
        )

        if not key:
            logger.warning(f"[AUTH] Missing API key — {request.remote_addr} → {request.path}")
            return jsonify({
                "error": "Missing API key",
                "hint": "Include your key in the X-API-Key header or ?api_key= query param",
            }), 401

        if key not in VALID_KEYS:
            logger.warning(f"[AUTH] Invalid key {_mask_key(key)} — {request.remote_addr}")
            return jsonify({"error": "Invalid API key"}), 403

        g.api_key = key
        logger.debug(f"[AUTH] Accepted key {_mask_key(key)}")
        return f(*args, **kwargs)

    return decorated


def rate_limit(limit: int = None, window: int = 60):
    """
    Decorator factory: applies sliding-window rate limit.
    Limit defaults to RATE_LIMIT_DEFAULT unless overridden.

    Usage:
        @rate_limit(limit=10)          # 10 requests / 60s
        @rate_limit(limit=5, window=30) # 5 requests / 30s
    """
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            _limit = limit if limit is not None else RATE_LIMIT_DEFAULT
            # Bucket = api_key (or IP if auth disabled)
            identity = getattr(g, "api_key", None) or request.remote_addr
            bucket = f"{identity}:{f.__name__}"

            allowed, info = _limiter.is_allowed(bucket, _limit, window)

            # Always attach rate-limit headers
            response_headers = {
                "X-RateLimit-Limit": str(info["limit"]),
                "X-RateLimit-Remaining": str(info["remaining"]),
                "X-RateLimit-Reset": str(info["reset_after"]),
            }

            if not allowed:
                logger.warning(
                    f"[RATE] Limit hit — {_mask_key(identity)} on {f.__name__} "
                    f"(reset in {info['reset_after']}s)"
                )
                resp = jsonify({
                    "error": "Rate limit exceeded",
                    "retry_after": info["reset_after"],
                    "limit": info["limit"],
                })
                resp.status_code = 429
                for k, v in response_headers.items():
                    resp.headers[k] = v
                return resp

            result = f(*args, **kwargs)

            # Attach headers to successful response too
            from flask import make_response
            resp = make_response(result)
            for k, v in response_headers.items():
                resp.headers[k] = v
            return resp

        return decorated
    return decorator

# ─────────────────────────────────────────────────────────────────────────────
# Key management utilities
# ─────────────────────────────────────────────────────────────────────────────

def generate_key(prefix: str = "wai") -> str:
    """Generate a new API key. Useful for admin scripts."""
    return f"{prefix}_{secrets.token_hex(20)}"

def add_key(key: str) -> None:
    """Add a key to the in-memory set (not persisted across restarts)."""
    VALID_KEYS.add(key)
    logger.info(f"[AUTH] Added key {_mask_key(key)}")

def revoke_key(key: str) -> bool:
    """Remove a key. Returns True if it existed."""
    existed = key in VALID_KEYS
    VALID_KEYS.discard(key)
    if existed:
        logger.info(f"[AUTH] Revoked key {_mask_key(key)}")
    return existed

def list_keys() -> list:
    """Return masked key list (safe for admin endpoint)."""
    return [_mask_key(k) for k in VALID_KEYS]
