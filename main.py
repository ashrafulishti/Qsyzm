"""
Qsyzm — FastAPI layer
Exposes /encrypt and /decrypt endpoints backed by QsyzmEngine.

Zero-Knowledge contract:
  - Route and Password are request-scoped only; never logged, stored, or echoed.
  - The server only ever sees (and returns) opaque tokens.
  - Failed-attempt rate limiting is tracked per client IP (in-memory).
"""

import logging
import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from engine import QsyzmEngine

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

# Load .env for local development; on Render the vars are injected directly.
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("qsyzm")


# ---------------------------------------------------------------------------
# Rate-limiter  (in-memory, per-IP, tracks failed decrypt attempts only)
# ---------------------------------------------------------------------------

FAIL_LIMIT    = 10        # max consecutive failures before lockout
LOCKOUT_SECS  = 600       # 10 minutes


@dataclass
class _ClientRecord:
    failures:    int   = 0
    locked_until: float = 0.0        # epoch seconds; 0 = not locked


class DecryptRateLimiter:
    """
    Thread-safe-enough for a single-process uvicorn worker.
    For multi-worker deployments replace with Redis-backed storage.
    """

    def __init__(self) -> None:
        self._records: dict[str, _ClientRecord] = defaultdict(_ClientRecord)

    def is_locked(self, ip: str) -> tuple[bool, float]:
        """Return (locked, seconds_remaining)."""
        rec = self._records[ip]
        now = time.monotonic()
        if rec.locked_until > now:
            return True, rec.locked_until - now
        return False, 0.0

    def record_failure(self, ip: str) -> None:
        rec = self._records[ip]
        rec.failures += 1
        if rec.failures >= FAIL_LIMIT:
            rec.locked_until = time.monotonic() + LOCKOUT_SECS
            log.warning("IP %s locked out after %d failed decrypt attempts.", ip, rec.failures)

    def record_success(self, ip: str) -> None:
        """Reset failure counter on a successful decrypt."""
        self._records[ip] = _ClientRecord()

    def purge_expired(self) -> int:
        """Remove stale records to keep memory bounded. Returns count removed."""
        now   = time.monotonic()
        stale = [
            ip for ip, rec in self._records.items()
            if rec.locked_until < now and rec.failures == 0
        ]
        for ip in stale:
            del self._records[ip]
        return len(stale)


limiter = DecryptRateLimiter()


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise the engine once at startup; fail fast if keys are missing."""
    log.info("Qsyzm starting — loading secret keys from environment …")
    try:
        app.state.engine = QsyzmEngine()
        log.info("QsyzmEngine initialised successfully (15 keys loaded).")
    except EnvironmentError as exc:
        log.critical("Key loading failed: %s", exc)
        raise SystemExit(1) from exc
    yield
    log.info("Qsyzm shutting down.")


app = FastAPI(
    title="Qsyzm",
    description="Zero-Knowledge XOR + AES-256-GCM encryption service.",
    version="1.0.0",
    lifespan=lifespan,
    # Never expose schema endpoints in production — remove the comments below
    # docs_url=None,
    # redoc_url=None,
)

# ---------------------------------------------------------------------------
# Static files — serve index.html at the root URL
# ---------------------------------------------------------------------------

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_STATIC_DIR = os.path.join(_BASE_DIR, "static")

# Mount /static for any future CSS/JS/image assets
if os.path.isdir(_STATIC_DIR):
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the Qsyzm single-page frontend."""
    index_path = os.path.join(_BASE_DIR, "index.html")
    if not os.path.exists(index_path):
        return JSONResponse(status_code=404, content={"detail": "Frontend not found."})
    return FileResponse(index_path, media_type="text/html")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _client_ip(request: Request) -> str:
    """
    Resolve the real client IP.
    Render (and most reverse proxies) set X-Forwarded-For.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _engine(request: Request) -> QsyzmEngine:
    return request.app.state.engine


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class EncryptRequest(BaseModel):
    message:  str = Field(..., min_length=1, max_length=65_536,
                          description="Plaintext message to encrypt.")
    route:    str = Field(..., pattern=r"^[1-5]{3}$",
                          description="3-digit route, each digit 1–5 (e.g. '325').")
    password: str = Field(..., min_length=1, max_length=1_024,
                          description="User password for the AES layer.")

    model_config = {"json_schema_extra": {
        "example": {"message": "Hello, Qsyzm!", "route": "325", "password": "s3cr3t"}
    }}


class EncryptResponse(BaseModel):
    token: str = Field(..., description="Base64-encoded ciphertext token.")


class DecryptRequest(BaseModel):
    token:    str = Field(..., min_length=1,
                          description="Token returned by /encrypt.")
    route:    str = Field(..., pattern=r"^[1-5]{3}$",
                          description="Route used during encryption.")
    password: str = Field(..., min_length=1, max_length=1_024,
                          description="Password used during encryption.")

    model_config = {"json_schema_extra": {
        "example": {"token": "<base64>", "route": "325", "password": "s3cr3t"}
    }}


class DecryptResponse(BaseModel):
    message: str = Field(..., description="Recovered plaintext.")


class ErrorResponse(BaseModel):
    detail: str


# ---------------------------------------------------------------------------
# Exception handler — never leak internal error details
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def _global_handler(request: Request, exc: Exception) -> JSONResponse:
    log.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", include_in_schema=False)
async def health():
    """Render health-check probe."""
    return {"status": "ok"}


@app.post(
    "/encrypt",
    response_model=EncryptResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Encrypt a message",
)
async def encrypt(body: EncryptRequest, request: Request):
    """
    Accepts a plaintext **message**, a 3-digit **route**, and a **password**.
    Returns an opaque base64 **token**.

    Neither the route nor the password is stored or logged.
    """
    ip = _client_ip(request)
    log.info("POST /encrypt  ip=%s  route=%s", ip, body.route)

    try:
        token = _engine(request).encrypt(
            body.message,
            route=body.route,
            password=body.password,
        )
    except ValueError as exc:
        # Bad route format (shouldn't reach here due to Pydantic, but be safe)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return EncryptResponse(token=token)


@app.post(
    "/decrypt",
    response_model=DecryptResponse,
    responses={
        400: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Decrypt a token",
)
async def decrypt(body: DecryptRequest, request: Request):
    """
    Accepts a **token**, a 3-digit **route**, and a **password**.
    Returns the original plaintext **message**.

    After **10 consecutive failures** the client IP is locked out for
    **10 minutes**.  A correct decryption resets the failure counter.
    """
    ip = _client_ip(request)
    log.info("POST /decrypt  ip=%s  route=%s", ip, body.route)

    # ── Rate-limit gate ──────────────────────────────────────────────────
    locked, remaining = limiter.is_locked(ip)
    if locked:
        mins, secs = divmod(int(remaining), 60)
        raise HTTPException(
            status_code=429,
            detail=(
                f"Too many failed attempts. "
                f"Try again in {mins}m {secs:02d}s."
            ),
            headers={"Retry-After": str(int(remaining))},
        )

    # ── Decrypt ──────────────────────────────────────────────────────────
    try:
        plaintext = _engine(request).decrypt(
            body.token,
            route=body.route,
            password=body.password,
        )
    except ValueError as exc:
        limiter.record_failure(ip)

        # Check immediately whether this failure triggered a lockout
        locked, remaining = limiter.is_locked(ip)
        if locked:
            log.warning("IP %s now locked out.", ip)
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Too many failed attempts. "
                    f"Locked out for {LOCKOUT_SECS // 60} minutes."
                ),
                headers={"Retry-After": str(LOCKOUT_SECS)},
            ) from exc

        # Generic auth failure — intentionally vague to prevent oracle attacks
        raise HTTPException(
            status_code=400,
            detail="Decryption failed. Check your route, password, and token.",
        ) from exc

    limiter.record_success(ip)
    log.info("POST /decrypt  ip=%s  success", ip)
    return DecryptResponse(message=plaintext)
