"""
Arise.ai Authentication Module
==============================
Provides SQLite user store, bcrypt password hashing, signed HttpOnly session cookies,
login rate limiting, admin bootstrapping, and daily per-user query quota guard.
"""

import os
import re
import time
import uuid
import hmac
import hashlib
import base64
import json
import secrets
import sqlite3
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple
from pydantic import BaseModel, EmailStr
from fastapi import APIRouter, Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
import bcrypt

SESSION_COOKIE_NAME = "arise_session"
SESSION_EXPIRY_SECONDS = 7 * 24 * 3600  # 7 days
EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")

# In-memory storage for login failure rate limiting (key: f"{ip}:{email}")
_login_failures: Dict[str, list] = {}

# In-memory storage for daily query count (key: f"{user_id}:{utc_date_str}")
_daily_query_counts: Dict[str, int] = {}


def get_db_path() -> str:
    path = os.getenv("USERS_DB_PATH", "data/users.db")
    abs_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    return abs_path


def get_db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes the SQLite users table."""
    conn = get_db_conn()
    try:
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT UNIQUE COLLATE NOCASE,
                    password_hash TEXT NOT NULL,
                    role TEXT CHECK(role IN ('user', 'admin')) NOT NULL DEFAULT 'user',
                    created_at REAL NOT NULL
                )
            """)
    finally:
        conn.close()


def get_session_secret() -> str:
    secret = os.getenv("SESSION_SECRET")
    env = os.getenv("ENV", os.getenv("ENVIRONMENT", "development")).lower()
    auth_enabled = os.getenv("AUTH_ENABLED", "true").lower() in ("true", "1", "yes")

    if not secret:
        if env == "production" and auth_enabled:
            raise RuntimeError("SESSION_SECRET must be set when ENV=production and AUTH_ENABLED=true")
        if not hasattr(get_session_secret, "_dev_secret"):
            print("[Auth Warning] SESSION_SECRET not set. Generating a random secret for development.")
            get_session_secret._dev_secret = secrets.token_hex(32)
        return get_session_secret._dev_secret
    return secret


def hash_password(password: str) -> str:
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def check_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


def validate_email_password(email: str, password: str) -> Optional[str]:
    if not email or not EMAIL_REGEX.match(email.strip()):
        return "Invalid email address format."
    if not password or len(password) < 8 or len(password) > 128:
        return "Password must be between 8 and 128 characters."
    return None


def sign_session(user_id: str) -> str:
    secret = get_session_secret()
    exp = int(time.time() + SESSION_EXPIRY_SECONDS)
    payload = {"uid": user_id, "exp": exp}
    payload_json = json.dumps(payload, separators=(",", ":"))
    payload_b64 = base64.urlsafe_b64encode(payload_json.encode("utf-8")).decode("utf-8").rstrip("=")
    sig = hmac.new(secret.encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256).digest()
    sig_b64 = base64.urlsafe_b64encode(sig).decode("utf-8").rstrip("=")
    return f"{payload_b64}.{sig_b64}"


def verify_session(cookie_val: str) -> Optional[str]:
    """Verifies session cookie and returns user_id if valid, else None."""
    if not cookie_val or "." not in cookie_val:
        return None
    try:
        payload_b64, sig_b64 = cookie_val.split(".", 1)
        secret = get_session_secret()

        # Re-compute signature
        expected_sig = hmac.new(secret.encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256).digest()

        padding = 4 - (len(sig_b64) % 4)
        if padding != 4:
            sig_b64 += "=" * padding
        actual_sig = base64.urlsafe_b64decode(sig_b64.encode("utf-8"))

        if not hmac.compare_digest(expected_sig, actual_sig):
            return None

        pad_p = 4 - (len(payload_b64) % 4)
        if pad_p != 4:
            payload_b64 += "=" * pad_p
        payload_json = base64.urlsafe_b64decode(payload_b64.encode("utf-8")).decode("utf-8")
        payload = json.loads(payload_json)

        if payload.get("exp", 0) < time.time():
            return None

        return payload.get("uid")
    except Exception:
        return None


def get_user_by_id(user_id: str) -> Optional[dict]:
    conn = get_db_conn()
    try:
        row = conn.execute("SELECT id, email, role, created_at FROM users WHERE id = ?", (user_id,)).fetchone()
        if row:
            return dict(row)
        return None
    finally:
        conn.close()


def get_user_by_email(email: str) -> Optional[dict]:
    conn = get_db_conn()
    try:
        row = conn.execute("SELECT id, email, password_hash, role, created_at FROM users WHERE email = ?", (email.strip().lower(),)).fetchone()
        if row:
            return dict(row)
        return None
    finally:
        conn.close()


def get_current_user(request: Request) -> Optional[dict]:
    cookie = request.cookies.get(SESSION_COOKIE_NAME)
    if not cookie:
        return None
    user_id = verify_session(cookie)
    if not user_id:
        return None
    return get_user_by_id(user_id)


def set_session_cookie(response: Response, token: str):
    secure = os.getenv("COOKIE_SECURE", "false").lower() in ("true", "1", "yes")
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        max_age=SESSION_EXPIRY_SECONDS,
        httponly=True,
        samesite="lax",
        secure=secure,
        path="/"
    )


def clear_session_cookie(response: Response):
    response.delete_cookie(key=SESSION_COOKIE_NAME, path="/")


def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "127.0.0.1"


def record_login_failure(ip: str, email: str):
    now = time.time()
    cutoff = now - 900  # 15 minutes window
    key = f"{ip}:{email.strip().lower()}"
    timestamps = [t for t in _login_failures.get(key, []) if t > cutoff]
    timestamps.append(now)
    _login_failures[key] = timestamps

    # Prune stale keys
    for k in list(_login_failures.keys()):
        _login_failures[k] = [t for t in _login_failures[k] if t > cutoff]
        if not _login_failures[k]:
            _login_failures.pop(k, None)


def is_login_rate_limited(ip: str, email: str) -> bool:
    now = time.time()
    cutoff = now - 900
    key = f"{ip}:{email.strip().lower()}"
    timestamps = [t for t in _login_failures.get(key, []) if t > cutoff]
    return len(timestamps) >= 5


def clear_login_failures(ip: str, email: str):
    key = f"{ip}:{email.strip().lower()}"
    _login_failures.pop(key, None)


def bootstrap_admin():
    """Bootstraps the admin user if ADMIN_EMAIL and ADMIN_PASSWORD are provided and user doesn't exist."""
    admin_email = os.getenv("ADMIN_EMAIL")
    admin_password = os.getenv("ADMIN_PASSWORD")
    if not admin_email or not admin_password:
        return
    admin_email = admin_email.strip().lower()
    conn = get_db_conn()
    try:
        user = conn.execute("SELECT id FROM users WHERE email = ?", (admin_email,)).fetchone()
        if not user:
            user_id = str(uuid.uuid4())
            pw_hash = hash_password(admin_password)
            with conn:
                conn.execute(
                    "INSERT INTO users (id, email, password_hash, role, created_at) VALUES (?, ?, ?, 'admin', ?)",
                    (user_id, admin_email, pw_hash, time.time())
                )
            print(f"[Auth] Bootstrapped admin user '{admin_email}'.")
    finally:
        conn.close()


def check_and_increment_daily_query(user_id: str) -> bool:
    """Returns True if user is allowed to query, False if daily quota exceeded."""
    limit = int(os.getenv("DAILY_QUERY_LIMIT", "0"))
    if limit <= 0:
        return True

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    key = f"{user_id}:{today}"

    # Drop old day entries
    for k in list(_daily_query_counts.keys()):
        if not k.endswith(today):
            _daily_query_counts.pop(k, None)

    current_count = _daily_query_counts.get(key, 0)
    if current_count >= limit:
        return False

    _daily_query_counts[key] = current_count + 1
    return True


# ---------------------------------------------------------
# Router & Endpoints
# ---------------------------------------------------------
auth_router = APIRouter(prefix="/api/auth", tags=["Authentication"])


class AuthPayload(BaseModel):
    email: str
    password: str


@auth_router.post("/register")
async def register(payload: AuthPayload, response: Response):
    allow_signup = os.getenv("ALLOW_SIGNUP", "true").lower() in ("true", "1", "yes")
    if not allow_signup:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User registration is currently disabled."
        )

    val_err = validate_email_password(payload.email, payload.password)
    if val_err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=val_err)

    email = payload.email.strip().lower()
    conn = get_db_conn()
    try:
        existing = conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered."
            )
        user_id = str(uuid.uuid4())
        pw_hash = hash_password(payload.password)
        with conn:
            conn.execute(
                "INSERT INTO users (id, email, password_hash, role, created_at) VALUES (?, ?, ?, 'user', ?)",
                (user_id, email, pw_hash, time.time())
            )
    finally:
        conn.close()

    token = sign_session(user_id)
    set_session_cookie(response, token)
    return {"id": user_id, "email": email, "role": "user"}


@auth_router.post("/login")
async def login(payload: AuthPayload, request: Request, response: Response):
    client_ip = get_client_ip(request)
    email = (payload.email or "").strip().lower()

    if is_login_rate_limited(client_ip, email):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many failed login attempts. Please wait 15 minutes."
        )

    conn = get_db_conn()
    user = None
    try:
        row = conn.execute("SELECT id, email, password_hash, role FROM users WHERE email = ?", (email,)).fetchone()
        if row:
            user = dict(row)
    finally:
        conn.close()

    if not user or not check_password(payload.password, user["password_hash"]):
        record_login_failure(client_ip, email)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    clear_login_failures(client_ip, email)
    token = sign_session(user["id"])
    set_session_cookie(response, token)
    return {"id": user["id"], "email": user["email"], "role": user["role"]}


@auth_router.post("/logout")
async def logout(response: Response):
    clear_session_cookie(response)
    return {"message": "Logged out successfully"}


@auth_router.get("/me")
async def get_me(request: Request):
    auth_enabled = os.getenv("AUTH_ENABLED", "true").lower() in ("true", "1", "yes")
    user = get_current_user(request)
    if not user:
        if not auth_enabled:
            return {"id": "anonymous", "email": "anonymous@arise.ai", "role": "admin"}
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
    return {"id": user["id"], "email": user["email"], "role": user["role"]}


@auth_router.get("/config")
async def get_auth_config():
    auth_enabled = os.getenv("AUTH_ENABLED", "true").lower() in ("true", "1", "yes")
    allow_signup = os.getenv("ALLOW_SIGNUP", "true").lower() in ("true", "1", "yes")
    return {
        "auth_enabled": auth_enabled,
        "allow_signup": allow_signup
    }
