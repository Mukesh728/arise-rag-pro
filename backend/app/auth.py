"""
Arise.ai Authentication Module
==============================
Provides SQLite user store, bcrypt password hashing, signed HttpOnly session cookies,
login rate limiting, admin bootstrapping, token invalidation on logout, per-IP signup limiting,
and per-user / global daily query quotas.
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
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple
from pydantic import BaseModel, EmailStr
from fastapi import APIRouter, Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
import bcrypt
from backend.app import supabase_persistence

logger = logging.getLogger("arise.auth")

SESSION_COOKIE_NAME = "arise_session"
SESSION_EXPIRY_SECONDS = 7 * 24 * 3600  # 7 days
EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")

# In-memory storage for login failure rate limiting (key: f"{ip}:{email}")
_login_failures: Dict[str, list] = {}

# In-memory storage for per-IP signup rate limiting (key: ip)
_signup_history: Dict[str, list] = {}

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
    """Initializes the SQLite users table with migration support and global usage table."""
    conn = get_db_conn()
    try:
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT UNIQUE COLLATE NOCASE,
                    password_hash TEXT NOT NULL,
                    role TEXT CHECK(role IN ('user', 'admin')) NOT NULL DEFAULT 'user',
                    token_version INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL
                )
            """)

            # Migrate existing users table if token_version is missing
            cols = [row["name"] for row in conn.execute("PRAGMA table_info(users)").fetchall()]
            if "token_version" not in cols:
                conn.execute("ALTER TABLE users ADD COLUMN token_version INTEGER NOT NULL DEFAULT 0")

            # Persistent table for global daily query tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_global_usage (
                    usage_date TEXT PRIMARY KEY,
                    query_count INTEGER NOT NULL DEFAULT 0
                )
            """)

            # Persistent table for user-uploaded documents
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_documents (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    unit TEXT NOT NULL,
                    total_pages INTEGER NOT NULL DEFAULT 0,
                    total_chunks INTEGER NOT NULL DEFAULT 0,
                    uploaded_at REAL NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_documents_user ON user_documents(user_id)")

            # Auto-backfill existing uploaded documents into admin user accounts if empty or missing
            admins = conn.execute("SELECT id FROM users WHERE role = 'admin'").fetchall()
            sample_docs = [
                ("doc_051a25df", "Human Computer Interaction Notes.pdf", "Human Computer Interaction", "Unit 3", 10, 18, 1730562840.0),
                ("doc_8ffa3f83", "CSE Sem 7 Syllabus.pdf", "CSE Sem 7 Syllabus", "Unit 1", 8, 13, 1729615740.0)
            ]
            for admin_row in admins:
                uid = admin_row["id"]
                for (doc_id, fname, subj, unit, pages, chunks, up_time) in sample_docs:
                    record_id = f"{doc_id}_{uid[:8]}" if uid != "reg-user" else doc_id
                    conn.execute("""
                        INSERT OR IGNORE INTO user_documents (id, user_id, filename, subject, unit, total_pages, total_chunks, uploaded_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (record_id, uid, fname, subj, unit, pages, chunks, up_time))
    finally:
        conn.close()


def restore_user_record(user: Dict[str, Any]) -> None:
    """Restore a durable Supabase user into the local SQLite cache."""
    conn = get_db_conn()
    try:
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO users (id, email, password_hash, role, token_version, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    user["id"], user["email"], user["password_hash"],
                    user.get("role", "user"), int(user.get("token_version", 0)),
                    float(user.get("created_at", time.time())),
                ),
            )
    finally:
        conn.close()


def get_all_users() -> list:
    conn = get_db_conn()
    try:
        rows = conn.execute(
            "SELECT id, email, password_hash, role, token_version, created_at FROM users"
        ).fetchall()
        return [dict(row) for row in rows]
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
            logger.warning("[Auth Warning] SESSION_SECRET not set. Generating a random secret for development.")
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


def sign_session(user_id: str, token_version: int = 0) -> str:
    secret = get_session_secret()
    exp = int(time.time() + SESSION_EXPIRY_SECONDS)
    payload = {"uid": user_id, "ver": token_version, "exp": exp}
    payload_json = json.dumps(payload, separators=(",", ":"))
    payload_b64 = base64.urlsafe_b64encode(payload_json.encode("utf-8")).decode("utf-8").rstrip("=")
    sig = hmac.new(secret.encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256).digest()
    sig_b64 = base64.urlsafe_b64encode(sig).decode("utf-8").rstrip("=")
    return f"{payload_b64}.{sig_b64}"


def verify_session(cookie_val: str) -> Optional[Tuple[str, int]]:
    """Verifies session cookie and returns (user_id, token_version) if valid, else None."""
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

        uid = payload.get("uid")
        if not uid:
            return None
        ver = payload.get("ver", 0)
        return uid, ver
    except Exception:
        return None


def get_user_by_id(user_id: str) -> Optional[dict]:
    conn = get_db_conn()
    try:
        row = conn.execute(
            "SELECT id, email, role, token_version, created_at FROM users WHERE id = ?",
            (user_id,)
        ).fetchone()
        if row:
            return dict(row)
        return None
    finally:
        conn.close()


def get_user_by_email(email: str) -> Optional[dict]:
    conn = get_db_conn()
    try:
        row = conn.execute(
            "SELECT id, email, password_hash, role, token_version, created_at FROM users WHERE email = ?",
            (email.strip().lower(),)
        ).fetchone()
        if row:
            return dict(row)
        return None
    finally:
        conn.close()


def get_current_user(request: Request) -> Optional[dict]:
    cookie = request.cookies.get(SESSION_COOKIE_NAME)
    if not cookie:
        return None
    session_data = verify_session(cookie)
    if not session_data:
        return None
    user_id, token_version = session_data
    user = get_user_by_id(user_id)
    if not user:
        return None
    # Reject sessions whose token_version does not match current DB version
    if user.get("token_version", 0) != token_version:
        return None
    return user


def set_session_cookie(response: Response, token: str):
    env = os.getenv("ENV", os.getenv("ENVIRONMENT", "development")).lower()
    is_prod = env == "production"
    cookie_secure_env = os.getenv("COOKIE_SECURE")
    if cookie_secure_env is None:
        secure = is_prod
    else:
        secure = cookie_secure_env.strip().lower() in ("true", "1", "yes")

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
    """
    Safely resolves client IP based on TRUSTED_PROXY_COUNT.
    If TRUSTED_PROXY_COUNT == 0 (default), uses request.client.host and ignores X-Forwarded-For.
    If TRUSTED_PROXY_COUNT is N > 0 and the header has at least N entries, uses the Nth entry
    from the RIGHT (-N, appended by trusted reverse proxies), never the leftmost entry.
    Otherwise falls back to request.client.host.
    """
    try:
        trusted_proxies = int(os.getenv("TRUSTED_PROXY_COUNT", "0"))
    except Exception:
        trusted_proxies = 0

    if trusted_proxies > 0:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            ips = [ip.strip() for ip in forwarded.split(",") if ip.strip()]
            if len(ips) >= trusted_proxies:
                return ips[-trusted_proxies]

    return request.client.host if (request.client and request.client.host) else "127.0.0.1"


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


def is_signup_rate_limited(ip: str) -> bool:
    now = time.time()
    cutoff = now - 3600  # 1 hour window
    timestamps = [t for t in _signup_history.get(ip, []) if t > cutoff]
    return len(timestamps) >= 10


def record_signup(ip: str):
    now = time.time()
    cutoff = now - 3600
    timestamps = [t for t in _signup_history.get(ip, []) if t > cutoff]
    timestamps.append(now)
    _signup_history[ip] = timestamps

    for k in list(_signup_history.keys()):
        _signup_history[k] = [t for t in _signup_history[k] if t > cutoff]
        if not _signup_history[k]:
            _signup_history.pop(k, None)


def bootstrap_admin():
    """Create or reconcile the configured administrator account."""
    admin_email = os.getenv("ADMIN_EMAIL")
    admin_password = os.getenv("ADMIN_PASSWORD")
    if not admin_email or not admin_password:
        return
    admin_email = admin_email.strip().lower()
    conn = get_db_conn()
    try:
        user = conn.execute(
            "SELECT id, role, password_hash FROM users WHERE email = ?",
            (admin_email,),
        ).fetchone()
        if not user:
            user_id = str(uuid.uuid4())
            pw_hash = hash_password(admin_password)
            with conn:
                conn.execute(
                    "INSERT INTO users (id, email, password_hash, role, token_version, created_at) VALUES (?, ?, ?, 'admin', 0, ?)",
                    (user_id, admin_email, pw_hash, time.time())
                )
            logger.info("[Auth] Bootstrapped initial admin user.")
        else:
            password_changed = not check_password(admin_password, user["password_hash"])
            if user["role"] != "admin" or password_changed:
                updates = ["role = 'admin'"]
                params = []
                if password_changed:
                    updates.append("password_hash = ?")
                    params.append(hash_password(admin_password))
                updates.append("token_version = token_version + 1")
                params.append(user["id"])
                with conn:
                    conn.execute(
                        f"UPDATE users SET {', '.join(updates)} WHERE id = ?",
                        tuple(params),
                    )
                logger.info("[Auth] Reconciled configured administrator account.")
    finally:
        conn.close()


def check_and_increment_daily_query(user_id: str) -> bool:
    """Returns True if user is allowed to query, False if daily per-user quota exceeded."""
    env = os.getenv("ENV", os.getenv("ENVIRONMENT", "development")).lower()
    is_prod = env == "production"
    raw_limit = os.getenv("DAILY_QUERY_LIMIT")
    if raw_limit is None or raw_limit.strip() == "":
        limit = 20 if is_prod else 0
    else:
        try:
            limit = int(raw_limit.strip())
        except ValueError:
            limit = 20 if is_prod else 0

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


def check_and_increment_global_daily_query() -> bool:
    """
    Checks and increments global daily query counter across all users.
    Persisted in users SQLite DB, auto-resets daily.
    """
    from backend.app.config import settings
    env = os.getenv("ENV", os.getenv("ENVIRONMENT", "development")).lower()
    is_prod = env == "production"
    raw_limit = os.getenv("GLOBAL_DAILY_QUERY_LIMIT")
    if raw_limit is not None and raw_limit.strip() != "":
        try:
            limit = int(raw_limit.strip())
        except ValueError:
            limit = 300 if is_prod else 0
    else:
        limit = getattr(settings, "GLOBAL_DAILY_QUERY_LIMIT", (300 if is_prod else 0))

    if limit <= 0:
        return True

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    conn = get_db_conn()
    try:
        with conn:
            row = conn.execute(
                "SELECT query_count FROM daily_global_usage WHERE usage_date = ?",
                (today,)
            ).fetchone()
            current_count = row["query_count"] if row else 0
            if current_count >= limit:
                return False

            conn.execute("""
                INSERT INTO daily_global_usage (usage_date, query_count)
                VALUES (?, 1)
                ON CONFLICT(usage_date) DO UPDATE SET query_count = query_count + 1
            """, (today,))
            return True
    finally:
        conn.close()


# ---------------------------------------------------------
# User Document Store Helpers
# ---------------------------------------------------------
def save_user_document(
    doc_id: str,
    user_id: str,
    filename: str,
    subject: str,
    unit: str,
    total_pages: int = 0,
    total_chunks: int = 0,
) -> Dict[str, Any]:
    now = time.time()
    conn = get_db_conn()
    try:
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO user_documents (id, user_id, filename, subject, unit, total_pages, total_chunks, uploaded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (doc_id, user_id, filename, subject, str(unit), total_pages, total_chunks, now)
            )
        return {
            "id": doc_id,
            "user_id": user_id,
            "filename": filename,
            "subject": subject,
            "unit": str(unit),
            "total_pages": total_pages,
            "total_chunks": total_chunks,
            "uploaded_at": now,
        }
    finally:
        conn.close()


def get_user_documents(user_id: str) -> list:
    conn = get_db_conn()
    try:
        rows = conn.execute(
            """
            SELECT id, user_id, filename, subject, unit, total_pages, total_chunks, uploaded_at
            FROM user_documents
            WHERE user_id = ?
            ORDER BY uploaded_at DESC
            """,
            (user_id,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_document_owner(doc_id: str) -> Optional[str]:
    conn = get_db_conn()
    try:
        row = conn.execute(
            "SELECT user_id FROM user_documents WHERE id = ? OR id LIKE ?",
            (doc_id, f"{doc_id}_%")
        ).fetchone()
        return row["user_id"] if row else None
    finally:
        conn.close()


def delete_user_document(doc_id: str):
    conn = get_db_conn()
    try:
        with conn:
            conn.execute("DELETE FROM user_documents WHERE id = ? OR id LIKE ?", (doc_id, f"{doc_id}_%"))
    finally:
        conn.close()


# ---------------------------------------------------------
# Router & Endpoints
# ---------------------------------------------------------
auth_router = APIRouter(prefix="/api/auth", tags=["Authentication"])


class AuthPayload(BaseModel):
    email: str
    password: str


@auth_router.post("/register")
async def register(payload: AuthPayload, request: Request, response: Response):
    allow_signup = os.getenv("ALLOW_SIGNUP", "true").lower() in ("true", "1", "yes")
    if not allow_signup:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User registration is currently disabled."
        )

    client_ip = get_client_ip(request)
    if is_signup_rate_limited(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many registrations from this IP. Please try again later."
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
                "INSERT INTO users (id, email, password_hash, role, token_version, created_at) VALUES (?, ?, ?, 'user', 0, ?)",
                (user_id, email, pw_hash, time.time())
            )
    finally:
        conn.close()

    try:
        supabase_persistence.upsert_user({
            "id": user_id,
            "email": email,
            "password_hash": pw_hash,
            "role": "user",
            "token_version": 0,
            "created_at": time.time(),
        })
    except Exception as e:
        logger.warning("[Supabase] User mirror failed after registration: %s", e)

    record_signup(client_ip)
    token = sign_session(user_id, 0)
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
        row = conn.execute(
            "SELECT id, email, password_hash, role, token_version FROM users WHERE email = ?",
            (email,)
        ).fetchone()
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
    token = sign_session(user["id"], user.get("token_version", 0))
    set_session_cookie(response, token)
    return {"id": user["id"], "email": user["email"], "role": user["role"]}


@auth_router.post("/logout")
async def logout(request: Request, response: Response):
    user = get_current_user(request)
    if user:
        conn = get_db_conn()
        try:
            with conn:
                conn.execute(
                    "UPDATE users SET token_version = token_version + 1 WHERE id = ?",
                    (user["id"],)
                )
        finally:
            conn.close()
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
