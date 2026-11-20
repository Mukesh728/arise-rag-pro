"""
Tests for Arise.ai Authentication System
========================================
Covers:
1. Registration & Login (Session Cookie)
2. /api/auth/me
3. Unauthenticated /api/query returns 401
4. Authenticated users can ingest personal PDFs
5. Admin /api/ingest also passes auth check
6. Logout clears cookie
7. Login rate limiting (429 after 5 failed attempts)
8. AUTH_ENABLED=false allows anonymous access
"""

import os
import sys
import tempfile
try:
    import pytest
except ImportError:
    class _DummyPytest:
        @staticmethod
        def fixture(*args, **kwargs):
            def decorator(fn):
                return fn
            return decorator
    pytest = _DummyPytest()
from pathlib import Path
from unittest.mock import MagicMock

# Add repository root to pythonpath
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Use isolated temp database for tests
temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
temp_db_path = temp_db.name
temp_db.close()

os.environ["USERS_DB_PATH"] = temp_db_path
os.environ["AUTH_ENABLED"] = "true"
os.environ["SESSION_SECRET"] = "test_super_secret_session_key_12345"
os.environ["ADMIN_TOKEN"] = "test_admin_token_xyz"
os.environ["DAILY_QUERY_LIMIT"] = "5"

from fastapi.testclient import TestClient
import backend.app.main as main_module
from backend.app.main import app
import backend.app.auth as auth_module


import bcrypt

def init_test_env():
    # Initialize test DB
    auth_module.init_db()
    # Fast hashing in test environment
    auth_module.hash_password = lambda pw: bcrypt.hashpw(pw.encode("utf-8"), bcrypt.gensalt(rounds=4)).decode("utf-8")
    # Mock LLM generator to prevent external Gemini API calls
    main_module.llm_generator.generate_answer = MagicMock(return_value="Mocked answer for testing")
    # Mock search service to prevent Chroma/BM25 disk scans during auth tests
    main_module.hybrid_service.search = MagicMock(return_value=[])


@pytest.fixture(scope="session", autouse=True)
def setup_auth_system():
    init_test_env()
    yield
    try:
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)
    except Exception:
        pass


def test_register_and_login_sets_cookie():
    client = TestClient(app)
    
    # Register new user
    reg_res = client.post("/api/auth/register", json={
        "email": "student_test1@mrsptu.ac.in",
        "password": "Password123!"
    })
    assert reg_res.status_code == 200, f"Registration failed: {reg_res.text}"
    data = reg_res.json()
    assert data["email"] == "student_test1@mrsptu.ac.in"
    assert data["role"] == "user"
    assert "arise_session" in reg_res.cookies

    # Login with user
    login_res = client.post("/api/auth/login", json={
        "email": "student_test1@mrsptu.ac.in",
        "password": "Password123!"
    })
    assert login_res.status_code == 200
    assert "arise_session" in login_res.cookies


def test_get_me_authenticated():
    client = TestClient(app)
    # Login to set session
    client.post("/api/auth/login", json={
        "email": "student_test1@mrsptu.ac.in",
        "password": "Password123!"
    })

    me_res = client.get("/api/auth/me")
    assert me_res.status_code == 200
    user = me_res.json()
    assert user["email"] == "student_test1@mrsptu.ac.in"
    assert user["role"] == "user"


def test_unauthenticated_api_query_returns_401():
    client = TestClient(app)
    # No session cookie provided
    res = client.post("/api/query", json={
        "question": "What is an Operating System?",
        "mode": "hybrid",
        "top_k": 3
    })
    assert res.status_code == 401, f"Expected 401, got {res.status_code}"
    assert res.json()["detail"] == "Not authenticated"


def test_authenticated_user_ingest_passes_auth_check():
    client = TestClient(app)
    # Login as student (role=user)
    client.post("/api/auth/login", json={
        "email": "student_test1@mrsptu.ac.in",
        "password": "Password123!"
    })

    # Personal uploads are available to every authenticated user.
    # Missing multipart fields should produce validation 422, not auth 401/403.
    res = client.post("/api/ingest", data={})
    assert res.status_code == 422, f"Expected validation 422, got {res.status_code}: {res.text}"


def test_admin_ingest_succeeds_auth_check():
    client = TestClient(app)
    # Create an admin user directly in DB
    conn = auth_module.get_db_conn()
    admin_id = "admin-uuid-test"
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO users (id, email, password_hash, role, created_at) VALUES (?, ?, ?, 'admin', ?)",
            (admin_id, "admin_user@mrsptu.ac.in", auth_module.hash_password("AdminPass123!"), 0.0)
        )
    conn.close()

    # Login as admin
    client.post("/api/auth/login", json={
        "email": "admin_user@mrsptu.ac.in",
        "password": "AdminPass123!"
    })

    # Call ingest - should pass auth (returns 422 for missing body, NOT 401 or 403)
    res = client.post("/api/ingest", data={})
    assert res.status_code not in (401, 403), f"Admin was rejected with {res.status_code}"

    # Also test non-admin with X-Admin-Token
    client_student = TestClient(app)
    client_student.post("/api/auth/login", json={
        "email": "student_test1@mrsptu.ac.in",
        "password": "Password123!"
    })
    res_token = client_student.post("/api/ingest", headers={"X-Admin-Token": "test_admin_token_xyz"}, data={})
    assert res_token.status_code not in (401, 403), f"Admin token was rejected with {res_token.status_code}"


def test_logout_clears_cookie():
    client = TestClient(app)
    client.post("/api/auth/login", json={
        "email": "student_test1@mrsptu.ac.in",
        "password": "Password123!"
    })

    logout_res = client.post("/api/auth/logout")
    assert logout_res.status_code == 200

    # Next /me call should be 401
    me_res = client.get("/api/auth/me")
    assert me_res.status_code == 401


def test_real_logout_invalidates_old_session():
    client = TestClient(app)
    # Register / login user
    email = "token_revocation@mrsptu.ac.in"
    reg = client.post("/api/auth/register", json={"email": email, "password": "Password123!"})
    cookie_token = reg.cookies.get("arise_session")
    assert cookie_token, "No session cookie received"

    # Verify session works
    me1 = client.get("/api/auth/me", cookies={"arise_session": cookie_token})
    assert me1.status_code == 200

    # Logout to increment token_version
    logout_res = client.post("/api/auth/logout", cookies={"arise_session": cookie_token})
    assert logout_res.status_code == 200

    # Old cookie token MUST now be rejected (401)
    me2 = client.get("/api/auth/me", cookies={"arise_session": cookie_token})
    assert me2.status_code == 401, f"Expected 401 after token_version revocation, got {me2.status_code}"


def test_ip_spoofing_defense_count_0():
    """When TRUSTED_PROXY_COUNT=0, X-Forwarded-For is ignored and client host is used."""
    os.environ["TRUSTED_PROXY_COUNT"] = "0"
    client = TestClient(app)
    bad_email = "spoof_victim_0@mrsptu.ac.in"
    auth_module._login_failures.clear()

    # Attempt 5 logins with changing fake X-Forwarded-For headers
    for i in range(5):
        res = client.post(
            "/api/auth/login",
            json={"email": bad_email, "password": "WrongPassword!"},
            headers={"X-Forwarded-For": f"10.0.0.{i+1}"}
        )
        assert res.status_code == 401

    # 6th attempt with another fake IP must be BLOCKED (429) because both share client.host
    blocked = client.post(
        "/api/auth/login",
        json={"email": bad_email, "password": "WrongPassword!"},
        headers={"X-Forwarded-For": "10.0.0.99"}
    )
    assert blocked.status_code == 429, f"Spoofed IP bypassed rate limit! Status: {blocked.status_code}"
    auth_module._login_failures.clear()


def test_ip_spoofing_defense_count_1():
    """When TRUSTED_PROXY_COUNT=1, rightmost IP is trusted, prepended spoofed IPs are ignored."""
    os.environ["TRUSTED_PROXY_COUNT"] = "1"
    client = TestClient(app)
    bad_email = "spoof_victim_1@mrsptu.ac.in"
    auth_module._login_failures.clear()

    trusted_proxy_ip = "198.51.100.25"

    for i in range(5):
        res = client.post(
            "/api/auth/login",
            json={"email": bad_email, "password": "WrongPassword!"},
            headers={"X-Forwarded-For": f"1.2.3.{i+1}, {trusted_proxy_ip}"}
        )
        assert res.status_code == 401

    # 6th attempt with new prepended fake client IP must still be BLOCKED (429)
    blocked = client.post(
        "/api/auth/login",
        json={"email": bad_email, "password": "WrongPassword!"},
        headers={"X-Forwarded-For": f"9.9.9.9, 8.8.8.8, {trusted_proxy_ip}"}
    )
    assert blocked.status_code == 429, f"Spoofed IP bypassed count=1 rate limit! Status: {blocked.status_code}"
    auth_module._login_failures.clear()
    os.environ["TRUSTED_PROXY_COUNT"] = "0"


def test_global_daily_query_limit():
    """When GLOBAL_DAILY_QUERY_LIMIT is reached, /api/query returns 429 with demo quota message."""
    client = TestClient(app)
    # Login as student
    client.post("/api/auth/login", json={
        "email": "student_test1@mrsptu.ac.in",
        "password": "Password123!"
    })

    os.environ["GLOBAL_DAILY_QUERY_LIMIT"] = "2"
    main_module.settings.GLOBAL_DAILY_QUERY_LIMIT = 2

    # Reset SQLite daily usage table for clean test
    conn = auth_module.get_db_conn()
    with conn:
        conn.execute("DELETE FROM daily_global_usage")
    conn.close()

    res1 = client.post("/api/query", json={"question": "Test query 1", "mode": "hybrid"})
    assert res1.status_code == 200, f"Query 1 failed: {res1.text}"

    res2 = client.post("/api/query", json={"question": "Test query 2", "mode": "hybrid"})
    assert res2.status_code == 200, f"Query 2 failed: {res2.text}"

    # 3rd query exceeds limit of 2
    res3 = client.post("/api/query", json={"question": "Test query 3", "mode": "hybrid"})
    assert res3.status_code == 429
    assert res3.json()["detail"] == "Daily demo quota reached, try tomorrow"

    # Reset
    os.environ.pop("GLOBAL_DAILY_QUERY_LIMIT", None)
    main_module.settings.GLOBAL_DAILY_QUERY_LIMIT = 0


def test_signup_rate_limiting():
    """10 signups per hour per IP allowed, 11th gets 429."""
    client = TestClient(app)
    auth_module._signup_history.clear()

    for i in range(10):
        res = client.post(
            "/api/auth/register",
            json={"email": f"signup_user_{i}@mrsptu.ac.in", "password": "Password123!"}
        )
        assert res.status_code == 200, f"Signup {i+1} failed: {res.text}"

    # 11th signup from same IP
    blocked = client.post(
        "/api/auth/register",
        json={"email": "signup_user_11@mrsptu.ac.in", "password": "Password123!"}
    )
    assert blocked.status_code == 429
    assert "Too many registrations" in blocked.json()["detail"]
    auth_module._signup_history.clear()


def test_query_input_limits():
    """Query > 1000 characters returns 400 or 422, top_k is clamped to 1..8."""
    client = TestClient(app)
    client.post("/api/auth/login", json={
        "email": "student_test1@mrsptu.ac.in",
        "password": "Password123!"
    })

    # Query > 1000 chars
    long_q = "A" * 1005
    res_long = client.post("/api/query", json={"question": long_q, "mode": "hybrid"})
    assert res_long.status_code in (400, 422), f"Expected 400/422 for long query, got {res_long.status_code}"

    # Query with top_k > 8 clamped to 8
    res_k = client.post("/api/query", json={"question": "Short valid query", "top_k": 50})
    assert res_k.status_code == 200


def test_security_headers():
    """Security headers are attached to responses and auth routes have no-store."""
    client = TestClient(app)
    res_auth = client.get("/api/auth/config")
    assert res_auth.headers.get("X-Content-Type-Options") == "nosniff"
    assert res_auth.headers.get("X-Frame-Options") == "DENY"
    assert res_auth.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
    assert "no-store" in res_auth.headers.get("Cache-Control", "")

    res_root = client.get("/")
    assert res_root.headers.get("X-Content-Type-Options") == "nosniff"
    assert res_root.headers.get("X-Frame-Options") == "DENY"
    assert res_root.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"


def test_safe_production_defaults():
    """When ENV=production and variables unset, safe defaults are enforced."""
    import backend.app.config as cfg
    orig_env = os.environ.get("ENV")
    orig_daily = os.environ.get("DAILY_QUERY_LIMIT")
    orig_hourly = os.environ.get("RATE_LIMIT_PER_HOUR")
    orig_docs = os.environ.get("ENABLE_DOCS")
    orig_cookie = os.environ.get("COOKIE_SECURE")
    orig_global = os.environ.get("GLOBAL_DAILY_QUERY_LIMIT")
    try:
        os.environ["ENV"] = "production"
        os.environ.pop("DAILY_QUERY_LIMIT", None)
        os.environ.pop("RATE_LIMIT_PER_HOUR", None)
        os.environ.pop("ENABLE_DOCS", None)
        os.environ.pop("COOKIE_SECURE", None)
        os.environ.pop("GLOBAL_DAILY_QUERY_LIMIT", None)
        prod_settings = cfg.Settings()
        assert prod_settings.DAILY_QUERY_LIMIT == 20
        assert prod_settings.RATE_LIMIT_PER_HOUR == 60
        assert prod_settings.ENABLE_DOCS is False
        assert prod_settings.COOKIE_SECURE is True
        assert prod_settings.GLOBAL_DAILY_QUERY_LIMIT == 300
    finally:
        if orig_env: os.environ["ENV"] = orig_env
        else: os.environ.pop("ENV", None)
        if orig_daily: os.environ["DAILY_QUERY_LIMIT"] = orig_daily
        if orig_hourly: os.environ["RATE_LIMIT_PER_HOUR"] = orig_hourly
        if orig_docs: os.environ["ENABLE_DOCS"] = orig_docs
        if orig_cookie: os.environ["COOKIE_SECURE"] = orig_cookie
        if orig_global: os.environ["GLOBAL_DAILY_QUERY_LIMIT"] = orig_global


def test_health_endpoint():
    """GET /api/health is public, requires no auth, and returns ok status, chunks, version, auth_enabled."""
    client = TestClient(app)
    res = client.get("/api/health")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    data = res.json()
    assert data["status"] == "ok"
    assert "chunks" in data
    assert "total_chunks" in data
    assert "version" in data
    assert "auth_enabled" in data


def test_login_rate_limiting_triggers_after_5_failures():
    client = TestClient(app)
    bad_email = "victim_rate_limit@mrsptu.ac.in"

    auth_module._login_failures.clear()

    # 5 failed attempts with wrong password
    for i in range(5):
        res = client.post("/api/auth/login", json={
            "email": bad_email,
            "password": "WrongPassword!"
        }, headers={"X-Forwarded-For": "192.0.2.1"})
        assert res.status_code == 401, f"Attempt {i+1} expected 401, got {res.status_code}"

    # 6th attempt should be blocked by rate limiter with 429
    blocked_res = client.post("/api/auth/login", json={
        "email": bad_email,
        "password": "WrongPassword!"
    }, headers={"X-Forwarded-For": "192.0.2.1"})
    assert blocked_res.status_code == 429, f"Attempt 6 expected 429, got {blocked_res.status_code}"
    assert "Too many failed login attempts" in blocked_res.json()["detail"]

    auth_module._login_failures.clear()


def test_auth_disabled_allows_anonymous_access():
    os.environ["AUTH_ENABLED"] = "false"
    client = TestClient(app)

    # Anonymous query should succeed without session cookie
    res = client.post("/api/query", json={
        "question": "Explain usability heuristics",
        "mode": "hybrid",
        "top_k": 3
    })
    assert res.status_code == 200, f"Expected 200 with AUTH_ENABLED=false, got {res.status_code}"

    # Reset back to true
    os.environ["AUTH_ENABLED"] = "true"


if __name__ == "__main__":
    init_test_env()
    tests = [
        test_register_and_login_sets_cookie,
        test_get_me_authenticated,
        test_unauthenticated_api_query_returns_401,
        test_authenticated_user_ingest_passes_auth_check,
        test_admin_ingest_succeeds_auth_check,
        test_logout_clears_cookie,
        test_real_logout_invalidates_old_session,
        test_ip_spoofing_defense_count_0,
        test_ip_spoofing_defense_count_1,
        test_global_daily_query_limit,
        test_signup_rate_limiting,
        test_query_input_limits,
        test_security_headers,
        test_safe_production_defaults,
        test_health_endpoint,
        test_login_rate_limiting_triggers_after_5_failures,
        test_auth_disabled_allows_anonymous_access,
    ]
    passed = 0
    for t in tests:
        print(f"Running {t.__name__}...", end=" ", flush=True)
        try:
            t()
            print("PASSED")
            passed += 1
        except Exception as e:
            print(f"FAILED: {e}")
            raise
    print(f"\nAll {passed}/{len(tests)} auth tests passed successfully!")
