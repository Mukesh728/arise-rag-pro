"""
Tests for Arise.ai Authentication System
========================================
Covers:
1. Registration & Login (Session Cookie)
2. /api/auth/me
3. Unauthenticated /api/query returns 401
4. Non-admin /api/ingest returns 403
5. Admin /api/ingest succeeds / passes auth check
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


def init_test_env():
    # Initialize test DB
    auth_module.init_db()
    # Mock LLM generator to prevent external Gemini API calls
    main_module.llm_generator.generate_answer = MagicMock(return_value="Mocked answer for testing")


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


def test_non_admin_ingest_returns_403():
    client = TestClient(app)
    # Login as student (role=user)
    client.post("/api/auth/login", json={
        "email": "student_test1@mrsptu.ac.in",
        "password": "Password123!"
    })

    # Attempt ingest without admin role or admin token
    res = client.post("/api/ingest", data={})
    assert res.status_code == 403, f"Expected 403, got {res.status_code}"
    assert "Admin privileges required" in res.json()["detail"]


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
        test_non_admin_ingest_returns_403,
        test_admin_ingest_succeeds_auth_check,
        test_logout_clears_cookie,
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
