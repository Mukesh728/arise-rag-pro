# Arise.ai Production Deployment Guide & Checklist

This document details the production deployment procedures, security architecture, environment variable reference, container volume configuration, and rollback instructions for **Arise.ai** (MRSPTU StudyRAG).

## 0. Recommended Free Architecture

- **Render Free** runs the existing Dockerized FastAPI backend and serves the existing frontend.
- **Supabase Free** stores PostgreSQL metadata, 384-dimensional pgvector embeddings, document chunks, chat history, and private PDF Storage objects.
- **Gemini integration** remains unchanged and is called only by the backend.
- Set `VECTOR_STORE_TYPE=pgvector`; local Chroma remains available only as a rollback/migration source.

Run `supabase/001_arise_schema.sql` first, verify the tables and private `arise-pdfs` bucket, then deploy the code and set Render variables. Do not delete the old Chroma data or Render service until health, login, upload, query, restart, and chat-history checks pass.

---

## 1. Pre-Deployment Checklist

Before deploying the container to production (Render, Railway, Fly.io, AWS ECS, DigitalOcean, or private VPS):

- [ ] **Generate High-Entropy Session Secret**:
  ```bash
  python -c "import secrets; print(secrets.token_hex(32))"
  ```
  Set this value as the `SESSION_SECRET` environment variable. In production (`ENV=production`), the application strictly halts startup if `SESSION_SECRET` is missing.

- [ ] **Configure Environment Mode**:
  Set `ENV=production`. This activates hardened production defaults automatically across all security layers.

- [ ] **Mount Persistent Volume for SQLite Database (`/data`)**:
  User accounts, bcrypt password hashes, token invalidation versions, and global daily query counters are persisted in SQLite at `/data/users.db`.
  Ensure your cloud provider mounts a persistent volume to `/data`:
  - **Docker CLI**: `-v arise_data:/data`
  - **Railway**: Add a Volume with Mount Path `/data`
  - **Render**: Add a Disk with Mount Path `/data`
  - **Fly.io**: `fly volumes create arise_data --size 1` and map to `/data` in `fly.toml`

- [ ] **Configure Reverse Proxy Trust (`TRUSTED_PROXY_COUNT`)**:
  Set `TRUSTED_PROXY_COUNT` to match the number of reverse proxies in front of your container:
  - **Render / Railway / Fly.io / Cloudflare + Nginx**: Set `TRUSTED_PROXY_COUNT=1`.
  - **Direct Docker exposure (no reverse proxy)**: Set `TRUSTED_PROXY_COUNT=0`.
  > **Security Note**: Setting this too high is unsafe (allows malicious clients to spoof IPs via `X-Forwarded-For`). Setting it too low is safe (only causes users behind the same proxy to share a rate-limit bucket).

- [ ] **Set Google Gemini API Key**:
  Provide a valid `GEMINI_API_KEY` from Google AI Studio. If unset or quota is exhausted, the application gracefully provides grounded citations with fallback passages without crashing.

- [ ] **Bootstrap Admin Account (Optional)**:
  Set `ADMIN_EMAIL="admin@yourdomain.edu"` and `ADMIN_PASSWORD="StrongPassword123!"`. The server automatically provisions this administrator account on first boot with bcrypt-hashed credentials.

---

## 2. Effective Production Defaults Table

When `ENV=production`, the application automatically enforces strict, safe defaults whenever variables are not explicitly overridden:

| Configuration Parameter | Development Default | Production Default (when unset) | Operational Purpose |
| :--- | :--- | :--- | :--- |
| `ENV` | `development` | `production` | Activates production security profile |
| `SESSION_SECRET` | Auto-generated dev key | **Required** (Fatal error if unset) | Cryptographic HMAC-SHA256 cookie signing |
| `COOKIE_SECURE` | `false` | `true` | Enforces `Secure` flag (HTTPS transmission only) |
| `ENABLE_DOCS` | `true` | `false` | Disables `/docs`, `/redoc`, and `/openapi.json` |
| `DAILY_QUERY_LIMIT` | `0` (unlimited) | `20` questions / day | Prevents individual user quota exhaustion |
| `RATE_LIMIT_PER_HOUR` | `0` (disabled) | `60` requests / hour | IP-level DDoS & brute-force defense |
| `GLOBAL_DAILY_QUERY_LIMIT` | `0` (unlimited) | `300` queries / day | Safeguards shared API demo budget |
| `TRUSTED_PROXY_COUNT` | `0` | `1` (in Dockerfile) | Anti-spoofing `X-Forwarded-For` parser |
| Signup Rate Limit | 10 / hour / IP | 10 / hour / IP | Prevents automated spam registrations |
| Query Character Limit | 1000 characters | 1000 characters | Blocks oversized prompt injection payloads |
| `top_k` Clamping | 1 .. 8 | 1 .. 8 | Bounds vector retrieval search overhead |

---

## 3. Container Specification & Health Check

The production container is defined in `Dockerfile`:
- **Base Image**: `python:3.11-slim`
- **Non-Root Execution**: Runs under system user `appuser` (`uid=10001`, `gid=10001`).
- **Container Volume**: `/data` owned by `appuser:appgroup` for `users.db`.
- **Pre-Cached Embeddings**: Chroma's ONNX mini-LM embedding model is pre-downloaded at build time so cold starts do not trigger model downloads.
- **Single Worker Process**: BM25 index lives in process memory; single worker prevents multi-process state desynchronization.
- **Proxy Flag**: `--no-proxy-headers` is used so Uvicorn does not blindly trust spoofable client headers.
- **Health Check Endpoint**:
  ```http
  GET /api/health
  ```
  Returns HTTP 200:
  ```json
  {
    "status": "ok",
    "chunks": 31,
    "total_chunks": 31,
    "version": "1.0.0",
    "auth_enabled": true,
    "app_name": "Arise",
    "vector_store": "chroma"
  }
  ```
  Docker Healthcheck instruction:
  ```dockerfile
  HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
      CMD curl -f http://localhost:${PORT:-8000}/api/health || exit 1
  ```

---

## 4. Security Hardening Summary

1. **Client IP Spoofing Prevention**:
   The shared helper `get_client_ip(request)` reads reverse-proxy appended entries from the right (`ips[-TRUSTED_PROXY_COUNT]`). Malicious client-supplied `X-Forwarded-For` prefixes are ignored.
2. **Session Invalidation on Logout (`token_version`)**:
   Session cookies embed `ver: token_version`. Calling `/api/auth/logout` increments `token_version` in the database, instantly invalidating all existing session tokens.
3. **Security Headers**:
   All responses include:
   - `X-Content-Type-Options: nosniff`
   - `X-Frame-Options: DENY`
   - `Referrer-Policy: strict-origin-when-cross-origin`
   - `/api/auth/*` responses enforce `Cache-Control: no-store, no-cache, must-revalidate, private` and `Pragma: no-cache`.
4. **Log Hygiene**:
   Sensitive payloads (passwords, emails, request bodies) are never logged to stdout or files.

---

## 5. Rollback Procedures

If issues arise after deployment, rollback can be performed cleanly per phase:

### Rollback Phase 2 (Production Container & Health)
To revert container permissions or healthcheck changes:
```bash
git revert 3697522 --no-edit
```

### Rollback Phase 1 (Security Hardening & Limits)
To revert security middleware, IP spoofing defenses, and production defaults:
```bash
git revert fc11106 --no-edit
```

### Revert to Clean Initial Deploy-Prep Branch
To reset the working tree back to the pre-deployment commit:
```bash
git reset --hard 4391a43
```
