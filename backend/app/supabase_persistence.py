"""Supabase-backed durability for Arise users and uploaded PDF documents."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional



def _base_url() -> str:
    return os.getenv("SUPABASE_URL", "").rstrip("/")


def _service_key() -> str:
    return os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")


def enabled() -> bool:
    return bool(_base_url() and _service_key())


def _request(method: str, path: str, payload: Any = None, content_type: str = "application/json"):
    if not enabled():
        return None
    body = None
    if payload is not None:
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
    headers = {
        "apikey": _service_key(),
        "Authorization": f"Bearer {_service_key()}",
    }
    if path.startswith("/rest/v1/") and method == "POST":
        headers["Prefer"] = "resolution=merge-duplicates,return=minimal"
    if payload is not None:
        headers["Content-Type"] = content_type
    req = urllib.request.Request(f"{_base_url()}{path}", data=body, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=30) as response:
        raw = response.read()
        if not raw:
            return None
        if "json" in (response.headers.get("content-type") or ""):
            return json.loads(raw.decode("utf-8"))
        return raw


def _safe_name(filename: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.basename(filename or "document.pdf"))[:180]


def storage_path(user_id: str, doc_id: str, filename: str) -> str:
    return f"{user_id}/{doc_id}/{_safe_name(filename)}"


def upsert_user(user: Dict[str, Any]) -> None:
    if not enabled():
        return
    _request(
        "POST",
        "/rest/v1/arise_users",
        {
            "id": user["id"],
            "email": user["email"],
            "password_hash": user["password_hash"],
            "role": user.get("role", "user"),
            "created_at": user.get("created_at", 0),
            "token_version": user.get("token_version", 0),
        },
    )


def list_users() -> List[Dict[str, Any]]:
    if not enabled():
        return []
    fields = "id,email,password_hash,role,created_at,token_version"
    result = _request("GET", f"/rest/v1/arise_users?select={fields}&order=created_at.asc")
    return result if isinstance(result, list) else []


def upsert_document(record: Dict[str, Any], file_bytes: bytes) -> str:
    if not enabled():
        return ""
    path = record.get("storage_path") or storage_path(record["user_id"], record["id"], record["filename"])
    bucket = os.getenv("SUPABASE_STORAGE_BUCKET", "arise-pdfs")
    encoded_path = urllib.parse.quote(path, safe="/")
    _request(
        "POST",
        f"/storage/v1/object/{bucket}/{encoded_path}",
        file_bytes,
        content_type="application/pdf",
    )
    _request(
        "POST",
        "/rest/v1/arise_documents",
        {
            "id": record["id"],
            "user_id": record["user_id"],
            "filename": record["filename"],
            "subject": record["subject"],
            "unit": record.get("unit"),
            "total_pages": record.get("total_pages", 0),
            "total_chunks": record.get("total_chunks", 0),
            "storage_path": path,
            "uploaded_at": record.get("uploaded_at", 0),
        },
    )
    return path


def list_documents() -> List[Dict[str, Any]]:
    if not enabled():
        return []
    fields = "id,user_id,filename,subject,unit,total_pages,total_chunks,storage_path,uploaded_at"
    result = _request("GET", f"/rest/v1/arise_documents?select={fields}&order=uploaded_at.asc")
    return result if isinstance(result, list) else []


def download_document(storage_path_value: str) -> bytes:
    bucket = os.getenv("SUPABASE_STORAGE_BUCKET", "arise-pdfs")
    encoded_path = urllib.parse.quote(storage_path_value, safe="/")
    result = _request("GET", f"/storage/v1/object/{bucket}/{encoded_path}")
    return result if isinstance(result, bytes) else b""


def delete_document(record: Dict[str, Any]) -> None:
    if not enabled():
        return
    bucket = os.getenv("SUPABASE_STORAGE_BUCKET", "arise-pdfs")
    path = record.get("storage_path", "")
    if path:
        encoded_path = urllib.parse.quote(path, safe="/")
        try:
            _request("DELETE", f"/storage/v1/object/{bucket}/{encoded_path}")
        except Exception:
            pass
    doc_id = urllib.parse.quote(str(record.get("id", "")), safe="")
    _request("DELETE", f"/rest/v1/arise_documents?id=eq.{doc_id}")
