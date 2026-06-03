"""JWT-based authentication helpers."""

import os
import jwt
from datetime import datetime, timedelta
from fastapi import HTTPException, Cookie, Header
from typing import Optional

SECRET_KEY  = os.getenv("SECRET_KEY", "company_gpt_secret_key_change_in_production")
ALGORITHM   = "HS256"
EXPIRE_HOURS = 8

ROLE_PERMISSIONS = {
    "admin":  ["upload", "delete", "query", "manage_users", "view_all", "view_restricted"],
    "editor": ["upload", "query", "view_all"],
    "viewer": ["query"],
}

ACCESS_LEVEL_ROLES = {
    "public":     ["viewer", "editor", "admin"],
    "employee":   ["viewer", "editor", "admin"],
    "hr_only":    ["editor", "admin"],
    "restricted": ["admin"],
}


def create_token(username: str, role: str) -> str:
    payload = {
        "sub":  username,
        "role": role,
        "exp":  datetime.utcnow() + timedelta(hours=EXPIRE_HOURS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Session expired. Please login again.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token.")


def get_current_user(token: Optional[str] = Cookie(default=None)):
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    return decode_token(token)


def require_permission(user: dict, permission: str):
    role = user.get("role", "viewer")
    if permission not in ROLE_PERMISSIONS.get(role, []):
        raise HTTPException(
            status_code=403,
            detail=f"Access denied. Your role '{role}' cannot perform '{permission}'.",
        )


def can_access_document(role: str, access_level: str) -> bool:
    allowed_roles = ACCESS_LEVEL_ROLES.get(access_level, ["admin"])
    return role in allowed_roles
