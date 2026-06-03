"""
SQLite database — users, documents, query history.
Data persists across restarts in data/company_gpt.db
"""

import sqlite3
import hashlib
import os
from pathlib import Path
from datetime import datetime

DB_PATH = Path("data/company_gpt.db")
DB_PATH.parent.mkdir(exist_ok=True)


def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Create tables and seed default users."""
    conn = get_conn()
    c = conn.cursor()

    # ── Users ──────────────────────────────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT UNIQUE NOT NULL,
            password    TEXT NOT NULL,
            role        TEXT NOT NULL DEFAULT 'viewer',
            full_name   TEXT,
            email       TEXT,
            created_at  TEXT DEFAULT (datetime('now'))
        )
    """)

    # ── Documents ──────────────────────────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id           TEXT PRIMARY KEY,
            filename     TEXT NOT NULL,
            size_kb      REAL,
            chunks       INTEGER,
            access_level TEXT NOT NULL DEFAULT 'public',
            uploaded_by  TEXT,
            uploaded_at  TEXT DEFAULT (datetime('now')),
            status       TEXT DEFAULT 'ready'
        )
    """)

    # ── Query history ──────────────────────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS query_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT,
            question    TEXT,
            answer      TEXT,
            source      TEXT,
            asked_at    TEXT DEFAULT (datetime('now'))
        )
    """)

    conn.commit()

    # ── Seed default users ─────────────────────────────────────────────────
    defaults = [
        ("admin",   "admin123",   "admin",  "Administrator", "admin@company.com"),
        ("editor",  "editor123",  "editor", "HR Editor",     "editor@company.com"),
        ("viewer",  "viewer123",  "viewer", "John Employee", "viewer@company.com"),
    ]
    for username, password, role, full_name, email in defaults:
        existing = c.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone()
        if not existing:
            c.execute(
                "INSERT INTO users (username,password,role,full_name,email) VALUES (?,?,?,?,?)",
                (username, hash_password(password), role, full_name, email),
            )
    conn.commit()
    conn.close()


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


# ── User helpers ───────────────────────────────────────────────────────────────

def get_user(username: str):
    conn = get_conn()
    row = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    conn.close()
    return dict(row) if row else None


def verify_user(username: str, password: str):
    user = get_user(username)
    if user and user["password"] == hash_password(password):
        return user
    return None


def get_all_users():
    conn = get_conn()
    rows = conn.execute("SELECT id,username,role,full_name,email,created_at FROM users").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def create_user(username, password, role, full_name="", email=""):
    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO users (username,password,role,full_name,email) VALUES (?,?,?,?,?)",
            (username, hash_password(password), role, full_name, email),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def delete_user(username: str):
    conn = get_conn()
    conn.execute("DELETE FROM users WHERE username=?", (username,))
    conn.commit()
    conn.close()


# ── Document helpers ───────────────────────────────────────────────────────────

def save_document(doc_id, filename, size_kb, chunks, access_level, uploaded_by):
    conn = get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO documents
           (id,filename,size_kb,chunks,access_level,uploaded_by)
           VALUES (?,?,?,?,?,?)""",
        (doc_id, filename, size_kb, chunks, access_level, uploaded_by),
    )
    conn.commit()
    conn.close()


def get_all_documents():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM documents ORDER BY uploaded_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_document(doc_id: str):
    conn = get_conn()
    row = conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def delete_document_db(doc_id: str):
    conn = get_conn()
    conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
    conn.commit()
    conn.close()


def get_accessible_docs(role: str):
    """Return documents the given role can access."""
    conn = get_conn()
    if role == "admin":
        rows = conn.execute("SELECT * FROM documents").fetchall()
    elif role == "editor":
        rows = conn.execute(
            "SELECT * FROM documents WHERE access_level IN ('public','employee','hr_only')"
        ).fetchall()
    else:  # viewer
        rows = conn.execute(
            "SELECT * FROM documents WHERE access_level IN ('public','employee')"
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Query history ──────────────────────────────────────────────────────────────

def save_query(username, question, answer, source):
    conn = get_conn()
    conn.execute(
        "INSERT INTO query_history (username,question,answer,source) VALUES (?,?,?,?)",
        (username, question, answer, source),
    )
    conn.commit()
    conn.close()


def get_query_history(username=None, limit=50):
    conn = get_conn()
    if username:
        rows = conn.execute(
            "SELECT * FROM query_history WHERE username=? ORDER BY asked_at DESC LIMIT ?",
            (username, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM query_history ORDER BY asked_at DESC LIMIT ?", (limit,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
