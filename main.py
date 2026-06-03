import os
import uuid
import shutil
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends, Response, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv(override=True)

from database import (
    init_db, verify_user, get_all_users, create_user, delete_user,
    save_document, get_all_documents, get_document, delete_document_db,
    get_accessible_docs, save_query, get_query_history,
)
from auth import create_token, get_current_user, require_permission, can_access_document
from document_processor import DocumentProcessor
from vector_store import VectorStore
from ai_engine import AIEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Init ───────────────────────────────────────────────────────────────────────
init_db()

app = FastAPI(title="Company GPT v2", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

doc_processor = DocumentProcessor()
vector_store  = VectorStore()
ai_engine     = AIEngine()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
ALLOWED_EXT = {".pdf", ".docx", ".txt", ".md", ".csv", ".xlsx"}

ACCESS_LEVELS = ["public", "employee", "hr_only", "restricted"]

# ── Schemas ────────────────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    username: str
    password: str

class QuestionRequest(BaseModel):
    question: str
    top_k: int = 5
    persona: str = "assistant"

class CreateUserRequest(BaseModel):
    username:  str
    password:  str
    role:      str
    full_name: str = ""
    email:     str = ""

# ── Pages ──────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    token = request.cookies.get("token")
    if not token:
        return RedirectResponse("/login")
    try:
        from auth import decode_token
        decode_token(token)
        return HTMLResponse(Path("templates/index.html").read_text(encoding="utf-8"))
    except Exception:
        return RedirectResponse("/login")

@app.get("/login", response_class=HTMLResponse)
async def login_page():
    return HTMLResponse(Path("templates/login.html").read_text(encoding="utf-8"))

# ── Auth endpoints ─────────────────────────────────────────────────────────────
@app.post("/auth/login")
async def login(req: LoginRequest, response: Response):
    user = verify_user(req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    token = create_token(user["username"], user["role"])
    response.set_cookie("token", token, httponly=True, max_age=28800, samesite="lax")
    return {"success": True, "username": user["username"], "role": user["role"], "full_name": user.get("full_name", "")}

@app.post("/auth/logout")
async def logout(response: Response):
    response.delete_cookie("token")
    return {"success": True}

@app.get("/auth/me")
async def me(user=Depends(get_current_user)):
    return user

# ── Personas ───────────────────────────────────────────────────────────────────
@app.get("/personas")
async def get_personas(user=Depends(get_current_user)):
    return {"personas": ai_engine.get_personas()}

# ── Upload ─────────────────────────────────────────────────────────────────────
@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    access_level: str = Form(default="public"),
    user=Depends(get_current_user),
):
    require_permission(user, "upload")

    if access_level not in ACCESS_LEVELS:
        raise HTTPException(status_code=400, detail=f"Invalid access level. Choose from: {ACCESS_LEVELS}")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}'.")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="File is empty.")
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Max 50MB.")

    doc_id    = str(uuid.uuid4())[:8]
    save_path = UPLOAD_DIR / f"{doc_id}_{file.filename}"
    save_path.write_bytes(content)

    text   = doc_processor.extract_text(save_path, file.filename)
    chunks = doc_processor.split_into_chunks(text)

    if not chunks:
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail="No readable text found.")

    vector_store.add_document(doc_id, file.filename, chunks, access_level)
    save_document(doc_id, file.filename, round(len(content)/1024, 2), len(chunks), access_level, user["sub"])

    return {
        "success": True, "doc_id": doc_id, "filename": file.filename,
        "chunks_created": len(chunks), "access_level": access_level,
        "message": f"✅ '{file.filename}' uploaded with {len(chunks)} chunks ({access_level} access).",
    }

# ── Ask ────────────────────────────────────────────────────────────────────────
@app.post("/ask")
async def ask_question(req: QuestionRequest, user=Depends(get_current_user)):
    require_permission(user, "query")
    role     = user.get("role", "viewer")
    question = req.question.strip()
    persona  = req.persona or "assistant"

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if vector_store.total_chunks() == 0:
        raise HTTPException(status_code=404, detail="Knowledge base is empty. Upload documents first.")

    results = vector_store.search(question, role=role, top_k=req.top_k)

    if not results:
        return {
            "question": question,
            "answer":   "I couldn't find relevant information in documents accessible to your role. Please contact your administrator if you need access to restricted documents.",
            "source":   "N/A",
            "confidence": "low",
            "chunks_used": 0,
            "persona": persona,
        }

    context = "\n\n---\n\n".join(f"[Source: {r['filename']}]\n{r['chunk']}" for r in results)
    sources = list(dict.fromkeys(r["filename"] for r in results))
    answer  = ai_engine.generate_answer(question, context, role, persona)

    save_query(user["sub"], question, answer, ", ".join(sources))

    return {
        "question":    question,
        "answer":      answer,
        "source":      ", ".join(sources),
        "confidence":  "high" if len(results) >= 3 else "medium",
        "chunks_used": len(results),
        "role":        role,
        "persona":     persona,
    }

# ── Documents ──────────────────────────────────────────────────────────────────
@app.get("/documents")
async def list_documents(user=Depends(get_current_user)):
    role = user.get("role", "viewer")
    docs = get_accessible_docs(role)
    return {"total": len(docs), "documents": docs, "role": role}

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, user=Depends(get_current_user)):
    require_permission(user, "delete")
    doc = get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    vector_store.delete_document(doc_id)
    delete_document_db(doc_id)
    for f in UPLOAD_DIR.glob(f"{doc_id}_*"):
        f.unlink(missing_ok=True)
    return {"success": True, "message": f"Document deleted."}

# ── Users (admin only) ─────────────────────────────────────────────────────────
@app.get("/users")
async def list_users(user=Depends(get_current_user)):
    require_permission(user, "manage_users")
    return {"users": get_all_users()}

@app.post("/users")
async def add_user(req: CreateUserRequest, user=Depends(get_current_user)):
    require_permission(user, "manage_users")
    if req.role not in ["admin", "editor", "viewer"]:
        raise HTTPException(status_code=400, detail="Role must be admin, editor, or viewer.")
    ok = create_user(req.username, req.password, req.role, req.full_name, req.email)
    if not ok:
        raise HTTPException(status_code=409, detail=f"Username '{req.username}' already exists.")
    return {"success": True, "message": f"User '{req.username}' created."}

@app.delete("/users/{username}")
async def remove_user(username: str, user=Depends(get_current_user)):
    require_permission(user, "manage_users")
    if username == user["sub"]:
        raise HTTPException(status_code=400, detail="Cannot delete yourself.")
    delete_user(username)
    return {"success": True}

# ── History & Stats ────────────────────────────────────────────────────────────
@app.get("/history")
async def history(user=Depends(get_current_user)):
    role = user.get("role", "viewer")
    username = user["sub"] if role != "admin" else None
    return {"history": get_query_history(username)}

@app.get("/stats")
async def stats(user=Depends(get_current_user)):
    docs = get_accessible_docs(user.get("role", "viewer"))
    return {
        "total_documents": len(docs),
        "total_chunks":    vector_store.total_chunks(),
        "role":            user.get("role"),
    }

@app.delete("/reset")
async def reset(user=Depends(get_current_user)):
    require_permission(user, "delete")
    vector_store.reset()
    from database import get_conn
    conn = get_conn()
    conn.execute("DELETE FROM documents")
    conn.commit()
    conn.close()
    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR)
        UPLOAD_DIR.mkdir()
    return {"success": True, "message": "Knowledge base reset."}

@app.get("/health")
async def health():
    return {"status": "ok"}
