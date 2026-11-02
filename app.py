"""FastAPI service — run: uvicorn app:app --reload  ->  http://localhost:8000/docs"""
import os, shutil, tempfile
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from pipeline import analyze_claim, analyze_text, rag_answer, build_index, get_policy, llm_available, Claim, validate_claim, \
    compute_settlement, fraud_score, rule_decision

app = FastAPI(title="Health Insurance Claims Copilot", version="1.0",
              description="OCR + RAG + multi-agent (LangGraph) claim adjudication assistant. Decision support only; the final decision rests with the claims officer.")
ALLOWED = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
MAX_MB = int(os.getenv("MAX_UPLOAD_MB", "15"))


class Ask(BaseModel):
    question: str


class TextClaim(BaseModel):
    text: str


def _save(f: UploadFile) -> str:
    ext = os.path.splitext(f.filename or "")[1].lower()
    if ext not in ALLOWED:
        raise HTTPException(400, f"Unsupported file type {ext}. Allowed: {sorted(ALLOWED)}")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    shutil.copyfileobj(f.file, tmp); tmp.close()
    if os.path.getsize(tmp.name) > MAX_MB * 1024 * 1024:
        os.unlink(tmp.name); raise HTTPException(413, f"File is larger than {MAX_MB} MB.")
    return tmp.name


@app.get("/health")
def health():
    return {"status": "ok", "mode": "llm" if llm_available() else "offline"}


@app.post("/analyze")
def analyze(file: UploadFile = File(...)):
    """One claim file (claim form + discharge summary + bill, PDF/scan) -> decision, payable, deductions, fraud, letter."""
    path = _save(file)
    try:
        return analyze_claim(path)
    finally:
        os.unlink(path)


@app.post("/analyze-text")
def analyze_text_ep(body: TextClaim):
    """Send already-OCR'd text (TPA portal / email)."""
    return analyze_text(body.text)


@app.post("/adjudicate")
def adjudicate(claim: Claim):
    """Structured claim JSON (from a core system) -> deterministic rules + settlement + fraud only. No LLM call: fast and cheap."""
    issues = [i.model_dump() for i in validate_claim(claim)]
    settle = compute_settlement(claim).model_dump()
    fraud = fraud_score(claim)
    return {"issues": issues, "settlement": settle, "fraud": fraud, "rule_decision": rule_decision(issues, settle, fraud)}


@app.post("/analyze-batch")
def analyze_batch_ep(files: List[UploadFile] = File(...)):
    results = []
    for f in files:
        path = _save(f)
        try:
            results.append({"file": f.filename, **analyze_claim(path)})
        except Exception as e:
            results.append({"file": f.filename, "error": str(e)})
        finally:
            os.unlink(path)
    summary, payable = {}, 0.0
    for r in results:
        d = (r.get("report") or {}).get("decision", "ERROR")
        summary[d] = summary.get(d, 0) + 1
        payable += (r.get("report") or {}).get("payable_amount", 0) or 0
    return {"summary": summary, "total_payable": round(payable, 2), "results": results}


@app.get("/policy/{policy_no:path}")
def policy(policy_no: str):
    p = get_policy(policy_no)
    if not p:
        raise HTTPException(404, "Policy not found")
    return p


@app.post("/ask")
def ask(body: Ask):
    """Questions on policy wording / IRDAI rules (RAG, with sources)."""
    return rag_answer(body.question)


@app.post("/reindex")
def reindex():
    if not llm_available():
        raise HTTPException(400, "Reindexing requires OPENAI_API_KEY (embeddings).")
    build_index()
    return {"status": "reindexed"}
