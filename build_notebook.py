"""Builds insurance_claims_copilot.ipynb from pipeline.py plus explanation/demo cells.
Run: python build_notebook.py            (build only)
     python build_notebook.py --execute  (build + execute in offline mode, saving outputs)"""
import re, sys, os
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent
M, C = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
src = (ROOT / "pipeline.py").read_text(encoding="utf-8")
parts = re.split(r"^# %% (.+)$", src, flags=re.M)[1:]
sections = list(zip(parts[0::2], parts[1::2]))          # [(title, code), ...] in pipeline.py order

HEADERS = [
    "### Config, LLM factory and offline switch\nWhen `llm_available()` is False (no key, or `OFFLINE_MODE=1`), every LLM step falls back to a deterministic implementation, so the demo and tests run without a key.",
    "## 3. Schemas (Pydantic)\n`Claim` (claim form + discharge summary + bill), `BillItem` (bill heads), `Issue`, `Settlement` + `Deduction`.",
    "## 4. OCR\nDigital PDF → pdfplumber text layer. Scanned PDF/image → pdf2image (300 dpi) + **table grid-line removal** (numpy) + Tesseract.",
    "## 5. Structured extraction\nPrimary path: LLM (`with_structured_output(Claim)`). Offline path: label-based regex plus an **OCR-tolerant bill-row parser** with fuzzy matching of bill heads.",
    "## 6. Policy master and reference data\n`data/policies.json` stands in for the policy administration system (members, pre-existing diseases, sum insured, room rent %, co-pay, waiting periods). `non_payable_items.csv` is a subset of IRDAI List I.",
    "## 7. Rule engine (deterministic)\nChecks: policy active, member insured, initial / PED / specific-disease waiting periods, permanent exclusions, 24-hour rule vs day-care, network hospital for cashless, ROHINI ID and doctor registration, intimation and submission timelines, bill arithmetic.",
    "## 8. Settlement calculator\nNon-payable items → room/ICU cap → **proportionate deduction** (doctor, surgery and nursing only; not pharmacy, consumables, implants, diagnostics or ICU) → co-pay → cap at claimed amount / remaining sum insured. **The payable amount is computed only in code.**",
    "## 9. Fraud red-flag scoring\nExplainable, points-based score (early claim, round bill, >80% of sum insured, no ROHINI ID, no doctor registration, bill lines outside the stay, duplicate fingerprint, high per-day cost). A score at or above the threshold routes the claim to `INVESTIGATE` and the SIU agent.",
    "## 10. RAG over policy wording and IRDAI rules\nFAISS + OpenAI embeddings in LLM mode; a keyword TF-IDF retriever offline. Every answer includes its sources.",
    "## 11. Multi-agent system (LangGraph)\nFive agents with **conditional routing**: a HIGH fraud score goes to the Fraud Investigator; a missing policy goes straight to the Adjudicator.\nAdjudicator guards: the payable amount comes from code; the LLM cannot be more lenient than the rule decision; the LLM cannot reject on its own (at most it escalates to QUERY).",
    "## 12. Orchestration\n`analyze_claim(path)`, `analyze_text(text)`, `analyze_batch(paths)`.",
]
assert len(sections) == len(HEADERS), f"pipeline.py has {len(sections)} sections but HEADERS has {len(HEADERS)}"

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
c = nb.cells
c += [M("""# 🏥 Health Insurance Claims Copilot — Capstone
**Problem:** Health insurance claims in India (cashless and reimbursement) are largely adjudicated by hand. A claims officer reads the claim form, discharge summary and itemised bill, applies policy terms (waiting periods, room rent limits, co-pay, exclusions), calculates deductions and looks for fraud. The process is slow under IRDAI timelines (1-hour cashless decision, 3-hour discharge, 30-day settlement), inconsistent across officers, and incorrect deductions lead to grievances and Ombudsman cases.

**Solution:** OCR → LLM extraction → deterministic policy rule engine, settlement calculator and fraud scoring → RAG over policy wording and IRDAI rules → a five-agent LangGraph workflow that returns a decision (APPROVE / PARTIAL_APPROVE / QUERY / INVESTIGATE / REJECT), the payable amount, a deduction sheet and a letter to the claimant.

**Combines four course projects:** Enterprise RAG · Multi-Agent Assistant · OCR · AWS Deployment.

**Contents:** 0 Setup · 1 Architecture · 2 Data · 3 Schemas · 4 OCR · 5 Extraction · 6 Policy master · 7 Rule engine · 8 Settlement · 9 Fraud · 10 RAG · 11 Agents · 12 Orchestration · 13 Demo · 14 Evaluation · 15 API + UI · 16 Deployment · 17 Limitations

> The outputs saved in this notebook were produced in **offline mode** (no API key). Add `OPENAI_API_KEY` to `.env` and re-run to use the LLM extractor, FAISS RAG and LLM adjudicator."""),
      M("## 0. Setup\nSystem packages: `sudo apt install tesseract-ocr poppler-utils` (macOS: `brew install tesseract poppler`).\nPython: `pip install -e \".[all]\"` · `cp .env.example .env` · samples: `python generate_samples.py`."),
      C("import shutil, os\ntry:\n    from dotenv import load_dotenv; load_dotenv()\nexcept ImportError:\n    pass\nfor b in ['tesseract', 'pdftoppm']:\n    print(b.ljust(15), '✅' if shutil.which(b) else '❌ not installed')\nprint('OPENAI_API_KEY'.ljust(15), '✅ LLM mode' if os.getenv('OPENAI_API_KEY') and os.getenv('OFFLINE_MODE') != '1' else '➖ offline mode (fallbacks)')"),
      M("""## 1. Architecture
```
Claim file (PDF/scan: claim form + discharge summary + bill)
   │ OCR (pdfplumber / Tesseract + grid-line removal)
   ▼
[1 Intake Agent] ── LLM → Claim (Pydantic)
   ▼
[2 Validator Agent] ── CODE: policy rules · settlement calc · fraud score · rule decision
   ├── fraud ≥ 50 ──► [3 Fraud Investigator Agent] (SIU checklist) ──┐
   ├── policy not found ───────────────────────────────────────────┐ │
   ▼                                                               │ ▼
[4 Policy Researcher Agent] ── RAG: policy wording + IRDAI ◄──────────┘
   ▼                                                               │
[5 Adjudicator Agent] ◄────────────────────────────────────────────┘
   → decision + payable + deductions + rationale + queries + letter
FastAPI (/analyze /analyze-batch /adjudicate /ask /policy /reindex /health) → Docker → ECR → App Runner
```
**Design decisions**
1. **Money and eligibility are decided by code, not the LLM.** Waiting periods, caps, co-pay and proportionate deduction are deterministic and unit-tested.
2. The LLM is used only to read messy documents, find and explain clauses, and draft letters.
3. **Hard guards:** the LLM cannot make a decision more lenient, and cannot reject on its own (IRDAI requires every repudiation to cite a specific clause and pass a review committee).
4. The fraud score is made of explainable points, so the SIU can see why a claim was flagged; no black-box ML score.
5. Every step has an offline fallback, so a demo never fails because of an API outage."""),
      M("## 2. Data\n- `data/policies.json` — synthetic policy master (4 policies, network hospitals, specific diseases, exclusions)\n- `data/policy_docs/` — RAG knowledge base (starter notes; add real policy wording PDFs and the IRDAI master circular here — see the checklist in `data/README.md`)\n- `data/non_payable_items.csv`, `data/claims_history.json` (duplicate detection)\n- `samples/` — 13 synthetic claim files + `expected.json`"),
      C("import json\nfrom pathlib import Path\nprint('policy docs:', [d.name for d in Path('data/policy_docs').glob('*')])\npol = json.load(open('data/policies.json'))\nprint('policies   :', list(pol['policies']))\nprint('samples    :', len(list(Path('samples').glob('*.pdf'))), 'PDFs')")]

for (title, code), hdr in zip(sections, HEADERS):
    code = code.strip().replace('if __name__ == "__main__":', "if False:  # CLI entry point, skipped in the notebook")
    c += [M(hdr), C(code)]
    if title.startswith("OCR"):
        c += [M("**Demo:** OCR text from the bill page of a scanned (image-only) claim."),
              C("t = ocr_document('samples/13_scanned_clean_approve.pdf')\nprint(t[t.find('FINAL'):][:700])")]
    if title.startswith("Extraction"):
        c += [C("claim = extract_claim(ocr_document('samples/02_room_rent_proportionate.pdf'))\nimport pandas as pd\nprint(claim.policy_no, '|', claim.patient_name, '|', claim.diagnosis, claim.icd10_code)\npd.DataFrame([b.model_dump() for b in claim.bill_items])")]
    if title.startswith("Deterministic"):
        c += [M("**Offline demo (no API key):** a senior citizen admitted for a hypertension-related condition (declared pre-existing disease, 36-month waiting period)."),
              C("demo = Claim(policy_no='HLT/2025/000377', patient_name='Venkat Rao', admission_date='2026-08-02', discharge_date='2026-08-06',\n"
                "             diagnosis='Hypertensive heart disease', icd10_code='I11.0', hospital_name='Charminar Heart Institute',\n"
                "             bill_items=[BillItem(head='icu', description='ICU', quantity=2, rate=9000, amount=18000)], total_billed=18000)\n"
                "for x in validate_claim(demo): print(x.severity.upper().ljust(8), x.code.ljust(26), x.message)")]
    if title.startswith("Settlement"):
        c += [M("**Walkthrough:** a deluxe suite at ₹9,000/day vs the eligible ₹5,000/day (1% of a ₹5 lakh sum insured)."),
              C("s = compute_settlement(claim)\nprint(f'Billed ₹{s.total_billed:,.2f}  →  Payable ₹{s.payable:,.2f}')\npd.DataFrame([d.model_dump() for d in s.deductions])")]
    if title.startswith("Fraud"):
        c += [C("fc = extract_claim(ocr_document('samples/09_fraud_red_flags_investigate.pdf'))\nf = fraud_score(fc)\nprint('score', f['score'], f['band'])\npd.DataFrame(f['flags'])")]
    if title.startswith("RAG"):
        c += [M("In LLM mode, build the index once (and again whenever the documents change) with `build_index()`. Offline mode uses the keyword retriever."),
              C("r = rag_answer('Is cataract surgery covered in the first year of the policy?')\nprint(r['answer'][:600]); print(r['sources'])")]
    if title.startswith("Multi-Agent"):
        c += [M("Graph structure in Mermaid syntax (paste into mermaid.live to view the diagram):"),
              C("print(build_graph().get_graph().draw_mermaid())")]

c += [M("## 13. End-to-end demo"),
      C("res = analyze_claim('samples/08_copay_nonpayable_partial.pdf')\nrep = res['report']\nprint('DECISION :', rep['decision'], '| payable ₹', f\"{rep['payable_amount']:,.2f}\", '| settle by', rep['settle_by'])\nprint()\nprint(rep['letter'])"),
      C("res = analyze_claim('samples/09_fraud_red_flags_investigate.pdf')\nprint(res['report']['decision'], res['fraud']['score'])\nfor s in res['report']['investigation']['checklist']: print(' -', s)"),
      M("## 14. Evaluation\n13 labelled synthetic claims (`samples/expected.json`): decision accuracy and payable amounts. Unit tests for the rule engine and calculator: `python test_validators.py` / `pytest -q`."),
      C("exp = json.load(open('samples/expected.json')); rows = []\nfor f, e in exp.items():\n    r = analyze_claim(f'samples/{f}'); got = r['report']['decision']\n"
        "    rows.append((f, e['expected_decision'], got, r['report']['payable_amount'], r['fraud']['score'], 'PASS' if got == e['expected_decision'] else 'FAIL'))\n"
        "df = pd.DataFrame(rows, columns=['file', 'expected', 'got', 'payable ₹', 'fraud', 'result'])\nprint(f\"Accuracy: {(df.result == 'PASS').mean():.0%}\")\ndf"),
      M("## 15. API + UI\n`uvicorn app:app --reload` → http://localhost:8000/docs · `streamlit run streamlit_app.py`\n\n| Endpoint | Purpose |\n|---|---|\n| `POST /analyze` | Claim file → full multi-agent report |\n| `POST /analyze-batch` | Multiple claims → summary + total payable |\n| `POST /analyze-text` | Already-extracted text → report |\n| `POST /adjudicate` | Structured JSON → rules, settlement and fraud only (no LLM) |\n| `POST /ask` | Policy / IRDAI Q&A (RAG) |\n| `GET /policy/{no}` · `POST /reindex` · `GET /health` | Utilities |"),
      C("from fastapi.testclient import TestClient\nimport sys; sys.path.insert(0, '.')\nfrom app import app\nclient = TestClient(app)\nprint(client.get('/health').json())\nwith open('samples/02_room_rent_proportionate.pdf', 'rb') as fh:\n    out = client.post('/analyze', files={'file': ('claim.pdf', fh, 'application/pdf')}).json()\nprint(out['report']['decision'], out['report']['payable_amount'])"),
      M("## 16. Deployment (AWS)\nDocker → ECR → **App Runner** (port 8000, `OPENAI_API_KEY` from Secrets Manager, health check `/health`, 1 vCPU / 2 GB). Script: `deploy/aws_deploy.sh`. Keep the FAISS index and policy documents in S3, and claim files in S3 with SSE-KMS. For production, consider Textract for OCR, DynamoDB/RDS for the policy master and CloudWatch logs with PII masking. Full steps are in `README.md`."),
      M("""## 17. Limitations and next steps
- The policy master is a JSON file; a real system would integrate with the core policy administration system or a TPA API.
- Proportionate deduction does not check whether the hospital follows differential billing (needs hospital tariff data).
- Pre/post-hospitalisation, disease-wise sub-limits, package rates, cumulative bonus and restoration benefits are not modelled yet.
- The fraud score is rule-based; with labelled data it could be extended with a gradient-boosting model and network analysis (hospital–agent rings).
- Tesseract is weak on handwritten discharge summaries; a vision-LLM or Textract fallback would help.
- IRDAI timelines and limits change over time; update `data/policy_docs` and the rule constants periodically.
- **Decision support only:** under IRDAI rules, repudiations are decided by a claims review committee. Real claims contain health data (DPDP Act), so consent, masking, encryption and access logging are required.""")]

out = ROOT / "insurance_claims_copilot.ipynb"
nbf.write(nb, out)
print("cells:", len(c), "->", out.name)

if "--execute" in sys.argv:
    from nbconvert.preprocessors import ExecutePreprocessor
    os.environ.setdefault("OFFLINE_MODE", "1")
    nb = nbf.read(out, as_version=4)
    ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(ROOT)}})
    nbf.write(nb, out)
    print("executed ✅")
