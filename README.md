# 🏥 Insurance Claims & Policy Intelligence

An AI-assisted claims adjudication system for Indian health insurance. Upload a claim file (claim form + discharge summary + itemised hospital bill, as a PDF or scan) and the system runs OCR, extracts structured data, applies policy rules, calculates the payable amount, flags fraud risk, retrieves the relevant policy and IRDAI clauses, and produces a **decision, deduction sheet and claimant letter**.

Built as a capstone that combines four projects: **Enterprise RAG · Multi-Agent Assistant (LangGraph) · OCR · AWS Deployment**.

## Architecture
```
Claim file ─► OCR (pdfplumber / Tesseract + table grid-line removal)
          ─► [1 Intake Agent]      LLM → Pydantic Claim  (offline: regex + OCR-tolerant bill parser)
          ─► [2 Validator Agent]   Deterministic code: policy active, member insured, initial/PED/specific-disease
                                   waiting periods, exclusions, 24-hour rule vs day-care, network check (cashless),
                                   timelines, bill arithmetic + settlement calculator + fraud score + rule decision
               ├─ fraud score ≥ 50 ─► [3 Fraud Investigator Agent]  SIU verification checklist
               └─ policy not found ─► skip RAG
          ─► [4 Policy Researcher] RAG over policy wording + IRDAI notes (FAISS / offline keyword retriever)
          ─► [5 Adjudicator]       decision, rationale, document queries, letter (with hard guards)
FastAPI: /analyze /analyze-batch /analyze-text /adjudicate /ask /policy/{no} /reindex /health → Docker → ECR → App Runner
```

### Key design decisions
- **Money and eligibility are decided by code, not the LLM.** The LLM reads messy documents, explains clauses and drafts letters. Waiting periods, caps, co-pay and deductions are deterministic and unit-tested.
- **Hard guards on the adjudicator:**
  1. The payable amount always comes from the settlement calculator.
  2. The LLM cannot make a decision more lenient than the rule engine's.
  3. The LLM cannot reject a claim on its own; at most it can escalate to `QUERY`. Under IRDAI rules, every repudiation must cite a specific clause and go through a claims review committee.
- **Explainable fraud scoring:** a points-based red-flag score instead of a black-box model, so investigators can see exactly why a claim was flagged.
- **Offline fallbacks everywhere:** the full pipeline runs without an API key (regex extractor, keyword retriever, template letters), so demos and CI never depend on an external service.

### Decisions
`APPROVE` · `PARTIAL_APPROVE` · `QUERY` · `INVESTIGATE` · `REJECT`

### Settlement logic
1. Remove non-payable items (IRDAI List I).
2. Apply the room rent cap (1% of sum insured per day) and the ICU cap (2% per day).
3. Apply proportionate deduction to doctor, surgery and nursing charges only. It does not apply to pharmacy, consumables, implants, diagnostics or ICU.
4. Apply co-payment.
5. Cap the result at the lower of the amount claimed and the remaining sum insured.

## Quick start
```bash
cp .env.example .env            # add OPENAI_API_KEY (optional: runs in offline mode without it)
make install samples test eval-offline
make api                         # http://localhost:8000/docs
make ui                          # http://localhost:8501
```
System packages required: `tesseract-ocr` and `poppler-utils` (macOS: `brew install tesseract poppler`).

**LLM mode:** set the key in `.env`, then run `make index` (builds the FAISS index) and `make eval`.

**Notebook:** `insurance_claims_copilot.ipynb` is generated from `pipeline.py` (the single source of truth). Regenerate it with `python build_notebook.py --execute`.

## Project structure
| File | Purpose |
|---|---|
| `pipeline.py` | OCR, extraction, rule engine, settlement calculator, fraud scoring, RAG, LangGraph agents |
| `app.py` | FastAPI service |
| `streamlit_app.py` | Web UI (single claim, batch queue, policy Q&A) |
| `insurance_claims_copilot.ipynb` | End-to-end walkthrough notebook |
| `generate_samples.py` | Generates 13 synthetic claim files + `expected.json` |
| `evaluate.py` | End-to-end decision accuracy on the samples |
| `test_validators.py` | Offline unit tests (rules, calculator, fraud, extractor) |
| `data/` | Policy master, non-payable list, claims history, RAG documents |
| `deploy/aws_deploy.sh` | ECR + App Runner deployment script |
| `Dockerfile`, `Makefile`, `pyproject.toml`, `.github/workflows/ci.yml` | Build, tooling, CI |

## API examples
```bash
curl -F file=@samples/02_room_rent_proportionate.pdf localhost:8000/analyze
curl -F files=@samples/01_clean_cashless_approve.pdf -F files=@samples/09_fraud_red_flags_investigate.pdf localhost:8000/analyze-batch
curl -X POST localhost:8000/ask -H 'Content-Type: application/json' -d '{"question":"Is cataract surgery covered in the first year?"}'
curl localhost:8000/policy/HLT/2025/000377
```

| Endpoint | Description |
|---|---|
| `POST /analyze` | Claim file → full multi-agent report |
| `POST /analyze-batch` | Multiple files → per-claim reports, decision summary, total payable |
| `POST /analyze-text` | Already-extracted text → report |
| `POST /adjudicate` | Structured claim JSON → rules, settlement and fraud only (no LLM call) |
| `POST /ask` | Policy / IRDAI question answering with sources |
| `GET /policy/{no}` | Policy lookup |
| `POST /reindex` | Rebuild the FAISS index (requires API key) |
| `GET /health` | Health check and current mode |

## Sample claims (synthetic)
Generate with `python generate_samples.py`.

| Sample | Expected | Scenario |
|---|---|---|
| 01_clean_cashless_approve | APPROVE ₹1,28,300 | Clean cashless surgery at a network hospital |
| 02_room_rent_proportionate | PARTIAL ₹65,277.78 | ₹9,000/day suite vs ₹5,000 cap + proportionate deduction |
| 03_initial_waiting_reject | REJECT | 30-day initial waiting period (non-accident) |
| 04_ped_waiting_reject | REJECT | Declared hypertension, 36-month PED waiting period |
| 05_specific_disease_waiting_reject | REJECT | Cataract, 24-month specific-disease waiting period |
| 06_policy_lapsed_reject | REJECT | Admission after policy expiry |
| 07_bill_total_mismatch_query | QUERY | Line items total ₹89,000 vs printed ₹98,000 |
| 08_copay_nonpayable_partial | PARTIAL ₹64,400 | 20% senior citizen co-pay + non-payable items |
| 09_fraud_red_flags_investigate | INVESTIGATE (score 85) | Early claim, round bill, >80% of SI, no ROHINI ID or doctor registration |
| 10_cosmetic_exclusion_reject | REJECT | Cosmetic rhinoplasty (permanent exclusion) |
| 11_cashless_non_network_query | QUERY | Cashless request at a non-network hospital |
| 12_duplicate_claim_investigate | INVESTIGATE (score 50) | Resubmission of an already-paid claim |
| 13_scanned_clean_approve | APPROVE | Image-only PDF, exercises the OCR path |

**Results (offline mode):** 13/13 decisions match the expected labels. All unit tests pass (`python test_validators.py` or `pytest -q`).

## Streamlit UI
Three tabs:
- **Single claim:** decision banner, claimed / payable / deductions / fraud score / settle-by date, rule findings, deduction sheet, red flags, generated letter.
- **Batch queue:** decision distribution chart, total payable, CSV export.
- **Ask policy / IRDAI:** RAG-based Q&A with sources.

Enter a deployed API URL in the sidebar to point the UI at the AWS service instead of the local pipeline.

## AWS deployment (ECR + App Runner)
```bash
REGION=ap-south-1 bash deploy/aws_deploy.sh
```
The script creates an ECR repository and pushes the image, stores the OpenAI key in Secrets Manager, and creates or updates an App Runner service (port 8000, 1 vCPU / 2 GB, health check on `/health`). Two IAM roles need to be created once beforehand; see the comments in the script.

**Production recommendations:**
- Store the FAISS index and policy documents in S3, and claim files in S3 with SSE-KMS.
- Move the policy master to DynamoDB or RDS.
- Use Amazon Textract as the OCR engine.
- Mask PII in CloudWatch logs and run inside a VPC with private endpoints.

**Alternative:** run on EC2 (t3.small) with `docker run -p 80:8000 --env-file .env claims-copilot`.

## Demo script (~6 minutes)
1. Clean cashless claim → APPROVE.
2. Deluxe suite claim → walk through the deduction sheet and explain proportionate deduction.
3. Senior citizen claim with co-pay and consumables → PARTIAL_APPROVE.
4. Cataract surgery in year one → REJECT with the clause retrieved via RAG.
5. Fraud sample → INVESTIGATE with the SIU checklist (conditional graph edge).
6. Batch tab → run all samples, show the chart and total payable.
7. Live API at `/docs` on the AWS URL.

## Limitations and next steps
- The policy master is a JSON file; a real deployment would integrate with a policy administration system or TPA API.
- Proportionate deduction does not check whether the hospital follows differential billing (needs hospital tariff data).
- Pre/post-hospitalisation, disease-wise sub-limits, package rates, cumulative bonus and restoration benefits are not yet modelled.
- Fraud scoring is rule-based; with labelled data it could be extended with a gradient-boosting model and network analysis.
- Tesseract struggles with handwritten discharge summaries; a vision-LLM or Textract fallback would help.

## Important notes
- All data is **synthetic** (policies, hospitals, patients). Real claims contain health data and PII and require consent, masking, encryption and access logging under the DPDP Act.
- Rule constants (waiting periods, caps, timelines) follow the sample policy wording. For a real product, update them from that product's policy wording and the latest IRDAI regulations.
- This is a **decision-support** tool. Final decisions rest with a licensed claims officer or claims review committee.
