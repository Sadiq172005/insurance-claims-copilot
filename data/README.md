# data/ — what goes here

## 1. `data/policy_docs/` (RAG knowledge base) — required
This folder currently contains two **synthetic starter** notes for demo purposes. For real use, add official sources (text-layer PDFs, not scans):

| Document | Source | Why |
|---|---|---|
| IRDAI Master Circular on Health Insurance Business (29 May 2024) | irdai.gov.in | Cashless 1-hour / discharge 3-hour timelines, documents, claim rejection process |
| IRDAI (Insurance Products) Regulations 2024 | irdai.gov.in | Maximum PED / specific-disease waiting periods, moratorium |
| IRDAI (Protection of Policyholders' Interests) Regulations 2024 | irdai.gov.in | Settlement timelines, interest on delay, grievance process |
| Standardised exclusions + list of non-payable items (Lists I–IV) | irdai.gov.in | Exclusions, consumables, items subsumed into room charges |
| Policy wordings and prospectus of the products being claimed against | Insurer website | Waiting periods, sub-limits, room rent, co-pay |
| Day-care procedures list | Policy annexure | Exceptions to the 24-hour hospitalisation rule |
| Bima Bharosa / Insurance Ombudsman rules | irdai.gov.in / cioins.co.in | Grievance text in letters |

## 2. Structured reference data
- `policies.json` — policy master (synthetic). In production this comes from the policy administration system or a TPA API.
- `non_payable_items.csv` — subset of IRDAI List I (keyword match). Extend it from the full list.
- `claims_history.json` — fingerprints of paid claims for duplicate detection (seeded by `generate_samples.py`).
- Possible additions: `hospital_tariffs.csv` (package rates / differential billing), `rohini_registry.csv` (ROHINI ID validation), `icd10_daycare_map.csv`.

## 3. Generated files (do not commit)
- `faiss_index/` — built by `make index` (LLM mode). Rebuild via `/reindex` whenever the documents change.

## 4. Privacy
Real claim documents contain health data and PII. Use only synthetic or masked data for testing; consent, encryption and access control are mandatory for real data.
