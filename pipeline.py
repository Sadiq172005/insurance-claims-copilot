# %% Setup & Config
"""Health Insurance Claims Copilot — OCR -> Extraction -> Rules + Settlement Calc + Fraud + RAG -> Multi-Agent (LangGraph)."""
import os, re, json, math, hashlib, difflib, datetime as dt
from pathlib import Path
from typing import List, Optional, TypedDict, Literal, Dict, Any

from pydantic import BaseModel, Field

try:                                   # load OPENAI_API_KEY etc. from .env (optional)
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
DOCS_DIR = Path(os.getenv("POLICY_DOCS_DIR", BASE_DIR / "data/policy_docs"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", BASE_DIR / "data/faiss_index"))
POLICY_DB = Path(os.getenv("POLICY_DB", BASE_DIR / "data/policies.json"))
CLAIMS_HISTORY = Path(os.getenv("CLAIMS_HISTORY", BASE_DIR / "data/claims_history.json"))
NON_PAYABLE_CSV = Path(os.getenv("NON_PAYABLE_CSV", BASE_DIR / "data/non_payable_items.csv"))
FRAUD_THRESHOLD = int(os.getenv("FRAUD_THRESHOLD", "50"))


def llm_available() -> bool:
    """Without OPENAI_API_KEY (or with OFFLINE_MODE=1) the pipeline uses offline fallbacks, so demos and tests run without a key."""
    return bool(os.getenv("OPENAI_API_KEY")) and os.getenv("OFFLINE_MODE", "0") != "1"


def get_llm():
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=LLM_MODEL, temperature=0)


# %% Schemas
BillHead = Literal["room", "icu", "doctor", "surgery", "nursing", "pharmacy", "consumables", "implants", "investigation", "other"]


class BillItem(BaseModel):
    date: Optional[str] = Field(None, description="YYYY-MM-DD")
    description: str = ""
    head: BillHead = Field("other", description="Bill head / category")
    quantity: float = 1
    rate: Optional[float] = None
    amount: float = 0.0


class Claim(BaseModel):
    claim_type: Literal["cashless", "reimbursement"] = "reimbursement"
    policy_no: Optional[str] = None
    patient_name: Optional[str] = None
    patient_age: Optional[int] = None
    hospital_name: Optional[str] = None
    hospital_rohini_id: Optional[str] = Field(None, description="IIB ROHINI hospital ID (13 chars)")
    admission_date: Optional[str] = Field(None, description="YYYY-MM-DD")
    discharge_date: Optional[str] = Field(None, description="YYYY-MM-DD")
    intimation_date: Optional[str] = Field(None, description="YYYY-MM-DD")
    submission_date: Optional[str] = Field(None, description="YYYY-MM-DD (reimbursement docs submitted)")
    is_accident: bool = False
    diagnosis: Optional[str] = None
    icd10_code: Optional[str] = None
    procedure: Optional[str] = None
    treatment_type: Literal["surgical", "medical", "daycare"] = "medical"
    room_type: Optional[str] = None
    doctor_name: Optional[str] = None
    doctor_reg_no: Optional[str] = None
    bill_items: List[BillItem] = []
    total_billed: Optional[float] = None
    amount_claimed: Optional[float] = None


class Issue(BaseModel):
    code: str
    severity: Literal["error", "warning", "info"]
    message: str
    fix: Optional[str] = None


class Deduction(BaseModel):
    reason: str
    amount: float
    clause: Optional[str] = None


class Settlement(BaseModel):
    total_billed: float = 0
    amount_claimed: float = 0
    non_payable: float = 0
    room_rent_excess: float = 0
    proportionate_deduction: float = 0
    copay: float = 0
    sum_insured_cap: float = 0
    payable: float = 0
    deductions: List[Deduction] = []


# %% OCR
def _remove_table_lines(im, h_len: int = 150, v_len: int = 100):
    """Table grid lines in scanned bills confuse Tesseract, so long horizontal/vertical dark runs are painted white (vectorised numpy)."""
    import numpy as np
    from PIL import Image
    a = np.array(im.convert("L"))
    dark = (a < 215).astype(np.int32)          # also catch light-grey grid lines (text strokes are never this long)

    def runs(mask, w, axis):
        c = np.cumsum(mask, axis=axis)
        pad = np.zeros_like(c.take([0], axis=axis))
        c = np.concatenate([pad, c], axis=axis)
        full = (c.take(range(w, c.shape[axis]), axis=axis) - c.take(range(0, c.shape[axis] - w), axis=axis)) == w
        # window start positions -> mark all w pixels covered
        f = full.astype(np.int32)
        fc = np.concatenate([np.zeros_like(f.take([0], axis=axis)), np.cumsum(f, axis=axis)], axis=axis)
        n = mask.shape[axis]
        idx = np.arange(n)
        lo = np.clip(idx - w + 1, 0, f.shape[axis]); hi = np.clip(idx + 1, 0, f.shape[axis])
        return (fc.take(hi, axis=axis) - fc.take(lo, axis=axis)) > 0

    def dilate(mask, axis, r=2):                 # skewed scans: lines drift 1-2px, so dilate slightly in the perpendicular direction
        out = mask.copy()
        for k in range(1, r + 1):
            out |= np.roll(mask, k, axis=axis) | np.roll(mask, -k, axis=axis)
        return out

    h = runs(dilate(dark.astype(bool), 0).astype(np.int32), h_len, 1)
    v = runs(dilate(dark.astype(bool), 1).astype(np.int32), v_len, 0)
    a[(h | v) & dark.astype(bool)] = 255
    return Image.fromarray(a)


def _ocr_image(im) -> str:
    import pytesseract
    from PIL import ImageOps
    return pytesseract.image_to_string(_remove_table_lines(ImageOps.grayscale(im)), config="--psm 6")


def ocr_document(path: str) -> str:
    """Handles PDFs (text-layer or scanned) and images. Multi-page claim file (claim form + discharge summary + bill)."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".pdf":
        text = ""
        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                text = "\n".join((pg.extract_text() or "") for pg in pdf.pages)
        except Exception:
            pass
        if len(text.strip()) > 80:          # digital PDF: no OCR needed
            return text
        from pdf2image import convert_from_path
        return "\n".join(_ocr_image(im) for im in convert_from_path(path, dpi=300))
    from PIL import Image
    return _ocr_image(Image.open(path))


# %% Extraction (LLM -> structured Claim, offline regex fallback)
EXTRACT_PROMPT = """You are an Indian health-insurance claim document parser (claim form + discharge summary + final hospital bill).
Extract fields from the OCR text. Rules: dates as YYYY-MM-DD; amounts as plain numbers (no commas/currency);
bill_items = every line of the itemised final bill with head in [room, icu, doctor, surgery, nursing, pharmacy, consumables, implants, investigation, other];
quantity for room/icu = number of days. If a field is absent use null. Do NOT guess or correct values — copy exactly what is printed.
OCR TEXT:
{text}"""

HEADS = ["room", "icu", "doctor", "surgery", "nursing", "pharmacy", "consumables", "implants", "investigation", "other"]
_LABELS = {
    "policy_no": r"Policy (?:Number|No\.?)", "claim_type": r"Claim Type", "patient_name": r"Patient Name", "patient_age": r"Patient Age",
    "hospital_name": r"Hospital Name", "hospital_rohini_id": r"(?:Hospital )?ROHINI ID", "admission_date": r"Date of Admission",
    "discharge_date": r"Date of Discharge", "intimation_date": r"Date of Intimation", "submission_date": r"Date of Submission",
    "is_accident": r"Accident(?: Case)?", "diagnosis": r"(?:Final )?Diagnosis", "icd10_code": r"ICD-?10 Code", "procedure": r"Procedure",
    "treatment_type": r"Treatment Type", "room_type": r"Room Type", "doctor_name": r"Treating Doctor", "doctor_reg_no": r"Doctor Reg\.? No\.?",
    "total_billed": r"Total Bill Amount", "amount_claimed": r"Amount Claimed",
}


def _num(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    m = re.search(r"-?[\d,]+(?:\.\d+)?", s.replace(" ", ""))
    return float(m.group(0).replace(",", "")) if m else None


def _iso(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    m = re.search(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})", s)
    if m:
        d, mo, y = map(int, m.groups())
        try:
            return dt.date(y, mo, d).isoformat()
        except ValueError:
            return None
    m = re.search(r"\d{4}-\d{2}-\d{2}", s)
    return m.group(0) if m else None


def rule_based_extract(text: str) -> Claim:
    """Offline extractor for claim documents in 'Label: value' format (samples/). Messy real-world documents use the LLM extractor."""
    f: Dict[str, Any] = {}
    for key, lab in _LABELS.items():
        m = re.search(rf"^\s*{lab}\s*[:\-]\s*(.*)$", text, flags=re.I | re.M)
        val = m.group(1).strip() if m else None
        f[key] = val if val not in ("", "-", "N/A", "NA") else None
    for k in ("admission_date", "discharge_date", "intimation_date", "submission_date"):
        f[k] = _iso(f[k])
    for k in ("total_billed", "amount_claimed"):
        f[k] = _num(f[k])
    f["patient_age"] = int(_num(f["patient_age"])) if f["patient_age"] and _num(f["patient_age"]) else None
    f["is_accident"] = bool(f["is_accident"] and f["is_accident"].lower().startswith("y"))
    ct = (f["claim_type"] or "").lower()
    f["claim_type"] = "cashless" if "cashless" in ct else "reimbursement"
    tt = (f["treatment_type"] or "").lower()
    f["treatment_type"] = next((t for t in ("surgical", "daycare", "medical") if t in tt.replace("-", "").replace(" ", "")), "medical")
    items = []
    money = re.compile(r"\d{1,3}(?:,\d{2,3})*\.\d{2}")
    for ln in text.splitlines():               # tolerant row parser: [date] description HEAD qty rate amount (OCR noise allowed)
        nums = money.findall(ln)
        if len(nums) < 2:
            continue
        toks = re.findall(r"[A-Za-z]+", ln)
        head = next((h for t in toks for h in difflib.get_close_matches(t.lower(), HEADS, n=1, cutoff=0.8)), None)
        if not head:
            continue
        rate, amt = _num(nums[-2]), _num(nums[-1])
        dm = re.search(r"\d{1,2}[-/.]\d{1,2}[-/.]\d{4}", ln)
        hm = re.search(r"[A-Za-z]+(?=[^A-Za-z]*\d+(?:\.\d+)?\s*[|\s]*" + re.escape(nums[-2]) + ")", ln)
        start = dm.end() if dm else 0
        end = ln.lower().rfind(hm.group(0).lower()) if hm else ln.find(nums[-2])
        desc = re.sub(r"[|_~—«“”‘’]+", " ", ln[start:max(end, start)]).strip(" -.:")
        qty = round(amt / rate, 2) if rate else 1.0
        items.append(BillItem(date=_iso(dm.group(0)) if dm else None, description=" ".join(desc.split()), head=head,
                              quantity=qty, rate=rate, amount=amt))
    f["bill_items"] = items
    return Claim(**f)


def extract_claim(text: str) -> Claim:
    if not llm_available():
        return rule_based_extract(text)
    llm = get_llm().with_structured_output(Claim, method="function_calling")
    return llm.invoke(EXTRACT_PROMPT.format(text=text[:15000]))


# %% Policy master + reference data
_POLICIES: Optional[dict] = None


def load_policies(path: Path = POLICY_DB) -> dict:
    """Stand-in for the policy administration system. In production this comes from the core insurance DB / API."""
    global _POLICIES
    if _POLICIES is None:
        _POLICIES = json.loads(Path(path).read_text(encoding="utf-8"))
    return _POLICIES


def get_policy(policy_no: Optional[str]) -> Optional[dict]:
    if not policy_no:
        return None
    key = re.sub(r"\s+", "", policy_no).upper()
    return load_policies()["policies"].get(key)


def load_non_payable(path: Path = NON_PAYABLE_CSV) -> List[str]:
    """Subset of IRDAI 'List I' (items not payable), as lowercase keywords."""
    if not Path(path).exists():
        return []
    rows = Path(path).read_text(encoding="utf-8").strip().splitlines()[1:]
    return [r.split(",")[0].strip().lower() for r in rows if r.strip()]


def load_history(path: Path = CLAIMS_HISTORY) -> List[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else []


# %% Deterministic claim validators (hard checks on policy rules, not the LLM)
ICD_RE = re.compile(r"^[A-TV-Z][0-9][0-9A-Z](\.[0-9A-Z]{1,4})?$")
ROHINI_RE = re.compile(r"^\d{13}$")
REJECT_CODES = {"POLICY_NOT_FOUND", "PATIENT_NOT_INSURED", "POLICY_INACTIVE", "INITIAL_WAITING", "PED_WAITING",
                "SPECIFIC_DISEASE_WAITING", "EXCLUSION"}


def _d(s: Optional[str]) -> Optional[dt.date]:
    try:
        return dt.date.fromisoformat(s) if s else None
    except ValueError:
        return None


def add_months(d: dt.date, months: int) -> dt.date:
    y, m = divmod(d.month - 1 + months, 12)
    y += d.year; m += 1
    last = [31, 29 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1]
    return dt.date(y, m, min(d.day, last))


def _norm(name: Optional[str]) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", (name or "").lower()).split())


def find_member(policy: dict, name: Optional[str]) -> Optional[dict]:
    n = _norm(name)
    for mem in policy.get("members", []):
        if _norm(mem["name"]) == n:
            return mem
    for mem in policy.get("members", []):   # "Ramesh Kumar" vs "Ramesh K": first name + initial
        a, b = _norm(mem["name"]).split(), n.split()
        if a and b and a[0] == b[0] and len(a) > 1 and len(b) > 1 and a[-1][0] == b[-1][0]:
            return mem
    return None


def matches_condition(claim: Claim, cond: dict) -> bool:
    """Condition = {name, icd: [prefixes], keywords: [...]}; matches on ICD prefix or a diagnosis/procedure keyword."""
    icd = (claim.icd10_code or "").upper().replace(" ", "")
    if icd and any(icd.startswith(p.upper()) for p in cond.get("icd", [])):
        return True
    text = f"{claim.diagnosis or ''} {claim.procedure or ''}".lower()
    return any(re.search(rf"\b{re.escape(k.lower())}", text) for k in cond.get("keywords", []))


def length_of_stay(c: Claim) -> Optional[int]:
    a, b = _d(c.admission_date), _d(c.discharge_date)
    return (b - a).days if a and b else None


def validate_claim(c: Claim, today: Optional[dt.date] = None) -> List[Issue]:
    today = today or dt.date.today()
    out: List[Issue] = []
    pol = get_policy(c.policy_no)
    adm, dis = _d(c.admission_date), _d(c.discharge_date)

    # --- documents / dates
    if not adm or not dis:
        out.append(Issue(code="DATE_INVALID", severity="error", message="Admission/discharge date is missing or invalid.",
                         fix="Confirm the dates from the discharge summary."))
    elif dis < adm:
        out.append(Issue(code="DATE_ORDER", severity="error", message=f"Discharge date ({dis}) is before the admission date ({adm})."))
    elif adm > today:
        out.append(Issue(code="DATE_FUTURE", severity="error", message="Admission date is in the future."))
    if not c.diagnosis:
        out.append(Issue(code="DOCS_MISSING", severity="error", message="Diagnosis / discharge summary not found.",
                         fix="Request the discharge summary (diagnosis, treatment, doctor's signature)."))
    if not c.icd10_code:
        out.append(Issue(code="ICD_MISSING", severity="warning", message="ICD-10 code missing.", fix="Obtain a coded discharge summary from the hospital."))
    elif not ICD_RE.match(c.icd10_code.strip().upper()):
        out.append(Issue(code="ICD_FORMAT", severity="warning", message=f"ICD-10 code '{c.icd10_code}' has an invalid format."))

    # --- policy lookup
    if not pol:
        out.append(Issue(code="POLICY_NOT_FOUND", severity="error", message=f"Policy '{c.policy_no}' was not found in the policy master.",
                         fix="Verify the policy number / e-card."))
        return out
    member = find_member(pol, c.patient_name)
    if not member:
        out.append(Issue(code="PATIENT_NOT_INSURED", severity="error",
                         message=f"Patient '{c.patient_name}' is not an insured member of this policy.",
                         fix="Check the member list / KYC and verify the name spelling."))
    start, end, first = _d(pol["policy_start"]), _d(pol["policy_end"]), _d(pol["first_inception"])
    if adm and not (start <= adm <= end):
        out.append(Issue(code="POLICY_INACTIVE", severity="error",
                         message=f"Admission {adm} is outside the policy period ({start} → {end}).", fix="Check renewal / grace period status."))

    # --- waiting periods (continuous coverage counted from first_inception)
    if adm and first:
        if not c.is_accident and adm < first + dt.timedelta(days=pol.get("initial_waiting_days", 30)):
            out.append(Issue(code="INITIAL_WAITING", severity="error",
                             message=f"Admission is within {pol.get('initial_waiting_days', 30)} days of first inception ({first}) (non-accident).",
                             fix="Initial waiting period exclusion: only accident claims are payable."))
        if member:
            ped_end = add_months(first, pol.get("ped_waiting_months", 36))
            for ped in member.get("ped", []):
                if matches_condition(c, ped) and adm < ped_end:
                    out.append(Issue(code="PED_WAITING", severity="error",
                                     message=f"Waiting period for declared pre-existing disease '{ped['name']}' runs until {ped_end}.",
                                     fix="Payable only after the PED waiting period; if the condition is unrelated, obtain a medical opinion."))
        sd_end = add_months(first, pol.get("specific_disease_waiting_months", 24))
        for sd in load_policies().get("specific_diseases", []):
            if matches_condition(c, sd) and adm < sd_end:
                out.append(Issue(code="SPECIFIC_DISEASE_WAITING", severity="error",
                                 message=f"Specific-disease waiting period for '{sd['name']}' runs until {sd_end}.",
                                 fix="Check the specific waiting period clause in the policy wording."))
    for ex in load_policies().get("exclusions", []):
        if matches_condition(c, ex) and not (c.is_accident and ex.get("unless_accident")):
            out.append(Issue(code="EXCLUSION", severity="error", message=f"Permanent exclusion: {ex['name']} ({ex.get('clause', '')}).",
                             fix="Claim falls under an exclusion; quote the clause in the rejection letter."))

    # --- hospitalisation rules
    los = length_of_stay(c)
    if los is not None and los < 1 and c.treatment_type != "daycare":
        out.append(Issue(code="MIN_HOSPITALIZATION", severity="error",
                         message="Hospitalisation under 24 hours and not a day-care procedure.",
                         fix="Check the day-care list or process under OPD cover."))
    if c.claim_type == "cashless" and c.hospital_name and not any(
            _norm(h) == _norm(c.hospital_name) for h in load_policies().get("network_hospitals", [])):
        out.append(Issue(code="NON_NETWORK_CASHLESS", severity="error",
                         message=f"'{c.hospital_name}' is not a network hospital; cashless is not allowed.",
                         fix="Ask the claimant to resubmit via the reimbursement route."))
    if not c.hospital_rohini_id or not ROHINI_RE.match(c.hospital_rohini_id.replace(" ", "")):
        out.append(Issue(code="ROHINI_MISSING", severity="warning", message="Hospital ROHINI ID missing/invalid.",
                         fix="Verify the hospital in the IIB ROHINI registry."))
    if not c.doctor_reg_no:
        out.append(Issue(code="DOCTOR_REG_MISSING", severity="warning", message="Treating doctor's registration number is missing.",
                         fix="Verify with the NMC / State Medical Council registry."))

    # --- timelines
    intim = _d(c.intimation_date)
    if adm and intim and c.claim_type == "reimbursement" and (intim - adm).days > pol.get("intimation_days", 1):
        out.append(Issue(code="LATE_INTIMATION", severity="warning",
                         message=f"Claim intimated {(intim - adm).days} days after admission (limit {pol.get('intimation_days', 1)} days).",
                         fix="IRDAI: a genuine claim must not be rejected solely for delay; record the reason."))
    sub = _d(c.submission_date)
    if dis and sub and c.claim_type == "reimbursement" and (sub - dis).days > pol.get("submission_days", 30):
        out.append(Issue(code="LATE_SUBMISSION", severity="warning",
                         message=f"Documents submitted {(sub - dis).days} days after discharge (limit {pol.get('submission_days', 30)}).",
                         fix="Condone the delay if justified (IRDAI guidance)."))

    # --- bill math
    items_sum = round(sum(i.amount for i in c.bill_items), 2)
    if not c.bill_items:
        out.append(Issue(code="BILL_MISSING", severity="error", message="Itemised final bill not found.", fix="Request the itemised bill and payment receipt."))
    if c.total_billed is not None and c.bill_items and abs(items_sum - c.total_billed) > 1:
        out.append(Issue(code="BILL_TOTAL_MISMATCH", severity="error",
                         message=f"Bill items add up to ₹{items_sum:,.2f}, but the printed total is ₹{c.total_billed:,.2f}.",
                         fix="Request a corrected final bill from the hospital."))
    for n, it in enumerate(c.bill_items, 1):
        if it.rate is not None and abs(it.rate * it.quantity - it.amount) > 1:
            out.append(Issue(code="LINE_MATH", severity="warning",
                             message=f"Bill line {n} ('{it.description}'): {it.quantity} × {it.rate} ≠ {it.amount}."))
        d = _d(it.date)
        if d and adm and dis and not (adm <= d <= dis):
            out.append(Issue(code="BILL_DATE_OUTSIDE_STAY", severity="warning",
                             message=f"Bill line {n} is dated {d}, outside the hospital stay ({adm} → {dis})."))
    claimed = c.amount_claimed if c.amount_claimed is not None else c.total_billed
    if claimed is not None and c.total_billed is not None and claimed > c.total_billed + 1:
        out.append(Issue(code="CLAIM_EXCEEDS_BILL", severity="error",
                         message=f"Claimed ₹{claimed:,.2f} exceeds the bill (₹{c.total_billed:,.2f})."))
    remaining = pol["sum_insured"] - pol.get("claims_paid", 0)
    if claimed and claimed > remaining:
        out.append(Issue(code="SI_INSUFFICIENT", severity="warning",
                         message=f"Claimed ₹{claimed:,.0f} exceeds the remaining sum insured ₹{remaining:,.0f}; payable will be capped."))
    return out


# %% Settlement calculator (deterministic: the LLM never decides the payable amount)
PROPORTIONATE_HEADS = {"doctor", "surgery", "nursing"}   # IRDAI: no proportionate deduction on pharmacy, consumables, implants, diagnostics


def is_non_payable(item: BillItem, keywords: List[str], consumables_cover: bool) -> bool:
    if item.head == "consumables" and not consumables_cover:
        return True
    d = item.description.lower()
    return any(k in d for k in keywords)


def compute_settlement(c: Claim, policy: Optional[dict] = None) -> Settlement:
    pol = policy or get_policy(c.policy_no)
    billed = round(sum(i.amount for i in c.bill_items), 2)
    claimed = c.amount_claimed if c.amount_claimed is not None else (c.total_billed if c.total_billed is not None else billed)
    s = Settlement(total_billed=billed, amount_claimed=claimed)
    if not pol:
        return s
    deds: List[Deduction] = []
    np_kw = load_non_payable()
    non_pay = [i for i in c.bill_items if is_non_payable(i, np_kw, pol.get("consumables_cover", False))]
    np_ids = {id(i) for i in non_pay}
    s.non_payable = round(sum(i.amount for i in non_pay), 2)
    for i in non_pay:
        deds.append(Deduction(reason=f"Non-payable item: {i.description}", amount=i.amount, clause="IRDAI List I / policy non-medical expenses"))

    si = pol["sum_insured"]
    ratios = []
    for head, pct_key in (("room", "room_rent_limit_pct"), ("icu", "icu_limit_pct")):
        lines = [i for i in c.bill_items if i.head == head and id(i) not in np_ids]
        pct = pol.get(pct_key)
        if not lines or not pct:
            continue
        days = sum(i.quantity for i in lines) or 1
        actual_per_day = sum(i.amount for i in lines) / days
        eligible_per_day = si * pct / 100
        if actual_per_day > eligible_per_day + 0.5:
            excess = round((actual_per_day - eligible_per_day) * days, 2)
            s.room_rent_excess += excess
            if head == "room":                 # IRDAI: ICU is capped but not used for proportionate deduction
                ratios.append(eligible_per_day / actual_per_day)
            deds.append(Deduction(reason=f"{head.upper()} rent ₹{actual_per_day:,.0f}/day > eligible ₹{eligible_per_day:,.0f}/day ({pct}% of SI) × {days:g} days",
                                  amount=excess, clause="Room rent limit clause"))
    if ratios and pol.get("proportionate_deduction", True):
        ratio = min(ratios)
        assoc = sum(i.amount for i in c.bill_items if i.head in PROPORTIONATE_HEADS and id(i) not in np_ids)
        s.proportionate_deduction = round(assoc * (1 - ratio), 2)
        if s.proportionate_deduction:
            deds.append(Deduction(reason=f"Proportionate deduction on associated medical expenses (doctor/surgery/nursing ₹{assoc:,.0f} × {1 - ratio:.2%})",
                                  amount=s.proportionate_deduction, clause="Proportionate deduction clause (not on pharmacy/consumables/implants/diagnostics)"))
    s.room_rent_excess = round(s.room_rent_excess, 2)
    admissible = billed - s.non_payable - s.room_rent_excess - s.proportionate_deduction
    copay_pct = pol.get("copay_pct", 0)
    if copay_pct:
        s.copay = round(admissible * copay_pct / 100, 2)
        deds.append(Deduction(reason=f"Co-payment {copay_pct}%", amount=s.copay, clause="Co-payment clause"))
    after = admissible - s.copay
    cap = min(after, claimed, si - pol.get("claims_paid", 0))
    if cap < after - 0.01:
        s.sum_insured_cap = round(after - cap, 2)
        deds.append(Deduction(reason="Capped at amount claimed / remaining sum insured", amount=s.sum_insured_cap, clause="Sum insured clause"))
    s.payable = round(max(cap, 0), 2)
    s.deductions = deds
    return s


# %% Fraud red-flag scoring (explainable, rule-based)
def claim_fingerprint(c: Claim) -> str:
    key = f"{(c.policy_no or '').upper()}|{_norm(c.patient_name)}|{c.admission_date}|{_norm(c.hospital_name)}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def fraud_score(c: Claim, history: Optional[List[dict]] = None) -> Dict[str, Any]:
    """0-100 score with reasons. Not an ML model: every flag is explainable (required for SIU review)."""
    pol = get_policy(c.policy_no) or {}
    flags: List[Dict[str, Any]] = []
    adm, first = _d(c.admission_date), _d(pol.get("first_inception"))
    total = c.total_billed or sum(i.amount for i in c.bill_items)

    def add(pts, why): flags.append({"points": pts, "reason": why})

    if adm and first and 0 <= (adm - first).days <= 90:
        add(25, f"Claim within {(adm - first).days} days of policy inception (early claim)")
    if total and total >= 10000 and total % 10000 == 0:
        add(10, f"Bill total is a round figure (₹{total:,.0f})")
    if pol and total and total > 0.8 * pol["sum_insured"]:
        add(15, "Bill exceeds 80% of the sum insured")
    if not c.hospital_rohini_id:
        add(15, "No hospital ROHINI ID; hospital may be unregistered")
    if not c.doctor_reg_no:
        add(10, "Doctor registration number missing")
    a, d = adm, _d(c.discharge_date)
    if a and d and any((_d(i.date) and not (a <= _d(i.date) <= d)) for i in c.bill_items):
        add(15, "Bill lines dated outside the hospital stay")
    fp = claim_fingerprint(c)
    hist = history if history is not None else load_history()
    if any(h.get("fingerprint") == fp for h in hist):
        add(30, "A claim with the same policy, patient, admission date and hospital already exists (duplicate)")
    los = length_of_stay(c)
    if los and los > 0 and total and c.treatment_type == "medical" and total / los > 60000:
        add(10, f"Per-day cost of ₹{total / los:,.0f} in a medical (non-surgical) case is unusually high")
    score = min(100, sum(f["points"] for f in flags))
    return {"score": score, "band": "HIGH" if score >= FRAUD_THRESHOLD else ("MEDIUM" if score >= 25 else "LOW"),
            "flags": flags, "fingerprint": fp}


def rule_decision(issues: List[dict], settlement: dict, fraud: dict) -> str:
    """Deterministic decision ladder: REJECT > INVESTIGATE > QUERY > PARTIAL_APPROVE > APPROVE."""
    codes = {i["code"] for i in issues if i["severity"] == "error"}
    if codes & REJECT_CODES:
        return "REJECT"
    if fraud["score"] >= FRAUD_THRESHOLD:
        return "INVESTIGATE"
    if codes:
        return "QUERY"
    if settlement["payable"] < settlement["amount_claimed"] - 1:
        return "PARTIAL_APPROVE"
    return "APPROVE"


# %% RAG — policy wordings + IRDAI regulations knowledge base
def _load_docs(docs_dir: Path = DOCS_DIR):
    from langchain_core.documents import Document
    docs = []
    for f in sorted(Path(docs_dir).glob("**/*")):
        if f.suffix.lower() == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader
            docs += PyPDFLoader(str(f)).load()
        elif f.suffix.lower() in {".txt", ".md"}:
            docs.append(Document(page_content=f.read_text(encoding="utf-8"), metadata={"source": f.name}))
    return docs


def _split(docs, size=900, overlap=150):
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        return RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap).split_documents(docs)
    except ImportError:                       # minimal fallback: paragraph chunks
        from langchain_core.documents import Document
        out = []
        for d in docs:
            for para in re.split(r"\n\s*\n", d.page_content):
                if para.strip():
                    out.append(Document(page_content=para.strip(), metadata=dict(d.metadata)))
        return out


def build_index(docs_dir: Path = DOCS_DIR, index_dir: Path = INDEX_DIR):
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings
    docs = _load_docs(docs_dir)
    if not docs:
        raise RuntimeError(f"{docs_dir} is empty; add policy wording PDFs and the IRDAI master circular here.")
    vs = FAISS.from_documents(_split(docs), OpenAIEmbeddings(model=EMBED_MODEL))
    vs.save_local(str(index_dir))
    global _VS; _VS = vs
    return vs


_VS = None
_CHUNKS = None


def _keyword_retrieve(q: str, k: int = 4):
    """Offline retriever (no embeddings): TF-IDF-style term overlap. For demos and tests only."""
    global _CHUNKS
    if _CHUNKS is None:
        _CHUNKS = _split(_load_docs())
    toks = lambda s: [t for t in re.findall(r"[a-z0-9]+", s.lower()) if len(t) > 2]
    N = len(_CHUNKS) or 1
    df: Dict[str, int] = {}
    for ch in _CHUNKS:
        for t in set(toks(ch.page_content)):
            df[t] = df.get(t, 0) + 1
    qt = set(toks(q))
    scored = sorted(_CHUNKS, key=lambda ch: -sum(math.log(1 + N / df[t]) for t in qt & set(toks(ch.page_content))))
    return scored[:k]


def retrieve(q: str, k: int = 4):
    global _VS
    if not llm_available():
        return _keyword_retrieve(q, k)
    if _VS is None:
        from langchain_community.vectorstores import FAISS
        from langchain_openai import OpenAIEmbeddings
        if Path(INDEX_DIR).exists():
            _VS = FAISS.load_local(str(INDEX_DIR), OpenAIEmbeddings(model=EMBED_MODEL), allow_dangerous_deserialization=True)
        else:
            build_index()
    return _VS.similarity_search(q, k=k)


def rag_answer(question: str, k: int = 4) -> dict:
    docs = retrieve(question, k)
    sources = [{"source": d.metadata.get("source"), "page": d.metadata.get("page")} for d in docs]
    ctx = "\n\n".join(f"[{i+1}] ({d.metadata.get('source','?')} p{d.metadata.get('page','-')}) {d.page_content}" for i, d in enumerate(docs))
    if not llm_available():
        best = docs[0].page_content.strip() if docs else "No context found."
        return {"answer": f"(offline extractive answer) {best[:700]}", "sources": sources}
    msg = ("You are a health-insurance claims expert (India, IRDAI). Answer ONLY from the context, citing like [1]. "
           "If the context is insufficient, say so — don't guess.\n\nCONTEXT:\n" + ctx + f"\n\nQUESTION: {question}")
    return {"answer": get_llm().invoke(msg).content, "sources": sources}


# %% Multi-Agent System (LangGraph): Intake -> Validator -> [Fraud Investigator] -> Policy Researcher -> Adjudicator
class State(TypedDict, total=False):
    file_path: str
    raw_text: str
    claim: dict
    issues: List[dict]
    settlement: dict
    fraud: dict
    rule_decision: str
    investigation: dict
    policy_notes: str
    report: dict


def intake_agent(s: State) -> State:
    """Agent 1: OCR + structured extraction."""
    text = s.get("raw_text") or ocr_document(s["file_path"])
    return {"raw_text": text, "claim": extract_claim(text).model_dump()}


def validator_agent(s: State) -> State:
    """Agent 2 (deterministic): policy rules + settlement calculator + fraud score + rule decision."""
    c = Claim(**s["claim"])
    issues = [i.model_dump() for i in validate_claim(c)]
    settle = compute_settlement(c).model_dump()
    fraud = fraud_score(c)
    return {"issues": issues, "settlement": settle, "fraud": fraud, "rule_decision": rule_decision(issues, settle, fraud)}


def fraud_investigator_agent(s: State) -> State:
    """Agent 3 (conditional, HIGH fraud band only): investigation plan for the SIU."""
    flags = [f["reason"] for f in s["fraud"]["flags"]]
    checklist = ["Hospital visit: cross-check indoor case papers, nursing notes and OT register",
                 "Interview patient/attendant and verify admission time (CCTV / gate register)",
                 "Reconcile pharmacy bills against original GST invoices",
                 "Verify the treating doctor's registration with the NMC / State council",
                 "Verify actual payment from bank statements / payment receipts"]
    if llm_available():
        try:
            resp = get_llm().invoke("You are an Indian health insurance SIU (fraud) investigator. Given red flags and claim JSON, "
                                    "give a JSON list (max 6) of specific verification steps in plain English. Return ONLY a JSON list.\n"
                                    f"FLAGS: {flags}\nCLAIM: {json.dumps(s['claim'])}")
            checklist = json.loads(re.sub(r"^```(?:json)?|```$", "", resp.content.strip(), flags=re.M).strip())
        except Exception:
            pass
    return {"investigation": {"red_flags": flags, "checklist": checklist}}


def policy_researcher_agent(s: State) -> State:
    """Agent 4 (RAG): finds the policy wording / IRDAI clause behind each issue and deduction."""
    c = Claim(**s["claim"])
    queries = [f"{i['code']}: {i['message']}" for i in s["issues"] if i["severity"] == "error"][:3]
    queries += [d["reason"] for d in s["settlement"]["deductions"]][:2]
    queries.append(f"Is treatment for {c.diagnosis or 'this illness'} ({c.procedure or 'medical management'}) covered or excluded? Any waiting period?")
    notes = []
    for q in queries:
        try:
            r = rag_answer(q)
            src = ", ".join(sorted({str(x["source"]) for x in r["sources"]}))
            notes.append(f"Q: {q}\nA: {r['answer']}\nSources: {src}")
        except Exception as e:                       # keep the pipeline running even without an index
            notes.append(f"Q: {q}\nA: (RAG unavailable: {e})")
            break
    return {"policy_notes": "\n\n".join(notes)}


ADJUDICATOR_PROMPT = """You are a senior health-insurance claims adjudicator in India (IRDAI-regulated). Write in clear, professional English.
Rule engine decision: {decision}. Payable amount (computed by code, DO NOT change): Rs {payable}.
Given claim, issues, settlement, fraud flags and policy notes, return JSON with keys:
summary (2-3 lines), rationale (list of points, each citing a policy clause / IRDAI rule from notes where possible),
queries (list of documents/clarifications needed, empty if none),
letter (short, polite letter to the claimant/hospital explaining the decision, amount and deductions; mention grievance route: insurer GRO -> Bima Bharosa / Insurance Ombudsman for rejections/deductions),
llm_decision (one of APPROVE, PARTIAL_APPROVE, QUERY, INVESTIGATE, REJECT — your independent view).
Don't invent clauses beyond the notes. Return ONLY JSON.

CLAIM: {claim}
ISSUES: {issues}
SETTLEMENT: {settlement}
FRAUD: {fraud}
POLICY NOTES: {notes}"""

DECISION_ORDER = ["APPROVE", "PARTIAL_APPROVE", "QUERY", "INVESTIGATE", "REJECT"]   # left = lenient, right = strict


def _template_report(s: State) -> dict:
    d, st = s["rule_decision"], s["settlement"]
    errs = [i for i in s["issues"] if i["severity"] == "error"]
    c = s["claim"]
    lines = {"APPROVE": f"Claim approved in full. Payable amount: ₹{st['payable']:,.2f}.",
             "PARTIAL_APPROVE": f"Claim partially approved. Claimed ₹{st['amount_claimed']:,.2f}, payable ₹{st['payable']:,.2f}.",
             "QUERY": "A query has been raised on this claim; the documents/clarifications listed below are required.",
             "INVESTIGATE": f"Claim referred for investigation due to fraud red flags (score {s['fraud']['score']}).",
             "REJECT": "Claim is not payable under the policy terms."}
    rationale = [f"{i['code']}: {i['message']}" for i in errs] + [f"Deduction ₹{x['amount']:,.2f} — {x['reason']}" for x in st["deductions"]]
    queries = [i["fix"] for i in errs if i.get("fix") and d == "QUERY"]
    letter = (f"Dear {c.get('patient_name') or 'Claimant'},\n\nStatus of your hospitalisation claim under policy {c.get('policy_no')} "
              f"at {c.get('hospital_name')} ({c.get('admission_date')} to {c.get('discharge_date')}): {d}.\n{lines[d]}\n"
              + ("".join(f"- {r}\n" for r in rationale)) +
              "\nIf you are not satisfied with this decision, you may write to the insurer's Grievance Redressal Officer. If the "
              "grievance is not resolved within 30 days, you may approach the Bima Bharosa portal or the Insurance Ombudsman.\n\nRegards,\nClaims Team")
    return {"summary": lines[d], "rationale": rationale, "queries": queries, "letter": letter, "llm_decision": d}


def adjudicator_agent(s: State) -> State:
    """Agent 5: final decision + explanation + letter. Hard guards: the amount comes from code; the LLM can only make the decision stricter."""
    rule = s["rule_decision"]
    if llm_available():
        resp = get_llm().invoke(ADJUDICATOR_PROMPT.format(
            decision=rule, payable=s["settlement"]["payable"], claim=json.dumps(s["claim"]), issues=json.dumps(s["issues"]),
            settlement=json.dumps(s["settlement"]), fraud=json.dumps(s["fraud"]), notes=s.get("policy_notes", "")))
        txt = re.sub(r"^```(?:json)?|```$", "", resp.content.strip(), flags=re.M).strip()
        try:
            rep = json.loads(txt)
        except json.JSONDecodeError:
            rep = {**_template_report(s), "summary": txt[:400]}
    else:
        rep = _template_report(s)
    llm_d = rep.get("llm_decision") if rep.get("llm_decision") in DECISION_ORDER else rule
    # Guard 1: the LLM cannot be more lenient than the rule decision. Guard 2: the LLM cannot REJECT on its own (it can escalate to QUERY at most).
    final = rule
    if DECISION_ORDER.index(llm_d) > DECISION_ORDER.index(rule):
        final = llm_d if llm_d != "REJECT" else "QUERY"
    rep["decision"] = final
    rep["rule_decision"] = rule
    rep["payable_amount"] = 0.0 if final in ("REJECT", "QUERY", "INVESTIGATE") else s["settlement"]["payable"]
    sub = _d(s["claim"].get("submission_date")) or _d(s["claim"].get("discharge_date"))
    rep["settle_by"] = (sub + dt.timedelta(days=30)).isoformat() if sub else None   # IRDAI: settle/reject within 30 days of the last document
    if s.get("investigation"):
        rep["investigation"] = s["investigation"]
    return {"report": rep}


def route_after_validation(s: State) -> str:
    if any(i["code"] == "POLICY_NOT_FOUND" for i in s["issues"]):
        return "adjudicator"                     # no policy, so RAG adds nothing
    return "fraud_investigator" if s["fraud"]["score"] >= FRAUD_THRESHOLD else "policy_researcher"


def build_graph():
    from langgraph.graph import StateGraph, END
    g = StateGraph(State)
    g.add_node("intake", intake_agent)
    g.add_node("validator", validator_agent)
    g.add_node("fraud_investigator", fraud_investigator_agent)
    g.add_node("policy_researcher", policy_researcher_agent)
    g.add_node("adjudicator", adjudicator_agent)
    g.set_entry_point("intake")
    g.add_edge("intake", "validator")
    g.add_conditional_edges("validator", route_after_validation,
                            {"fraud_investigator": "fraud_investigator", "policy_researcher": "policy_researcher", "adjudicator": "adjudicator"})
    g.add_edge("fraud_investigator", "policy_researcher")
    g.add_edge("policy_researcher", "adjudicator")
    g.add_edge("adjudicator", END)
    return g.compile()


# %% Orchestration — end-to-end
_GRAPH = None
_KEYS = ["claim", "issues", "settlement", "fraud", "policy_notes", "report"]


def analyze_claim(file_path: str) -> dict:
    global _GRAPH
    _GRAPH = _GRAPH or build_graph()
    out = _GRAPH.invoke({"file_path": file_path})
    return {k: out.get(k) for k in _KEYS}


def analyze_text(raw_text: str) -> dict:
    """Skips OCR, for already-extracted text (e.g. email body / TPA portal)."""
    global _GRAPH
    _GRAPH = _GRAPH or build_graph()
    out = _GRAPH.invoke({"raw_text": raw_text})
    return {k: out.get(k) for k in _KEYS}


def analyze_batch(paths: List[str]) -> List[dict]:
    """Multiple claims, e.g. a TPA's daily inward queue."""
    return [{"file": p, **analyze_claim(p)} for p in paths]


if __name__ == "__main__":
    import sys
    print(json.dumps(analyze_claim(sys.argv[1]), indent=2, ensure_ascii=False, default=str))
