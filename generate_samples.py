"""Synthetic health-insurance claim files: each PDF = Claim Form + Discharge Summary + Itemised Final Bill (3 pages).
Each sample tests a different adjudication scenario. All data is FAKE (policies: data/policies.json).
Run: python generate_samples.py  ->  samples/*.pdf + samples/expected.json + data/claims_history.json
"""
import json
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, PageBreak

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "samples"; OUT.mkdir(exist_ok=True)
ss = getSampleStyleSheet()

DEFAULT = dict(claim_type="Reimbursement", rohini="8900080123456", accident="No", treatment="Medical", doctor="Dr. S. Reddy",
               doctor_reg="TSMC/2011/04521", procedure="Conservative medical management", room_type="Single Private AC")


def make(fname, *, policy, patient, age, hospital, adm, dis, diagnosis, icd, items, intimation=None, submission=None,
         printed_total=None, claimed=None, **kw):
    o = {**DEFAULT, **kw}
    total = round(sum(q * r for _, _, _, q, r in items), 2)
    printed_total = total if printed_total is None else printed_total
    claimed = printed_total if claimed is None else claimed
    line = lambda k, v: Paragraph(f"<b>{k}:</b> {v if v not in (None, '') else '-'}", ss["Normal"])
    doc = SimpleDocTemplate(str(OUT / fname), pagesize=A4, topMargin=36, bottomMargin=36, leftMargin=40, rightMargin=40)
    el = [Paragraph("HEALTH INSURANCE CLAIM FORM", ss["Title"]), Paragraph("Part A — To be filled by the insured", ss["Heading3"])]
    el += [line("Policy Number", policy), line("Claim Type", o["claim_type"]), line("Patient Name", patient), line("Patient Age", age),
           line("Hospital Name", hospital), line("Date of Admission", adm), line("Date of Discharge", dis),
           line("Date of Intimation", intimation or adm),
           line("Date of Submission", submission if o["claim_type"] == "Reimbursement" else "-"),
           line("Accident Case", o["accident"]), line("Amount Claimed", f"{claimed:,.2f}"), Spacer(1, 16),
           Paragraph("I declare that the information furnished is true and correct. (Signature of insured)", ss["Italic"]), PageBreak()]
    el += [Paragraph(f"{hospital.upper()}", ss["Title"]), Paragraph("DISCHARGE SUMMARY", ss["Heading2"]),
           line("Hospital ROHINI ID", o["rohini"]), line("Patient Name", patient), line("Date of Admission", adm), line("Date of Discharge", dis),
           line("Final Diagnosis", diagnosis), line("ICD-10 Code", icd), line("Procedure", o["procedure"]),
           line("Treatment Type", o["treatment"]), line("Room Type", o["room_type"]),
           line("Treating Doctor", o["doctor"]), line("Doctor Reg. No", o["doctor_reg"]), Spacer(1, 10),
           Paragraph("Condition at discharge: Stable. Advised follow-up after 7 days.", ss["Normal"]), PageBreak()]
    el += [Paragraph(f"{hospital.upper()}", ss["Title"]), Paragraph("FINAL HOSPITAL BILL (ITEMISED)", ss["Heading2"]),
           Paragraph(f"Patient: {patient} &nbsp;&nbsp; Bill period: {adm} to {dis}", ss["Normal"]), Spacer(1, 8)]
    rows = [["Date", "Description", "Head", "Qty", "Rate", "Amount"]]
    for d, desc, head, q, r in items:
        rows.append([d, desc, head, f"{q:g}", f"{r:,.2f}", f"{q * r:,.2f}"])
    t = Table(rows, colWidths=[62, 210, 70, 30, 70, 75])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .5, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                           ("FONTSIZE", (0, 0), (-1, -1), 8), ("ALIGN", (3, 1), (-1, -1), "RIGHT")]))
    el += [t, Spacer(1, 12), line("Total Bill Amount", f"{printed_total:,.2f}"), line("Amount Paid by Patient", f"{printed_total:,.2f}")]
    doc.build(el)


SUN, DCH, GMC, CHI = "Sunrise Multispeciality Hospital", "Deccan Care Hospital", "Godavari Medical Centre", "Charminar Heart Institute"

SCENARIOS = {
    "01_clean_cashless_approve.pdf": (dict(
        policy="HLT/2023/000101", patient="Sunita Kumar", age=48, hospital=DCH, adm="12-08-2026", dis="15-08-2026",
        claim_type="Cashless", diagnosis="Acute appendicitis", icd="K35.80", treatment="Surgical", procedure="Laparoscopic appendectomy",
        items=[("12-08-2026", "Room rent - Single Private AC", "Room", 3, 4500), ("13-08-2026", "Surgeon fee - appendectomy", "Surgery", 1, 45000),
               ("13-08-2026", "OT charges", "Surgery", 1, 25000), ("13-08-2026", "Anaesthetist fee", "Doctor", 1, 12000),
               ("12-08-2026", "Nursing charges", "Nursing", 3, 1500), ("15-08-2026", "Pharmacy - drugs and IV fluids", "Pharmacy", 1, 18500),
               ("12-08-2026", "CBC, USG abdomen, LFT", "Investigation", 1, 9800)]),
        "APPROVE", "Clean claim, no deductions"),
    "02_room_rent_proportionate.pdf": (dict(
        policy="HLT/2023/000101", patient="Ramesh Kumar", age=52, hospital=SUN, adm="03-09-2026", dis="07-09-2026", submission="20-09-2026",
        diagnosis="Dengue fever with thrombocytopenia", icd="A90", room_type="Deluxe Suite",
        items=[("03-09-2026", "Room rent - Deluxe Suite", "Room", 4, 9000), ("03-09-2026", "Physician visits", "Doctor", 4, 2000),
               ("03-09-2026", "Nursing charges", "Nursing", 4, 1500), ("07-09-2026", "Pharmacy - drugs and platelets", "Pharmacy", 1, 22000),
               ("04-09-2026", "NS1, platelet counts, LFT", "Investigation", 1, 15500)]),
        "PARTIAL_APPROVE", "ROOM_RENT excess + proportionate deduction (doctor/nursing)"),
    "03_initial_waiting_reject.pdf": (dict(
        policy="HLT/2026/000245", patient="Priya Sharma", age=29, hospital=SUN, adm="15-09-2026", dis="18-09-2026", submission="25-09-2026",
        diagnosis="Typhoid fever", icd="A01.0",
        items=[("15-09-2026", "Room rent - Twin sharing", "Room", 3, 2500), ("15-09-2026", "Physician visits", "Doctor", 3, 1500),
               ("18-09-2026", "Pharmacy", "Pharmacy", 1, 9500), ("15-09-2026", "Widal, blood culture", "Investigation", 1, 4200)]),
        "REJECT", "INITIAL_WAITING (30 days, non-accident)"),
    "04_ped_waiting_reject.pdf": (dict(
        policy="HLT/2025/000377", patient="Venkat Rao", age=67, hospital=CHI, adm="02-08-2026", dis="06-08-2026", submission="14-08-2026",
        diagnosis="Hypertensive heart disease with acute decompensation", icd="I11.0", doctor="Dr. A. Menon", doctor_reg="TSMC/2004/01187",
        items=[("02-08-2026", "ICU charges", "ICU", 2, 9000), ("04-08-2026", "Room rent - Single Private AC", "Room", 2, 4800),
               ("02-08-2026", "Cardiologist visits", "Doctor", 4, 3000), ("06-08-2026", "Pharmacy", "Pharmacy", 1, 31000),
               ("02-08-2026", "ECHO, ECG, troponin, BNP", "Investigation", 1, 16500)]),
        "REJECT", "PED_WAITING (declared hypertension, 36 months)"),
    "05_specific_disease_waiting_reject.pdf": (dict(
        policy="HLT/2025/000377", patient="Lakshmi Rao", age=63, hospital=GMC, adm="15-07-2026", dis="15-07-2026", submission="22-07-2026",
        diagnosis="Senile cataract, right eye", icd="H25.9", treatment="Daycare", procedure="Phacoemulsification with IOL implant", room_type="Daycare",
        items=[("15-07-2026", "Surgeon fee - phaco", "Surgery", 1, 18000), ("15-07-2026", "IOL lens implant", "Implants", 1, 22000),
               ("15-07-2026", "OT and daycare charges", "Surgery", 1, 8000), ("15-07-2026", "Pharmacy", "Pharmacy", 1, 3500)]),
        "REJECT", "SPECIFIC_DISEASE_WAITING (cataract, 24 months)"),
    "06_policy_lapsed_reject.pdf": (dict(
        policy="HLT/2022/000512", patient="Farhan Ali", age=41, hospital=SUN, adm="05-08-2026", dis="08-08-2026", submission="18-08-2026",
        diagnosis="Acute bronchitis", icd="J20.9",
        items=[("05-08-2026", "Room rent - Single Private AC", "Room", 3, 4000), ("05-08-2026", "Physician visits", "Doctor", 3, 1500),
               ("08-08-2026", "Pharmacy", "Pharmacy", 1, 12500), ("05-08-2026", "Chest X-ray, CBC", "Investigation", 1, 3800)]),
        "REJECT", "POLICY_INACTIVE (expired 30-06-2026)"),
    "07_bill_total_mismatch_query.pdf": (dict(
        policy="HLT/2023/000101", patient="Kiran Kumar", age=21, hospital=SUN, adm="20-08-2026", dis="23-08-2026", submission="30-08-2026",
        accident="Yes", diagnosis="Fracture of shaft of radius (road traffic accident)", icd="S52.3", treatment="Surgical",
        procedure="ORIF with plating", printed_total=98000,
        items=[("20-08-2026", "Room rent - Single Private AC", "Room", 3, 4500), ("21-08-2026", "Orthopaedic surgeon fee", "Surgery", 1, 30000),
               ("21-08-2026", "OT charges", "Surgery", 1, 15000), ("21-08-2026", "Titanium plate and screws", "Implants", 1, 18000),
               ("23-08-2026", "Pharmacy", "Pharmacy", 1, 7500), ("20-08-2026", "X-ray forearm, pre-op tests", "Investigation", 1, 5000)]),
        "QUERY", "BILL_TOTAL_MISMATCH (items 89,000 vs printed 98,000)"),
    "08_copay_nonpayable_partial.pdf": (dict(
        policy="HLT/2025/000377", patient="Venkat Rao", age=67, hospital=GMC, adm="10-09-2026", dis="15-09-2026", submission="22-09-2026",
        diagnosis="Community acquired pneumonia", icd="J18.9", doctor="Dr. K. Prasad", doctor_reg="TSMC/2008/02234",
        items=[("10-09-2026", "Room rent - Single Private AC", "Room", 5, 4000), ("10-09-2026", "Physician visits", "Doctor", 5, 2500),
               ("10-09-2026", "Nursing charges", "Nursing", 5, 1200), ("15-09-2026", "Pharmacy - antibiotics", "Pharmacy", 1, 28000),
               ("11-09-2026", "Chest CT, cultures, CBC", "Investigation", 1, 14000), ("10-09-2026", "Disposable gloves and masks", "Consumables", 1, 2400),
               ("10-09-2026", "Admission kit", "Other", 1, 1500), ("10-09-2026", "Attendant food charges", "Other", 1, 3500)]),
        "PARTIAL_APPROVE", "20% co-pay + non-payable items (gloves, admission kit, attendant)"),
    "09_fraud_red_flags_investigate.pdf": (dict(
        policy="HLT/2026/000245", patient="Priya Sharma", age=29, hospital="Sai Krishna Nursing Home", adm="05-10-2026", dis="08-10-2026",
        submission="12-10-2026", diagnosis="Acute gastroenteritis with dehydration", icd="A09", rohini="", doctor="Dr. M. Rao", doctor_reg="",
        items=[("05-10-2026", "Room rent - Single room", "Room", 3, 3000), ("05-10-2026", "Physician visits", "Doctor", 3, 5000),
               ("05-10-2026", "Nursing charges", "Nursing", 3, 2000), ("08-10-2026", "Pharmacy", "Pharmacy", 1, 98000),
               ("05-10-2026", "Lab investigations", "Investigation", 1, 72000), ("06-10-2026", "Medical procedures and monitoring", "Other", 1, 50000)]),
        "INVESTIGATE", "Early claim + round total + >80% SI + no ROHINI + no doctor reg + high per-day cost"),
    "10_cosmetic_exclusion_reject.pdf": (dict(
        policy="HLT/2023/000101", patient="Kiran Kumar", age=21, hospital=SUN, adm="02-09-2026", dis="03-09-2026", submission="10-09-2026",
        diagnosis="Nasal hump - cosmetic correction", icd="Z41.1", treatment="Surgical", procedure="Cosmetic rhinoplasty",
        items=[("02-09-2026", "Room rent - Single Private AC", "Room", 1, 4500), ("02-09-2026", "Plastic surgeon fee", "Surgery", 1, 85000),
               ("02-09-2026", "OT charges", "Surgery", 1, 20000), ("03-09-2026", "Pharmacy", "Pharmacy", 1, 6500)]),
        "REJECT", "EXCLUSION (cosmetic surgery, Excl. 08)"),
    "11_cashless_non_network_query.pdf": (dict(
        policy="HLT/2023/000101", patient="Sunita Kumar", age=48, hospital="Lotus Multispeciality Clinic", adm="01-10-2026", dis="03-10-2026",
        claim_type="Cashless", diagnosis="Acute gastroenteritis", icd="A09", rohini="8900080765432",
        items=[("01-10-2026", "Room rent - Single Private AC", "Room", 2, 4000), ("01-10-2026", "Physician visits", "Doctor", 2, 1500),
               ("03-10-2026", "Pharmacy - IV fluids", "Pharmacy", 1, 8500), ("01-10-2026", "Stool and blood tests", "Investigation", 1, 3200)]),
        "QUERY", "NON_NETWORK_CASHLESS -> reimbursement route"),
    "12_duplicate_claim_investigate.pdf": (dict(
        policy="HLT/2023/000101", patient="Ramesh Kumar", age=52, hospital=DCH, adm="12-05-2026", dis="16-05-2026", submission="02-10-2026",
        intimation="12-05-2026", diagnosis="Viral pneumonia", icd="J12.9", doctor_reg="",
        items=[("12-05-2026", "Room rent - Single Private AC", "Room", 4, 5000), ("12-05-2026", "Physician visits", "Doctor", 4, 2500),
               ("12-05-2026", "Nursing charges", "Nursing", 4, 1500), ("16-05-2026", "Pharmacy", "Pharmacy", 1, 52000),
               ("13-05-2026", "HRCT chest, cultures", "Investigation", 1, 32000)]),
        "INVESTIGATE", "Duplicate of already-paid claim + round total + doctor reg missing (+ LATE_SUBMISSION warning)"),
}


def seed_history():
    """Seed an old 'already paid' claim into history so sample 12 is detected as a duplicate."""
    import sys
    sys.path.insert(0, str(ROOT))
    from pipeline import Claim, claim_fingerprint
    c = Claim(policy_no="HLT/2023/000101", patient_name="Ramesh Kumar", admission_date="2026-05-12", hospital_name=DCH)
    hist = [{"claim_id": "CLM-2026-004417", "policy_no": c.policy_no, "patient": c.patient_name, "admission_date": c.admission_date,
             "hospital": c.hospital_name, "status": "PAID", "paid_amount": 118400, "paid_on": "2026-06-02", "fingerprint": claim_fingerprint(c)}]
    (ROOT / "data" / "claims_history.json").write_text(json.dumps(hist, indent=2))


if __name__ == "__main__":
    exp = {}
    for fname, (kw, verdict, why) in SCENARIOS.items():
        make(fname, **kw); exp[fname] = {"expected_decision": verdict, "expected_flags": why}
    # Scanned (image-only) PDF to test the OCR path: rasterise sample 01 with slight blur/rotation
    try:
        from pdf2image import convert_from_path
        from PIL import ImageFilter
        pages = [p.convert("L").rotate(0.6, expand=True, fillcolor=255).filter(ImageFilter.GaussianBlur(0.5))
                 for p in convert_from_path(str(OUT / "01_clean_cashless_approve.pdf"), dpi=200)]
        pages[0].save(OUT / "13_scanned_clean_approve.pdf", save_all=True, append_images=pages[1:], resolution=200)
        pages[0].save(OUT / "13b_scanned_claim_form_page.png")
        exp["13_scanned_clean_approve.pdf"] = {"expected_decision": "APPROVE", "expected_flags": "OCR path (image-only PDF)"}
    except Exception as e:
        print("Scanned sample skip:", e)
    seed_history()
    (OUT / "expected.json").write_text(json.dumps(exp, indent=2, ensure_ascii=False))
    print(f"{len(exp)} samples -> {OUT}/  (+ data/claims_history.json)")
