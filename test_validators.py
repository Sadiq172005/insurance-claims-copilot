"""Offline tests that run without an LLM/API key: python test_validators.py  (or: pytest -q)"""
import datetime as dt
from pipeline import (Claim, BillItem, validate_claim, compute_settlement, fraud_score, rule_decision, add_months,
                      rule_based_extract, find_member, get_policy)

TODAY = dt.date(2026, 10, 19)


def claim(**kw):
    base = dict(claim_type="reimbursement", policy_no="HLT/2023/000101", patient_name="Sunita Kumar", hospital_name="Deccan Care Hospital",
                hospital_rohini_id="8900080123456", admission_date="2026-08-12", discharge_date="2026-08-15", intimation_date="2026-08-12",
                submission_date="2026-08-20", diagnosis="Acute appendicitis", icd10_code="K35.80", procedure="Laparoscopic appendectomy",
                treatment_type="surgical", doctor_name="Dr. S. Reddy", doctor_reg_no="TSMC/2011/04521",
                bill_items=[BillItem(date="2026-08-12", description="Room rent", head="room", quantity=3, rate=4500, amount=13500),
                            BillItem(date="2026-08-13", description="Surgeon fee", head="surgery", quantity=1, rate=45000, amount=45000),
                            BillItem(date="2026-08-15", description="Pharmacy", head="pharmacy", quantity=1, rate=18500, amount=18500)],
                total_billed=77000, amount_claimed=77000)
    base.update(kw)
    return Claim(**base)


def codes(c):
    return {i.code for i in validate_claim(c, TODAY)}


def decide(c):
    issues = [i.model_dump() for i in validate_claim(c, TODAY)]
    return rule_decision(issues, compute_settlement(c).model_dump(), fraud_score(c, history=[]))


def test_clean_claim():
    c = claim()
    assert codes(c) == set(), codes(c)
    assert compute_settlement(c).payable == 77000
    assert decide(c) == "APPROVE"


def test_policy_and_member():
    assert "POLICY_NOT_FOUND" in codes(claim(policy_no="HLT/9999/000000"))
    assert "PATIENT_NOT_INSURED" in codes(claim(patient_name="Rahul Verma"))
    assert find_member(get_policy("HLT/2023/000101"), "Sunita K") is not None          # initial match
    assert "POLICY_INACTIVE" in codes(claim(policy_no="HLT/2022/000512", patient_name="Farhan Ali"))


def test_waiting_periods():
    p = dict(policy_no="HLT/2026/000245", patient_name="Priya Sharma")
    assert "INITIAL_WAITING" in codes(claim(**p, admission_date="2026-09-15", discharge_date="2026-09-18"))
    assert "INITIAL_WAITING" not in codes(claim(**p, admission_date="2026-09-15", discharge_date="2026-09-18", is_accident=True))
    v = dict(policy_no="HLT/2025/000377", patient_name="Venkat Rao")
    assert "PED_WAITING" in codes(claim(**v, diagnosis="Hypertensive heart disease", icd10_code="I11.0"))
    assert "PED_WAITING" not in codes(claim(**v, diagnosis="Pneumonia", icd10_code="J18.9"))
    assert "SPECIFIC_DISEASE_WAITING" in codes(claim(**v, diagnosis="Senile cataract", icd10_code="H25.9", treatment_type="daycare",
                                                     discharge_date="2026-08-12"))
    assert add_months(dt.date(2024, 1, 31), 1) == dt.date(2024, 2, 29)


def test_exclusion_and_accident_carveout():
    assert "EXCLUSION" in codes(claim(diagnosis="Cosmetic correction", procedure="Cosmetic rhinoplasty", icd10_code="Z41.1"))
    assert "EXCLUSION" not in codes(claim(diagnosis="Nasal bone fracture", procedure="Rhinoplasty after accident", icd10_code="S02.2",
                                          is_accident=True))


def test_dates_and_bill_math():
    assert "DATE_ORDER" in codes(claim(discharge_date="2026-08-10"))
    assert "MIN_HOSPITALIZATION" in codes(claim(discharge_date="2026-08-12", treatment_type="medical"))
    assert "MIN_HOSPITALIZATION" not in codes(claim(discharge_date="2026-08-12", treatment_type="daycare"))
    assert "BILL_TOTAL_MISMATCH" in codes(claim(total_billed=90000, amount_claimed=90000))
    assert "CLAIM_EXCEEDS_BILL" in codes(claim(amount_claimed=80000))
    assert "LATE_SUBMISSION" in codes(claim(submission_date="2026-10-15"))
    assert "NON_NETWORK_CASHLESS" in codes(claim(claim_type="cashless", hospital_name="Lotus Multispeciality Clinic"))
    assert "ICD_FORMAT" in codes(claim(icd10_code="123.4"))


def test_room_rent_proportionate():
    items = [BillItem(head="room", description="Deluxe", quantity=4, rate=9000, amount=36000),
             BillItem(head="doctor", description="Visits", quantity=4, rate=2000, amount=8000),
             BillItem(head="nursing", description="Nursing", quantity=4, rate=1500, amount=6000),
             BillItem(head="pharmacy", description="Drugs", amount=22000),
             BillItem(head="investigation", description="Labs", amount=15500)]
    s = compute_settlement(claim(bill_items=items, total_billed=87500, amount_claimed=87500))
    assert s.room_rent_excess == 16000
    assert abs(s.proportionate_deduction - 14000 * 4 / 9) < 0.01          # no proportionate deduction on pharmacy/labs
    assert abs(s.payable - (87500 - 16000 - 14000 * 4 / 9)) < 0.01


def test_icu_no_proportionate():
    items = [BillItem(head="icu", description="ICU", quantity=2, rate=15000, amount=30000),
             BillItem(head="doctor", description="Intensivist", quantity=2, rate=3000, amount=6000)]
    s = compute_settlement(claim(bill_items=items, total_billed=36000, amount_claimed=36000))
    assert s.room_rent_excess == 10000 and s.proportionate_deduction == 0      # ICU cap 2% = 10k/day


def test_copay_nonpayable_and_si_cap():
    v = dict(policy_no="HLT/2025/000377", patient_name="Venkat Rao")
    items = [BillItem(head="pharmacy", description="Drugs", amount=50000),
             BillItem(head="consumables", description="Disposable gloves", amount=2000),
             BillItem(head="other", description="Admission kit", amount=1000)]
    s = compute_settlement(claim(**v, bill_items=items, total_billed=53000, amount_claimed=53000))
    assert s.non_payable == 3000 and s.copay == 10000 and s.payable == 40000
    big = [BillItem(head="surgery", description="Surgery", amount=700000)]
    s = compute_settlement(claim(bill_items=big, total_billed=700000, amount_claimed=700000))
    assert s.payable == 500000                                                # SI cap


def test_fraud_and_decision_ladder():
    p = dict(policy_no="HLT/2026/000245", patient_name="Priya Sharma", admission_date="2026-10-05", discharge_date="2026-10-08",
             hospital_rohini_id=None, doctor_reg_no=None, treatment_type="medical", total_billed=250000, amount_claimed=250000,
             bill_items=[BillItem(date="2026-10-06", head="pharmacy", description="Drugs", amount=250000)])
    f = fraud_score(claim(**p), history=[])
    assert f["score"] >= 50 and f["band"] == "HIGH"
    assert decide(claim(**p)) == "INVESTIGATE"
    c = claim()
    from pipeline import claim_fingerprint
    assert fraud_score(c, history=[{"fingerprint": claim_fingerprint(c)}])["score"] >= 30   # duplicate
    assert decide(claim(policy_no="HLT/2022/000512", patient_name="Farhan Ali")) == "REJECT"
    assert decide(claim(total_billed=90000, amount_claimed=90000)) == "QUERY"


def test_offline_extractor():
    txt = """Policy Number: HLT/2023/000101
Claim Type: Cashless
Patient Name: Sunita Kumar
Date of Admission: 12-08-2026
Date of Discharge: 15-08-2026
Final Diagnosis: Acute appendicitis
ICD-10 Code: K35.80
Treatment Type: Surgical
12-08-2026 Room rent - Single Private AC Room 3 4,500.00 13,500.00
13-08-2026 | OT charges | Surgery 1 | 25,000.00 | 25,000.00
Total Bill Amount: 38,500.00"""
    c = rule_based_extract(txt)
    assert c.claim_type == "cashless" and c.admission_date == "2026-08-12" and c.treatment_type == "surgical"
    assert [i.head for i in c.bill_items] == ["room", "surgery"] and c.bill_items[0].quantity == 3 and c.total_billed == 38500


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t(); print("✓", t.__name__)
    print(f"All {len(tests)} validator tests passed ✅")
