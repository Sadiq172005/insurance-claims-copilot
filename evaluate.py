"""End-to-end evaluation on the samples: compares decisions against expected.json and prints payable amounts.
Runs in offline mode without OPENAI_API_KEY (regex extractor + keyword RAG + template letters).
Run: python evaluate.py            (LLM mode if a key is set)
     OFFLINE_MODE=1 python evaluate.py
"""
import json, sys, time
from pathlib import Path
from pipeline import analyze_claim, llm_available

ROOT = Path(__file__).resolve().parent
exp = json.loads((ROOT / "samples/expected.json").read_text())
print(f"Mode: {'LLM (' + __import__('pipeline').LLM_MODEL + ')' if llm_available() else 'OFFLINE fallbacks'}\n")
ok, rows = 0, []
for fname, e in exp.items():
    t0 = time.time()
    try:
        res = analyze_claim(str(ROOT / "samples" / fname))
        got = res["report"]["decision"]; codes = sorted({i["code"] for i in res["issues"] if i["severity"] != "info"})
        pay = res["report"]["payable_amount"]; fraud = res["fraud"]["score"]
    except Exception as ex:
        got, codes, pay, fraud = f"ERROR {ex}", [], 0, "-"
    hit = got == e["expected_decision"]; ok += hit
    print(f"{'PASS' if hit else 'FAIL'}  {fname:40s} expected={e['expected_decision']:15s} got={got:15s} payable={pay:>11,.2f} "
          f"fraud={fraud!s:>3}  {time.time() - t0:4.1f}s  codes={codes}")
print(f"\n{ok}/{len(exp)} decisions match")
sys.exit(0 if ok == len(exp) else 1)
