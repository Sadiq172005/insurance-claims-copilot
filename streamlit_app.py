"""Streamlit UI — run: streamlit run streamlit_app.py
By default the pipeline is imported directly. Enter an API URL in the sidebar to use the deployed FastAPI service (AWS)."""
import os, tempfile
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Claims Copilot", page_icon="🏥", layout="wide")
ROOT = Path(__file__).resolve().parent
DECISION_UI = {"APPROVE": ("✅ Approve", "success"), "PARTIAL_APPROVE": ("🟡 Partial approve", "warning"), "QUERY": ("📨 Query raised", "info"),
               "INVESTIGATE": ("🕵️ Send to SIU / investigate", "error"), "REJECT": ("⛔ Reject", "error")}
SEV_ICON = {"error": "🔴 error", "warning": "🟠 warning", "info": "🔵 info"}

with st.sidebar:
    st.title("🏥 Claims Copilot")
    api_url = st.text_input("API URL (optional)", placeholder="https://<app-runner-url>", help="Leave empty to use the local pipeline")
    if not api_url and not os.getenv("OPENAI_API_KEY"):
        key = st.text_input("OPENAI_API_KEY (optional)", type="password", help="Without a key, offline mode is used (regex extractor + keyword RAG)")
        if key:
            os.environ["OPENAI_API_KEY"] = key
    st.caption("Sample claims: `samples/` (python generate_samples.py). All data is synthetic.")
    st.caption("⚠️ Decision-support tool. The final decision rests with a licensed claims officer.")


def run_analyze(name: str, data: bytes) -> dict:
    if api_url:
        import requests
        r = requests.post(api_url.rstrip("/") + "/analyze", files={"file": (name, data)}, timeout=300)
        r.raise_for_status(); return r.json()
    from pipeline import analyze_claim
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(name).suffix) as f:
        f.write(data)
    try:
        return analyze_claim(f.name)
    finally:
        os.unlink(f.name)


def inr(x):
    return f"₹{(x or 0):,.2f}"


def render(res: dict):
    rep, issues, c, st_, fr = res["report"], res["issues"], res["claim"], res["settlement"], res["fraud"]
    label, kind = DECISION_UI.get(rep.get("decision"), (rep.get("decision", "?"), "info"))
    getattr(st, kind)(f"**{label}** — {rep.get('summary', '')}")
    m = st.columns(5)
    m[0].metric("Claimed", inr(st_["amount_claimed"]))
    m[1].metric("Payable", inr(rep.get("payable_amount")))
    m[2].metric("Deductions", inr(sum(d["amount"] for d in st_["deductions"])))
    m[3].metric("Fraud score", f"{fr['score']}/100", fr["band"], delta_color="off")
    m[4].metric("Settle by (IRDAI 30d)", rep.get("settle_by") or "-")
    left, right = st.columns([3, 2])
    with left:
        if issues:
            st.subheader("Rule-check findings")
            st.dataframe(pd.DataFrame([{"Severity": SEV_ICON[i["severity"]], "Code": i["code"], "Finding": i["message"], "Fix": i.get("fix") or ""}
                                       for i in issues]), width="stretch", hide_index=True)
        else:
            st.success("No issues found in the policy rule checks.")
        if st_["deductions"]:
            st.subheader("Settlement deductions")
            st.dataframe(pd.DataFrame(st_["deductions"]).rename(columns={"reason": "Reason", "amount": "Amount ₹", "clause": "Clause"}),
                         width="stretch", hide_index=True)
    with right:
        st.subheader("Patient & stay")
        st.markdown(f"**{c.get('patient_name')}** · {c.get('policy_no')}  \n{c.get('hospital_name')} · {c.get('claim_type')}  \n"
                    f"{c.get('admission_date')} → {c.get('discharge_date')}  \n{c.get('diagnosis')} (`{c.get('icd10_code')}`)")
        if fr["flags"]:
            st.subheader("Fraud red flags")
            for f in fr["flags"]:
                st.markdown(f"- **+{f['points']}** {f['reason']}")
    if rep.get("rationale"):
        with st.expander("Rationale (policy clauses)", expanded=True):
            for r in rep["rationale"]:
                st.markdown(f"- {r}")
    if rep.get("queries"):
        st.subheader("Queries / documents needed")
        for q in rep["queries"]:
            st.markdown(f"- {q}")
    if rep.get("investigation"):
        with st.expander("SIU investigation checklist", expanded=True):
            for q in rep["investigation"]["checklist"]:
                st.markdown(f"- {q}")
    if rep.get("letter"):
        st.subheader("Letter to claimant / hospital")
        st.code(rep["letter"], language=None)
    with st.expander("Extracted claim data"):
        st.json(c)
    if res.get("policy_notes"):
        with st.expander("Policy research (RAG)"):
            st.text(res["policy_notes"])


tab1, tab2, tab3 = st.tabs(["📄 Single claim", "📚 Batch queue", "💬 Ask policy / IRDAI"])

with tab1:
    up = st.file_uploader("Upload a claim file (PDF/PNG/JPG: claim form + discharge summary + bill)", type=["pdf", "png", "jpg", "jpeg"], key="one")
    samples = sorted((ROOT / "samples").glob("*.pdf")) + sorted((ROOT / "samples").glob("*.png"))
    pick = st.selectbox("...or pick a sample", ["—"] + [p.name for p in samples])
    if st.button("Adjudicate", type="primary"):
        if up:
            name, data = up.name, up.getvalue()
        elif pick != "—":
            name, data = pick, (ROOT / "samples" / pick).read_bytes()
        else:
            st.stop()
        with st.spinner("OCR → extraction → policy rules → settlement → fraud → policy research → adjudication..."):
            try:
                render(run_analyze(name, data))
            except Exception as e:
                st.error(f"Failed: {e}")

with tab2:
    files = st.file_uploader("Multiple claims", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True, key="many")
    use_samples = st.checkbox("Run all sample claims")
    if st.button("Process queue") and (files or use_samples):
        batch = [(f.name, f.getvalue()) for f in (files or [])] + ([(p.name, p.read_bytes()) for p in samples] if use_samples else [])
        rows, results = [], {}
        bar = st.progress(0.0)
        for n, (name, data) in enumerate(batch, 1):
            try:
                res = run_analyze(name, data); results[name] = res
                rows.append({"File": name, "Decision": res["report"].get("decision"), "Claimed ₹": res["settlement"]["amount_claimed"],
                             "Payable ₹": res["report"].get("payable_amount"), "Fraud": res["fraud"]["score"],
                             "Errors": sum(i["severity"] == "error" for i in res["issues"])})
            except Exception as e:
                rows.append({"File": name, "Decision": f"ERROR: {e}"})
            bar.progress(n / len(batch))
        df = pd.DataFrame(rows)
        c1, c2, c3 = st.columns(3)
        c1.metric("Claims", len(df)); c2.metric("Total payable", inr(df.get("Payable ₹", pd.Series(dtype=float)).fillna(0).sum()))
        c3.metric("Flagged for SIU", int((df["Decision"] == "INVESTIGATE").sum()))
        st.bar_chart(df["Decision"].value_counts())
        st.dataframe(df, width="stretch", hide_index=True)
        st.download_button("Download CSV", df.to_csv(index=False), "claims_batch_report.csv", "text/csv")
        for name, res in results.items():
            with st.expander(name):
                render(res)

with tab3:
    q = st.text_input("Question about the policy / IRDAI rules", placeholder="What deduction applies when the room rent limit is exceeded?")
    if st.button("Ask") and q:
        with st.spinner("Searching policy wording + IRDAI notes..."):
            try:
                if api_url:
                    import requests
                    r = requests.post(api_url.rstrip("/") + "/ask", json={"question": q}, timeout=120); r.raise_for_status(); out = r.json()
                else:
                    from pipeline import rag_answer
                    out = rag_answer(q)
                st.markdown(out["answer"])
                st.caption("Sources: " + ", ".join(f"{s['source']} (p{s['page']})" if s.get("page") is not None else str(s["source"]) for s in out["sources"]))
            except Exception as e:
                st.error(f"Failed: {e} — have you added documents to data/policy_docs/?")
