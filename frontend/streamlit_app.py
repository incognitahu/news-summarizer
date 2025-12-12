import streamlit as st
import requests
import json
import re
from typing import Dict, Any, List

st.set_page_config(page_title="News Summarizer (QA-aware)", layout="wide")

# We don't use st.secrets here to avoid StreamlitSecretNotFoundError.
API_URL_DEFAULT = "http://127.0.0.1:8000/summarize"


# ---- Utilities ----
def call_api(api_url: str, payload: Dict[str, Any], timeout: int = 120) -> Dict[str, Any]:
    resp = requests.post(api_url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()

def render_entity(entity: Dict[str, Any]) -> str:
    """Return a small markdown string for an entity row."""
    label = entity.get("label") or "UNK"
    in_src = entity.get("in_source", False)
    qa_supported = entity.get("qa_supported", False)
    qa_score = entity.get("qa_score", 0.0)
    qa_answer = entity.get("qa_answer", "")
    badge = "✅" if (in_src or qa_supported) else "❌"
    return f"{badge} **{entity['text']}** ({label}) — in_source: {in_src} — qa_score: {qa_score:.2f}  \n> {qa_answer}"

def color_sentence_html(sentence: str, flagged: bool) -> str:
    if flagged:
        bg = "#FFE5E5"           # soft red
        border = "2px solid #FF4D4D"
        color = "#660000"
    else:
        bg = "#E6F0FF"           # soft blue
        border = "1px solid #4D88FF"
        color = "#003366"
    return f'<div style="background:{bg}; color:{color}; padding:10px; border-radius:8px; border:{border}; margin-bottom:8px">{sentence}</div>'

# ---- Sidebar: controls ----
st.sidebar.title("Controls")
api_url = st.sidebar.text_input("API URL", API_URL_DEFAULT)
if api_url.endswith("/"):
    api_url = api_url[:-1]

st.sidebar.subheader("Model Selection")
model_options = {
    "DistilBART (fast, dev)": "sshleifer/distilbart-cnn-12-6",
    "BART Large (best quality)": "facebook/bart-large-cnn",
    "T5 Base": "google/flan-t5-base",
    "PEGASUS": "google/pegasus-cnn_dailymail"
}
selected_model = st.sidebar.selectbox("Summarization Model", list(model_options.keys()), index=0)
model_name = model_options[selected_model]

st.sidebar.subheader("Summarization Method")
method_choice = st.sidebar.radio(
    "Method",
    ["Abstractive (Transformer)", "Extractive (TF-IDF)", "Extractive (TextRank)", "Extractive (Lead)"]
)

extractive_method = None
if "TF-IDF" in method_choice:
    extractive_method = "tfidf"
elif "TextRank" in method_choice:
    extractive_method = "textrank"
elif "Lead" in method_choice:
    extractive_method = "lead"

st.sidebar.subheader("Generation")
min_len = st.sidebar.slider("min_length", 10, 120, 40)
max_len = st.sidebar.slider("max_length", 40, 400, 160)
beams = st.sidebar.slider("num_beams", 1, 8, 4)
length_penalty = st.sidebar.slider("length_penalty", 0.5, 2.0, 1.0, step=0.1)
no_repeat_ngram = st.sidebar.slider("no_repeat_ngram_size", 0, 4, 3)

st.sidebar.subheader("Reranker & QA")
use_reranker = st.sidebar.checkbox("Use SBERT reranker", True)
top_k = st.sidebar.slider("top_k (reranker)", 1, 10, 5)
run_qa = st.sidebar.checkbox("Run QA factuality checks", True)
min_qa_score = st.sidebar.slider("QA min score threshold", 0.0, 1.0, 0.25, step=0.01)

st.sidebar.markdown("---")
st.sidebar.markdown("Tip: set `run_qa=false` to speed up demo runs on CPU.")
st.sidebar.markdown("Make sure backend is running and accessible from this machine.")

# ---- Main UI ----
st.title("News Summarizer — Explainable Demo")
st.markdown("Paste article text or HTML on the left, tweak controls in the sidebar, and hit **Summarize**. Flagged sentences indicate potential hallucinations (entity not found + low QA support).")

col1, col2 = st.columns([1, 1.4])

with col1:
    st.subheader("Input (article text or HTML)")
    article_input = st.text_area("Paste article text or HTML here", height=350, key="article_input")

    with st.expander("Example text"):
        st.write("Title: Global Consortium Launches Ambitious Plan to Accelerate Clean Energy Transition by 2035")
        st.write("In a joint announcement on Tuesday, the International Renewable Energy Partnership (IREP), a 42-nation consortium formed in 2018, revealed what experts are calling the most aggressive clean-energy acceleration plan ever attempted. The initiative, known internally as Project Ascendant, aims to reduce global reliance on fossil fuels by 34 percent by 2030 and 62 percent by 2035. While the numbers echo previous climate pledges, analysts point out that this plan includes several unprecedented enforcement mechanisms and technology-sharing agreements that have not appeared in earlier compacts.")
        st.write("According to IREP’s newly published whitepaper, the plan centers around three pillars: rapid expansion of grid-scale solar and wind capacity, deployment of next-generation long-duration batteries, and coordinated investment in geothermal resources in regions previously considered too unstable or too costly. Notably, the report claims that a prototype geothermal stabilization system successfully operated for 91 consecutive days in Northern Kenya earlier this year, though external researchers have not yet verified the dataset.")
        st.write("One of the more controversial elements involves the creation of a new Carbon Transition Accountability Board (CTAB), which will have limited authority to impose financial penalties on member nations that miss quarterly benchmarks. While the penalties remain modest in the first phase — capped at 0.03 percent of national GDP — the charter allows the board to increase fines if nations repeatedly fail to meet commitments. Several economists argue that the board’s influence will be symbolic rather than punitive, but others say it could meaningfully alter domestic energy policies, especially in mid-sized economies like Malaysia, Chile, and South Africa.")
        st.write("The plan has also sparked debate over technology-sharing requirements. Under the agreement, member nations must publish standardized blueprints for renewable-energy infrastructure that can be freely adopted by other signatories. However, three countries — Japan, Germany, and South Korea — filed reservations during the drafting stage, warning that unrestricted sharing of proprietary battery-cooling systems could undermine existing intellectual-property frameworks. IREP’s leadership insists that exemptions will be handled “case by case,” but the final text of the agreement offers little clarity on how those exemptions would be assessed.")
        st.write("Financial markets reacted cautiously to the announcement. Shares of major renewable-energy manufacturers rose slightly in early trading, with Solarys Technologies and WindPath Global each gaining around 1.8 percent. However, oil and gas firms saw mixed outcomes: Northshore Petrochem dipped 0.7 percent, while HelioGas International remained mostly unchanged. Analysts at Veritas Capital suggested that markets may be waiting for more concrete evidence that the consortium can enforce its benchmarks rather than issue aspirational statements.")
        st.write("Political responses have been uneven as well. In the United States, Secretary of Energy Linda Marquez praised the initiative, describing it as “a crucial step toward a stable energy future.” Yet several members of Congress criticized the plan, claiming it could pressure developing countries into adopting technologies they cannot yet afford. Meanwhile, in Europe, the reaction has been largely positive, though a minority bloc in the European Parliament’s Committee on Industry, Research and Energy warned that the timeline appears “borderline unrealistic” without significantly increased public investment.")
        st.write("ritics have also pointed out inconsistencies in the whitepaper. In one section, the document forecasts that long-duration batteries will constitute 28 percent of global storage capacity by 2032, while another section places that figure at 26 percent. A footnote attributes the discrepancy to differing modeling assumptions, but independent analysts say conflicting numbers weaken the report’s credibility. Additionally, energy-sector researchers questioned the claim that hydrogen-based turbines would achieve commercial viability by 2027, a target they describe as “optimistic at best and misleading at worst.”")
        st.write("Despite these concerns, Project Ascendant has gathered momentum among climate-focused NGOs. The Global Climate Observatory praised its emphasis on verifiable metrics, while the Clean Earth Coalition said that even if the consortium achieves only half of its stated goals, the outcome would still represent the largest coordinated emissions-reduction effort in history. A spokesperson for the coalition added that “ambition, even imperfect ambition, remains preferable to stagnation.”")
        st.write("The next six months will be critical. Member nations are required to submit Phase-One implementation plans by June 14, 2026, including detailed projections for infrastructure upgrades, expected job creation, and budget allocations. The CTAB will then evaluate compliance quarterly beginning in October 2026. Countries that fail to submit reports on time may be issued formal warnings, although no financial penalties will be applied until at least mid-2027.")
        st.write("Observers note that the real test will not be the initial submissions but the ability of nations to maintain progress in the face of geopolitical tensions, fluctuating commodity prices, and public resistance to large-scale infrastructure projects. As one senior IREP advisor summarized, “Technology is the easy part. Coordination is the hard one.”")

    if st.button("Summarize", key="summarize"):
        if not article_input.strip():
            st.warning("Paste article text first.")
            st.stop()

        payload = {
            "text": article_input,
            "input_is_html": False,
            "model_name": model_name,
            "extractive_method": extractive_method,
            "min_length": min_len if not extractive_method else None,
            "max_length": max_len if not extractive_method else None,
            "num_beams": beams,
            "length_penalty": length_penalty,
            "no_repeat_ngram_size": no_repeat_ngram,
            "use_reranker": use_reranker,
            "top_k": top_k,
            "run_qa": run_qa,
        }

        with st.spinner("Calling summarizer..."):
            try:
                result = call_api(api_url, payload, timeout=240)
            except Exception as e:
                st.error(f"API call failed: {e}")
                st.stop()

        # Store result in session state for later use / download
        st.session_state["last_result"] = result

        st.success("Got result — see right panel for summary & QA report.")

with col2:
    
    st.subheader("Summary & QA report")
    last = st.session_state.get("last_result")
    if not last:
        st.info("No result yet. Run summarization from the left panel.")
    else:
        summary = last.get("summary", "")
        qa_report = last.get("qa_report", {})
        debug = last.get("debug", {})

        # Render debug in small text
        with st.expander("Debug info", expanded=False):
            st.json(debug)

        # ---------------------------------------------------------
        # INSERTED: HUMAN-FRIENDLY SUMMARY BLOCK
        # ---------------------------------------------------------
        def _split_sentences(text: str) -> List[str]:
            if not text:
                return []
            sents = re.split(r'(?<=[.!?])\s+', text.strip())
            return [s.strip() for s in sents if s.strip()]

        def render_human_summary(summary_text: str, qa_report: dict):
            st.markdown("## Final Summary (Human-readable)")

            mode = st.radio(
                "Summary view",
                ("Paragraph (full)", "Concise (1–2 sentences)", "Key points (bullets)"),
                index=0,
                horizontal=True
            )

            sents = _split_sentences(summary_text)

            if mode == "Paragraph (full)":
                st.write(summary_text)

            elif mode == "Concise (1–2 sentences)":
                if len(sents) >= 2:
                    st.write(" ".join(sents[:2]))
                elif len(sents) == 1:
                    st.write(sents[0])
                else:
                    st.write(summary_text)

            else:  # bullets
                for s in sents:
                    st.markdown(f"- {s}")

            # Key facts synthesis from QA
            sc = qa_report.get("sentence_checks", [])
            facts = []
            for sent_obj in sc:
                for ent in sent_obj.get("entities", []):
                    txt = ent.get("text")
                    if txt and txt not in facts and len(facts) < 8:
                        label = ent.get("label", "UNK")
                        in_src = ent.get("in_source", False)
                        facts.append(f"{txt} ({label}) — {'in source' if in_src else 'not found'}")

            if facts:
                st.markdown("### Key facts (QA/NER extracted)")
                for f in facts:
                    st.markdown(f"- {f}")

        # CALL THE NEW SUMMARY BLOCK
        render_human_summary(summary, qa_report)
        # ---------------------------------------------------------

        # ---------------------------------------------------------
        # EXISTING SECTION: SENTENCE-LEVEL EXPLAINABLE HIGHLIGHTING
        # ---------------------------------------------------------
        st.markdown("### Summary (sentence-level highlighting)")
        sentence_checks = qa_report.get("sentence_checks")
        if sentence_checks is None:
            st.write(summary)
        else:
            for s in sentence_checks:
                sent = s.get("sentence", "").strip()
                flagged = s.get("flagged", False)
                st.markdown(color_sentence_html(sent, flagged), unsafe_allow_html=True)

                ents = s.get("entities", [])
                if ents:
                    for ent in ents:
                        st.markdown(render_entity(ent))
                else:
                    st.write("_No named entities detected in this sentence._")

        # ---------------------------------------------------------
        # Evidence / source viewer
        # ---------------------------------------------------------
        st.markdown("### Evidence & Source")
        if qa_report.get("errors"):
            st.warning("QA reported errors. See debug for details.")
            st.write(qa_report.get("errors"))

        if sentence_checks:
            with st.expander("Show QA evidence answers (per sentence)"):
                for s in sentence_checks:
                    st.markdown(f"**Sentence:** {s.get('sentence')}")
                    ents = s.get("entities", [])
                    if not ents:
                        st.write("No entities in sentence.")
                    else:
                        for ent in ents:
                            qa_answer = ent.get("qa_answer", "")
                            qa_score = ent.get("qa_score", 0.0)
                            in_src = ent.get("in_source", False)
                            st.write(f"- **{ent['text']}** — in_source: {in_src} — qa_score: {qa_score:.2f}")
                            if qa_answer:
                                st.write(f"  > evidence: {qa_answer}")

        with st.expander("Show cleaned article text"):
            st.text_area("Article (original)", article_input, height=200)

        st.download_button(
            "Download result JSON",
            data=json.dumps(last, indent=2),
            file_name="summarization_result.json",
            mime="application/json"
        )

# Footer / small notes
st.markdown("---")
st.markdown("Built for demo: sentence-level QA checks flag potential hallucinations. Not perfect — tune thresholds and add claim generation for better coverage.")
