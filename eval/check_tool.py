"""Hand-check tool for the open human checks. Every click is saved straight into the CSV, so you can stop and
resume at any time.

    streamlit run eval/check_tool.py

Tasks:
  citation extraction   eval/citation_precision_sample.csv, column extraction_correct (guide: CITATION_CHECK_GUIDE.md)
  citation resolution   eval/citation_precision_sample.csv, column resolution_correct
  NLI filter            eval/nli_check_sample.csv, column supported (blinded: kept/rejected is not shown)

When done:
    python -m stage0_graph.precision --provenance "<who checked>"
    python -m stage5_summary.nli_check --score --provenance "<who checked>"
"""
import html
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CIT = ROOT / "eval" / "citation_precision_sample.csv"
NLI = ROOT / "eval" / "nli_check_sample.csv"
TASKS = {
    "Citation extraction": (CIT, "extraction", "extraction_correct"),
    "Citation resolution": (CIT, "resolution", "resolution_correct"),
    "NLI filter": (NLI, None, "supported"),
}
QUESTION = {
    "Citation extraction": "Does the **highlighted string** identify a judgment cited in this text?",
    "Citation resolution": "Is the **resolved case** the one this text actually cites?",
    "NLI filter": "Is the **summary sentence** supported by the judgment?",
}
HELP = {
    "Citation extraction": "Yes: a real cited judgment (a footnote, or a name with a stray “In”/“See”, still counts). "
                           "No: the citing judgment's own page header or title, a name cut to the wrong parties, "
                           "a name running into the next case, or a proceeding that is not a cited precedent.",
    "Citation resolution": "Compare the parties, the year and the reporter citation in the text with the resolved "
                           "title and date. Can't tell: leave it blank.",
    "NLI filter": "Yes: everything the sentence says is stated or clearly implied by the judgment. No: any part is "
                  "wrong or not in the judgment (wrong section, party, outcome, negation flip, invented detail). "
                  "The evidence is the 5 most similar judgment sentences; open the full judgment if they are not "
                  "enough.",
}

st.set_page_config(page_title="NyayaSethu checks", layout="wide")


def load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, dtype=str)


def save(df: pd.DataFrame, path: Path):
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def highlight(text: str, needle: str) -> str:
    t = html.escape(text)
    if needle:
        n = html.escape(needle)
        t = re.sub(re.escape(n), lambda m: f"<mark>{m.group(0)}</mark>", t, count=1)
    return t


task = st.sidebar.radio("Task", list(TASKS))
path, check, col = TASKS[task]
if not path.exists():
    st.error(f"{path.name} not found. For the NLI task run: python -m stage5_summary.nli_check --make")
    st.stop()
df = load(path)
rows = df.index[df.check == check].tolist() if check else df.index.tolist()
done = [i for i in rows if df.at[i, col].strip()]
st.sidebar.progress(len(done) / len(rows), text=f"{len(done)} / {len(rows)} answered")
st.sidebar.caption("Blank = not answered yet. “Can't tell” saves `unsure`, which the scorers skip.")
only_open = st.sidebar.checkbox("Skip answered rows", value=True)
show_ai = task != "NLI filter" and st.sidebar.checkbox("Show AI pre-label", value=False,
                                                       help="AI hint for speed. It is never scored.")

queue = [i for i in rows if not (only_open and df.at[i, col].strip())]
key = f"pos_{task}"
if not queue:
    st.success(f"All {len(rows)} rows answered. Run the scorer listed at the top of eval/check_tool.py.")
    st.stop()
pos = min(st.session_state.get(key, 0), len(queue) - 1)
i = queue[pos]
r = df.loc[i]

st.markdown(f"### {QUESTION[task]}")
st.caption(HELP[task])
st.markdown(f"Row **{r.row_id}** · {pos + 1} of {len(queue)} in this queue")

if task == "NLI filter":
    st.markdown(f"**Case:** {html.escape(r.title)}  ·  **field:** `{r.field}`")
    st.info(r.sentence)
    st.markdown("**Most similar sentences in the judgment**")
    st.markdown(f"<div style='font-size:0.95em;line-height:1.5'>{html.escape(r.evidence)}</div>",
                unsafe_allow_html=True)
    with st.expander("Full judgment"):
        from stage5_summary.docs import get_judgment
        doc = get_judgment(r.case_id)
        st.text("\n".join(s.text for s in doc.sentences) if doc else "(not found)")
else:
    st.markdown(f"<div style='font-size:1.05em;line-height:1.6;padding:0.8em;border:1px solid #8884;"
                f"border-radius:6px'>{highlight(r.sentence_or_context, r.extracted)}</div>",
                unsafe_allow_html=True)
    st.markdown(f"**Extracted:** `{r.extracted}`" + (f"  ·  **parties near it:** {r.party_name}" if r.party_name else ""))
    if task == "Citation resolution":
        st.markdown(f"**Resolved to:** {r.resolved_title}  ·  **date:** {r.resolved_date}  ·  "
                    f"method `{r.method}` (score {r.score})")
    if show_ai:
        st.caption(f"AI pre-label: {r.assistant_correct or '-'} {('· ' + r.assistant_note) if r.assistant_note else ''}")

note_col = "note" if task == "NLI filter" else None
note = st.text_input("Note (optional)", value=r[note_col] if note_col else "", key=f"note_{task}_{i}") if note_col else ""
if r[col].strip():
    st.caption(f"Current answer: **{r[col]}**")


def answer(v: str):
    d = load(path)
    d.at[i, col] = v
    if note_col:
        d.at[i, note_col] = st.session_state.get(f"note_{task}_{i}", "")
    save(d, path)
    if not only_open:
        st.session_state[key] = pos + 1


c = st.columns([1, 1, 1, 1, 1])
c[0].button("✅ Yes", on_click=answer, args=("yes",), use_container_width=True, type="primary")
c[1].button("❌ No", on_click=answer, args=("no",), use_container_width=True)
c[2].button("🤷 Can't tell", on_click=answer, args=("unsure",), use_container_width=True)
c[3].button("◀ Back", on_click=lambda: st.session_state.update({key: max(pos - 1, 0)}), use_container_width=True,
            disabled=pos == 0)
c[4].button("Skip ▶", on_click=lambda: st.session_state.update({key: pos + 1}), use_container_width=True,
            disabled=pos >= len(queue) - 1)
