"""Hand-check tool for the open human checks. Every click is saved straight into the CSV, so you can stop and
resume at any time.

    streamlit run eval/check_tool.py

Tasks:
  citation extraction   eval/citation_precision_sample.csv, column extraction_correct (guide: CITATION_CHECK_GUIDE.md)
  citation resolution   eval/citation_precision_sample.csv, column resolution_correct
  NLI filter            eval/nli_check_sample.csv, column supported (blinded: kept/rejected is not shown)
  Sentence splitter     data/validation/splitter_annotate.txt, line breaks only (step 6)
  Sentence roles        data/validation/roles_to_annotate.csv, column gold (step 11b, adds the Issue role)

When done:
    python -m stage0_graph.precision --provenance "<who checked>"
    python -m stage5_summary.nli_check --score --provenance "<who checked>"
    python -m stage0_corpus.splitter_validate
    python -m stage1_roles.annotate import
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

VAL = ROOT / "data" / "validation"
SPLIT_TXT, SPLIT_RAW, SPLIT_DONE = VAL / "splitter_annotate.txt", VAL / "splitter_sample.jsonl", VAL / "splitter_done.txt"
ROLES_CSV, ROLES_DONE = VAL / "roles_to_annotate.csv", VAL / "roles_done.txt"
ROLE_HELP = {
    "Facts": "what happened between the parties before the case reached this court",
    "Ruling by Lower Court": "what the trial court / High Court decided",
    "Issue": "the legal question(s) this court has to answer (often “whether …”, “the question is …”)",
    "Argument": "what counsel for either side submitted",
    "Statute": "text of a statute or rule quoted or described",
    "Precedent": "discussion of an earlier judgment",
    "Ratio of the decision": "this court's own reasoning",
    "Ruling by Present Court": "this court's final order (appeal allowed / dismissed, directions)",
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


def done_set(p: Path) -> set:
    return set(p.read_text().split()) if p.exists() else set()


def mark_done(p: Path, key: str):
    p.write_text("\n".join(sorted(done_set(p) | {key})) + "\n")


def splitter_task():
    import json
    sq = lambda t: re.sub(r"\s+", "", t)
    raw = {json.loads(l)["pid"]: json.loads(l) for l in open(SPLIT_RAW, encoding="utf-8")}
    lines = SPLIT_TXT.read_text(encoding="utf-8").split("\n")
    head, blocks, cur = [], {}, None
    for ln in lines:
        if ln.startswith("###"):
            cur = ln.split()[1]
            blocks[cur] = {"header": ln, "sents": []}
        elif cur is None:
            head.append(ln)
        elif ln.strip():
            blocks[cur]["sents"].append(ln.strip())
    done = done_set(SPLIT_DONE)
    pids = list(blocks)
    st.sidebar.progress(len(done & set(pids)) / len(pids), text=f"{len(done & set(pids))} / {len(pids)} passages saved")
    if st.session_state.get("sp_pick") not in pids:
        st.session_state["sp_pick"] = next((p for p in pids if p not in done), pids[0])
    pid = st.sidebar.selectbox("Passage", pids, key="sp_pick")
    st.markdown("### Put exactly one true sentence on each line")
    st.caption("Join lines that are one sentence (e.g. split after “v.”, “No.”, “S.”, “i.e.”); press Enter where "
               "two sentences share a line. Change line breaks only, never the words.")
    st.markdown(f"Passage **{pid}** · {raw[pid]['case_id']}" + (" · ✅ saved" if pid in done else ""))
    st.text_area("Sentences", "\n".join(blocks[pid]["sents"]), height=380, key=f"sp_{pid}")

    def save_passage():
        new = [x.strip() for x in st.session_state[f"sp_{pid}"].split("\n") if x.strip()]
        if sq(" ".join(new)) != sq(raw[pid]["text"]):
            st.session_state["sp_msg"] = ("error", f"{pid} not saved: the words changed. Only add or remove line "
                                          "breaks (undo with Cmd+Z).")
            return
        blocks[pid]["sents"] = new
        out = "\n".join(head).rstrip("\n") + "\n"
        for b in blocks.values():
            out += "\n" + b["header"] + "\n" + "\n".join(b["sents"]) + "\n"
        tmp = SPLIT_TXT.with_suffix(".tmp")
        tmp.write_text(out, encoding="utf-8")
        tmp.replace(SPLIT_TXT)
        mark_done(SPLIT_DONE, pid)
        left = [p for p in pids if p not in done_set(SPLIT_DONE)]
        st.session_state["sp_msg"] = ("success", f"Saved {pid} ({len(new)} sentences)." +
                                      (f" Now on {left[0]}." if left else " All passages saved."))
        if left:
            st.session_state["sp_pick"] = left[0]

    st.button("💾 Save passage and go to the next", type="primary", on_click=save_passage)
    if "sp_msg" in st.session_state:
        kind, msg = st.session_state.pop("sp_msg")
        (st.error if kind == "error" else st.success)(msg)


def roles_task():
    df = load(ROLES_CSV)
    labels = list(ROLE_HELP)
    cases = list(dict.fromkeys(df.case_id))
    done = done_set(ROLES_DONE)
    st.sidebar.progress(len(done & set(cases)) / len(cases), text=f"{len(done & set(cases))} / {len(cases)} judgments saved")
    if st.session_state.get("ro_pick") not in cases:
        st.session_state["ro_pick"] = next((c for c in cases if c not in done), cases[0])
    cid = st.sidebar.selectbox("Judgment", cases, key="ro_pick",
                               format_func=lambda c: f"{c} ({(df.case_id == c).sum()} sentences)" + (" ✅" if c in done else ""))
    with st.sidebar.expander("What the roles mean", expanded=True):
        for k, v in ROLE_HELP.items():
            st.markdown(f"**{k}**: {v}")
    st.markdown("### Correct the role of every sentence")
    st.caption("Pre-filled by the current model (Issue by cue words). Read top to bottom and change the `gold` "
               "dropdown where it is wrong. Every sentence counts as training data, so check all of them.")
    sub = df[df.case_id == cid][["sent_idx", "sentence", "gold"]].copy()
    sub["sent_idx"] = sub.sent_idx.astype(int)
    ed = st.data_editor(sub, hide_index=True, width="stretch", height=560, key=f"ro_{cid}",
                        disabled=["sent_idx", "sentence"],
                        column_config={"sentence": st.column_config.TextColumn(width="large"),
                                       "gold": st.column_config.SelectboxColumn(options=labels, required=True)})

    def save_judgment():
        d = load(ROLES_CSV)
        m = d.case_id == cid
        gold = sub["gold"].tolist()
        for row, ch in st.session_state[f"ro_{cid}"].get("edited_rows", {}).items():
            if "gold" in ch:
                gold[int(row)] = ch["gold"]
        d.loc[m, "gold"] = gold
        save(d, ROLES_CSV)
        mark_done(ROLES_DONE, cid)
        n = sum(g != p for g, p in zip(gold, d.loc[m, "predicted"]))
        left = [c for c in cases if c not in done_set(ROLES_DONE)]
        st.session_state["ro_msg"] = f"Saved {cid}: {n} labels changed from the pre-label." + (
            f" Now on {left[0]}." if left else " All judgments saved.")
        if left:
            st.session_state["ro_pick"] = left[0]

    st.button("💾 Save judgment and go to the next", type="primary", on_click=save_judgment)
    if "ro_msg" in st.session_state:
        st.success(st.session_state.pop("ro_msg"))


task = st.sidebar.radio("Task", list(TASKS) + ["Sentence splitter", "Sentence roles"])
if task == "Sentence splitter":
    splitter_task() if SPLIT_TXT.exists() else st.error("Run: python -m stage0_corpus.splitter_sample")
    st.stop()
if task == "Sentence roles":
    roles_task() if ROLES_CSV.exists() else st.error("Run: python -m stage1_roles.annotate export --n 10 --max-sentences 120")
    st.stop()
path, check, col = TASKS[task]
if not path.exists():
    st.error(f"{path.name} not found. For the NLI task run: python -m stage5_summary.nli_check --make")
    st.stop()
df = load(path)
rows = df.index[df.check == check].tolist() if check else df.index.tolist()
done = [i for i in rows if df.at[i, col].strip()]
st.sidebar.progress(len(done) / len(rows), text=f"{len(done)} / {len(rows)} answered")
st.sidebar.caption("Blank = not answered yet. “Can't tell” saves `unsure`, which the scorers skip.")
review = st.sidebar.text_area("Review only these row ids", key=f"review_{task}",
                              help="Comma- or space-separated, e.g. r046, r047. Shows them even if answered.")
review_ids = set(re.findall(r"[rn]\d{3}", review))
if review_ids:
    rows = [i for i in rows if df.at[i, "row_id"] in review_ids]
only_open = st.sidebar.checkbox("Skip answered rows", value=not review_ids, disabled=bool(review_ids))
only_open = only_open and not review_ids
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
    st.warning(f"Current answer: **{r[col]}**. Click again to change it, or Skip to keep it.")


def answer(v: str):
    d = load(path)
    d.at[i, col] = v
    if note_col:
        d.at[i, note_col] = st.session_state.get(f"note_{task}_{i}", "")
    save(d, path)
    if not only_open:
        st.session_state[key] = pos + 1


c = st.columns([1, 0.4, 1, 0.4, 1, 1, 1])        # spacers between the answer buttons against misclicks
c[0].button("✅ Yes", on_click=answer, args=("yes",), width="stretch", type="primary")
c[2].button("❌ No", on_click=answer, args=("no",), width="stretch")
c[4].button("🤷 Can't tell", on_click=answer, args=("unsure",), width="stretch")
c[5].button("◀ Back", on_click=lambda: st.session_state.update({key: max(pos - 1, 0)}), width="stretch",
            disabled=pos == 0)
c[6].button("Skip ▶", on_click=lambda: st.session_state.update({key: pos + 1}), width="stretch",
            disabled=pos >= len(queue) - 1)
