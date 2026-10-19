"""Member B, Stage 5 summary: field mapping, fallbacks, NLI filter, summarize(). Offline (stub models, mock data)."""
import json
from pathlib import Path

import pytest

from stage5_summary.fields import FIELDS, field_sources
from stage5_summary.generate import ExtractiveGenerator, field_input, strip_echo
from stage5_summary.nli import filter_text

REPO = Path(__file__).resolve().parents[1]

DOC = [("The tenant paid an advance of Rs 50,000.", "Facts"),
       ("The landlord retained the advance on vacation.", "Facts"),
       ("The point for consideration is whether retention needs proof of loss.", "Ratio of the decision"),
       ("Section 74 of the Contract Act allows only reasonable compensation.", "Statute"),
       ("Forfeiture without proof of loss is a penalty.", "Ratio of the decision"),
       ("A lease of agricultural land is governed by the tenancy law.", "Ratio of the decision"),
       ("The appeal is allowed.", "Ruling by Present Court")]


def test_field_mapping_and_fallbacks():
    src, prov = field_sources(DOC, "landlord kept deposit advance without proving loss", relevance_top=2)
    assert set(src) == set(FIELDS)
    assert src["material_facts"] == [DOC[0][0], DOC[1][0]]
    assert src["decision"] == ["The appeal is allowed."]
    assert src["ratio"][-1] == "The appeal is allowed."                # Ratio + Ruling by Present Court
    assert prov["issue"] == "fallback:question_cues" and "whether" in src["issue"][0]
    assert "agricultural" not in " ".join(src["relevance_to_query"])     # ranked by query similarity
    no_ruling = [(t, "Facts") for t, _ in DOC]
    s2, p2 = field_sources(no_ruling, "", decision_tail=2)
    assert p2["decision"] == "fallback:last_2_sentences" and s2["decision"] == [DOC[5][0], DOC[6][0]]
    assert p2["ratio"] == "empty" and s2["ratio"] == []                 # nothing invented


def test_query_conditioning_and_extractive():
    assert field_input("relevance_to_query", ["a."], "q").startswith("Query: q || ")
    assert field_input("ratio", ["a."], "q") == "a."
    assert strip_echo("Was natural justice violated? || The lessee was not heard.") == "The lessee was not heard."
    assert strip_echo("The lessee was not heard.") == "The lessee was not heard."
    g = ExtractiveGenerator()
    assert g.generate("decision", ["The appeal is allowed.", "No costs."]) == "The appeal is allowed. No costs."


def _stub_scorer(pairs):
    # entailed iff every content word of the hypothesis appears in the premise
    out = []
    for prem, hyp in pairs:
        words = [w.strip(".,").lower() for w in hyp.split() if len(w) > 3]
        out.append(0.9 if all(w in prem.lower() for w in words) else 0.1)
    return out


def test_nli_filter_rejects_unsupported():
    src = [t for t, _ in DOC]
    kept, rej = filter_text("Forfeiture without proof of loss is a penalty. The SEC obtained an asset freeze.",
                            src, _stub_scorer, threshold=0.5, k=3)
    assert kept == "Forfeiture without proof of loss is a penalty."
    assert len(rej) == 1 and "SEC" in rej[0]["sentence"] and rej[0]["p_entail"] == 0.1 and rej[0]["premise"]


@pytest.fixture
def mock_cfg(monkeypatch, tmp_path):
    from common import config
    real = config.cfg()
    monkeypatch.setattr(config, "cfg", lambda path=None: {**real, "use_mock": True})
    from stage5_summary import docs
    docs._segments.cache_clear()
    yield
    docs._segments.cache_clear()
    for p in (REPO / "data" / "mock").glob("*.offsets.json"):
        p.unlink()


class _Hallucinating:
    name = "stub"

    def generate(self, field, sources, query=""):
        return (" ".join(sources[:1]) + " The SEC obtained an asset freeze.").strip()


def test_summarize_on_mock(mock_cfg):
    from stage5_summary import summarize
    s = summarize("deposit forfeited without proof of loss", "sc:MOCK1963001", generator=_Hallucinating(),
                  scorer=_stub_scorer)
    assert s["status"] == "ok" and s["roles_source"] == "segments"
    assert s["material_facts"] == "The appellant agreed to purchase land and paid Rs 25,000 as earnest money."
    assert all("SEC" not in s[f] for f in FIELDS)
    assert len(s["rejected"]) == 5 and all(r.endswith("asset freeze.") for r in s["rejected"])
    ex = summarize("q", "sc:MOCK1963001", mode="extractive")
    assert ex["nli"] == "skipped" and ex["decision"] == "The appeal is partly allowed."
    assert summarize("q", "sc:NOPE")["status"] == "not_found"


def test_position_fallback_roles():
    from common.schema import Judgment, Sentence
    from stage5_summary import docs
    d = Judgment("x:1", sentences=[Sentence(i, f"S{i}.") for i in range(12)])
    import unittest.mock as um
    with um.patch.object(docs, "_segments", return_value={}), um.patch.object(docs, "_labeller", return_value=None):
        sents, srcname = docs.roles_for(d)
    assert srcname == "position_fallback"
    assert [r for _, r in sents][:3] == ["Facts"] * 3 and sents[-1][1] == "Ruling by Present Court"


def test_notebook_settings():
    nb = json.loads((REPO / "notebooks" / "stage5_led_finetune.ipynb").read_text())
    src = "".join("".join(c["source"]) for c in nb["cells"])
    for s in ("nsi319/legal-led-base-16384", "(8192, 512)", "(2 if SMOKE else 16)", "torch.float16",
              "gradient_checkpointing_enable", "GradScaler", "global_attention_mask", "SMOKE = False"):
        assert s in src
