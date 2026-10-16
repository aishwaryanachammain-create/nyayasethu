from stage0_corpus.metadata import extract_bench_size, extract_court, extract_date
from stage0_corpus.citations import canonical_key, extract_citations
from stage0_corpus.splitter import evaluate_splitter, split_sentences
from common.text import tokenize


def test_splitter_keeps_legal_abbreviations():
    t = ("The view in Fateh Chand v. Balkishan Das, AIR 1963 SC 1405, was applied. See S. 74 of the Act i.e. "
         "the Contract Act. 12. The next issue is whether Mr. A. K. Rao was right. The rate was 7.5 per cent.")
    s = split_sentences(t)
    assert s == ["The view in Fateh Chand v. Balkishan Das, AIR 1963 SC 1405, was applied.",
                 "See S. 74 of the Act i.e. the Contract Act.",
                 "12. The next issue is whether Mr. A. K. Rao was right.",
                 "The rate was 7.5 per cent."]


def test_splitter_evaluation_metric():
    gold = [["First sentence here.", "Second one, citing (2019) 5 SCC 123."]]
    p, r, f = evaluate_splitter(gold, [" ".join(gold[0])])
    assert f == 1.0


def test_citation_formats_normalise_to_same_key():
    assert canonical_key("(2019) 5 SCC 123") == canonical_key("2019 (5) SCC 123")
    kinds = {c.reporter for c in extract_citations(
        "AIR 1963 SC 1405; (1964) 1 SCR 515; 2023 INSC 456; 2015 SCC OnLine SC 123; MANU/SC/0123/2005")}
    assert {"AIR", "SCR", "INSC", "SCCOL", "MANU"} <= kinds
    hc = extract_citations("See 2023:DHC:1234 and 2024:BHC-AS:5678-DB.")
    assert [c.key for c in hc] == ["NEUTRAL|2023|DHC|1234", "NEUTRAL|2024|BHC-AS|5678"]


def test_tokenizer_keeps_statutory_refs():
    toks = tokenize("Under Section 74 and Art. 141 the learned counsel argued")
    assert "section_74" in toks and "article_141" in toks and "counsel" not in toks


def test_bench_extraction():
    assert extract_bench_size("Bench: A. Rao, B. Menon, C. Iyer.\n\nJUDGMENT") == 3
    assert extract_bench_size("Before a Constitution Bench of this Court") == 5






def test_metadata_regex():
    head = "IN THE SUPREME COURT OF INDIA\nCIVIL APPELLATE JURISDICTION\nCORAM: A. RAO, B. MENON, C. IYER\n\nJUDGMENT\n"
    assert extract_court(head) == "SC"
    assert extract_bench_size(head) == 3
    assert extract_bench_size("HON'BLE MR. JUSTICE A. K. SIKRI\nHON'BLE MR. JUSTICE ASHOK BHUSHAN\n") == 2
    assert extract_date("Date of Judgment: 12.03.2019") == "2019-03-12"
    assert extract_date("... NEW DELHI; MARCH 12, 2019") == "2019-03-12"
    assert extract_date("no date here") is None


def test_bhattacharya_loader(tmp_path):
    import zipfile
    from stage0_corpus.loaders import load_bhattacharya
    z = tmp_path / "b.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("x/data/text/1990_A_1.txt", "The appellant was employed\tFacts\nWe hold it is a penalty\tRatio of the decision\n")
        f.writestr("x/categories.txt", "Labour\t1990_A_1")
    d = load_bhattacharya(z)[0]
    assert d.case_id == "bhatt:1990_A_1" and d.source == "Bhattacharya2019/Labour" and d.date == "1990"
    assert [s.role for s in d.sentences] == ["Facts", "Ratio of the decision"]
