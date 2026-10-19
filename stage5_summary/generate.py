"""Field-wise generation with Legal-LED (Longformer encoder-decoder).

* `LEDGenerator`: loads the fine-tuned model from models/legal_led_inabs if present (trained with
  notebooks/stage5_led_finetune.ipynb on Kaggle), else the untuned `models.summarizer`. Global attention on
  the first token, inputs truncated to stage5.max_input tokens, beam search. Runs on CUDA, MPS or CPU.
* `ExtractiveGenerator`: no model; returns the first source sentences of each field up to a word budget.
  Used for quick UI runs, tests, and as an extractive baseline.

relevance_to_query is conditioned on the query by prefixing it to the input ("Query: ... || ...").
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from common.config import cfg, path

MAX_NEW = {"issue": 64, "material_facts": 160, "ratio": 200, "decision": 64, "relevance_to_query": 160}
WORDS = {"issue": 50, "material_facts": 120, "ratio": 150, "decision": 50, "relevance_to_query": 120}


def device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def default_model() -> str:
    tuned = path("models") / "legal_led_inabs"
    return str(tuned) if (tuned / "config.json").exists() else cfg()["models"]["summarizer"]


def field_input(field: str, sources: List[str], query: str = "") -> str:
    body = " ".join(sources)
    return f"Query: {query} || {body}" if field == "relevance_to_query" and query else body


def strip_echo(text: str) -> str:
    """The model sometimes copies the "Query: ... ||" input prefix into its output; drop it."""
    return text.split("||", 1)[1].strip() if "||" in text else text


class ExtractiveGenerator:
    name = "extractive"

    def generate(self, field: str, sources: List[str], query: str = "") -> str:
        out, n = [], 0
        for s in sources:
            if n and n + len(s.split()) > WORDS[field]:
                break
            out.append(s)
            n += len(s.split())
        return " ".join(out)


class LEDGenerator:
    def __init__(self, model: Optional[str] = None, dev: Optional[str] = None, num_beams: Optional[int] = None):
        from transformers import AutoTokenizer, LEDForConditionalGeneration
        s5 = cfg()["stage5"]
        self.name = model or default_model()
        self.device = dev or device()
        self.tok = AutoTokenizer.from_pretrained(self.name)
        self.model = LEDForConditionalGeneration.from_pretrained(self.name).to(self.device).eval()
        self.max_input = s5["max_input"]
        self.num_beams = num_beams or s5.get("num_beams", 2)

    def generate(self, field: str, sources: List[str], query: str = "") -> str:
        import torch
        text = field_input(field, sources, query)
        if not text.strip():
            return ""
        enc = self.tok(text, truncation=True, max_length=self.max_input, return_tensors="pt").to(self.device)
        gam = torch.zeros_like(enc["input_ids"])
        gam[:, 0] = 1
        with torch.no_grad():
            out = self.model.generate(**enc, global_attention_mask=gam, max_new_tokens=MAX_NEW[field],
                                      num_beams=self.num_beams, no_repeat_ngram_size=3, length_penalty=1.0,
                                      early_stopping=True)
        return strip_echo(self.tok.decode(out[0], skip_special_tokens=True).strip())


def make_generator(mode: str = "abstractive", model: Optional[str] = None):
    return ExtractiveGenerator() if mode == "extractive" else LEDGenerator(model)


def generate_fields(gen, sources: Dict[str, List[str]], query: str = "") -> Dict[str, str]:
    return {f: gen.generate(f, s, query) for f, s in sources.items()}


if __name__ == "__main__":                              # quick manual check: python -m stage5_summary.generate
    g = make_generator("abstractive")
    print(Path(g.name).name, g.device)
    print(g.generate("decision", ["The appeal is allowed.", "The impugned order is set aside with costs."]))
