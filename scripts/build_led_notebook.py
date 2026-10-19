"""Write notebooks/stage5_led_finetune.ipynb (Kaggle) from the cells below, so the notebook stays reviewable as code.

    python scripts/build_led_notebook.py
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "stage5_led_finetune.ipynb"

MD_INTRO = """# NyayaSethu Stage 5: fine-tune Legal-LED on IN-Abs (Kaggle)

Fine-tunes `nsi319/legal-led-base-16384` on IN-Abs (7,030 Indian Supreme Court judgment/summary pairs;
Shukla et al., AACL 2022, Zenodo 10.5281/zenodo.7152317). It then scores the untuned and fine-tuned
models on the 100 IN-Abs test documents with ROUGE-1/2/L and BERTScore.

**Kaggle settings:** Accelerator **GPU T4 x2** (one GPU is used) or P100, **Internet on**. Both need a phone-verified Kaggle account. If you use "Save Version", choose "Save & Run All".

**Settings (from the brief):**
* fp16, gradient checkpointing
* batch 1, gradient accumulation 16
* 8,192 input tokens, 512 output tokens

**Kaggle stops a run after 12 hours, and the full job takes longer.** The notebook therefore stops itself
after `TIME_BUDGET_H` hours, saves its progress (training checkpoint and the test summaries done so far) to
`/kaggle/working`, and ends normally. To continue, open the notebook, choose **Add Input → Your Work →
Notebook Output Files**, add this notebook's previous version, and run **Save Version → Save & Run All**
again. The next run copies the progress from `/kaggle/input` and continues. Repeat until the last cell
prints the results.

**Outputs** (download them from the notebook's Output tab):
* `legal_led_inabs.zip`: unzip it into the repo as `models/legal_led_inabs/`
* `stage5_inabs_results.json`: ROUGE and BERTScore results. Copy it to `eval/`.
* `stage5_inabs_predictions.csv`: the test summaries from both models

`SMOKE = True` runs the same code on 4 documents with a tiny random model. That is a code check only; its numbers are meaningless."""

CELLS = [
    ("md", MD_INTRO),
    ("code", """SMOKE = False            # True: 4 docs + tiny random LED, to check the code end to end
MODEL = "hf-internal-testing/tiny-random-LEDForConditionalGeneration" if SMOKE else "nsi319/legal-led-base-16384"
MAX_IN, MAX_OUT = (256, 32) if SMOKE else (8192, 512)
LR, EPOCHS, ACCUM = 3e-5, 1, (2 if SMOKE else 16)
N_VAL = 2 if SMOKE else 200     # held out from train for a loss check
N_TEST = 4 if SMOKE else 100    # IN-Abs test documents to score
BEAMS, GEN_BATCH = 2, 2
SAVE_EVERY = 1 if SMOKE else 50  # optimiser steps between checkpoints
TIME_BUDGET_H = 11.0             # stop and save before Kaggle's 12-hour limit
SEED = 13"""),
    ("code", """import os, sys, subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rouge-score", "bert-score"], check=False)
import json, math, random, shutil, time, zipfile, urllib.request
from pathlib import Path
T_START = time.time()
def time_left():
    return TIME_BUDGET_H * 3600 - (time.time() - T_START)
import torch
from transformers import AutoTokenizer, LEDForConditionalGeneration
WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("outputs/kaggle_local")
WORK.mkdir(parents=True, exist_ok=True)
# Progress from a previous run attached as input (Add Input -> Notebook Output Files): copy it in once.
INP = Path("/kaggle/input")
if INP.exists() and not (WORK / "ckpt" / "state.json").exists():
    for st in sorted(INP.rglob("ckpt/state.json")):
        prev = st.parent.parent
        shutil.copytree(prev / "ckpt", WORK / "ckpt")
        for f in prev.glob("preds_*.jsonl"):
            shutil.copy(f, WORK / f.name)
        print("resuming from", prev, json.loads(st.read_text()))
        break
DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
FP16 = DEV == "cuda"
random.seed(SEED); torch.manual_seed(SEED)
print(DEV, torch.__version__)
import socket
try:
    socket.create_connection(("huggingface.co", 443), timeout=10).close()
except OSError:
    raise SystemExit("No internet. The model (Hugging Face) and IN-Abs (Zenodo) are downloaded at run time. "
                     "On Kaggle: verify your phone (Settings > Phone verification), then turn Internet on "
                     "in the right-hand panel (Session options) and restart the session.")
if DEV != "cuda" and not SMOKE:
    print("WARNING: no GPU. Set Accelerator to a GPU (T4) before running; on CPU this will not finish.")"""),
    ("md", "## Data: IN-Abs"),
    ("code", """DATA = WORK / "in_abs_ext"
if not (DATA / "dataset" / "IN-Abs").exists():
    DATA.mkdir(parents=True, exist_ok=True)
    zp = DATA / "dataset.zip"
    if not zp.exists():
        urllib.request.urlretrieve("https://zenodo.org/records/7152317/files/dataset.zip?download=1", zp)
    zipfile.ZipFile(zp).extractall(DATA)
ROOT = DATA / "dataset" / "IN-Abs"

def pairs(split):
    jd, sd = ROOT / split / "judgement", ROOT / split / "summary"
    out = []
    for f in sorted(jd.glob("*.txt")):
        s = sd / f.name
        if s.exists():
            out.append((f.stem, f.read_text(errors="ignore").strip(), s.read_text(errors="ignore").strip()))
    return out

train = pairs("train-data"); test = pairs("test-data")
random.Random(SEED).shuffle(train)
val, train = train[:N_VAL], train[N_VAL:]
if SMOKE:
    train, test = train[:4], test[:N_TEST]
test = test[:N_TEST]
print(len(train), "train /", len(val), "val /", len(test), "test")"""),
    ("md", "## Fine-tuning (fp16, gradient checkpointing, batch 1, accumulation 16)"),
    ("code", """tok = AutoTokenizer.from_pretrained(MODEL)
CKPT = WORK / "ckpt"
start_model = str(CKPT) if (CKPT / "config.json").exists() else MODEL
model = LEDForConditionalGeneration.from_pretrained(start_model).to(DEV)
model.gradient_checkpointing_enable()
model.config.use_cache = False
state = json.loads((CKPT / "state.json").read_text()) if (CKPT / "state.json").exists() else {"done": 0, "step": 0}
print("starting from", start_model, state)

def encode(doc, summ):
    x = tok(doc, truncation=True, max_length=MAX_IN, return_tensors="pt")
    y = tok(text_target=summ, truncation=True, max_length=MAX_OUT, return_tensors="pt")["input_ids"]
    y[y == tok.pad_token_id] = -100
    g = torch.zeros_like(x["input_ids"]); g[:, 0] = 1
    return {"input_ids": x["input_ids"].to(DEV), "attention_mask": x["attention_mask"].to(DEV),
            "global_attention_mask": g.to(DEV), "labels": y.to(DEV)}

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
if (CKPT / "opt.pt").exists():
    opt.load_state_dict(torch.load(CKPT / "opt.pt", map_location=DEV))
total = math.ceil(len(train) * EPOCHS / ACCUM); warm = max(1, total // 20)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (total - s) / max(1, total - warm))))
for _ in range(state["step"]):
    sched.step()
scaler = torch.amp.GradScaler("cuda", enabled=FP16)

def save_ckpt():
    model.save_pretrained(CKPT); tok.save_pretrained(CKPT)
    torch.save(opt.state_dict(), CKPT / "opt.pt"); (CKPT / "state.json").write_text(json.dumps(state))

order = [i for _ in range(EPOCHS) for i in random.Random(SEED).sample(range(len(train)), len(train))]
model.train(); t0 = time.time(); run_loss = 0.0
for k in range(state["done"], len(order)):
    if time_left() < 0:
        break
    _, doc, summ = train[order[k]]
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=FP16):
        loss = model(**encode(doc, summ)).loss / ACCUM
    scaler.scale(loss).backward(); run_loss += loss.item()
    if (k + 1) % ACCUM == 0 or k + 1 == len(order):
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()
        state["step"] += 1; state["done"] = k + 1
        if state["step"] % 10 == 0 or SMOKE:
            print(f"step {state['step']}/{total} loss {run_loss:.4f} {time.time() - t0:.0f}s", flush=True)
        run_loss = 0.0
        if state["step"] % SAVE_EVERY == 0:
            save_ckpt()
TRAINED = state["done"] >= len(order)
save_ckpt()
print("training done" if TRAINED else "TIME BUDGET REACHED, training paused; re-run with this output as input", state)"""),
    ("code", """@torch.no_grad()
def val_loss(m):
    m.eval(); ls = []
    for _, doc, summ in val:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=FP16):
            ls.append(m(**encode(doc, summ)).loss.item())
    return sum(ls) / len(ls)
model.config.use_cache = True
FINAL = WORK / "legal_led_inabs"
VAL_FILE = WORK / "val_loss.json"
if TRAINED:
    if VAL_FILE.exists():
        tuned_val = json.loads(VAL_FILE.read_text())["val_loss_finetuned"]
    else:
        tuned_val = val_loss(model)
        VAL_FILE.write_text(json.dumps({"val_loss_finetuned": tuned_val}))
    print("val loss (fine-tuned):", tuned_val)
    model.save_pretrained(FINAL); tok.save_pretrained(FINAL)"""),
    ("md", "## Test: untuned vs fine-tuned on IN-Abs test (ROUGE-1/2/L, BERTScore)"),
    ("code", """@torch.no_grad()
def summarise(m, items, out_file):
    # {name: summary} for (name, doc) pairs; appended to out_file as they finish, names already there are skipped.
    # Incomplete if the time budget runs out.
    done = {}
    if out_file.exists():
        for line in open(out_file):
            r = json.loads(line); done[r["doc"]] = r["pred"]
    todo = [(n, d) for n, d in items if n not in done]
    m.eval()
    for i in range(0, len(todo), GEN_BATCH):
        if time_left() < 0:
            print("TIME BUDGET REACHED during generation; re-run with this output as input"); break
        b = [d for _, d in todo[i:i + GEN_BATCH]]
        x = tok(b, truncation=True, max_length=MAX_IN, padding=True, return_tensors="pt").to(DEV)
        g = torch.zeros_like(x["input_ids"]); g[:, 0] = 1
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=FP16):
            y = m.generate(**x, global_attention_mask=g, max_new_tokens=MAX_OUT, num_beams=BEAMS,
                           no_repeat_ngram_size=3, early_stopping=True)
        with open(out_file, "a") as f:
            for (n, _), p in zip(todo[i:i + GEN_BATCH], tok.batch_decode(y, skip_special_tokens=True)):
                done[n] = p; f.write(json.dumps({"doc": n, "pred": p}) + "\\n")
        print(f"  {out_file.stem}: {len(done)}/{len(items)} ({time.time() - T_START:.0f}s)", flush=True)
    return done

def scores(preds, refs):
    from rouge_score import rouge_scorer
    rs = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True)
    r = [rs.score(t, p) for p, t in zip(preds, refs)]
    res = {k: round(sum(x[k].fmeasure for x in r) / len(r), 4) for k in ("rouge1", "rouge2", "rougeL", "rougeLsum")}
    try:
        from bert_score import score as bs
        _, _, f = bs(preds, refs, lang="en", batch_size=8, device=DEV if DEV != "mps" else "cpu")
        res["bertscore_f1"] = round(f.mean().item(), 4)
        res["bertscore_model"] = "roberta-large (bert-score default for en), no baseline rescaling"
    except Exception as e:
        res["bertscore_f1"] = None; res["bertscore_error"] = str(e)[:200]
    return res

items, refs = [(n, d) for n, d, _ in test], [s for _, _, s in test]
names = [n for n, _ in items]
tuned = summarise(model, items, WORK / "preds_finetuned.jsonl") if TRAINED else {}
del model; torch.cuda.empty_cache() if DEV == "cuda" else None
base = {}
if TRAINED and len(tuned) == len(items):
    base = summarise(LEDForConditionalGeneration.from_pretrained(MODEL).to(DEV), items, WORK / "preds_untuned.jsonl")
COMPLETE = TRAINED and len(tuned) == len(base) == len(items)
if not COMPLETE:
    print(f"NOT FINISHED: trained={TRAINED}, fine-tuned summaries {len(tuned)}/{len(items)}, untuned {len(base)}/{len(items)}. "
          "Add this version's output as input and run Save & Run All again.")
else:
    tuned_preds, base_preds = [tuned[n] for n in names], [base[n] for n in names]
    results = {"dataset": "IN-Abs test", "n_docs": len(test), "smoke_test": SMOKE, "model": MODEL,
               "settings": {"max_input": MAX_IN, "max_output": MAX_OUT, "lr": LR, "epochs": EPOCHS, "grad_accum": ACCUM,
                            "batch": 1, "fp16": FP16, "grad_checkpointing": True, "beams": BEAMS, "n_train": len(train)},
               "val_loss_finetuned": tuned_val, "untuned": scores(base_preds, refs), "finetuned": scores(tuned_preds, refs),
               "note": "rougeL = sentence-agnostic LCS; rougeLsum = summary-level LCS over newline-split sentences; references longer than MAX_OUT tokens cannot be fully reproduced"}
    (WORK / "stage5_inabs_results.json").write_text(json.dumps(results, indent=2))
    import csv
    with open(WORK / "stage5_inabs_predictions.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["doc", "reference", "untuned", "finetuned"])
        for (n, _, r), b, t in zip(test, base_preds, tuned_preds):
            w.writerow([n, r, b, t])
    print(json.dumps(results, indent=2))"""),
    ("code", """if COMPLETE:
    shutil.make_archive(str(WORK / "legal_led_inabs"), "zip", WORK / "legal_led_inabs")
print(sorted(p.name for p in WORK.iterdir()))"""),
]


def cell(kind, src):
    lines = src.splitlines(keepends=True)
    if kind == "md":
        return {"cell_type": "markdown", "metadata": {}, "source": lines}
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines}


nb = {"cells": [cell(k, s) for k, s in CELLS], "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                   "language_info": {"name": "python"}, "kaggle": {"accelerator": "gpu", "isInternetEnabled": True}}}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1))
print("->", OUT)
