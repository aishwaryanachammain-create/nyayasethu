import os, subprocess
from pathlib import Path
from huggingface_hub import login, snapshot_download
from datasets import load_dataset, get_dataset_config_names

ROOT = Path(__file__).resolve().parents[1]
RAW, MODELS = ROOT / "data" / "raw", ROOT / "models"
RAW.mkdir(parents=True, exist_ok=True); MODELS.mkdir(exist_ok=True)

login(token=os.environ["HF_TOKEN"]) if os.environ.get("HF_TOKEN") else login()

# 1. IL-TUR: cjpe = ILDC, pcr = IL-PCR, rr = rhetorical roles, summ = summarization
REPO = "Exploration-Lab/IL-TUR"
try:
    configs = get_dataset_config_names(REPO)
except Exception:
    configs = ["cjpe", "pcr", "rr", "summ"]
print("IL-TUR configs:", configs)

for want in ["cjpe", "pcr", "rr", "summ"]:
    match = [c for c in configs if c.lower() == want]
    out = RAW / "il_tur" / want
    if not match or out.exists():
        print("skip", want); continue
    try:
        ds = load_dataset(REPO, match[0])
    except Exception as e:
        print(f"{want}: default load failed ({type(e).__name__}), retrying revision='script'")
        ds = load_dataset(REPO, match[0], revision="script")
    ds.save_to_disk(str(out))
    print(want, ds)

# 2. Models
for m in ["law-ai/InLegalBERT", "nsi319/legal-led-base-16384",
          "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"]:
    dest = MODELS / m.split("/")[-1]
    if not dest.exists():
        snapshot_download(m, local_dir=str(dest))
        print("model:", m)

# 3. GitHub repos (data or download links are in their READMEs)
REPOS = {
    "LegalSeg": "https://github.com/ShubhamKumarNigam/LegalSeg.git",
    "opennyai_rr": "https://github.com/Legal-NLP-EkStep/rhetorical-role-baseline.git",
    "law_ai_summarization": "https://github.com/Law-AI/summarization.git",
}
for name, url in REPOS.items():
    dest = RAW / name
    if not dest.exists():
        subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=False)

print("\nDone. Manual steps left: see each cloned repo's README for data links.")