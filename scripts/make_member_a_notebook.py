"""Write notebooks/member_a_gpu.ipynb: Member A's GPU steps on a Kaggle T4, run unattended.

    python scripts/make_member_a_notebook.py

The notebook expects the Kaggle dataset made by scripts/make_kaggle_bundle.py (attached as `nyayasethu-data`).
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "member_a_gpu.ipynb"

MD, CODE = "markdown", "code"
CELLS = [
    (MD, """# NyayaSethu: Member A GPU steps (Kaggle T4)

Runs steps 19-27 and 32-34 on the same corpus, role labels and SC-PCR benchmark as the local machine, then packs
the models and logs into `/kaggle/working/member_a_gpu_outputs.zip`.

**Setup**
1. *Add Input* → your dataset **nyayasethu-data** (upload of `nyayasethu-data.zip`).
2. *Settings* → Accelerator **GPU T4 x1** (or P100), Internet **on**.
3. *Save Version* → **Save & Run All (Commit)**. It runs in the background (about 3-5 hours); you can close the tab.
4. When it finishes, open the version → *Output* → download `member_a_gpu_outputs.zip`.

Every step logs to `logs/kaggle_*.out`; if one fails, the later ones still run where they can."""),
    (CODE, """import subprocess, sys, os, time
!nvidia-smi --query-gpu=name,memory.total --format=csv
!git clone -q https://github.com/aishwaryanachammain-create/nyayasethu /kaggle/working/nyayasethu
%cd /kaggle/working/nyayasethu
!git log --oneline -1"""),
    (CODE, """# Dependencies (Kaggle already has torch/transformers; this adds the rest)
!pip install -q -r requirements.txt 2>&1 | tail -3"""),
    (CODE, """# Data: symlink the dataset into the repo layout (read-only files, nothing copied)
import glob
DATA = sorted(glob.glob('/kaggle/input/nyayasethu-data*') + glob.glob('/kaggle/input/*/nyayasethu-data*'))[0]
while not os.path.exists(os.path.join(DATA, 'data')) and len(os.listdir(DATA)) == 1:
    DATA = os.path.join(DATA, os.listdir(DATA)[0])          # unwrap a single top-level folder
print('dataset:', DATA)
!cp -rs {DATA}/. .
!ls -la data/processed data/processed/sc-pcr models/stage1
!python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')\""""),
    (CODE, """def step(name, cmd):
    \"\"\"Run one pipeline step, tee its output to logs/kaggle_<name>.out, keep going on failure.\"\"\"
    t = time.time()
    print(f'===== {name}: {cmd}', flush=True)
    with open(f'logs/kaggle_{name}.out', 'w') as log:
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            log.write(line)
            if 'it/s]' not in line and 's/it]' not in line:
                print(line, end='', flush=True)
        p.wait()
    status = 'OK' if p.returncode == 0 else f'FAILED ({p.returncode})'
    print(f'===== {name}: {status} in {(time.time() - t) / 60:.1f} min', flush=True)
    with open('logs/kaggle_steps.tsv', 'a') as f:
        f.write(f'{name}\\t{status}\\t{(time.time() - t) / 60:.1f}\\n')
    return p.returncode == 0"""),
    (MD, "## Step 20: fine-tune the bi-encoder (InLegalBERT, SC-PCR train queries)"),
    (CODE, "step('biencoder', 'python -m stage2_retrieval.train_biencoder')"),
    (MD, "## Steps 19 + 21: role-partitioned BM25 + FAISS index of the whole corpus (used by the API and the demo)"),
    (CODE, "step('index_corpus', 'python -m stage2_retrieval.build_index --corpus')"),
    (MD, "## Step 25: train the cross-encoder"),
    (CODE, "step('cross_encoder', 'python -m stage3_rerank.cross_encoder train')"),
    (MD, "## Steps 26-27: rerank, alpha tuned on dev, scored on test (SC-PCR last, so its alpha is the one the API uses)"),
    (CODE, """step('rerank_aila', 'python -m stage3_rerank.rerank --bench aila')
step('rerank_sc-pcr', 'python -m stage3_rerank.rerank --bench sc-pcr')"""),
    (MD, "## Steps 24 + 33: ablation table (BM25 → + dense → + roles → + rerank → + PageRank)"),
    (CODE, "step('ablation', 'python -m eval.ablation')"),
    (MD, "## Step 34: error analysis of the 5 worst SC-PCR test queries"),
    (CODE, "step('error_analysis', 'python -m eval.error_analysis --bench sc-pcr')"),
    (MD, """## Steps 12-17: Stage 1 with frozen InLegalBERT features (report only)

Trained into `models/stage1_inlegalbert`; the LSA labeller in `models/stage1` (used for the corpus role labels and
query splitting above) is put back afterwards. Includes the hand-annotated Issue judgments if the dataset has
`data/processed/roles_gold.jsonl`."""),
    (CODE, """!mv models/stage1 models/stage1_lsa && mkdir -p models/stage1
step('stage1_inlegalbert', 'python -m stage1_roles.train --features inlegalbert')
!rm -rf models/stage1_inlegalbert && mv models/stage1 models/stage1_inlegalbert && mv models/stage1_lsa models/stage1
!ls models"""),
    (MD, "## Pack the outputs"),
    (CODE, """!cat logs/kaggle_steps.tsv
!cd /kaggle/working/nyayasethu && zip -qr /kaggle/working/member_a_gpu_outputs.zip models/biencoder models/cross_encoder models/index_corpus models/stage3_alpha.json models/stage1_inlegalbert logs 2>&1 | tail -2
!ls -la /kaggle/working/member_a_gpu_outputs.zip
!rm -rf /kaggle/working/nyayasethu   # keep only the zip in the notebook output"""),
]


def main():
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                       "language_info": {"name": "python"},
                       "kaggle": {"accelerator": "gpu", "isInternetEnabled": True}},
          "cells": []}
    for i, (kind, src) in enumerate(CELLS):
        cell = {"cell_type": kind, "id": f"c{i:02d}", "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == CODE:
            cell.update(execution_count=None, outputs=[])
        nb["cells"].append(cell)
    OUT.write_text(json.dumps(nb, indent=1))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
