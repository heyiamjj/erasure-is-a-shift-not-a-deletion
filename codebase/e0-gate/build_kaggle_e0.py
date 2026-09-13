"""Builds the Kaggle E0 smoke-test notebook (e0-erasure-smoke-test.ipynb)."""
import json, os

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "kaggle_e0")
os.makedirs(OUT_DIR, exist_ok=True)

md_intro = """# E0 Smoke Test — Erasure is a Shift, Not a Deletion

**Goal (this run):** validate assets, environment, model loading, and the *premise gate*.
Tiny frame counts on purpose — we want to catch every technical bug before burning T4 quota.

- **BC** = `VLA-Adapter/LIBERO-Spatial-Pro` (standard BC, should be "forgotten")
- **AA** = `Dwipz/Anchor-Align/libero-spatial` (Anchor-Align, should be "preserved")
- **base** = `Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b` (pretrained VLM, intact)

**Gate metric (v3):** linear **CKA of text-token hidden states vs base** (the paper's own erasure
metric, Fig. 14: BC 0.34 vs AA ~0.95). Gate passes if `CKA(base, AA) - CKA(base, BC) >= 0.2`
(AA clearly closer to base's pretrained geometry than BC).

**Design lesson from smoke v1/v2 (now in the code):** direction *decodability* at the pre-action
position is action-adjacent — BC legitimately maximizes it (the paper's Fig. 13: BC has high action
decodability). So direction probes are reported as diagnostics only; CKA is the gate.

Also reported: frozen-head readout (diagnostic), CV direction probes (diagnostic), AP accuracy
(sanity of the action pathway). Set `SMOKE = False` for the full E0 gate (500 frames)."""

md_done = """## Summary

- **GATE PASS** → the released BC checkpoint is measurably forgotten vs base → proceed to E1 (patching).
- **GATE FAIL** → try the non-Pro `VLA-Adapter/LIBERO-Spatial` checkpoint, then reassess.

Results JSON was saved to `/kaggle/working/e0_smoke_results.json`."""

cells = []

def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)})

def code(src):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                  "source": src.splitlines(keepends=True)})

md(md_intro)

# ---------------- Cell: config ----------------
code("""SMOKE = True                       # True = tiny frames (fast, catches bugs); False = full E0 gate
NUM_VERIFY = 2 if SMOKE else 5
NUM_PROBE  = 60 if SMOKE else 500

REPO_DIR  = "/kaggle/working/Anchor-Align"
CKPT_DIR  = "/kaggle/working/ckpts"
BC_PATH   = f"{CKPT_DIR}/bc"                 # VLA-Adapter/LIBERO-Spatial-Pro
AA_PATH   = f"{CKPT_DIR}/aa/libero-spatial"  # Dwipz/Anchor-Align
BASE_PATH = f"{CKPT_DIR}/base"               # Stanford-ILIAD prism base
RLDS_NAME = "libero_spatial_no_noops"
OUT_JSON  = "/kaggle/working/e0_smoke_results.json"
print("SMOKE =", SMOKE)""")

# ---------------- Cell: env ----------------
code("""import os
if not os.path.isdir(REPO_DIR):
    # tarball download (wget) is far more reliable than git clone on Kaggle (clone can hang)
    # NOTE: one '!' per shell line only -- a second '!' inside the chain becomes '!mv' (command not found)
    !wget -q -T 120 -O /kaggle/working/Anchor-Align.tar.gz https://github.com/dwipddalal/Anchor-Align/archive/refs/heads/main.tar.gz || wget -q -T 120 -O /kaggle/working/Anchor-Align.tar.gz https://github.com/dwipddalal/Anchor-Align/archive/refs/heads/main.tar.gz
    !tar -xzf /kaggle/working/Anchor-Align.tar.gz -C /kaggle/working/ && mv /kaggle/working/Anchor-Align-main {REPO_DIR} && echo TAR_OK
!ls {REPO_DIR} | head -6
os.chdir(REPO_DIR)
print("cwd:", os.getcwd())

# Fork-based deps FIRST (slow git installs, run once)
# NOTE: dlimp fork hard-requires tensorflow==2.15.0 (no py3.12 wheel) -> install --no-deps
!pip install -q "dlimp @ git+https://github.com/moojink/dlimp_openvla" --no-deps && echo DIMP_OK || echo DIMP_FAIL
!pip install -q "transformers @ git+https://github.com/moojink/transformers-openvla-oft.git" && echo TRANSFORMERS_OK || echo TRANSFORMERS_FAIL

# Remaining pinned deps, installed manually for controlled failure (torch stays as Kaggle's)
# NOTE: DO NOT touch tensorflow/tfds/keras -- Kaggle's TF 2.20.0 + tfds 4.9.9 import cleanly;
#       TF 2.15/2.16 downgrades break the tensorboard/keras stack (RecursionError on import).
!pip install -q accelerate einops huggingface_hub json-numpy jsonlines matplotlib peft==0.11.1 protobuf rich wandb diffusers==0.30.3 imageio uvicorn fastapi draccus==0.8.0 2>&1 | tail -n 2
!pip install -q timm==0.9.10 tokenizers==0.19.1 2>&1 | tail -n 2
!pip install -q sentencepiece 2>&1 | tail -n 1

# Editable install of the repo package itself (no deps -> we installed them above; explicit path, no cwd reliance)
!pip install -q -e {REPO_DIR} --no-deps 2>&1 | tail -n 3

# Version sanity check (fails loudly on missing module)
!python -c "import torch, transformers, tokenizers, timm, tensorflow as tf, tensorflow_datasets as tfds, dlimp, absl; print('torch', torch.__version__); print('tf', tf.__version__); print('transformers', transformers.__version__); print('tokenizers', tokenizers.__version__); print('timm', timm.__version__); print('tfds', tfds.__version__); print('dlimp', getattr(dlimp,'__version__','?'))" """)

# ---------------- Cell: checkpoints ----------------
code("""from huggingface_hub import snapshot_download
import os, time
os.makedirs(CKPT_DIR, exist_ok=True)

def download_with_retry(**kwargs):
    for attempt in range(4):
        try:
            return snapshot_download(**kwargs)
        except Exception as e:
            print(f"  download attempt {attempt+1} failed: {type(e).__name__}: {str(e)[:120]}; retrying...")
            time.sleep(15)
    raise RuntimeError(f"download failed after 4 attempts: {kwargs.get('repo_id')}")

if not os.path.exists(os.path.join(BC_PATH, "model.safetensors")):
    print(">> downloading BC (VLA-Adapter-Pro) ...")
    download_with_retry(repo_id="VLA-Adapter/LIBERO-Spatial-Pro", local_dir=BC_PATH)
print("BC files:", sorted(os.listdir(BC_PATH)))

if not os.path.exists(os.path.join(AA_PATH, "model.safetensors")):
    print(">> downloading AA (libero-spatial) ...")
    download_with_retry(repo_id="Dwipz/Anchor-Align", local_dir=f"{CKPT_DIR}/aa",
                        allow_patterns=["config.json", "libero-spatial/**"])
print("AA files:", sorted(os.listdir(AA_PATH)))
print("AA has align_dir_proj:", any(f.startswith("align_dir_proj") for f in os.listdir(AA_PATH)))

if not os.path.exists(os.path.join(BASE_PATH, "checkpoints", "step-020792-epoch-01-loss=0.5268.pt")):
    print(">> downloading base VLM ...")
    download_with_retry(repo_id="Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b", local_dir=BASE_PATH,
                        allow_patterns=["config.json", "config.yaml",
                                        "checkpoints/step-020792-epoch-01-loss=0.5268.pt"])
print("base files:", sorted(os.listdir(BASE_PATH)))
!du -sh {CKPT_DIR}/*""")

# ---------------- Cell: RLDS ----------------
code("""rlds_target = f"{REPO_DIR}/data/libero/{RLDS_NAME}"
if not os.path.isdir(rlds_target):
    print(">> downloading RLDS libero_spatial_no_noops ...")
    download_with_retry(repo_id="openvla/modified_libero_rlds", repo_type="dataset",
                        local_dir="/kaggle/working/rlds_dl",
                        allow_patterns=["libero_spatial_no_noops/**"])
    os.makedirs(f"{REPO_DIR}/data/libero", exist_ok=True)
    !mv /kaggle/working/rlds_dl/libero_spatial_no_noops {REPO_DIR}/data/libero/
print("RLDS top-level:", os.listdir(f"{REPO_DIR}/data/libero/{RLDS_NAME}"))
!du -sh {rlds_target}""")

# ---------------- Cell: imports ----------------
code("""import os, sys, json, time
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, f"{REPO_DIR}/experiments/robot/libero")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import numpy as np
import torch
import torch.nn.functional as F
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")

import run_alignment_test as rat
from experiments.robot.openvla_utils import get_vla, get_action_head, get_proprio_projector, get_processor
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DIR_WORDS = rat.DIRECTION_WORDS""")

# ---------------- Cell: load BC + AA ----------------
code("""def load_vla(path, dataset_key=RLDS_NAME):
    cfg = rat._Cfg(path, dataset_key)
    vla = get_vla(cfg)
    ah  = get_action_head(cfg, vla.llm_dim)
    pp  = get_proprio_projector(cfg, vla.llm_dim, proprio_dim=8)
    proc = get_processor(cfg)
    head = rat.unwrap_lm_head(vla.language_model.lm_head)
    print(f"[{os.path.basename(path)}] llm_dim={vla.llm_dim} lm_head={type(head).__name__} "
          f"has_w={hasattr(head, 'weight')} norm_keys={list(vla.norm_stats.keys())}")
    return {"vla": vla, "ah": ah, "pp": pp, "proc": proc, "cfg": cfg}

print(">>> BC (VLA-Adapter-Pro)")
bc = load_vla(BC_PATH)
torch.cuda.empty_cache()
print(">>> AA (Anchor-Align libero-spatial)")
aa = load_vla(AA_PATH)
torch.cuda.empty_cache()""")

# ---------------- Cell: base VLM ----------------
code("""from prismatic.models.load import load as load_base_vlm
print(">>> base VLM (native Prismatic load)")
base_vlm = load_base_vlm(BASE_PATH, image_sequence_len=2)
base_vlm.to(DEVICE, dtype=torch.bfloat16).eval()
torch.cuda.empty_cache()

lm_head_w = base_vlm.llm_backbone.llm.lm_head.weight.detach().to("cpu").float()  # canonical frozen head
print("lm_head", tuple(lm_head_w.shape), lm_head_w.dtype)

tok = getattr(base_vlm.llm_backbone, "tokenizer", None)
if tok is None:  # fallback: build from checkpoint
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(AA_PATH)
dir_ids = {w: tok.encode(w, add_special_tokens=False)[0] for w in DIR_WORDS}
print("dir_ids (base tokenizer):", dir_ids)
for name, m in {"bc": bc, "aa": aa}.items():
    ck = {w: m["proc"].tokenizer.encode(w, add_special_tokens=False)[0] for w in DIR_WORDS}
    print(f"  {name} ckpt-tokenizer ids == base:", ck == dir_ids, ck)""")

# ---------------- Cell: forward + readout helpers ----------------
code("""def base_hidden_full(base_vlm, inputs):
    # processor pixel_values: (1, 12, 224, 224) = 2 images x (siglip3 + dino3)
    # base_vlm loaded in bf16 -> cast inputs to bf16 (fp32 input vs bf16 params = dtype error)
    pv = inputs["pixel_values"].to(torch.bfloat16).to(DEVICE)
    imgs = torch.split(pv, [6] * 2, dim=1)                       # 2 x (1, 6, H, W)
    siglip = torch.stack([im[:, :3] for im in imgs], dim=1)      # (1, T=2, 3, H, W)
    dino   = torch.stack([im[:, 3:] for im in imgs], dim=1)
    with torch.inference_mode():
        out = base_vlm(input_ids=inputs["input_ids"].to(DEVICE),
                       pixel_values={"dino": dino, "siglip": siglip},
                       output_hidden_states=True)
    return out.hidden_states  # tuple of 25, each (1, 1+512+N, 896)

def vla_hidden_full(model, inputs, proprio, unnorm):
    holder = {}
    inner = model["vla"].language_model.model
    orig = inner.forward
    def patched(*a, **k):
        k["output_hidden_states"] = True
        out = orig(*a, **k)
        holder["all"] = out.hidden_states
        return out
    inner.forward = patched
    try:
        with torch.inference_mode():
            action, _ = model["vla"].predict_action(**inputs, unnorm_key=unnorm, do_sample=False,
                                                    proprio=proprio, proprio_projector=model["pp"],
                                                    action_head=model["ah"], use_film=False)
    finally:
        inner.forward = orig
    return np.asarray(action), holder["all"]

def last_text_pos(inputs):
    n = inputs["input_ids"].shape[-1] - 1   # text tokens (excl BOS)
    return 512 + n                          # index of last text token (pre-action) in multimodal seq

def frozen_word(hvec, lm_head_w, dir_ids):
    logits = F.linear(hvec.float().cpu(), lm_head_w)   # (vocab,)
    ids = [dir_ids[w] for w in DIR_WORDS]
    return DIR_WORDS[int(logits[ids].argmax().item())]

TAIL_K = 8   # text-tail tokens used for CKA (aligned by negative offset; prompt length varies)

def get_hidden_and_action(model, inputs, proprio):
    # returns (h_last_text_token, action, text_tail_hidden_per_layer)
    pos = last_text_pos(inputs)
    if model["kind"] == "base":
        hs = base_hidden_full(model["vla"], inputs)
        tail = [hs[u][0, pos - TAIL_K + 1: pos + 1].detach().float() for u in range(len(hs))]
        return hs[-1][0, pos].detach().float(), None, tail
    unnorm = rat._resolve_unnorm_key(model["vla"], RLDS_NAME)
    act, hs = vla_hidden_full(model, inputs, proprio, unnorm)
    tail = [hs[u][0, pos - TAIL_K + 1: pos + 1].detach().float() for u in range(len(hs))]
    return hs[-1][0, pos].detach().float(), act, tail

def linear_cka(A, B):
    A = A - A.mean(0, keepdims=True)
    B = B - B.mean(0, keepdims=True)
    return (torch.norm(B.T @ A, "fro") ** 2) / (torch.norm(A.T @ A, "fro") * torch.norm(B.T @ B, "fro"))

models = {
    "base": {"kind": "base", "vla": base_vlm},
    "bc":   {"kind": "vla", **bc},
    "aa":   {"kind": "vla", **aa},
}""")

# ---------------- Cell: verify loop ----------------
code("""it = rat.make_rlds_iterator(RLDS_NAME)
for fi in range(NUM_VERIFY):
    sample = next(it)
    inputs = rat.build_inputs(bc["proc"], sample["primary"], sample["wrist"], sample["lang"], num_images=2)
    proprio = sample["proprio"][:8] if sample["proprio"] is not None else None
    gt = np.asarray(sample["actions"])[:, :3].mean(axis=0)
    gt_lab = rat.ap_top1_label_from_vec3(gt)
    pos = last_text_pos(inputs)
    print(f"\\n--- frame {fi} | lang={sample['lang'][:45]!r} | gt_lab={gt_lab} | last_text_pos={pos}")
    for name, m in models.items():
        try:
            h, act, _ = get_hidden_and_action(m, inputs, proprio)
            fw = frozen_word(h, lm_head_w, dir_ids)
            ap = rat.ap_top1_label_from_vec3(np.asarray(act)[:, :3].mean(0)) if act is not None else None
            print(f"    {name:5s} frozen_word={fw:9s} ap_label={ap}")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"    {name:5s} FAILED: {e}")""")

# ---------------- Cell: gate probe ----------------
code("""probe = {name: {"frozen_correct": 0, "n": 0, "ap_correct": 0, "feats": [], "labels": [], "words": [],
                "tails": []}  # per-layer text-tail states: list over frames of list over layers (K,d)
         for name in models}
skipped = 0
it = rat.make_rlds_iterator(RLDS_NAME)
for fi in range(NUM_PROBE):
    sample = next(it)
    gt = np.asarray(sample["actions"])[:, :3].mean(axis=0)
    gt_lab = rat.ap_top1_label_from_vec3(gt)
    if gt_lab is None:
        skipped += 1
        continue
    inputs = rat.build_inputs(bc["proc"], sample["primary"], sample["wrist"], sample["lang"], num_images=2)
    proprio = sample["proprio"][:8] if sample["proprio"] is not None else None
    for name, m in models.items():
        try:
            h, act, tail = get_hidden_and_action(m, inputs, proprio)
        except Exception as e:
            print(f"  probe frame {fi} {name} FAILED: {type(e).__name__}: {e}")
            continue
        fw = frozen_word(h, lm_head_w, dir_ids)
        p = probe[name]
        p["n"] += 1
        p["frozen_correct"] += int(fw == gt_lab)
        p["feats"].append(h.cpu().numpy()); p["labels"].append(gt_lab); p["words"].append(fw)
        p["tails"].append([t.cpu() for t in tail])
        if act is not None:
            ap = rat.ap_top1_label_from_vec3(np.asarray(act)[:, :3].mean(0))
            p["ap_correct"] += int(ap == gt_lab)
print(f"skipped stationary frames: {skipped}")

from collections import Counter
summary = {}
for name, p in probe.items():
    summary[name] = {"n": p["n"], "frozen_acc": p["frozen_correct"] / max(1, p["n"]),
                     "ap_acc": p["ap_correct"] / max(1, p["n"]),
                     "word_dist": dict(Counter(p["words"]).most_common())}
print("\\n=== DIAGNOSTIC: FROZEN-HEAD DIRECTION ACCURACY (last text token) ===")
for name, s in summary.items():
    print(f"  {name:5s} frozen_acc={s['frozen_acc']*100:5.1f}%  ap_acc={s['ap_acc']*100:5.1f}%  (n={s['n']})  dist={s['word_dist']}")

# === GATE: text-token CKA vs base (paper's erasure metric, Fig 14) ===
# Early layers are trivially ~1.0 (LoRA doesn't touch embeddings); the erasure lives in DEEP
# layers (paper Fig 14: BC collapses to 0.34 at layer 24). Gate on the last 6 layers.
n_layers = len(probe["base"]["tails"][0])
cka = {"bc": [], "aa": []}
for name in ["bc", "aa"]:
    for u in range(n_layers):
        A = torch.cat([f[u] for f in probe["base"]["tails"][: probe[name]["n"]]], dim=0)  # (n*K, d)
        B = torch.cat([f[u] for f in probe[name]["tails"]], dim=0)
        cka[name].append(float(linear_cka(A, B).item()))
print("\\n=== GATE: TEXT-TOKEN CKA vs BASE (per-layer; paper Fig 14: BC~0.34 deep, AA~0.95) ===")
print("  layer curve bc:", " ".join(f"{v*100:4.0f}" for v in cka["bc"]))
print("  layer curve aa:", " ".join(f"{v*100:4.0f}" for v in cka["aa"]))
deep_bc = float(np.mean(cka["bc"][-6:]))
deep_aa = float(np.mean(cka["aa"][-6:]))
print(f"  deep(6-layer) CKA: bc={deep_bc*100:5.1f}%  aa={deep_aa*100:5.1f}%  gap={100*(deep_aa-deep_bc):.1f} pts")
gate = (deep_aa - deep_bc) >= 0.20
print("GATE:", "PASS" if gate else "FAIL", f"(deep-layer CKA gap >= 20 pts required)")

# === DIAGNOSTIC: CV direction probes (action-adjacent; BC may legitimately win here) ===
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
lin = {}
for name, p in probe.items():
    if p["n"] >= 20:
        X = np.stack(p["feats"]); y = np.array(p["labels"])
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)
        accs = []
        for tr, te in rskf.split(X, y):
            clf = LogisticRegression(max_iter=2000, C=0.1)
            clf.fit(X[tr], y[tr])
            accs.append(clf.score(X[te], y[te]))
        lin[name] = float(np.mean(accs))
print("\\n=== DIAGNOSTIC: LINEAR-PROBE DIRECTION ACC (5-fold x3 CV, action-adjacent) ===")
for name, acc in lin.items():
    print(f"  {name:5s} probe_acc={acc*100:5.1f}%")

results = {"smoke": SMOKE, "num_verify": NUM_VERIFY, "num_probe": NUM_PROBE,
           "summary": summary, "linear_probe_cv": lin,
           "cka_mean_vs_base": {"bc": float(np.mean(cka["bc"])), "aa": float(np.mean(cka["aa"]))},
           "cka_deep6_vs_base": {"bc": deep_bc, "aa": deep_aa},
           "cka_by_layer_vs_base": {"bc": cka["bc"], "aa": cka["aa"]},
           "gate_pass": bool(gate), "skipped_stationary": skipped, "dir_ids": dir_ids,
           "env": {"torch": torch.__version__, "cuda": torch.cuda.is_available(),
                   "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}}
with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)
print("saved:", OUT_JSON)""")

# ---------------- Cell: summary ----------------
md(md_done)

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = os.path.join(OUT_DIR, "e0-erasure-smoke-test.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)
print("wrote", out_path)

metadata = {
    "id": "your-kaggle-username/e0-erasure-smoke-test",
    "title": "E0 Erasure Smoke Test",
    "code_file": "e0-erasure-smoke-test.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": True,
    "machine_shape": "NvidiaTeslaT4",
}
with open(os.path.join(OUT_DIR, "kernel-metadata.json"), "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4)
print("wrote kernel-metadata.json")