"""Builds the Kaggle E1 notebook (e1-erasure-patching.ipynb): phenomenon baseline + drift + patching."""
import json, os

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "kaggle_e1")
os.makedirs(OUT_DIR, exist_ok=True)

md_intro = """# E1 — Erasure Patching (Phenomenon + Drift + Recovery)

**Goal:** (1) full-scale phenomenon baseline (= E0 at 500 frames: per-layer CKA curves, CV
direction probes, AP sanity), (2) compute the drift Delta = base - BC per layer (vision + text
regions only — base has no action tokens), (3) patch BC at inference with alpha x Delta on
layer subsets and measure semantic recovery (CKA vs base) against action fidelity (AP acc +
action L2), (4) controls: reverse patch on base, random directions, per-sample ceiling,
AA-to-BC drift.

**Models:** base = `Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b` (bf16 native),
BC = `VLA-Adapter/LIBERO-Spatial-Pro`, AA = `Dwipz/Anchor-Align/libero-spatial`.

**Position alignment (verified in E0):** sequence = [BOS, 512 vision, N text, 64 actions, STOP];
text tokens at identical absolute indices in base and BC; Delta covers vision [1:513] and text
(aligned by negative offset from the pre-action token). Action region is NOT patched (no base
counterpart) — the bridge-attention action head still sees patched vision/text states through
self-attention, so the trade-off is measurable.

Set `SMOKE = False` for the full run (500/400/100 frames)."""

cells = []

def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)})

def code(src):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                  "source": src.splitlines(keepends=True)})

md(md_intro)

# ---------------- config ----------------
code("""SMOKE = False                       # True = tiny frames (bug-hunting); False = full run
STAGE1_PROBE = 60 if SMOKE else 500   # phenomenon baseline frames (E0 full-scale)
STAGE2_TRAIN = 30 if SMOKE else 400   # drift computation frames
STAGE3_EVAL  = 20 if SMOKE else 100   # patching eval frames

REPO_DIR  = "/kaggle/working/Anchor-Align"
CKPT_DIR  = "/kaggle/working/ckpts"
BC_PATH   = f"{CKPT_DIR}/bc"                 # VLA-Adapter/LIBERO-Spatial-Pro
AA_PATH   = f"{CKPT_DIR}/aa/libero-spatial"  # Dwipz/Anchor-Align
BASE_PATH = f"{CKPT_DIR}/base"               # Stanford-ILIAD prism base
RLDS_NAME = "libero_spatial_no_noops"
OUT_JSON  = "/kaggle/working/e1_results.json"

PATCH_LAYERS = list(range(24))                       # hook targets (Qwen2DecoderLayer 0..23)
SUBSETS = {"band21_24": (21, 24), "deep6": (18, 24), "all": (0, 24)}   # half-open [lo, hi)
ALPHA_GRID = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
TAIL_K = 8                                           # text-tail tokens for CKA
VIS_STRIDE = 8                                       # strided vision patches for CKA (64 patches)
RNG_SEED = 0
print("SMOKE =", SMOKE)""")

# ---------------- env (E0-proven recipe) ----------------
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

# ---------------- checkpoints (retry) ----------------
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

if not os.path.exists(os.path.join(AA_PATH, "model.safetensors")):
    print(">> downloading AA (libero-spatial) ...")
    download_with_retry(repo_id="Dwipz/Anchor-Align", local_dir=f"{CKPT_DIR}/aa",
                        allow_patterns=["config.json", "libero-spatial/**"])

if not os.path.exists(os.path.join(BASE_PATH, "checkpoints", "step-020792-epoch-01-loss=0.5268.pt")):
    print(">> downloading base VLM ...")
    download_with_retry(repo_id="Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b", local_dir=BASE_PATH,
                        allow_patterns=["config.json", "config.yaml",
                                        "checkpoints/step-020792-epoch-01-loss=0.5268.pt"])
!du -sh {CKPT_DIR}/*""")

# ---------------- RLDS ----------------
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

# ---------------- imports ----------------
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

# ---------------- load models ----------------
code("""def load_vla(path, dataset_key=RLDS_NAME):
    cfg = rat._Cfg(path, dataset_key)
    vla = get_vla(cfg)
    ah  = get_action_head(cfg, vla.llm_dim)
    pp  = get_proprio_projector(cfg, vla.llm_dim, proprio_dim=8)
    proc = get_processor(cfg)
    print(f"[{os.path.basename(path)}] llm_dim={vla.llm_dim} norm_keys={list(vla.norm_stats.keys())}")
    return {"vla": vla, "ah": ah, "pp": pp, "proc": proc, "cfg": cfg}

print(">>> BC (VLA-Adapter-Pro)")
bc = load_vla(BC_PATH)
torch.cuda.empty_cache()
print(">>> AA (Anchor-Align libero-spatial)")
aa = load_vla(AA_PATH)
torch.cuda.empty_cache()

from prismatic.models.load import load as load_base_vlm
print(">>> base VLM (native Prismatic load, bf16)")
base_vlm = load_base_vlm(BASE_PATH, image_sequence_len=2)
base_vlm.to(DEVICE, dtype=torch.bfloat16).eval()
torch.cuda.empty_cache()

lm_head_w = base_vlm.llm_backbone.llm.lm_head.weight.detach().to("cpu").float()
print("lm_head", tuple(lm_head_w.shape))
tok = getattr(base_vlm.llm_backbone, "tokenizer", None)
if tok is None:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(AA_PATH)
dir_ids = {w: tok.encode(w, add_special_tokens=False)[0] for w in DIR_WORDS}
print("dir_ids:", dir_ids)

models = {
    "base": {"kind": "base", "vla": base_vlm},
    "bc":   {"kind": "vla", **bc},
    "aa":   {"kind": "vla", **aa},
}""")

# ---------------- helpers ----------------
code("""def base_pixel_dict(pixel_values):
    # processor pixel_values: (1, 12, 224, 224) = 2 images x (siglip3 + dino3) -> native dict (1, T, 3, H, W)
    pv = pixel_values.to(torch.bfloat16).to(DEVICE)
    imgs = torch.split(pv, [6] * 2, dim=1)
    return {"dino": torch.stack([im[:, 3:] for im in imgs], dim=1),
            "siglip": torch.stack([im[:, :3] for im in imgs], dim=1)}

def base_hidden_full(base_vlm, inputs):
    with torch.inference_mode():
        out = base_vlm(input_ids=inputs["input_ids"].to(DEVICE),
                       pixel_values=base_pixel_dict(inputs["pixel_values"]),
                       output_hidden_states=True)
    return out.hidden_states

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
    return 512 + n

def linear_cka(A, B):
    A = A - A.mean(0, keepdims=True)
    B = B - B.mean(0, keepdims=True)
    num = torch.norm(B.T @ A, "fro") ** 2
    den = torch.norm(A.T @ A, "fro") * torch.norm(B.T @ B, "fro")
    return float((num / den).item())

def text_tail(hs, pos, k=TAIL_K):
    return [hs[u][0, pos - k + 1: pos + 1].detach().float() for u in range(len(hs))]

def vis_subset(hs, stride=VIS_STRIDE):
    return [hs[u][0, 1:513:stride].detach().float() for u in range(len(hs))]

def frozen_word(hvec, lm_head_w, dir_ids):
    logits = F.linear(hvec.float().cpu(), lm_head_w)
    ids = [dir_ids[w] for w in DIR_WORDS]
    return DIR_WORDS[int(logits[ids].argmax().item())]

def gt_and_ap_labels(sample_actions, action):
    gt = np.asarray(sample_actions)[:, :3].mean(axis=0)
    gt_lab = rat.ap_top1_label_from_vec3(gt)
    ap_lab = rat.ap_top1_label_from_vec3(np.asarray(action)[:, :3].mean(0)) if action is not None else None
    return gt_lab, ap_lab

def action_l2(sample_actions, action):
    gt = np.asarray(sample_actions).astype(np.float32)          # (8, 7)
    pred = np.asarray(action).astype(np.float32)                # (8, 7)
    return float(np.linalg.norm(pred - gt, axis=1).mean())

def install_patch_hooks(inner_model, offsets, layers=None, holder=None, diag=None):
    # offsets: {u: (S,d) bf16}; hooks add the offset to the layer's residual-stream output
    handles = []
    for u, off in offsets.items():
        if layers is not None and u not in layers:
            continue
        def make_hook(off_u):
            def hook(module, args, output):
                if hasattr(output, "last_hidden_state"):
                    h = output.last_hidden_state
                    add = off_u[: h.shape[1]].unsqueeze(0).to(h.dtype)
                    try:
                        output.last_hidden_state = h + add
                    except RuntimeError as e:
                        print(f"  [hook u={u}] h={tuple(h.shape)} off_u={tuple(off_u.shape)} err={e}")
                        raise
                    return output
                h = output[0]
                add = off_u[: h.shape[1]].unsqueeze(0).to(h.dtype)
                try:
                    return (h + add,) + output[1:]
                except RuntimeError as e:
                    print(f"  [hook u={u}] h={tuple(h.shape)} off_u={tuple(off_u.shape)} err={e}")
                    raise
            return hook
        handles.append(inner_model.layers[u].register_forward_hook(make_hook(off)))
    return handles

def run_patched(model, inputs, proprio, unnorm, offsets, layers=None):
    holder = {}
    inner = model["vla"].language_model.model
    orig_fwd = inner.forward
    handles = install_patch_hooks(inner, offsets, layers=layers)
    def wrapped(*a, **k):
        k["output_hidden_states"] = True
        out = orig_fwd(*a, **k)
        holder["all"] = out.hidden_states
        return out
    inner.forward = wrapped
    try:
        with torch.inference_mode():
            action, _ = model["vla"].predict_action(**inputs, unnorm_key=unnorm, do_sample=False,
                                                    proprio=proprio, proprio_projector=model["pp"],
                                                    action_head=model["ah"], use_film=False)
    finally:
        inner.forward = orig_fwd
        for h in handles:
            h.remove()
    return np.asarray(action), holder["all"]

def run_patched_base(base_vlm, inputs, offsets, layers=None):
    holder = {}
    inner = base_vlm.llm_backbone.llm.model   # HF Qwen2Model (native backbone wraps llm)
    orig_fwd = inner.forward
    handles = install_patch_hooks(inner, offsets, layers=layers)
    def wrapped(*a, **k):
        k["output_hidden_states"] = True
        out = orig_fwd(*a, **k)
        holder["all"] = out.hidden_states
        return out
    inner.forward = wrapped
    try:
        with torch.inference_mode():
            base_vlm(input_ids=inputs["input_ids"].to(DEVICE),
                     pixel_values=base_pixel_dict(inputs["pixel_values"]),
                     output_hidden_states=True)
    finally:
        inner.forward = orig_fwd
        for h in handles:
            h.remove()
    return None, holder["all"]

def build_offsets(n, alpha, mask, delta_map):
    # returns {u: (S', d) bf16}; S' = 1+512+n+64+1 ; action region = 0 (base has no action tokens)
    S = 1 + 512 + n + 65
    offs = {}
    for u in PATCH_LAYERS:
        off = torch.zeros(S, 896, device=DEVICE)
        if mask in ("all", "vision"):
            off[1:513] = delta_map[u]["vis"] * alpha
        if mask in ("all", "text"):
            k = min(n, delta_map[u]["txt"].shape[0])
            off[513:513 + k] = delta_map[u]["txt"][:k].flip(0) * alpha
        offs[u] = off.to(torch.bfloat16)
    return offs

def random_offsets(n, alpha, layers, delta_map, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    S = 1 + 512 + n + 65
    offs = {}
    for u in layers:
        base_norm = torch.norm(delta_map[u]["vis"]) ** 2 + torch.norm(delta_map[u]["txt"][:n]) ** 2
        r = torch.randn(S, 896, generator=g, device=DEVICE)
        q, _ = torch.linalg.qr(r.T, mode="complete")  # q (896, 896) orthonormal; q[:S] -> (S, 896) rows
        rows = q[:S]
        scale = torch.sqrt(base_norm / max(1e-8, S))
        offs[u] = (rows * scale * alpha).to(torch.bfloat16)
    return offs""")

# ---------------- Stage 1: phenomenon baseline (E0 full-scale) ----------------
code("""print("\\n########## STAGE 1: PHENOMENON BASELINE ##########")
probe = {name: {"frozen_correct": 0, "n": 0, "ap_correct": 0, "feats": [], "labels": [], "words": [], "tails": []}
         for name in models}
skipped = 0
frame_cache = []   # first STAGE3_EVAL frames, reused by stage 3 (phenomenon + patching on SAME frames)
it = rat.make_rlds_iterator(RLDS_NAME)
for fi in range(STAGE1_PROBE):
    sample = next(it)
    gt = np.asarray(sample["actions"])[:, :3].mean(axis=0)
    gt_lab = rat.ap_top1_label_from_vec3(gt)
    if gt_lab is None:
        skipped += 1
        continue
    inputs = rat.build_inputs(bc["proc"], sample["primary"], sample["wrist"], sample["lang"], num_images=2)
    proprio = sample["proprio"][:8] if sample["proprio"] is not None else None
    pos = last_text_pos(inputs)
    if len(frame_cache) < STAGE3_EVAL:
        frame_cache.append({"inputs": inputs, "proprio": proprio, "gt_lab": gt_lab,
                            "actions": sample["actions"], "n": inputs["input_ids"].shape[-1] - 1})
    for name, m in models.items():
        try:
            if m["kind"] == "base":
                hs = base_hidden_full(m["vla"], inputs)
                h = hs[-1][0, pos].detach().float(); act = None
            else:
                unnorm = rat._resolve_unnorm_key(m["vla"], RLDS_NAME)
                act, hs = vla_hidden_full(m, inputs, proprio, unnorm)
                h = hs[-1][0, pos].detach().float()
        except Exception as e:
            print(f"  stage1 frame {fi} {name} FAILED: {type(e).__name__}: {e}")
            continue
        fw = frozen_word(h, lm_head_w, dir_ids)
        p = probe[name]
        p["n"] += 1
        p["frozen_correct"] += int(fw == gt_lab)
        p["feats"].append(h.cpu().numpy()); p["labels"].append(gt_lab); p["words"].append(fw)
        p["tails"].append([t.cpu() for t in text_tail(hs, pos)])
        if act is not None:
            _, ap_lab = gt_and_ap_labels(sample["actions"], act)
            p["ap_correct"] += int(ap_lab == gt_lab)
print(f"skipped stationary: {skipped}")

from collections import Counter
summary = {}
for name, p in probe.items():
    summary[name] = {"n": p["n"], "frozen_acc": p["frozen_correct"] / max(1, p["n"]),
                     "ap_acc": p["ap_correct"] / max(1, p["n"]),
                     "word_dist": dict(Counter(p["words"]).most_common())}
print("=== DIAGNOSTIC: FROZEN-HEAD DIRECTION ACC ===")
for name, s in summary.items():
    print(f"  {name:5s} frozen_acc={s['frozen_acc']*100:5.1f}%  ap_acc={s['ap_acc']*100:5.1f}%  (n={s['n']})  dist={s['word_dist']}")

n_layers = len(probe["base"]["tails"][0])
cka = {"bc": [], "aa": []}
for name in ["bc", "aa"]:
    for u in range(n_layers):
        A = torch.cat([f[u] for f in probe["base"]["tails"][: probe[name]["n"]]], dim=0)
        B = torch.cat([f[u] for f in probe[name]["tails"]], dim=0)
        cka[name].append(linear_cka(A, B))
print("=== GATE: TEXT-TOKEN CKA vs BASE (per-layer) ===")
print("  bc:", " ".join(f"{v*100:4.0f}" for v in cka["bc"]))
print("  aa:", " ".join(f"{v*100:4.0f}" for v in cka["aa"]))
deep_bc = float(np.mean(cka["bc"][-6:])); deep_aa = float(np.mean(cka["aa"][-6:]))
print(f"  deep6: bc={deep_bc*100:.1f}% aa={deep_aa*100:.1f}% gap={100*(deep_aa-deep_bc):.1f} pts")
print("  GATE:", "PASS" if (deep_aa - deep_bc) >= 0.20 else "FAIL")

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
lin = {}
for name, p in probe.items():
    if p["n"] >= 20:
        X = np.stack(p["feats"]); y = np.array(p["labels"])
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RNG_SEED)
        accs = []
        for tr, te in rskf.split(X, y):
            clf = LogisticRegression(max_iter=2000, C=0.1)
            clf.fit(X[tr], y[tr])
            accs.append(clf.score(X[te], y[te]))
        lin[name] = float(np.mean(accs))
print("=== DIAGNOSTIC: CV DIRECTION PROBES (action-adjacent) ===")
for name, acc in lin.items():
    print(f"  {name:5s} probe_acc={acc*100:5.1f}%")

stage1 = {"summary": summary, "linear_probe_cv": lin, "cka_by_layer": {"bc": cka["bc"], "aa": cka["aa"]},
          "deep6": {"bc": deep_bc, "aa": deep_aa}, "skipped": skipped}""")

# ---------------- Stage 2: drift computation ----------------
code("""print("\\n########## STAGE 2: DRIFT COMPUTATION (train frames) ##########")
# Delta per layer u: vision (512,d), text (maxN,d by offset from pre-action). Action region: none (base has no actions).
n_dev = 0
vis_sums = {u: torch.zeros(512, 896, dtype=torch.float32, device=DEVICE) for u in PATCH_LAYERS}
txt_sums = {u: torch.zeros(512, 896, dtype=torch.float32, device=DEVICE) for u in PATCH_LAYERS}  # maxN <= 512 guard
txt_cnt  = {u: 0 for u in PATCH_LAYERS}
vis_cnt  = {u: 0 for u in PATCH_LAYERS}
max_n = 0
it = rat.make_rlds_iterator(RLDS_NAME)
for fi in range(STAGE2_TRAIN):
    sample = next(it)
    inputs = rat.build_inputs(bc["proc"], sample["primary"], sample["wrist"], sample["lang"], num_images=2)
    proprio = sample["proprio"][:8] if sample["proprio"] is not None else None
    n = inputs["input_ids"].shape[-1] - 1
    max_n = max(max_n, n)
    pos = 512 + n
    hs_b = base_hidden_full(base_vlm, inputs)
    unnorm = rat._resolve_unnorm_key(bc["vla"], RLDS_NAME)
    _, hs_c = vla_hidden_full(bc, inputs, proprio, unnorm)
    for u in PATCH_LAYERS:
        hb = hs_b[u + 1][0].float()          # (1+512+n, d)  -- base has NO action tokens
        hc = hs_c[u + 1][0].float()          # (1+512+n+65, d)
        # region-wise subtraction: vision [1:513] and text [513:513+n] share absolute indices
        vis_sums[u] += (hb[1:513] - hc[1:513])
        txt_sums[u][:n] += (hb[513:513 + n] - hc[513:513 + n]).flip(0)   # flip: idx 0 = pre-action token
        vis_cnt[u] += 1
        txt_cnt[u] += 1
    n_dev += 1
    if (fi + 1) % 10 == 0:
        print(f"  train frame {fi+1}/{STAGE2_TRAIN} (max_n={max_n})")
delta = {}
for u in PATCH_LAYERS:
    delta[u] = {"vis": (vis_sums[u] / max(1, vis_cnt[u])), "txt": (txt_sums[u] / max(1, txt_cnt[u]))[:max_n]}
print(f"computed delta for {len(delta)} layers, max_n={max_n}, frames={n_dev}")""")

# ---------------- Stage 3+4: patching sweeps + controls ----------------
code("""print("\\n########## STAGE 3+4: PATCHING SWEEPS + CONTROLS ##########")

# ---- build config list ----
configs = []
configs.append(("row_base", "base", None))
configs.append(("row_bc_a0", "bc", None))
for sname, (lo, hi) in SUBSETS.items():
    lset = list(range(lo, hi))
    for a in ALPHA_GRID:
        configs.append((f"bc_{sname}_a{a}", "bc", ("all", lset, a)))
configs.append(("bc_deep6_text_a0.5", "bc", ("text", list(range(18, 24)), 0.5)))
configs.append(("bc_deep6_text_a1.0", "bc", ("text", list(range(18, 24)), 1.0)))
configs.append(("bc_deep6_vis_a1.0", "bc", ("vision", list(range(18, 24)), 1.0)))
controls = [
    ("ctl_reverse_base",  "base", ("all", list(range(18, 24)), -1.0)),
    ("ctl_random",        "bc",   ("RANDOM", list(range(18, 24)), 1.0)),
    ("ctl_persample",     "bc",   ("PERSAMPLE", list(range(18, 24)), 1.0)),
    ("ctl_aa_toward_bc",  "aa",   ("AABC", list(range(18, 24)), 1.0)),
]
print(f"{len(configs)} configs + {len(controls)} controls")

res = {name: {"cka_text_deep": [], "cka_text_l24": [], "cka_vis_deep": [], "ap_correct": 0, "n": 0,
              "l2": [], "frozen_correct": 0}
       for name, _, _ in configs + controls}

def eval_row(rname, hs, act, gt_lab, sample_actions, tail_b, vis_b):
    r = res[rname]; r["n"] += 1
    tail = text_tail(hs, last_text_pos(inputs)); vis = vis_subset(hs)
    deep = [linear_cka(tail_b[u], tail[u]) for u in range(len(hs))]
    r["cka_text_deep"].append(float(np.mean(deep[-6:])))
    r["cka_text_l24"].append(float(deep[-1]))
    r["cka_vis_deep"].append(float(np.mean([linear_cka(vis_b[u], vis[u]) for u in range(len(hs))][-6:])))
    h = hs[-1][0, last_text_pos(inputs)].detach().float()
    r["frozen_correct"] += int(frozen_word(h, lm_head_w, dir_ids) == gt_lab)
    if act is not None:
        _, ap_lab = gt_and_ap_labels(sample_actions, act)
        r["ap_correct"] += int(ap_lab == gt_lab)
        r["l2"].append(action_l2(sample_actions, act))

it = rat.make_rlds_iterator(RLDS_NAME)
t0 = time.time()
print(f"stage3 using {len(frame_cache)} cached frames (same as stage 1 phenomenon frames)")
for fi, fr in enumerate(frame_cache):
    sample_actions = fr["actions"]
    gt_lab = fr["gt_lab"]
    inputs = fr["inputs"]
    proprio = fr["proprio"]
    n = fr["n"]
    unnorm = rat._resolve_unnorm_key(bc["vla"], RLDS_NAME)

    # per-frame caches: base all-layer states, bc unpatched states + action
    hs_base = base_hidden_full(base_vlm, inputs)
    act0, hs_bc0 = vla_hidden_full(bc, inputs, proprio, unnorm)
    tail_b = text_tail(hs_base, last_text_pos(inputs)); vis_b = vis_subset(hs_base)

    eval_row("row_base", hs_base, None, gt_lab, sample_actions, tail_b, vis_b)
    eval_row("row_bc_a0", hs_bc0, act0, gt_lab, sample_actions, tail_b, vis_b)

    for cname, mname, spec in configs[2:]:
        mask, lset, a = spec
        if fi == 0:
            print("  cfg:", cname)
        offs = build_offsets(n, a, mask, delta)
        act, hs = run_patched(bc, inputs, proprio, unnorm, offs, layers=set(lset))
        eval_row(cname, hs, act, gt_lab, sample_actions, tail_b, vis_b)

    for cname, mname, spec in controls:
        kind, lset, a = spec
        if fi == 0:
            print("  ctl:", cname)
        if kind == "RANDOM":
            offs = random_offsets(n, a, lset, delta, RNG_SEED + fi)
            act, hs = run_patched(bc, inputs, proprio, unnorm, offs, layers=set(lset))
        elif kind == "PERSAMPLE":
            offs = {}
            for u in lset:
                off = torch.zeros(1 + 512 + n + 65, 896, device=DEVICE)
                off[1:513] = (hs_base[u + 1][0, 1:513] - hs_bc0[u + 1][0, 1:513]).float()
                off[513:513 + n] = (hs_base[u + 1][0, 513:513 + n] - hs_bc0[u + 1][0, 513:513 + n]).float()
                offs[u] = off.to(torch.bfloat16)
            act, hs = run_patched(bc, inputs, proprio, unnorm, offs, layers=set(lset))
        elif kind == "AABC":
            # patch AA toward BC: aa' = aa + (bc - aa)  ->  hidden states become BC's at patched layers
            _, hs_aa = vla_hidden_full(aa, inputs, proprio, rat._resolve_unnorm_key(aa["vla"], RLDS_NAME))
            offs = {}
            for u in lset:
                off = torch.zeros(1 + 512 + n + 65, 896, device=DEVICE)
                off[1:513] = (hs_bc0[u + 1][0, 1:513] - hs_aa[u + 1][0, 1:513]).float()
                off[513:513 + n] = (hs_bc0[u + 1][0, 513:513 + n] - hs_aa[u + 1][0, 513:513 + n]).float()
                offs[u] = off.to(torch.bfloat16)
            act, hs = run_patched(aa, inputs, proprio, rat._resolve_unnorm_key(aa["vla"], RLDS_NAME), offs,
                                  layers=set(lset))
        elif mname == "base":
            # reverse control: base + (-Delta) = BC's states at patched layers (directionality check)
            offs = build_offsets(n, a, kind, delta)
            act, hs = run_patched_base(base_vlm, inputs, offs, layers=set(lset))
        else:
            offs = build_offsets(n, a, kind, delta)
            act, hs = run_patched(bc, inputs, proprio, unnorm, offs, layers=set(lset))
        eval_row(cname, hs, act, gt_lab, sample_actions, tail_b, vis_b)

    if (fi + 1) % 10 == 0:
        print(f"  eval frame {fi+1}/{STAGE3_EVAL} elapsed={time.time()-t0:.0f}s")

print("\\n=== RESULTS ===")
for cname, _, _ in configs + controls:
    r = res[cname]
    nn = max(1, r["n"])
    l2 = np.mean(r["l2"]) if r["l2"] else float("nan")
    print(f"  {cname:26s} n={r['n']:3d} cka_text_deep={np.mean(r['cka_text_deep'])*100:5.1f}% "
          f"cka_text_l24={np.mean(r['cka_text_l24'])*100:5.1f}% cka_vis_deep={np.mean(r['cka_vis_deep'])*100:5.1f}% "
          f"ap_acc={100*r['ap_correct']/nn:5.1f}% l2={l2:6.3f}")

# === MECHANICS CHECK (validates the patching machinery + experiment direction) ===
if "ctl_persample" in res and "ctl_reverse_base" in res and "row_bc_a0" in res:
    pc = float(np.mean(res["ctl_persample"]["cka_text_deep"]))
    rv = float(np.mean(res["ctl_reverse_base"]["cka_text_deep"]))
    bc0 = float(np.mean(res["row_bc_a0"]["cka_text_deep"]))
    aa_tb = float(np.mean(res["ctl_aa_toward_bc"]["cka_text_deep"]))
    print("\\n=== MECHANICS CHECK ===")
    print(f"  persample ceiling  = {pc*100:5.1f}%   (expect ~100: patching must reproduce base states)")
    print(f"  reverse base-Delta = {rv*100:5.1f}%   (expect ~bc {bc0*100:.1f}%: base+(-Delta) ~= bc states)")
    print(f"  aa toward bc       = {aa_tb*100:5.1f}% (expect drop toward bc {bc0*100:.1f}%: drift axis is semantic)")
    ok = (pc > 0.95) and (rv < bc0 + 0.15) and (aa_tb < bc0 + 0.15)
    print("  MECHANICS:", "OK" if ok else "SUSPICIOUS - inspect before full run")

out = {"stage1": stage1, "configs": [c[0] for c in configs + controls], "rows": {}}
for cname, _, _ in configs + controls:
    r = res[cname]
    row = {"n": r["n"], "ap_correct": r["ap_correct"], "frozen_correct": r["frozen_correct"]}
    for k in ["cka_text_deep", "cka_text_l24", "cka_vis_deep"]:
        row[k] = float(np.mean(r[k])) if r[k] else None
    row["l2_mean"] = float(np.mean(r["l2"])) if r["l2"] else None
    out["rows"][cname] = row
with open(OUT_JSON, "w") as f:
    json.dump(out, f, indent=2)
print("saved:", OUT_JSON)""")

# ---------------- md summary ----------------
md("""## Summary

- **Stage 1** = full-scale phenomenon (CKA curves, probes, AP) — the paper's Phenomenon figure data.
- **Stage 2** = drift Delta (vision+text regions only) per layer, computed on train frames.
- **Stage 3** = patched-BC semantic recovery (CKA vs base) vs action fidelity (AP acc + L2) across
  alpha x layer subsets x token masks.
- **Stage 4** = controls: reverse (base - Delta), random directions, per-sample ceiling, AA+drift.

Headline: if CKA(base, patched-BC) rises with alpha toward base levels while AP/L2 degrade only
modestly, "erasure is a shift" is supported. Results saved to `/kaggle/working/e1_results.json`.""")

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = os.path.join(OUT_DIR, "e1-erasure-patching.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)
print("wrote", out_path)

metadata = {
    "id": "your-kaggle-username/e1-erasure-patching",
    "title": "E1 Erasure Patching",
    "code_file": "e1-erasure-patching.ipynb",
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