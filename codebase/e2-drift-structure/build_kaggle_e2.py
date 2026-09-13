"""Builds the Kaggle E2 notebook (e2-drift-structure.ipynb): drift SVD, top-k recovery, depth profile."""
import json, os

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "kaggle_e2")
os.makedirs(OUT_DIR, exist_ok=True)

md_intro = """# E2 — Drift Structure: Why a Single Correction Suffices

**Goal:** characterize the drift Delta = base - BC (low-rank? deep-concentrated? consistent across
frames?) and produce the **top-k recovery curve** — the "drift bundle" payload for the design
principle ("ship a few principal drift directions per deep layer, not the full tensor").

**Stages:**
1. **Per-frame drift spectra** (train frames): drift vectors at pre-action / text-tail-mean /
   vision-mean positions, per layer; SVD -> singular spectra, participation ratio, top-1 energy.
   Answers: *is the drift low-rank and shared across frames?* (flat spectrum = sample-specific =
   weakens the deployable-bundle story; peaked spectrum = shared = supports it)
2. **Mean-Delta SVD per layer/region** (vision 512xd, text maxNxd): top-k reconstruction error
   curve, effective rank.
3. **Top-k recovery** (headline, on eval frames): patch BC with alpha * (U_k S_k V_k^T) per region,
   k in {1,4,8,16,32,full}, deep6 layers; measure vis CKA, text l24 CKA, AP, L2.
   If k=8 ~ k=full -> a tiny drift bundle captures the correction.
4. **Depth profile**: ||Delta_u||_F per layer vs per-layer CKA collapse (mini phenomenon on the
   same eval frames) -> Pearson r (drift norm should be high where CKA collapses).
5. **Weight-space (secondary, guarded)**: AA's released LoRA adapter (B@A per module, exact rank-64
   update) spectra per layer; BC merged-minus-base sanity (bf16 cast caveat).

**Reuses E1's verified machinery** (hooks, build_offsets, frame cache, CKA). New code: SVD +
top-k reconstruction only.

Set `SMOKE = False` for the full run (400 train / 60 eval)."""

cells = []

def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)})

def code(src):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                  "source": src.splitlines(keepends=True)})

md(md_intro)

# ---------------- config ----------------
code("""SMOKE = False                       # True = tiny frames (bug-hunting); False = full run
STAGE1_TRAIN = 30 if SMOKE else 400   # per-frame drift spectra frames
STAGE3_EVAL  = 15 if SMOKE else 60    # top-k recovery eval frames
K_GRID = [1, 4, 8, 16, 32, "full"]
ALPHA_K = 1.0                         # top-k patching strength (E1 reference: deep6 a1.0)

REPO_DIR  = "/kaggle/working/Anchor-Align"
CKPT_DIR  = "/kaggle/working/ckpts"
BC_PATH   = f"{CKPT_DIR}/bc"                 # VLA-Adapter/LIBERO-Spatial-Pro
AA_PATH   = f"{CKPT_DIR}/aa/libero-spatial"  # Dwipz/Anchor-Align
BASE_PATH = f"{CKPT_DIR}/base"               # Stanford-ILIAD prism base
RLDS_NAME = "libero_spatial_no_noops"
OUT_JSON  = "/kaggle/working/e2_results.json"

PATCH_LAYERS = list(range(24))
DEEP6 = list(range(18, 24))            # top-k patching target
TAIL_K = 8
VIS_STRIDE = 8
RNG_SEED = 0
print("SMOKE =", SMOKE)""")

# ---------------- env (E0/E1-proven recipe) ----------------
code("""import os
if not os.path.isdir(REPO_DIR):
    !wget -q -T 120 -O /kaggle/working/Anchor-Align.tar.gz https://github.com/dwipddalal/Anchor-Align/archive/refs/heads/main.tar.gz || wget -q -T 120 -O /kaggle/working/Anchor-Align.tar.gz https://github.com/dwipddalal/Anchor-Align/archive/refs/heads/main.tar.gz
    !tar -xzf /kaggle/working/Anchor-Align.tar.gz -C /kaggle/working/ && mv /kaggle/working/Anchor-Align-main {REPO_DIR} && echo TAR_OK
!ls {REPO_DIR} | head -6
os.chdir(REPO_DIR)
print("cwd:", os.getcwd())

!pip install -q "dlimp @ git+https://github.com/moojink/dlimp_openvla" --no-deps && echo DIMP_OK || echo DIMP_FAIL
!pip install -q "transformers @ git+https://github.com/moojink/transformers-openvla-oft.git" && echo TRANSFORMERS_OK || echo TRANSFORMERS_FAIL
!pip install -q accelerate einops huggingface_hub json-numpy jsonlines matplotlib peft==0.11.1 protobuf rich wandb diffusers==0.30.3 imageio uvicorn fastapi draccus==0.8.0 2>&1 | tail -n 2
!pip install -q timm==0.9.10 tokenizers==0.19.1 2>&1 | tail -n 2
!pip install -q sentencepiece 2>&1 | tail -n 1
!pip install -q -e {REPO_DIR} --no-deps 2>&1 | tail -n 3
!python -c "import torch, transformers, tokenizers, timm, tensorflow as tf, tensorflow_datasets as tfds, dlimp, absl; print('torch', torch.__version__); print('tf', tf.__version__); print('transformers', transformers.__version__); print('tfds', tfds.__version__)" """)

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
torch.cuda.empty_cache()""")

# ---------------- helpers (E1-verified) ----------------
code("""def base_pixel_dict(pixel_values):
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
    n = inputs["input_ids"].shape[-1] - 1
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

def gt_and_ap_labels(sample_actions, action):
    gt = np.asarray(sample_actions)[:, :3].mean(axis=0)
    gt_lab = rat.ap_top1_label_from_vec3(gt)
    ap_lab = rat.ap_top1_label_from_vec3(np.asarray(action)[:, :3].mean(0)) if action is not None else None
    return gt_lab, ap_lab

def action_l2(sample_actions, action):
    gt = np.asarray(sample_actions).astype(np.float32)
    pred = np.asarray(action).astype(np.float32)
    return float(np.linalg.norm(pred - gt, axis=1).mean())

def install_patch_hooks(inner_model, offsets, layers=None):
    handles = []
    for u, off in offsets.items():
        if layers is not None and u not in layers:
            continue
        def make_hook(off_u):
            def hook(module, args, output):
                if hasattr(output, "last_hidden_state"):
                    h = output.last_hidden_state
                    add = off_u[: h.shape[1]].unsqueeze(0).to(h.dtype)
                    output.last_hidden_state = h + add
                    return output
                h = output[0]
                add = off_u[: h.shape[1]].unsqueeze(0).to(h.dtype)
                return (h + add,) + output[1:]
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

def build_offsets(n, alpha, mask, delta_map):
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

# --- E2 new: SVD helpers ---
def participation_ratio(s):
    s2 = s ** 2
    return float((s2.sum() ** 2) / (s2 ** 2).sum())

def topk_reconstruct(U, S, Vh, k):
    if k == "full":
        k = S.shape[0]
    return U[:, :k] @ torch.diag(S[:k]) @ Vh[:k]""")

# ---------------- Stage 1: per-frame drift spectra ----------------
code("""print("\\n########## STAGE 1: PER-FRAME DRIFT SPECTRA (train frames) ##########")
# per-frame drift at 3 aggregates, per layer: pre-action (1x896), text-tail mean, vision mean
n_used = 0
agg = {a: {u: [] for u in PATCH_LAYERS} for a in ["pre", "text", "vis"]}
it = rat.make_rlds_iterator(RLDS_NAME)
for fi in range(STAGE1_TRAIN):
    sample = next(it)
    gt = np.asarray(sample["actions"])[:, :3].mean(axis=0)
    if rat.ap_top1_label_from_vec3(gt) is None:
        continue
    inputs = rat.build_inputs(bc["proc"], sample["primary"], sample["wrist"], sample["lang"], num_images=2)
    proprio = sample["proprio"][:8] if sample["proprio"] is not None else None
    n = inputs["input_ids"].shape[-1] - 1
    pos = 512 + n
    hs_b = base_hidden_full(base_vlm, inputs)
    _, hs_c = vla_hidden_full(bc, inputs, proprio, rat._resolve_unnorm_key(bc["vla"], RLDS_NAME))
    for u in PATCH_LAYERS:
        hb = hs_b[u + 1][0].float()
        hc = hs_c[u + 1][0].float()
        db = torch.zeros_like(hc).cpu()          # base has no action tokens -> region-wise only
        db[1:513] = (hb[1:513] - hc[1:513]).cpu()
        db[513:513 + n] = (hb[513:513 + n] - hc[513:513 + n]).cpu()
        agg["pre"][u].append(db[pos].unsqueeze(0))                     # pre-action (last text token)
        agg["text"][u].append(db[pos - TAIL_K + 1: pos + 1].mean(0).unsqueeze(0))
        agg["vis"][u].append(db[1:513].mean(0).unsqueeze(0))
    n_used += 1
print(f"used frames: {n_used}")

spectra = {}
print("=== per-layer SVD of the (frames x 896) drift matrix at each aggregate ===")
print(f"  {'layer':>5s} | {'pre':>22s} | {'text':>22s} | {'vis':>22s}")
print(f"  {'':>5s} | {'PR':>8s} {'E1%':>6s} {'E8%':>6s} | {'PR':>8s} {'E1%':>6s} {'E8%':>6s} | {'PR':>8s} {'E1%':>6s} {'E8%':>6s}")
for u in PATCH_LAYERS:
    row = []
    for a in ["pre", "text", "vis"]:
        X = torch.cat(agg[a][u], dim=0).to(DEVICE)          # (frames, 896)
        X = X - X.mean(0, keepdims=True)
        _, S, _ = torch.linalg.svd(X, full_matrices=False)
        e1 = float((S[:1] ** 2).sum() / (S ** 2).sum())
        e8 = float((S[:8] ** 2).sum() / (S ** 2).sum())
        row.append((participation_ratio(S), e1, e8))
        spectra[f"{a}_l{u}"] = {"pr": participation_ratio(S), "e1": e1, "e8": e8}
    print(f"  {u:5d} | {row[0][0]:8.1f} {row[0][1]*100:5.1f}% {row[0][2]*100:5.1f}% "
          f"| {row[1][0]:8.1f} {row[1][1]*100:5.1f}% {row[1][2]*100:5.1f}% "
          f"| {row[2][0]:8.1f} {row[2][1]*100:5.1f}% {row[2][2]*100:5.1f}%")
deep_pre = [spectra[f"pre_l{u}"]["e1"] for u in range(18, 24)]
deep_vis = [spectra[f"vis_l{u}"]["e1"] for u in range(18, 24)]
print(f"deep6 top-1 energy: pre={np.mean(deep_pre)*100:.1f}% vis={np.mean(deep_vis)*100:.1f}% "
      f"(high = drift shared across frames, supports mean-drift bundles)")""")

# ---------------- Stage 2: mean-Delta SVD + reconstruction ----------------
code("""print("\\n########## STAGE 2: MEAN-DELTA SVD PER LAYER/REGION ##########")
vis_sums = {u: torch.zeros(512, 896, dtype=torch.float32, device=DEVICE) for u in PATCH_LAYERS}
txt_sums = {u: torch.zeros(512, 896, dtype=torch.float32, device=DEVICE) for u in PATCH_LAYERS}
vis_cnt = {u: 0 for u in PATCH_LAYERS}
txt_cnt = {u: 0 for u in PATCH_LAYERS}
max_n = 0
it = rat.make_rlds_iterator(RLDS_NAME)
for fi in range(STAGE1_TRAIN):
    sample = next(it)
    gt = np.asarray(sample["actions"])[:, :3].mean(axis=0)
    if rat.ap_top1_label_from_vec3(gt) is None:
        continue
    inputs = rat.build_inputs(bc["proc"], sample["primary"], sample["wrist"], sample["lang"], num_images=2)
    proprio = sample["proprio"][:8] if sample["proprio"] is not None else None
    n = inputs["input_ids"].shape[-1] - 1
    max_n = max(max_n, n)
    hs_b = base_hidden_full(base_vlm, inputs)
    _, hs_c = vla_hidden_full(bc, inputs, proprio, rat._resolve_unnorm_key(bc["vla"], RLDS_NAME))
    for u in PATCH_LAYERS:
        hb = hs_b[u + 1][0].float()
        hc = hs_c[u + 1][0].float()
        vis_sums[u] += (hb[1:513] - hc[1:513])
        txt_sums[u][:n] += (hb[513:513 + n] - hc[513:513 + n]).flip(0)
        vis_cnt[u] += 1
        txt_cnt[u] += 1

delta = {}
svd_store = {}
print(f"  {'layer':>5s} | {'vis: PR':>8s} {'E1%':>6s} {'E8%':>6s} {'E32%':>6s} | {'text: PR':>8s} {'E1%':>6s} {'E8%':>6s} {'E32%':>6s}")
for u in PATCH_LAYERS:
    vis_d = (vis_sums[u] / max(1, vis_cnt[u]))[:]
    txt_d = (txt_sums[u] / max(1, txt_cnt[u]))[:max_n]
    delta[u] = {"vis": vis_d, "txt": txt_d}
    svd_store[u] = {}
    row = []
    for name, M in [("vis", vis_d), ("txt", txt_d)]:
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        svd_store[u][name] = (U, S, Vh)
        e1 = float((S[:1] ** 2).sum() / (S ** 2).sum())
        e8 = float((S[:8] ** 2).sum() / (S ** 2).sum())
        e32 = float((S[:32] ** 2).sum() / (S ** 2).sum())
        row.append((participation_ratio(S), e1, e8, e32))
    print(f"  {u:5d} | {row[0][0]:8.1f} {row[0][1]*100:5.1f}% {row[0][2]*100:5.1f}% {row[0][3]*100:5.1f}% "
          f"| {row[1][0]:8.1f} {row[1][1]*100:5.1f}% {row[1][2]*100:5.1f}% {row[1][3]*100:5.1f}%")

# reconstruction error curve (mean over layers 18-23, both regions)
print("=== top-k reconstruction error (deep6 mean, Frobenius, relative) ===")
rel_err = {}
for k in [1, 4, 8, 16, 32, "full"]:
    errs = []
    for u in range(18, 24):
        for name in ["vis", "txt"]:
            U, S, Vh = svd_store[u][name]
            full_norm = torch.norm(delta[u][name])
            rec = topk_reconstruct(U, S, Vh, k)
            errs.append(float((torch.norm(delta[u][name] - rec) / full_norm).item()))
    rel_err[str(k)] = float(np.mean(errs))
    print(f"  k={str(k):>4s}: rel err = {rel_err[str(k)]*100:6.2f}%")
print("  (monotone decreasing = reconstruction math correct)")""")

# ---------------- Stage 3: top-k recovery (headline) ----------------
code("""print("\\n########## STAGE 3: TOP-K RECOVERY (eval frames, deep6, alpha=%s) ##########" % ALPHA_K)
# cache eval frames
frame_cache = []
it = rat.make_rlds_iterator(RLDS_NAME)
while len(frame_cache) < STAGE3_EVAL:
    sample = next(it)
    gt = np.asarray(sample["actions"])[:, :3].mean(axis=0)
    gt_lab = rat.ap_top1_label_from_vec3(gt)
    if gt_lab is None:
        continue
    inputs = rat.build_inputs(bc["proc"], sample["primary"], sample["wrist"], sample["lang"], num_images=2)
    proprio = sample["proprio"][:8] if sample["proprio"] is not None else None
    frame_cache.append({"inputs": inputs, "proprio": proprio, "gt_lab": gt_lab,
                        "actions": sample["actions"], "n": inputs["input_ids"].shape[-1] - 1})
print(f"cached {len(frame_cache)} eval frames")

def projected_delta_map(k):
    dm = {}
    for u in PATCH_LAYERS:
        dm[u] = {}
        for name in ["vis", "txt"]:
            U, S, Vh = svd_store[u][name]
            dm[u][name] = topk_reconstruct(U, S, Vh, k)
    return dm

res = {}
def eval_row(rname, hs, act, fr, tail_b, vis_b):
    r = res.setdefault(rname, {"cka_text_deep": [], "cka_text_l24": [], "cka_vis_deep": [],
                               "ap_correct": 0, "n": 0, "l2": []})
    r["n"] += 1
    pos = last_text_pos(fr["inputs"])
    tail = text_tail(hs, pos); vis = vis_subset(hs)
    deep = [linear_cka(tail_b[u], tail[u]) for u in range(len(hs))]
    r["cka_text_deep"].append(float(np.mean(deep[-6:])))
    r["cka_text_l24"].append(float(deep[-1]))
    r["cka_vis_deep"].append(float(np.mean([linear_cka(vis_b[u], vis[u]) for u in range(len(hs))][-6:])))
    if act is not None:
        _, ap_lab = gt_and_ap_labels(fr["actions"], act)
        r["ap_correct"] += int(ap_lab == fr["gt_lab"])
        r["l2"].append(action_l2(fr["actions"], act))

configs = [("row_bc_a0", None)] + [(f"bc_k{k}", k) for k in K_GRID]
print("configs:", [c[0] for c in configs])
t0 = time.time()
eval_hs_base, eval_hs_bc0 = [], []
for fi, fr in enumerate(frame_cache):
    inputs = fr["inputs"]; proprio = fr["proprio"]; n = fr["n"]
    unnorm = rat._resolve_unnorm_key(bc["vla"], RLDS_NAME)
    hs_base = base_hidden_full(base_vlm, inputs)
    act0, hs_bc0 = vla_hidden_full(bc, inputs, proprio, unnorm)
    eval_hs_base.append(hs_base)
    eval_hs_bc0.append(hs_bc0)
    tail_b = text_tail(hs_base, last_text_pos(inputs)); vis_b = vis_subset(hs_base)
    eval_row("row_bc_a0", hs_bc0, act0, fr, tail_b, vis_b)
    for cname, k in configs[1:]:
        dm = projected_delta_map(k)
        offs = build_offsets(n, ALPHA_K, "all", dm)
        act, hs = run_patched(bc, inputs, proprio, unnorm, offs, layers=set(DEEP6))
        eval_row(cname, hs, act, fr, tail_b, vis_b)
    if (fi + 1) % 10 == 0:
        print(f"  eval frame {fi+1}/{STAGE3_EVAL} elapsed={time.time()-t0:.0f}s")

print("\\n=== TOP-K RECOVERY ===")
print(f"  {'config':>12s} | {'text deep':>9s} {'text l24':>9s} {'vis deep':>9s} {'AP%':>5s} {'L2':>6s}")
for cname, k in configs:
    r = res[cname]
    nn = max(1, r["n"])
    l2 = np.mean(r["l2"]) if r["l2"] else float("nan")
    print(f"  {cname:>12s} | {np.mean(r['cka_text_deep'])*100:8.1f}% {np.mean(r['cka_text_l24'])*100:8.1f}% "
          f"{np.mean(r['cka_vis_deep'])*100:8.1f}% {100*r['ap_correct']/nn:4.0f}% {l2:6.3f}")

# mechanics: k=full should match E1's deep6 alpha=1.0 reference (vis ~62%, text l24 ~91%)
ref = res["bc_kfull"]
print("\\nMECHANICS: k=full vs E1 reference (deep6 a1.0): "
      f"vis {np.mean(ref['cka_vis_deep'])*100:.1f}% (expect ~62%), "
      f"text l24 {np.mean(ref['cka_text_l24'])*100:.1f}% (expect ~91%)")""")

# ---------------- Stage 4: depth profile + CKA correlation ----------------
code("""print("\\n########## STAGE 4: DEPTH PROFILE vs CKA COLLAPSE ##########")
# drift norm per layer (from stage 2 sums, both regions) and CKA(base, bc) per layer on eval frames
drift_norm = {}
for u in PATCH_LAYERS:
    drift_norm[u] = float((torch.norm(delta[u]["vis"]) ** 2 + torch.norm(delta[u]["txt"]) ** 2).sqrt().item())
cka_curve = []
for u in range(len(eval_hs_base[0])):   # 25 entries: embedding + 24 layers
    A = torch.cat([text_tail(hs, last_text_pos(fr["inputs"]))[u] for hs, fr in zip(eval_hs_base, frame_cache)], dim=0)
    B = torch.cat([text_tail(hs, last_text_pos(fr["inputs"]))[u] for hs, fr in zip(eval_hs_bc0, frame_cache)], dim=0)
    cka_curve.append(linear_cka(A, B))
print("  CKA curve (bc vs base, eval frames):", " ".join(f"{v*100:3.0f}" for v in cka_curve))
print("  drift norm per layer:", " ".join(f"{drift_norm[u]:.0f}" for u in PATCH_LAYERS))
corr = float(np.corrcoef([drift_norm[u] for u in PATCH_LAYERS],
                         [1 - cka_curve[u + 1] for u in PATCH_LAYERS])[0, 1])
print(f"  Pearson r(drift norm, CKA-drop) = {corr:+.3f}  (positive = drift concentrates where erasure happens)")""")

# ---------------- Stage 5: weight-space (guarded, secondary) ----------------
code("""print("\\n########## STAGE 5: WEIGHT-SPACE DRIFT (secondary) ##########")
try:
    from safetensors.torch import load_file
    adapter = load_file(os.path.join(AA_PATH, "lora_adapter", "adapter_model.safetensors"))
    keys = [k for k in adapter if "lora_B" in k]
    layers_seen = sorted({int(k.split(".layers.")[1].split(".")[0]) for k in keys if ".layers." in k})
    print(f"AA LoRA adapter: {len(keys)} lora_B matrices, layers {layers_seen[0]}..{layers_seen[-1]}")
    print("  top-8 singular values of B@A per layer (sorted desc, first 8 of the stack):")
    for u in layers_seen[:6]:
        svs = []
        for k in keys:
            if f".layers.{u}." in k:
                b = adapter[k].float()
                a = adapter[k.replace("lora_B", "lora_A")].float()
                svs.append(torch.linalg.svdvals(b @ a))
        if svs:
            all_s = torch.sort(torch.cat(svs), descending=True).values
            print(f"    layer {u}: " + " ".join(f"{v:.2f}" for v in all_s[:8].tolist()))
except Exception as e:
    print(f"AA adapter analysis skipped: {type(e).__name__}: {e}")

try:
    import gc
    # BC weight-space drift (sanity): merged - base, per layer, bf16-cast both
    w_bc = {k: v for k, v in bc["vla"].state_dict().items() if "language_model.model.layers" in k and "weight" in k}
    w_ba = {k: v for k, v in base_vlm.llm_backbone.llm.state_dict().items() if "model.layers" in k and "weight" in k}
    print(f"BC merged {len(w_bc)} layer weights; base {len(w_ba)} layer weights (sanity only, dtype caveat)")
    for probe_layer in [21, 23]:
        bc_w = [w for k, w in w_bc.items() if f"layers.{probe_layer}." in k]
        ba_w = [w for k, w in w_ba.items() if f"model.layers.{probe_layer}." in k]
        if bc_w and ba_w:
            d = (bc_w[0].float() - ba_w[0].float())
            sv = torch.linalg.svdvals(d)
            pr = participation_ratio(sv)
            e8 = float((sv[:8] ** 2).sum() / (sv ** 2).sum())
            print(f"  layer {probe_layer} q_proj drift: top1={sv[0]:.3f} PR={pr:.1f} E8={e8*100:.1f}%")
except Exception as e:
    print(f"BC weight-space skipped: {type(e).__name__}: {e}")""")

# ---------------- summary ----------------
code("""out = {"smoke": SMOKE, "frames_train": n_used, "frames_eval": len(frame_cache),
       "spectra": spectra, "rel_err": rel_err, "topk": {}, "drift_norm": drift_norm,
       "cka_curve": cka_curve, "corr_drift_cka": corr}
for cname, k in configs:
    r = res[cname]
    nn = max(1, r["n"])
    out["topk"][cname] = {"n": nn,
                          "cka_text_deep": float(np.mean(r["cka_text_deep"])),
                          "cka_text_l24": float(np.mean(r["cka_text_l24"])),
                          "cka_vis_deep": float(np.mean(r["cka_vis_deep"])),
                          "ap_acc": r["ap_correct"] / nn,
                          "l2": (float(np.mean(r["l2"])) if r["l2"] else None)}
with open(OUT_JSON, "w") as f:
    json.dump(out, f, indent=2)
print("saved:", OUT_JSON)""")

md("""## Summary

- **Stage 1** = per-frame drift spectra (is the drift shared across frames? participation ratio / top-1 energy)
- **Stage 2** = mean-Delta SVD per layer/region + reconstruction error curve
- **Stage 3** = top-k recovery curve (the "drift bundle" payload): if k=8 ~ k=full, a tiny bundle suffices
- **Stage 4** = drift-norm depth profile vs CKA collapse (Pearson r)
- **Stage 5** = weight-space drift spectra (AA LoRA adapter = exact rank-64; BC merged-base = sanity)

Mechanics gates: reconstruction error monotone in k; k=full reproduces E1's deep6 alpha=1.0 numbers.
Results saved to `/kaggle/working/e2_results.json`.""")

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = os.path.join(OUT_DIR, "e2-drift-structure.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)
print("wrote", out_path)

metadata = {
    "id": "your-kaggle-username/e2-drift-structure",
    "title": "E2 Drift Structure",
    "code_file": "e2-drift-structure.ipynb",
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