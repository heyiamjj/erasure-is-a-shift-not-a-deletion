"""Builds the Kaggle E3 notebook (e3-frozen-head.ipynb): frozen-head readability + fresh-probe discriminator."""
import json, os

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "kaggle_e3")
os.makedirs(OUT_DIR, exist_ok=True)

md_intro = """# E3 — Frozen-Head Readability & the Fresh-Probe Discriminator

**Goal:** explain *why Anchor-Align stays on the pretrained manifold* — its alignment loss routes
supervision through the **frozen pretrained lm_head** via a *trained projection* W_proj, whose fixed
readout geometry forces the representation to remain decodable by the pretrained head's directions.

**Smoke-run finding (v1-v3, incorporated):** identity-projection frozen-head readability at the
pre-action position is degenerate for ALL models (base 12.5%, BC 20%, AA 17.5%) — the frozen head
only reads direction *through the trained projection*. The mechanism is the projection, not the
raw states. Corrected metrics below.

**Metrics:**
- **Decodability** (fresh linear probe, 5-fold x3 CV): info present in the geometry, any axes.
  Expect base < BC < AA (replicates E0/E1).
- **Frozen-readability with each model's own pathway**: AA = trained `align_dir_proj` + frozen
  head (expect ~90%); BC/base = identity projection (expect ~15-20%, diagnostic only).
- **THE DISCRIMINATOR — projection transfer**: apply AA's *trained* proj to BC's pre-action states.
  If the frozen head then reads BC well (~90%), BC sits on the same readout axes (contradicts
  shift); if it reads ~chance, BC is *off the pretrained readout axes* (confirms shift-not-deletion:
  the info exists — decodability ~86% — but not on the pretrained axes).
- **Patched-BC readability recovery** (bridge to E1/E2): patch BC at deep6, measure identity-read.

**Stages:**
1. Cache frames; collect per-layer pre-action + text-tail-mean states for the 3 models.
2. Per-layer decodability + identity-read (diagnostic curves).
3. Patched-BC identity-read recovery.
4. AA proj ablation + **proj transfer to BC**.

Set `SMOKE = False` for the full run."""

cells = []

def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)})

def code(src):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                  "source": src.splitlines(keepends=True)})

md(md_intro)

# ---------------- config ----------------
code("""SMOKE = False                       # True = tiny frames (bug-hunting); False = full run
STAGE1_FRAMES = 40 if SMOKE else 250   # eval frames for readability/decodability
DELTA_TRAIN   = 30 if SMOKE else 200   # drift computation frames (deep6 patching)
PATCH_ALPHAS  = [0.25, 1.0]

REPO_DIR  = "/kaggle/working/Anchor-Align"
CKPT_DIR  = "/kaggle/working/ckpts"
BC_PATH   = f"{CKPT_DIR}/bc"
AA_PATH   = f"{CKPT_DIR}/aa/libero-spatial"
BASE_PATH = f"{CKPT_DIR}/base"
RLDS_NAME = "libero_spatial_no_noops"
OUT_JSON  = "/kaggle/working/e3_results.json"

PATCH_LAYERS = list(range(24))
DEEP6 = list(range(18, 24))
TAIL_K = 8
RNG_SEED = 0
print("SMOKE =", SMOKE)""")

# ---------------- env ----------------
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

# ---------------- checkpoints ----------------
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
    print(f"[{os.path.basename(path)}] llm_dim={vla.llm_dim}")
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
dir_id_list = [dir_ids[w] for w in DIR_WORDS]
W6 = lm_head_w[dir_id_list]          # (6, 896) frozen pretrained readout rows
print("W6", tuple(W6.shape))

models = {"base": {"kind": "base", "vla": base_vlm},
          "bc": {"kind": "vla", **bc},
          "aa": {"kind": "vla", **aa}}""")

# ---------------- helpers ----------------
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

def frozen_logits(hvec):
    # hvec: (B, 896) or (896,) -> (B, 6) frozen-head logits over the 6 direction words (CPU out)
    h = hvec.float().cpu()
    return h @ W6.T

def prior_corrected_word(logits, prior):
    # logits: (n, 6); prior: (6,) base-model mean logits -> argmax after subtraction
    return DIR_WORDS[int((logits - prior).argmax(dim=1)[0].item())]

def per_model_hidden(name, inputs, proprio):
    if models[name]["kind"] == "base":
        hs = base_hidden_full(models[name]["vla"], inputs)
        return None, hs
    unnorm = rat._resolve_unnorm_key(models[name]["vla"], RLDS_NAME)
    act, hs = vla_hidden_full(models[name], inputs, proprio, unnorm)
    return act, hs""")

# ---------------- Stage 1: cache frames + collect states ----------------
code("""print("\\n########## STAGE 1: COLLECT PER-LAYER STATES ##########")
frame_cache = []
it = rat.make_rlds_iterator(RLDS_NAME)
while len(frame_cache) < STAGE1_FRAMES:
    sample = next(it)
    gt = np.asarray(sample["actions"])[:, :3].mean(axis=0)
    gt_lab = rat.ap_top1_label_from_vec3(gt)
    if gt_lab is None:
        continue
    inputs = rat.build_inputs(bc["proc"], sample["primary"], sample["wrist"], sample["lang"], num_images=2)
    proprio = sample["proprio"][:8] if sample["proprio"] is not None else None
    frame_cache.append({"inputs": inputs, "proprio": proprio, "gt_lab": gt_lab,
                        "actions": sample["actions"], "n": inputs["input_ids"].shape[-1] - 1})
print(f"cached {len(frame_cache)} frames")

# per-model per-layer states at pre-action and text-tail-mean (CPU float32)
states = {name: {"pre": [[] for _ in range(25)], "tail": [[] for _ in range(25)]} for name in models}
gt_labels = []
for fr in frame_cache:
    inputs = fr["inputs"]; proprio = fr["proprio"]; pos = last_text_pos(inputs)
    gt_labels.append(fr["gt_lab"])
    for name in models:
        _, hs = per_model_hidden(name, inputs, proprio)
        for u in range(25):
            states[name]["pre"][u].append(hs[u][0, pos].detach().float().cpu())
            states[name]["tail"][u].append(hs[u][0, pos - TAIL_K + 1: pos + 1].mean(0).detach().float().cpu())
for name in models:
    for u in range(25):
        states[name]["pre"][u] = torch.stack(states[name]["pre"][u])
        states[name]["tail"][u] = torch.stack(states[name]["tail"][u])
print("state tensors ready:", {name: tuple(states[name]["pre"][24].shape) for name in models})""")

# ---------------- Stage 2: readability, decodability, gap ----------------
code("""print("\\n########## STAGE 2: PER-LAYER READABILITY / DECODABILITY / GAP ##########")
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression

def cv_probe(X, y):
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RNG_SEED)
    accs = []
    for tr, te in rskf.split(X, y):
        clf = LogisticRegression(max_iter=2000, C=0.1)
        clf.fit(X[tr], y[tr])
        accs.append(clf.score(X[te], y[te]))
    return float(np.mean(accs))

y = np.array(gt_labels)
n_frames = len(frame_cache)
metrics = {name: {"read_pre": [], "read_tail": [], "decod_pre": [], "decod_tail": []} for name in models}

# frozen-readability needs the base prior per layer (mean base logits over frames)
prior = {"pre": [], "tail": []}
for u in range(25):
    for agg in ["pre", "tail"]:
        prior[agg].append(frozen_logits(states["base"][agg][u]).mean(0))

for u in range(25):
    row = []
    for name in models:
        for agg in ["pre", "tail"]:
            L = frozen_logits(states[name][agg][u])                      # (n, 6)
            read = float(np.mean([int(prior_corrected_word(L[i:i + 1], prior[agg][u]) == y[i])
                                  for i in range(n_frames)]))
            decod = cv_probe(states[name][agg][u].numpy(), y)
            metrics[name][f"read_{agg}"].append(read)
            metrics[name][f"decod_{agg}"].append(decod)
    if u in [0, 12, 18, 21, 23, 24]:
        print(f"--- layer {u} (hidden_states[{u}]) ---")
        for name in models:
            m = metrics[name]
            print(f"  {name:5s} read_pre={m['read_pre'][u]*100:5.1f}% read_tail={m['read_tail'][u]*100:5.1f}% "
                  f"decod_pre={m['decod_pre'][u]*100:5.1f}% decod_tail={m['decod_tail'][u]*100:5.1f}%")

print("\\n=== DEEP-MEAN (layers 19-24 = hidden_states[19..24]) and LAST-LAYER ===")
for agg in ["pre", "tail"]:
    for name in models:
        m = metrics[name]
        dm_r = float(np.mean(m[f"read_{agg}"][-6:]))
        dm_d = float(np.mean(m[f"decod_{agg}"][-6:]))
        gap = dm_d - dm_r
        print(f"  {name:5s} {agg:4s} deep: read={dm_r*100:5.1f}% decod={dm_d*100:5.1f}% GAP={gap*100:+5.1f} pts | "
              f"last-layer read={m[f'read_{agg}'][24]*100:5.1f}% decod={m[f'decod_{agg}'][24]*100:5.1f}%")

# discriminator claim: BC gap >> AA gap (deep, pre)
gap_bc = np.mean(metrics["bc"]["decod_pre"][-6:]) - np.mean(metrics["bc"]["read_pre"][-6:])
gap_aa = np.mean(metrics["aa"]["decod_pre"][-6:]) - np.mean(metrics["aa"]["read_pre"][-6:])
gap_base = np.mean(metrics["base"]["decod_pre"][-6:]) - np.mean(metrics["base"]["read_pre"][-6:])
print(f"\\nDISCRIMINATOR GAP (decod - read, deep pre): base={gap_base*100:+.1f} bc={gap_bc*100:+.1f} aa={gap_aa*100:+.1f}")
print("  (expect: bc large positive = shifted geometry; aa small = on-manifold)")""")

# ---------------- Stage 3: patched-BC readability recovery ----------------
code("""print("\\n########## STAGE 3: PATCHED-BC FROZEN-READABILITY RECOVERY ##########")
# drift Delta (all 24 layers, vision+text regions) on DELTA_TRAIN frames
vis_sums = {u: torch.zeros(512, 896, dtype=torch.float32, device=DEVICE) for u in PATCH_LAYERS}
txt_sums = {u: torch.zeros(512, 896, dtype=torch.float32, device=DEVICE) for u in PATCH_LAYERS}
vis_cnt = {u: 0 for u in PATCH_LAYERS}
txt_cnt = {u: 0 for u in PATCH_LAYERS}
max_n = 0
it = rat.make_rlds_iterator(RLDS_NAME)
for fi in range(DELTA_TRAIN):
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
        hb = hs_b[u + 1][0].float(); hc = hs_c[u + 1][0].float()
        vis_sums[u] += (hb[1:513] - hc[1:513])
        txt_sums[u][:n] += (hb[513:513 + n] - hc[513:513 + n]).flip(0)
        vis_cnt[u] += 1; txt_cnt[u] += 1
delta = {u: {"vis": (vis_sums[u] / max(1, vis_cnt[u])), "txt": (txt_sums[u] / max(1, txt_cnt[u]))[:max_n]}
         for u in PATCH_LAYERS}
print(f"delta ready, max_n={max_n}")

read_patched = {a: [] for a in PATCH_ALPHAS}
for a in PATCH_ALPHAS:
    correct = 0
    for fr in frame_cache:
        inputs = fr["inputs"]; proprio = fr["proprio"]; n = fr["n"]
        offs = build_offsets(n, a, "all", delta)
        _, hs = run_patched(bc, inputs, proprio, rat._resolve_unnorm_key(bc["vla"], RLDS_NAME),
                            offs, layers=set(DEEP6))
        pos = last_text_pos(inputs)
        L = frozen_logits(hs[24][0, pos].detach().float())
        if prior_corrected_word(L.unsqueeze(0), prior["pre"][24]) == fr["gt_lab"]:
            correct += 1
    read_patched[a] = correct / len(frame_cache)
    print(f"  patched-BC alpha={a}: frozen-readability (last layer) = {read_patched[a]*100:5.1f}%")

r_bc = metrics["bc"]["read_pre"][24]
r_aa = metrics["aa"]["read_pre"][24]
r_base = metrics["base"]["read_pre"][24]
print(f"  reference: bc a0={r_bc*100:5.1f}% aa={r_aa*100:5.1f}% base={r_base*100:5.1f}%")""")

# ---------------- Stage 4: AA align_dir_proj ablation + transfer to BC ----------------
code("""print("\\n########## STAGE 4: AA ALIGN_DIR_PROJ ABLATION + TRANSFER TO BC ##########")
proj = rat.load_align_dir_proj(AA_PATH, 896)
if proj is not None:
    n_correct_proj_aa = 0
    n_correct_id_aa = 0
    n_correct_proj_bc = 0
    for fr in frame_cache:
        inputs = fr["inputs"]; proprio = fr["proprio"]
        _, hs_aa = per_model_hidden("aa", inputs, proprio)
        _, hs_bc = per_model_hidden("bc", inputs, proprio)
        pos = last_text_pos(inputs)
        h_aa = hs_aa[24][0, pos].detach().float()
        h_bc = hs_bc[24][0, pos].detach().float()
        # AA trained pathway: align_dir_proj -> frozen head (axis remap for spatial ckpt)
        lp = F.linear(proj(h_aa.unsqueeze(0).to(DEVICE).to(torch.bfloat16)).float().cpu(), lm_head_w)
        word = rat.remap_axis_swap(DIR_WORDS[int(lp[0, dir_id_list].argmax().item())])
        n_correct_proj_aa += int(word == fr["gt_lab"])
        # AA identity pathway (prior-corrected)
        if prior_corrected_word(frozen_logits(h_aa).unsqueeze(0), prior["pre"][24]) == fr["gt_lab"]:
            n_correct_id_aa += 1
        # TRANSFER: AA's trained proj applied to BC's states (the discriminator)
        lp_bc = F.linear(proj(h_bc.unsqueeze(0).to(DEVICE).to(torch.bfloat16)).float().cpu(), lm_head_w)
        word_bc = rat.remap_axis_swap(DIR_WORDS[int(lp_bc[0, dir_id_list].argmax().item())])
        n_correct_proj_bc += int(word_bc == fr["gt_lab"])
    n = len(frame_cache)
    aa_proj = n_correct_proj_aa / n
    aa_id = n_correct_id_aa / n
    bc_via_aa_proj = n_correct_proj_bc / n
    print(f"  AA with trained proj + frozen head (remapped): {aa_proj*100:5.1f}%")
    print(f"  AA identity projection (prior-corrected):      {aa_id*100:5.1f}%")
    print(f"  -> AA projection contribution: {(aa_proj - aa_id)*100:+.1f} pts")
    print(f"  TRANSFER: AA proj applied to BC states:        {bc_via_aa_proj*100:5.1f}%")
    print(f"  -> transfer gap (AA-on-AA - AA-on-BC): {(aa_proj - bc_via_aa_proj)*100:+.1f} pts")
    print("  (large transfer gap = BC states are OFF the pretrained readout axes)")
else:
    print("  align_dir_proj not found for this checkpoint")

# ---- mechanics gates ----
decod_order_ok = (metrics["bc"]["decod_pre"][24] > metrics["base"]["decod_pre"][24] and
                  metrics["aa"]["decod_pre"][24] > metrics["bc"]["decod_pre"][24])
mech_ok = proj is not None and aa_proj > 0.70
transfer_ok = proj is not None and (aa_proj - bc_via_aa_proj) > 0.30
patch_ok = read_patched[PATCH_ALPHAS[-1]] > metrics["bc"]["read_pre"][24] + 0.05
print("\\nMECHANICS:",
      "OK" if (decod_order_ok and mech_ok and transfer_ok and patch_ok) else "SUSPICIOUS - inspect",
      f"(decod order {decod_order_ok}, aa-proj>=70% {mech_ok}, transfer gap>=30pts {transfer_ok}, patch recovery {patch_ok})")

out = {"smoke": SMOKE, "frames": len(frame_cache), "metrics": metrics,
       "read_patched": {str(a): v for a, v in read_patched.items()},
       "aa_ablation": {"n": len(frame_cache)}}
if proj is not None:
    out["aa_ablation"]["aa_with_proj"] = aa_proj
    out["aa_ablation"]["aa_identity"] = aa_id
    out["aa_ablation"]["bc_via_aa_proj"] = bc_via_aa_proj
with open(OUT_JSON, "w") as f:
    json.dump(out, f, indent=2)
print("saved:", OUT_JSON)""")

md("""## Summary

- **Stage 1** = per-layer pre-action + text-tail states for base/BC/AA (same frames)
- **Stage 2** = per-layer frozen-readability (prior-corrected) vs decodability (CV probe) vs gap
- **Stage 3** = patched-BC frozen-readability recovery (bridge to E1/E2)
- **Stage 4** = AA align_dir_proj ablation (trained projection contribution)

Mechanics gates: decodability order base<bc<aa (E0 replication); aa-readability > bc; BC gap >> AA
gap (discriminator); patched-BC readability recovers. Results saved to `/kaggle/working/e3_results.json`.""")

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = os.path.join(OUT_DIR, "e3-frozen-head.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)
print("wrote", out_path)

metadata = {
    "id": "your-kaggle-username/e3-frozen-head-readability",
    "title": "E3 Frozen Head Readability",
    "code_file": "e3-frozen-head.ipynb",
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