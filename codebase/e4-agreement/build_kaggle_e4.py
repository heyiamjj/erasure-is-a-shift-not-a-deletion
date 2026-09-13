"""Builds the Kaggle E4 notebook (e4-next-token-agreement.ipynb): frozen-head next-token agreement with the pretrained model."""
import json, os

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "kaggle_e4")
os.makedirs(OUT_DIR, exist_ok=True)

md_intro = """# E4 — Next-Token Agreement with the Pretrained Model

**Motivation (first-principles correction of E4):** erasure = *displacement* preserves linearly
decodable information, so decodability readouts cannot show forgetting (E4's null was the correct
prediction). Forgetting appears in *alignment*: do the finetuned model's states still produce the
**pretrained model's behavior**? E4 measures exactly that, data-free:

> For each prompt/frame, compute the **frozen pretrained head's next-token logits** at the answer
> position for base, BC, AA, and patched-BC. Metrics vs base: (1) full-vocab logit cosine,
> (2) **centered** logit cosine (Pearson on logits — shape similarity), (3) top-1 agreement,
> (4) top-5 hit (model's argmax in base's top-5).

**Prediction:** BC's displaced states -> next-token behavior diverges from base (agreement drops);
AA (CKA 97%) -> agreement stays high; patched-BC -> agreement **recovers** toward base. This is the
readout-level mechanization of the paper's Fig. 9 (BC loses 94% GQA accuracy).

**Mechanics gates (smoke, must hold before full scale):**
1. **Base informativeness**: base's next-token varies across frames (>= 2 distinct top-1 tokens and
   mean pairwise top-1 diversity > 0) — otherwise the metric is degenerate (trivial agreement).
2. **Erasure visible behaviorally**: agreement(bc) < agreement(aa) (cosine gap or top-1 gap).
3. **Recovery visible behaviorally**: agreement(bc_patch a=1.0) > agreement(bc).
4. Sanity: base-vs-base = 1.0.

Prompts (3 templates, LIBERO spatial frames): instruction (VLA format), object question, direction
question. Set `SMOKE = False` for the full run (200 frames)."""

cells = []

def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)})

def code(src):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                  "source": src.splitlines(keepends=True)})

md(md_intro)

code("""SMOKE = False                       # True = tiny frames (bug-hunting); False = full run
FRAMES   = 40 if SMOKE else 200      # frames per template
DELTA_TRAIN = 30 if SMOKE else 200   # drift frames (deep6 patching)
PATCH_AS = [0.25, 1.0]

REPO_DIR  = "/kaggle/working/Anchor-Align"
CKPT_DIR  = "/kaggle/working/ckpts"
BC_PATH   = f"{CKPT_DIR}/bc"
AA_PATH   = f"{CKPT_DIR}/aa/libero-spatial"
BASE_PATH = f"{CKPT_DIR}/base"
RLDS_NAME = "libero_spatial_no_noops"
OUT_JSON  = "/kaggle/working/e4_results.json"

PATCH_LAYERS = list(range(24))
DEEP6 = list(range(18, 24))
RNG_SEED = 0
print("SMOKE =", SMOKE)""")

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

code("""target = f"{REPO_DIR}/data/libero/{RLDS_NAME}"
if not os.path.isdir(target):
    print(">> downloading RLDS libero_spatial_no_noops ...")
    download_with_retry(repo_id="openvla/modified_libero_rlds", repo_type="dataset",
                        local_dir="/kaggle/working/rlds_dl",
                        allow_patterns=["libero_spatial_no_noops/**"])
    os.makedirs(f"{REPO_DIR}/data/libero", exist_ok=True)
    !mv /kaggle/working/rlds_dl/{RLDS_NAME} {REPO_DIR}/data/libero/
print("RLDS:", os.listdir(f"{REPO_DIR}/data/libero/{RLDS_NAME}"))
!du -sh {REPO_DIR}/data/libero/{RLDS_NAME}""")

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

models = {"base": {"kind": "base", "vla": base_vlm},
          "bc": {"kind": "vla", **bc},
          "aa": {"kind": "vla", **aa}}""")

code("""def base_pixel_dict(pixel_values):
    pv = pixel_values.to(torch.bfloat16).to(DEVICE)
    t = pv.shape[1] // 6
    imgs = torch.split(pv, [6] * t, dim=1)
    return {"dino": torch.stack([im[:, 3:] for im in imgs], dim=1),
            "siglip": torch.stack([im[:, :3] for im in imgs], dim=1)}

def base_readout_hidden(base_vlm, inputs):
    with torch.inference_mode():
        out = base_vlm(input_ids=inputs["input_ids"].to(DEVICE),
                       pixel_values=base_pixel_dict(inputs["pixel_values"]),
                       output_hidden_states=True)
    return out.hidden_states[-1][0, -1].detach().float()

def vla_readout_hidden(model, inputs):
    holder = {}
    inner = model["vla"].language_model.model
    orig = inner.forward
    def wrapped(*a, **k):
        k["output_hidden_states"] = True
        out = orig(*a, **k)
        holder["last"] = out.hidden_states[-1]
        return out
    inner.forward = wrapped
    try:
        with torch.inference_mode():
            text_embeds = model["vla"].get_input_embeddings()(inputs["input_ids"].to(DEVICE))
            proj_patches = model["vla"]._process_vision_features(inputs["pixel_values"],
                                                                 language_embeddings=text_embeds, use_film=False)
            combined = torch.cat([proj_patches, text_embeds], dim=1)
            attn = torch.ones(combined.shape[:2], dtype=torch.long, device=combined.device)
            inner(input_ids=None, attention_mask=attn, inputs_embeds=combined,
                  use_cache=False, output_attentions=False, output_hidden_states=True, return_dict=True)
    finally:
        inner.forward = orig
    return holder["last"][0, -1].detach().float()

def vla_readout_patched(model, inputs, offsets, layers=None):
    holder = {}
    inner = model["vla"].language_model.model
    orig = inner.forward
    handles = []
    for u, off in offsets.items():
        if layers is not None and u not in layers:
            continue
        def make_hook(off_u):
            def hook(module, args, output):
                if hasattr(output, "last_hidden_state"):
                    h = output.last_hidden_state
                    output.last_hidden_state = h + off_u[: h.shape[1]].unsqueeze(0).to(h.dtype)
                    return output
                h = output[0]
                return (h + off_u[: h.shape[1]].unsqueeze(0).to(h.dtype),) + output[1:]
            return hook
        handles.append(inner.layers[u].register_forward_hook(make_hook(off)))
    def wrapped(*a, **k):
        k["output_hidden_states"] = True
        out = orig(*a, **k)
        holder["last"] = out.hidden_states[-1]
        return out
    inner.forward = wrapped
    try:
        with torch.inference_mode():
            text_embeds = model["vla"].get_input_embeddings()(inputs["input_ids"].to(DEVICE))
            proj_patches = model["vla"]._process_vision_features(inputs["pixel_values"],
                                                                 language_embeddings=text_embeds, use_film=False)
            combined = torch.cat([proj_patches, text_embeds], dim=1)
            attn = torch.ones(combined.shape[:2], dtype=torch.long, device=combined.device)
            inner(input_ids=None, attention_mask=attn, inputs_embeds=combined,
                  use_cache=False, output_attentions=False, output_hidden_states=True, return_dict=True)
    finally:
        inner.forward = orig
        for h in handles:
            h.remove()
    return holder["last"][0, -1].detach().float()

def build_offsets_readout(n, alpha, delta_map, layers):
    # VLA readout path: seq = [512 vision patches; BOS + n question tokens] (patches at 0..511,
    # q tokens at 513..512+n; position p in readout <-> position p+1 in the action path).
    # The ANSWER position is the last q token (512+n) -- MUST be patched.
    S = 512 + n + 1
    offs = {}
    for u in layers:
        off = torch.zeros(S, 896, device=DEVICE)
        off[0:512] = delta_map[u]["vis"] * alpha
        k = min(n, delta_map[u]["txt"].shape[0])
        off[513:513 + k] = delta_map[u]["txt"][:k].flip(0) * alpha
        offs[u] = off.to(torch.bfloat16)
    return offs

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
    return np.asarray(action), holder["all"]""")

code("""print("\\n########## STAGE 0: DRIFT DELTA (spatial train frames, all layers) ##########")
vis_sums = {u: torch.zeros(512, 896, dtype=torch.float32, device=DEVICE) for u in PATCH_LAYERS}
txt_sums = {u: torch.zeros(512, 896, dtype=torch.float32, device=DEVICE) for u in PATCH_LAYERS}
vis_cnt = {u: 0 for u in PATCH_LAYERS}; txt_cnt = {u: 0 for u in PATCH_LAYERS}
max_n = 0
it = rat.make_rlds_iterator(RLDS_NAME)
n_d = 0
while n_d < DELTA_TRAIN:
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
    n_d += 1
delta = {u: {"vis": vis_sums[u] / max(1, vis_cnt[u]), "txt": (txt_sums[u] / max(1, txt_cnt[u]))[:max_n]} for u in PATCH_LAYERS}
print(f"delta ready: {n_d} frames, max_n={max_n}")""")

code("""print("\\n########## STAGE 1: NEXT-TOKEN AGREEMENT (vs base, frozen head) ##########")
TEMPLATES = {
    "instruction": lambda lang: f"In: {lang}\\nOut:",
    "object":      lambda lang: "In: What object should the robot pick up? Answer in one word.\\nOut:",
    "direction":   lambda lang: "In: In which direction should the end effector move? Answer in one word.\\nOut:",
}
def logits_of(h):
    return F.linear(h.float().cpu(), lm_head_w)     # (vocab,)

def agree_metrics(log_base, log_model):
    c = float(F.cosine_similarity(log_base.unsqueeze(0), log_model.unsqueeze(0)).item())
    lb, lm = log_base - log_base.mean(), log_model - log_model.mean()
    cc = float(F.cosine_similarity(lb.unsqueeze(0), lm.unsqueeze(0)).item())
    t1 = int(log_base.argmax().item() == log_model.argmax().item())
    top5 = set(log_base.topk(5).indices.tolist())
    t5 = int(log_model.argmax().item() in top5)
    return c, cc, t1, t5

results = {}
for tname, tfn in TEMPLATES.items():
    stats = {m: {"cos": [], "ccos": [], "t1": 0, "t5": 0, "n": 0} for m in ["base", "bc", "aa"] + [f"bc_p{a}" for a in PATCH_AS]}
    base_tokens = []
    it = rat.make_rlds_iterator(RLDS_NAME)
    n_frames = 0
    while n_frames < FRAMES:
        sample = next(it)
        lang = sample["lang"]
        prompt = tfn(lang)
        inputs = rat.build_inputs(bc["proc"], sample["primary"], sample["wrist"], prompt, num_images=2)
        n_text = inputs["input_ids"].shape[-1] - 1
        h_base = base_readout_hidden(base_vlm, inputs)
        l_base = logits_of(h_base)
        base_tokens.append(int(l_base.argmax().item()))
        h_bc = vla_readout_hidden(bc, inputs)
        h_aa = vla_readout_hidden(aa, inputs)
        l_bc = logits_of(h_bc); l_aa = logits_of(h_aa)
        lp = {}
        for a in PATCH_AS:
            offs = build_offsets_readout(n_text, a, delta, DEEP6)
            lp[a] = logits_of(vla_readout_patched(bc, inputs, offs, layers=set(DEEP6)))
        for m, l in [("base", l_base), ("bc", l_bc), ("aa", l_aa)] + [(f"bc_p{a}", lp[a]) for a in PATCH_AS]:
            c, cc, t1, t5 = agree_metrics(l_base, l)
            stats[m]["cos"].append(c); stats[m]["ccos"].append(cc)
            stats[m]["t1"] += t1; stats[m]["t5"] += t5; stats[m]["n"] += 1
        n_frames += 1
    # base informativeness: distinct top-1 tokens
    distinct = len(set(base_tokens))
    results[tname] = {"distinct_base_top1": distinct}
    print(f"\\n=== template '{tname}' (n={n_frames}, base top-1 distinct tokens={distinct}) ===")
    for m, s in stats.items():
        cos = float(np.mean(s["cos"])); ccos = float(np.mean(s["ccos"]))
        t1 = s["t1"] / max(1, s["n"]); t5 = s["t5"] / max(1, s["n"])
        results[tname][m] = {"cos": cos, "ccos": ccos, "t1": t1, "t5": t5}
        print(f"  {m:8s} cos={cos:6.4f} ccos={ccos:6.4f} t1={t1*100:5.1f}% t5={t5*100:5.1f}%")

# ---- mechanics gates ----
gates = {}
for tname in TEMPLATES:
    r = results[tname]
    bc_r = r["bc"]; aa_r = r["aa"]; bp_r = r[f"bc_p{PATCH_AS[-1]}"]
    informative = r["distinct_base_top1"] >= 2
    erasure = bc_r["t1"] < aa_r["t1"] - 0.03 or bc_r["ccos"] < aa_r["ccos"] - 0.01
    recovery = bp_r["t1"] > bc_r["t1"] + 0.03 or bp_r["ccos"] > bc_r["ccos"] + 0.01
    gates[tname] = (informative, erasure, recovery)
    print(f"\\nGATES[{tname}]: informative={informative} erasure={erasure} recovery={recovery}")

print("\\nMECHANICS:",
      "OK" if any(all(g) for g in gates.values()) else "SUSPICIOUS - inspect",
      "(at least one template must pass all three gates)")

with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)
print("saved:", OUT_JSON)""")

md("""## Summary

- Frozen-head **next-token agreement with the pretrained model** (logit cosine, centered cosine,
  top-1, top-5) per template, for base / BC / AA / patched-BC.
- Gates: base informativeness (top-1 varies), erasure (bc < aa), recovery (patched > bc).
- This is the readout-level mechanization of the paper's Fig. 9 (BC loses 94% GQA): if BC's states
  no longer reproduce the pretrained model's next-token behavior, its pretrained QA competence is
  behaviorally erased — and patching should restore it.

Results saved to `/kaggle/working/e4_results.json`.""")

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = os.path.join(OUT_DIR, "e4-next-token-agreement.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)
print("wrote", out_path)

metadata = {
    "id": "your-kaggle-username/e4-next-token-agreement",
    "title": "E4 Next Token Agreement",
    "code_file": "e4-next-token-agreement.ipynb",
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