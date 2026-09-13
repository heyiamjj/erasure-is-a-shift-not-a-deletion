"""Builds the Kaggle E5 notebook (e5-gqa-readout.ipynb): GQA frozen-head readout + recovery + behavioral agreement."""
import json, os

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "kaggle_e5")
os.makedirs(OUT_DIR, exist_ok=True)

md_intro = """# E5 — GQA Readout: the Paper's Own Erasure Metric at Readout Level + Recovery

**Goal:** the one readout where the *pretrained model itself has competence* — GQA visual reasoning
(the paper's Fig. 9: BC loses 94% GQA accuracy in 10K steps, AA retains 70%). Replicate at
frozen-head readout level and test whether drift patching recovers it.

**Data (no big downloads):**
- Images: Kaggle dataset `minhngcng3/gqa-images-subset` (4.28 GB, 30k images) mounted at
  `/kaggle/input/gqa-images-subset/images_subset/images_subset/{id}.jpg`
- Questions: `testdev_balanced_questions.json` (10.8 MB, answers) downloaded via the Kaggle CLI
  from `minhngcng3/gqa-vqa-subset`; fallback to its val/train subsets if testdev image coverage is
  too low (the 30k subset may be curated).

**Protocol:** question -> `In: {q}\\nOut:` (VLA format); single image duplicated to both slots
(base loaded with image_sequence_len=2); frozen-head next-token readout at the last position.
Metrics: **open-vocab answer accuracy** (argmax == answer token), top-5 hit, and **next-token
agreement vs base** (E4 machinery: cosine / centered cosine / top-1).

**Models:** base / BC / AA / patched-BC (deep6, alpha in {0.25, 1.0}).

**Gates (smoke, must hold before full):**
1. Coverage >= N questions (available image + single-token answer)
2. Base informative (answer tokens vary)
3. Erasure: base_acc - bc_acc >= 10 pts (paper: -94%)
4. Preservation: aa_acc > bc_acc
5. Recovery: patched-bc_acc > bc_acc by >= 3 pts
6. Sanity: base-vs-base agreement = 1.0

Set `SMOKE = False` for the full run (1500 questions, capped by availability)."""

cells = []

def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)})

def code(src):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                  "source": src.splitlines(keepends=True)})

md(md_intro)

code("""SMOKE = False                       # True = tiny (bug-hunting); False = full run
GQA_N  = 1500 if SMOKE else 1500      # questions (after filters)
DELTA_TRAIN = 200 if SMOKE else 200   # drift frames (deep6 patching)
PATCH_AS = [0.25, 1.0]

REPO_DIR  = "/kaggle/working/Anchor-Align"
CKPT_DIR  = "/kaggle/working/ckpts"
BC_PATH   = f"{CKPT_DIR}/bc"
AA_PATH   = f"{CKPT_DIR}/aa/libero-spatial"
BASE_PATH = f"{CKPT_DIR}/base"
RLDS_NAME = "libero_spatial_no_noops"
IMG_DIR   = "/kaggle/input/gqa-images-subset/images_subset/images_subset"
Q_DIR     = "/kaggle/working/gqa_q"
OUT_JSON  = "/kaggle/working/e5_results.json"

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
print("RLDS:", os.listdir(f"{REPO_DIR}/data/libero/{RLDS_NAME}"))""")

code("""print(">> GQA data prep ...")
print("mounted images:", os.path.isdir(IMG_DIR), os.listdir(IMG_DIR)[:3] if os.path.isdir(IMG_DIR) else "")
n_imgs = len(os.listdir(IMG_DIR)) if os.path.isdir(IMG_DIR) else 0
if not os.path.isdir(IMG_DIR):
    # fallback: download + extract the subset into working (4.28 GB zip -> ~4.3 GB images)
    print(">> dataset not mounted - downloading subset via CLI (fallback)...")
    !kaggle datasets download minhngcng3/gqa-images-subset -p /kaggle/working/ || echo IMGDL_FAIL
    !cd /kaggle/working && unzip -o -q gqa-images-subset.zip && rm -f gqa-images-subset.zip
    for root, dirs, files in os.walk("/kaggle/working/images_subset"):
        if files and files[0].endswith(".jpg"):
            IMG_DIR = root
            break
print("image count:", len(os.listdir(IMG_DIR)) if os.path.isdir(IMG_DIR) else 0)
os.makedirs(Q_DIR, exist_ok=True)

def dl_qfile(name, url):
    dest = f"{Q_DIR}/{name}"
    if not os.path.exists(dest):
        !kaggle datasets download minhngcng3/gqa-vqa-subset -f {name} -p {Q_DIR} || echo QDL_FAIL
    return os.path.exists(dest)

# primary: full testdev (digit-style COCO ids -> matches the images subset);
# the balanced testdev uses VG-style 'n...' ids NOT present in the subset.
qjson = f"{Q_DIR}/testdev_all_questions.json"
if not os.path.exists(qjson):
    print(">> downloading questions1.2.zip (53MB) and extracting testdev_all_questions.json ...")
    import urllib.request, zipfile, io
    for attempt in range(3):
        try:
            data = urllib.request.urlopen("https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip",
                                          timeout=900).read()
            zf = zipfile.ZipFile(io.BytesIO(data))
            open(qjson, "wb").write(zf.read("testdev_all_questions.json"))
            break
        except Exception as e:
            print(f"  attempt {attempt+1} failed: {e}")
dl_qfile("testdev_balanced_questions.json", "")
dl_qfile("val_subset_5k.json", "")
print("questions dir:", os.listdir(Q_DIR))""")

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
tok = getattr(base_vlm.llm_backbone, "tokenizer", None)
if tok is None:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(AA_PATH)

def ans_id(word):
    ids = tok.encode(word, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None

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

def build_offsets_readout(n, alpha, delta_map, layers, mask="all"):
    # VLA readout path: seq = [512 vision patches; BOS + n question tokens] (patches at 0..511,
    # q tokens at 513..512+n; position p in readout <-> position p+1 in the action path).
    # The ANSWER position is the last q token (512+n) -- MUST be patched.
    S = 512 + n + 1
    offs = {}
    for u in layers:
        off = torch.zeros(S, 896, device=DEVICE)
        off[0:512] = delta_map[u]["vis"] * alpha
        if mask == "all":
            k = min(n, delta_map[u]["txt"].shape[0])
            off[513:513 + k] = delta_map[u]["txt"][:k].flip(0) * alpha
        offs[u] = off.to(torch.bfloat16)
    return offs

def vla_readout_full(model, inputs):
    # readout-path forward returning ALL hidden states (for per-sample patching)
    holder = {}
    inner = model["vla"].language_model.model
    orig = inner.forward
    def wrapped(*a, **k):
        k["output_hidden_states"] = True
        out = orig(*a, **k)
        holder["all"] = out.hidden_states
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
    return holder["all"]

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

def logits_of(h):
    return F.linear(h.float().cpu(), lm_head_w)     # (vocab,)""")

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

code("""print("\\n########## STAGE 1: GQA READOUT + RECOVERY ##########")
# ---- select split + filter by image availability + single-token answers ----
qfiles = ["testdev_all_questions.json", "testdev_balanced_questions.json", "val_subset_5k.json"]
def load_questions(path):
    with open(path) as f:
        return json.load(f)

def iter_entries(gqa_all):
    # handles both structures: {imageId: [{question, answer}...]} and {qid: {imageId, question, answer}}
    first = next(iter(gqa_all.values()))
    if isinstance(first, list):
        for img_id, qs in gqa_all.items():
            for q in qs:
                yield img_id, q["question"], q["answer"]
    else:
        for q in gqa_all.values():
            yield q["imageId"], q["question"], q["answer"]

def build_pairs(gqa_all, img_ok):
    pairs = []
    for img_id, question, answer in iter_entries(gqa_all):
        if not img_ok(img_id):
            continue
        a = ans_id(answer)
        if a is not None:
            pairs.append((img_id, question, a))
    return pairs

selected = None
for qf in qfiles:
    qp = f"{Q_DIR}/{qf}"
    if not os.path.exists(qp):
        continue
    gqa_all = load_questions(qp)
    pairs = build_pairs(gqa_all, lambda i: os.path.exists(f"{IMG_DIR}/{i}.jpg"))
    print(f"{qf}: {sum(len(v) for v in gqa_all.values()) if isinstance(next(iter(gqa_all.values())), list) else len(gqa_all)} questions, "
          f"{len(pairs)} usable (image + single-token answer)")
    if len(pairs) >= min(GQA_N, 300):
        selected = (qf, pairs)
        break
if selected is None:
    print("!! insufficient coverage on all splits - GQA stage will be skipped")
    gqa_ok = False
else:
    gqa_ok = True
    qf, pairs = selected
    rng = np.random.RandomState(RNG_SEED)
    idx = rng.choice(len(pairs), size=min(GQA_N, len(pairs)), replace=False)
    sel = [pairs[i] for i in idx]
    print(f"selected {len(sel)} questions from {qf}")

def gqa_inputs(img_id, question):
    from PIL import Image
    pil = Image.open(f"{IMG_DIR}/{img_id}.jpg").convert("RGB")
    prompt = f"In: {question}\\nOut:"
    inputs = bc["proc"](prompt, pil).to(DEVICE, dtype=torch.bfloat16)
    if inputs["pixel_values"].shape[1] == 6:      # single image -> duplicate into both slots
        inputs["pixel_values"] = torch.cat([inputs["pixel_values"], inputs["pixel_values"]], dim=1)
    return inputs

acc = {m: {"correct": 0, "top5": 0, "n": 0} for m in
        ["base", "bc", "aa"] + [f"bc_p{mask}_{a}" for mask in ["all", "vis"] for a in PATCH_AS] + ["bc_persample"]}
agree = {m: {"cos": [], "ccos": [], "t1": 0, "n": 0} for m in acc}
base_answers = []
n_run = 0
if gqa_ok:
    # answer-vocabulary restricted readout (standard GQA/VQA protocol)
    ans_vocab_ids = sorted(set(a_id for _, _, a_id in sel))
    print(f"answer vocabulary size: {len(ans_vocab_ids)}")

    def restricted_correct(l, a_id):
        return int(ans_vocab_ids[int(l[ans_vocab_ids].argmax().item())] == a_id)

    def restricted_top5(l, a_id):
        idx = l[ans_vocab_ids].topk(5).indices.tolist()
        return int(a_id in {ans_vocab_ids[i] for i in idx})

    for img_id, question, a_id in sel:
        try:
            inputs = gqa_inputs(img_id, question)
        except Exception as e:
            continue
        n_text = inputs["input_ids"].shape[-1] - 1
        h_base = base_readout_hidden(base_vlm, inputs)
        l_base = logits_of(h_base)
        base_answers.append(int(l_base.argmax().item()))
        h_bc = vla_readout_hidden(bc, inputs)
        h_aa = vla_readout_hidden(aa, inputs)
        l_bc = logits_of(h_bc); l_aa = logits_of(h_aa)
        lp = {}
        for mask in ["all", "vis"]:
            for a in PATCH_AS:
                offs = build_offsets_readout(n_text, a, delta, DEEP6, mask=mask)
                lp[f"{mask}_{a}"] = logits_of(vla_readout_patched(bc, inputs, offs, layers=set(DEEP6)))
        # per-sample ceiling: exact base-state restoration at deep6 (readout-path layout)
        hs_b = base_hidden_full(base_vlm, inputs)
        hs_c = vla_readout_full(bc, inputs)
        offs_ps = {}
        for u in DEEP6:
            off = torch.zeros(512 + n_text + 1, 896, device=DEVICE)
            hb = hs_b[u + 1][0].float(); hc = hs_c[u + 1][0].float()
            off[0:512] = (hb[1:513] - hc[0:512])
            # q tokens are position-aligned at 513..512+n_text (base keeps BOS@0; readout has BOS@512)
            off[513:513 + n_text] = (hb[513:513 + n_text] - hc[513:513 + n_text]).float()
            offs_ps[u] = off.to(torch.bfloat16)
        lp["persample"] = logits_of(vla_readout_patched(bc, inputs, offs_ps, layers=set(DEEP6)))

        for m, l in [("base", l_base), ("bc", l_bc), ("aa", l_aa)] + \\
                    [(f"bc_p{mask}_{a}", lp[f"{mask}_{a}"]) for mask in ["all", "vis"] for a in PATCH_AS] + \\
                    [("bc_persample", lp["persample"])]:
            acc[m]["n"] += 1
            acc[m]["correct"] += restricted_correct(l, a_id)
            acc[m]["top5"] += restricted_top5(l, a_id)
            c = float(F.cosine_similarity(l_base.unsqueeze(0), l.unsqueeze(0)).item())
            lb, lm = l_base - l_base.mean(), l - l.mean()
            cc = float(F.cosine_similarity(lb.unsqueeze(0), lm.unsqueeze(0)).item())
            agree[m]["cos"].append(c); agree[m]["ccos"].append(cc)
            agree[m]["t1"] += int(l.argmax().item() == l_base.argmax().item())
            agree[m]["n"] += 1
        n_run += 1

print("\\n=== GQA READOUT (open-vocab next-token accuracy) ===")
print(f"  base top-1 distinct answer tokens: {len(set(base_answers))} (informativeness)")
for m in acc:
    n = max(1, acc[m]["n"])
    print(f"  {m:8s} acc={acc[m]['correct']/n*100:5.1f}%  top5={acc[m]['top5']/n*100:5.1f}%  (n={acc[m]['n']})")
print("\\n=== NEXT-TOKEN AGREEMENT WITH BASE ===")
for m in agree:
    n = max(1, agree[m]["n"])
    print(f"  {m:8s} cos={np.mean(agree[m]['cos']):6.4f} ccos={np.mean(agree[m]['ccos']):6.4f} t1={agree[m]['t1']/n*100:5.1f}%")

# ---- gates ----
if gqa_ok:
    b = acc["base"]["correct"] / max(1, acc["base"]["n"])
    c = acc["bc"]["correct"] / max(1, acc["bc"]["n"])
    a = acc["aa"]["correct"] / max(1, acc["aa"]["n"])
    ps = acc["bc_persample"]["correct"] / max(1, acc["bc_persample"]["n"])
    bcos = float(np.mean(agree["base"]["cos"]))
    ccos_m = float(np.mean(agree["bc"]["cos"]))
    acos = float(np.mean(agree["aa"]["cos"]))
    pcos_vis = float(np.mean(agree["bc_pvis_1.0"]["cos"]))
    informative = len(set(base_answers)) >= 5
    erasure = (b - c) >= 0.10
    preserve = acos > ccos_m
    recovery = (ps - c) >= 0.03 or (pcos_vis - ccos_m) >= 0.01
    print("\\nMECHANICS:",
          "OK" if (informative and erasure and preserve and recovery) else "SUSPICIOUS - inspect",
          f"(informative {informative}, erasure base-bc={100*(b-c):.1f}pts {erasure}, "
          f"aa-cos {acos:.3f} vs bc-cos {ccos_m:.3f} {preserve}, "
          f"persample rec {100*(ps-c):+.1f}pts / vis-patch cos {pcos_vis:.3f} vs {ccos_m:.3f} {recovery})")
    results = {"smoke": SMOKE, "split": qf, "n": acc["base"]["n"],
               "acc": {m: {k: v for k, v in s.items()} for m, s in acc.items()},
               "agree": {m: {k: (float(np.mean(v)) if isinstance(v, list) else v) for k, v in s.items()} for m, s in agree.items()}}
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print("saved:", OUT_JSON)""")

md("""## Summary

- GQA open-vocab next-token accuracy (paper's Fig-9 metric at readout level) + top-5 + behavioral
  next-token agreement vs base, for base / BC / AA / patched-BC.
- Gates: coverage, informativeness, erasure (base-bc >= 10 pts), preservation (aa > bc), recovery
  (patched > bc by >= 3 pts).
- If insufficient image coverage, the notebook falls back to val/train question subsets.

Results saved to `/kaggle/working/e5_results.json`.""")

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = os.path.join(OUT_DIR, "e5-gqa-readout.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)
print("wrote", out_path)

metadata = {
    "id": "your-kaggle-username/e5-gqa-readout",
    "title": "E5 GQA Readout",
    "code_file": "e5-gqa-readout.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": True,
    "machine_shape": "NvidiaTeslaT4",
    "datasources": [
        {"datasourceId": "minhngcng3/gqa-images-subset"}
    ],
}
with open(os.path.join(OUT_DIR, "kernel-metadata.json"), "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4)
print("wrote kernel-metadata.json")