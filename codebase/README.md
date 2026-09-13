# Erasure is a Shift, Not a Deletion — experiment code

Code and run logs for the paper:

> **Erasure is a Shift, Not a Deletion: Recovering Pretrained Semantics from Behavior-Cloned VLAs at Inference Time**

Every number in the paper is produced by the scripts and logs in this folder. The pipeline runs entirely on publicly released checkpoints: no training, no simulator, and no proprietary data.

## The claim in one paragraph

Behavior-cloning finetuning of a pretrained vision-language model (VLM) into a robot policy does not delete the pretrained semantics; it displaces them. The drift is low-rank, concentrated in the deepest layers, measurable from the checkpoint pair alone, and can be added back at inference ("displacement correction") to recover pretrained geometry (+15.4 vision CKA points), pretrained next-token behavior (+0.08–0.10 logit agreement), and a substantial fraction of pretrained answers (restricted-answer GQA accuracy 5.7% → 20.2%). Thirty-two principal directions per deep layer reproduce the correction to within 0.8 points of the full tensor, as a roughly 1 MB "drift bundle".

## Experiments

| Folder | Experiment | Protocol (short) | Where in the paper |
|---|---|---|---|
| `e0-gate/` | E0 — phenomenon (gate) | Per-layer text-token CKA to the pretrained VLM; 60–500 non-stationary frames in three sets; vision CKA on a strided 64-patch subset | Fig. 1, Table 1 |
| `e1-recovery/` | E1 — displacement correction | Drift from 400 training frames; grid α ∈ {0.25–2.0} × layers {21–24, 18–23, all} × masks {all, text-only, vision-only}; evaluated on 100 held-out frames; random / reverse / per-sample / anchored-push controls | Fig. 2, Table 2; full sweep in Appendix Table 5 |
| `e2-drift-structure/` | E2 — drift structure | Per-layer, per-region SVD of the drift; rank-k bundle reconstruction error and recovery vs. k; 60 held-out frames; drift-norm/CKA correlation | Fig. 3; Appendix Table 6 |
| `e3-frozen-head/` | E3 — frozen-head readability | Motion-direction readout through the released trained projection vs. identity projection; fresh linear probes (5-fold × 3 CV); 250 frames | Fig. 4; Appendix Table 7 |
| `e4-agreement/` | E4 — behavioral agreement | Frozen-head next-token agreement with the pretrained model (logit cosine, top-1 / top-5); three prompt templates × 200 frames; corrected rows | Table 3; Appendix Table 8 |
| `e5-gqa/` | E5 — answer recovery | Restricted-answer GQA readout (1,500 questions over a 180-word answer vocabulary) and recovery under per-sample displacement correction | Table 4 |

## Repository layout

Each experiment folder holds a host-side builder and the complete log of the released run:

```
codebase/
├── e0-gate/
│   ├── build_kaggle_e0.py           # writes the Kaggle notebook + kernel-metadata.json
│   └── e0-gate_final_log.txt        # raw output of the released run
├── e1-recovery/
│   ├── build_kaggle_e1.py
│   └── e1-recovery_final_log.txt
├── e2-drift-structure/
├── e3-frozen-head/
├── e4-agreement/
└── e5-gqa/
```

The builders embed the full notebook source, which is the actual experiment code. They only *generate* the notebook; they do not run the experiment on your machine.

## Running an experiment

1. Generate the notebook (host-side, no GPU needed):
   ```
   python codebase/e3-frozen-head/build_kaggle_e3.py
   ```
   This writes `<repo root>/kaggle_e3/` containing `e3-frozen-head.ipynb` and `kernel-metadata.json`.
2. Put your Kaggle username into the generated `kernel-metadata.json` (`"id": "your-kaggle-username/…"`) and push with the Kaggle CLI (requires a Kaggle API token):
   ```
   kaggle kernels push -p kaggle_e3
   ```
3. The notebook runs end-to-end on a single NVIDIA T4 with internet enabled; checkpoints and data are downloaded in the first cells. Wall-clock times vary from minutes (E0) to a few hours (E5); the logs record the exact durations.
4. Each run writes `eX_results.json` plus console output. The `*_final_log.txt` files here are the raw outputs of the runs whose numbers appear in the paper.

## Models and data (downloaded, not redistributed)

- Pretrained VLM — `Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b`
- Behavior-cloned VLA — `VLA-Adapter/LIBERO-Spatial-Pro`
- Anchored VLA — `Dwipz/Anchor-Align`
- LIBERO-Spatial frames (RLDS) — `openvla/modified_libero_rlds`
- GQA — questions from the official GQA release; images via the Kaggle dataset `minhngcng3/gqa-images-subset`

Each notebook pins its Python dependencies in its first cells (PyTorch, Transformers, timm, PEFT, TensorFlow/TFDS with dlimp, and others).

## Conventions worth knowing (also in Appendix A)

- Readout path: `[512 vision patches, BOS + N question tokens]`, with no action tokens. Position p of the readout path corresponds to position p+1 of the action path; the answer position is 512+N.
- The frozen head is always the pretrained VLM's language-model head, applied at the answer position; the same head is used for all three models.
- The drift Δ is the per-layer, per-position activation difference (pretrained − finetuned) averaged over frames. Displacement correction adds α·Δ at inference through residual-stream hooks.

## Notes

- Some generated kernel slugs keep their original run names (`e0-erasure-smoke-test`, `e1-erasure-patching`, `e4-next-token-agreement`); folder names here follow the paper's E0–E5 labels.
- The logs are kept verbatim as released. They contain no personal or machine-specific data.
- Model checkpoints remain under their original licenses and are fetched from their original hubs.

## Citation

If you build on this code, please cite the paper:

```bibtex
@misc{erasure_is_a_shift_2026,
  title = {Erasure is a Shift, Not a Deletion: Recovering Pretrained Semantics from Behavior-Cloned VLAs at Inference Time},
  year  = {2026},
  note  = {Preprint; add the arXiv ID once available}
}
```
