# Erasure is a Shift, Not a Deletion

**Recovering Pretrained Semantics from Behavior-Cloned VLAs at Inference Time**

This repository contains the paper and the complete experiment code and run logs.

Paper: [10.5281/zenodo.22733623](https://doi.org/10.5281/zenodo.22733623)

## Summary

Behavior-cloning finetuning of a pretrained vision-language model (VLM) into a robot policy does not delete the pretrained semantics; it displaces them. The displacement is low-rank, concentrated in the deepest layers, and measurable from the checkpoint pair alone, and it can be added back at inference ("displacement correction") with zero training:

- pretrained geometry returns: **+15.4 vision CKA points**;
- pretrained next-token behavior returns: **+0.08–0.10 logit agreement**;
- pretrained answers return: restricted-answer GQA accuracy **5.7% → 20.2%** (×3.5);
- a **32-direction, ~1 MB drift bundle** reproduces the correction to within **0.8 points** of the full tensor;
- a transfer discriminator locates the mechanism: the frozen head reads the anchored model's states at **90.0%** vs. **58.8%** for the behavior-cloned model's, through the same trained projection, while the latter's information remains **89.9%** linearly decodable.

All experiments run on publicly released checkpoints at the readout level, with no training and no simulator.

## Repository layout

```
├── paper/
│   └── paper.pdf           the compiled paper
└── codebase/               experiment code and complete run logs (E0–E5)
    ├── README.md           (protocol map, how to run, conventions)
    ├── e0-gate/
    ├── e1-recovery/
    ├── e2-drift-structure/
    ├── e3-frozen-head/
    ├── e4-agreement/
    └── e5-gqa/
```

## Experiments

| | Experiment | In one line |
|---|---|---|
| E0 | phenomenon (gate) | Per-layer CKA to the pretrained VLM: deep-layer erasure is real and deep-concentrated |
| E1 | displacement correction | Inject the measured drift back at inference; recover geometry and behavior |
| E2 | drift structure | The drift is low-rank; 32 directions per layer capture the correction as a ~1 MB bundle |
| E3 | frozen-head readability | The anchored model stays readable through the trained projection; BC's states do not |
| E4 | behavioral agreement | Frozen-head next-token agreement with the pretrained model, before and after correction |
| E5 | answer recovery | Restricted-answer GQA readout recovers under per-sample displacement correction |

Protocols, file-level details, and run instructions are in [`codebase/README.md`](codebase/README.md).

## Reproducing the experiments

The `codebase/` builders generate the Kaggle notebooks used for every experiment; the `*_final_log.txt` files are the raw outputs of the released runs. See [`codebase/README.md`](codebase/README.md) for the step-by-step instructions.

All model checkpoints and datasets are downloaded from their original sources and are not redistributed here:

- pretrained VLM — `Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b`
- behavior-cloned VLA — `VLA-Adapter/LIBERO-Spatial-Pro`
- anchored VLA — `Dwipz/Anchor-Align`
- LIBERO-Spatial frames (RLDS) — `openvla/modified_libero_rlds`
- GQA — official questions; images via the Kaggle dataset `minhngcng3/gqa-images-subset`

## Citation

If you build on this work, please cite the paper:

```bibtex
@misc{jani_2026_22733623,
  author       = {Jani, Jatin},
  title        = {Erasure is a Shift, Not a Deletion: Recovering Pretrained Semantics from Behavior-Cloned VLAs at Inference Time},
  month        = sep,
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.22733623},
  url          = {https://doi.org/10.5281/zenodo.22733623}
}
```

Jani, J. (2026). *Erasure is a Shift, Not a Deletion: Recovering Pretrained Semantics from Behavior-Cloned VLAs at Inference Time*. Zenodo. https://doi.org/10.5281/zenodo.22733623
