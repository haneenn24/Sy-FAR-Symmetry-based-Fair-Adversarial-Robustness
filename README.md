# Sy-FAR-Symmetry-based-Fair-Adversarial-Robustness
This repository provides the official implementation of Sy-FAR, a novel training framework designed to improve adversarial robustness while ensuring fairness across classes
# Sy-FAR: Symmetry-based Fair Adversarial Robustness

Sy-FAR is a training/evaluation toolkit for **fair** and **robust** image classification. It combines:

* a **symmetry regularizer** that penalizes asymmetric confusions between classes,
* **ROA** (Rectangular Occlusion Attack) adversarial training in the pixel domain,
* optional **FAAL** (KL-DRO) class reweighting,
* clean, modular code with CLIs for standard training, ViT/VGG/ResNet backbones, AutoAttack evaluation, randomized smoothing tests, and rich fairness/robustness metrics & visualizations.

---

## Highlights

* **Symmetry penalty (Sy-FAR):** minimizes gaps $|\text{cm}_{i,j}-\text{cm}_{j,i}|$ on the adversarial confusion matrix to reduce group/class disparities.
* **Physical-ish adversaries (ROA):** search + constrained PGD over rectangles in input pixels (optionally targeted), compatible with VGG-Face mean.
* **Fair reweighting (FAAL):** plug-in KL-divergence DRO to emphasize hard or under-served classes.
* **Batteries included:** AutoAttack, randomized smoothing against “glasses” & “patch” attacks, rich metrics, heatmaps with example faces.

---

## Repository structure (suggested)

```
.
├─ models/
│  ├─ vgg16.py           # VGG-16 (VGG-Face convs + flexible FC; e2e or head-only; multiple inits)
│  ├─ resnet.py          # ResNet{18,34,50,…}
│  └─ vit.py             # ViT via timm (vit_base_patch16_224)
│
├─ utils/
│  ├─ data_process.py    # dataset transforms & dataloaders (edit data_dir)
│  ├─ carlini_wagner.py  # CW margin loss
│  └─ faal.py            # FAAL KL-DRO weights (optional)
│
├─ attacks/
│  ├─ glass_attack.py            # eyeglass-frame attack (digit/pixel space)
│  ├─ smooth_glassattack.py      # smoothed classifier + glasses
│  ├─ smooth_patch.py            # smoothed classifier + learned patch
│  ├─ sticker_attack.py          # ROA baseline (script)
│  └─ roa.py                     # ROA library (search + cPGD, normalized domain)
│
├─ defenses/
│  ├─ standard_train.py          # standard ERM training (clean)
│  ├─ specnorm.py                # spectral-norm flavored training loop
│  ├─ roa_adv_train.py           # adversarial training via ROA
│  └─ syfar_train.py             # **Sy-FAR** (clean + ROA + symmetry penalty [+ FAAL])
│
├─ evaluation/
│  ├─ origin_test.py             # clean accuracy
│  ├─ autoattack.py              # AutoAttack evaluation on CIFAR-10/100
│  ├─ metrics_all.py             # fairness/robustness metrics from a confusion matrix
│  └─ plot_visual_metrics.py     # heatmaps + candidate images strip
│
└─ requirements.txt
```

> Some filenames above reflect the refactors we discussed. If your local names differ, keep the README examples consistent with your actual file paths.

---

## Setup

**Python:** 3.8.5+
**CUDA (example):** Torch 2.2.2 + cu118 (as pinned)

```bash
# 1) Create env
conda create -n syfar python=3.8 -y
conda activate syfar

# 2) Install deps
pip install -r requirements.txt
```

**Extra model files**

* VGG-Face weights (`VGG_FACE.t7`) required if you call `VGG_16.load_weights(...)`.

  * Place it where your `models/vgg16.py` expects it, or pass a path to `load_weights(path=...)`.

**Dataset**

* We used a PubFig-10–style split (and a “siblings” mix in some experiments).
* Edit `utils/data_process.py` to point `data_dir` to your dataset root structured as:

  ```
  <DATA_DIR>/
    train/<class_name>/*.jpg
    val/<class_name>/*.jpg
    test/<class_name>/*.jpg
  ```

---

## Quickstart

### 1) Standard training (clean ERM)

```bash
python defenses/standard_train.py \
  --batch-size 8 --epochs 25 --lr 1e-3 --opt sgd --step-size 7 --gamma 0.1
```

### 2) Sy-FAR training (ours)

```bash
python defenses/syfar_train.py \
  --batch-size 8 --epochs 5 --lr 1e-3 \
  --loss carlini_wagner \
  --clean-weight 0.1 --adv-weight 10 --sym-weight 10 --epsilon 0.1 \
  --alpha 20 --iters 100 --width 70 --height 70 --xskip 10 --yskip 10 \
  --out-dir runs/syfar --tag pubfig_vgg16
```

Optional FAAL reweighting:

```bash
python defenses/syfar_train.py \
  ... --use-faal --faal-radius 0.1
```

### 3) Adversarial training via ROA (baseline)

```bash
python defenses/roa_adv_train.py \
  --batch-size 8 --epochs 5 --lr 1e-3 \
  --alpha 20 --iters 100 --width 70 --height 70 --xskip 10 --yskip 10
```

### 4) Evaluate clean accuracy

```bash
python evaluation/origin_test.py \
  --model-path path/to/model.pt
```

### 5) AutoAttack (CIFAR-10/100)

```bash
python evaluation/autoattack.py \
  --data-dir ./cifar-data --batch-size 200 \
  --model WRN --pre-trained MART --model-name best \
  --epsilon 8 --normalization 01
```

### 6) Randomized smoothing vs glasses

```bash
python attacks/smooth_glassattack.py my_gaussian_model.pt \
  -sigma 1.0 -outfile out_glass.txt --batch 32 --N 1000 --alpha 0.001
```

### 7) Visualize fairness/robustness

```bash
python evaluation/plot_visual_metrics.py
# saves: *_full_heatmap.png, *_upper_triangle_diff.png, *_upper_triangle_absdiff.png
```

---

## VGG-16 training modes & inits

`models/vgg16.py` exposes knobs to control training and initialization of the final layer(s).

* **Training mode:** end-to-end vs. head-only (freeze convs)
* **Init:** `xavier_uniform`, `xavier_normal`, `kaiming_uniform`, `kaiming_normal`, `trunc_normal`, etc.

Example (inside your script):

```python
from models.vgg16 import VGG_16

model = VGG_16(
    num_classes=10,
    train_mode="e2e",               # or "head_only"
    head_init="xavier_uniform",     # or "kaiming_normal", ...
)
# Optionally load VGG-Face conv weights
model.load_weights(path="path/to/VGG_FACE.t7")
```

---

## Metrics & fairness

We provide utilities to compute:

* **Robust accuracy** (overall, per-class)
* **Subgroup Robust Accuracy (SRA)**, **Robustness Disparity (RD)**
* **Target-perspective misclassification shares** and gaps
* **Max asymmetry gap**: $\max_{i<j} |\text{cm}_{i,j}-\text{cm}_{j,i}|$
* **Symmetry penalty** used in training

See `evaluation/metrics_all.py` and the plotting script for heatmaps + exemplar strips.

---

## Reproducibility

* We set seeds and enable deterministic flags where viable.
* Use `--seed` on training scripts.
* GPUs and cudnn settings may still cause slight nondeterminism in some ops.

---

## Tips / Troubleshooting

* **cvxpy / FAAL:** If you enable `--use-faal`, make sure the solvers in `requirements.txt` are installed. `mosek` is optional and requires a license; otherwise ECOS/SCS are used.
* **Data path:** If you see “file not found” in dataloaders, edit `utils/data_process.py` to your dataset root.
* **VGG-Face weights:** If `load_weights` fails, check path and Torch/Lua tensor ordering; we copy conv weights only.
* **CUDA version:** Match your local CUDA to the pinned PyTorch wheels (we used cu118 in `requirements.txt`).

## Contact
Questions, issues, or contributions are welcome. Please open a GitHub issue or pull request.
