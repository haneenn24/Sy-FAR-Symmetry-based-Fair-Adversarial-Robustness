# Sy-FAR: Symmetry-based Fair Adversarial Robustness

This repository provides the official implementation of **Sy-FAR**, a training and evaluation toolkit for **fair** and **robust** image classification. Sy-FAR couples:

* a **symmetry regularizer** that penalizes asymmetric confusions between classes,
* **ROA** (Rectangular Occlusion Attack)–based adversarial training,
* optional **FAAL** (KL-DRO) class reweighting,
* clean, modular code with CLIs for standard and adversarial training, VGG/ResNet/ViT backbones, AutoAttack evaluation, randomized smoothing tests, and rich fairness/robustness metrics & visualizations.

---

## Repository structure

```
.
├─ attacks/
│  ├─ autoattack.py           # AutoAttack evaluation (CIFAR-10/100)
│  ├─ glass_attack.py         # eyeglass-frame attack
│  ├─ smooth_glassattack.py   # randomized smoothing vs glasses
│  ├─ smooth_patch.py         # randomized smoothing vs learned patch
│  └─ sticker_attack.py       # ROA as a standalone attack script
│
├─ datasets/                  # (placeholder) put dataset helpers or symlinks here
│  └─ __init__.py
│
├─ defenses/
│  ├─ PGD.py                  # PGD utilities (training/eval helpers)
│  └─ ROA.py                  # ROA-based adversarial training loop (targeted option)
│
├─ evaluation/
│  ├─ metrics_report.py       # compute fairness/robustness metrics from a CM
│  └─ test_clean_accuracy.py  # clean test accuracy for a saved model
│
├─ models/
│  ├─ resnet.py               # ResNet{18,34,50,…}
│  ├─ vgg16.py                # VGG-Face convs + flexible FC; e2e/head-only; multiple inits
│  └─ vit.py                  # ViT via timm ('vit_base_patch16_224')
│
├─ pretrained_models/
│  └─ __init__.py             # place VGG_FACE.t7 here (or pass its path explicitly)
│
├─ training_schemes/
│  ├─ baselines/              # room for additional baselines
│  ├─ adversarial.py          # adversarial training (ROA baseline)
│  ├─ standard.py             # standard ERM training
│  └─ syfar.py                # Sy-FAR: clean + ROA + symmetry penalty (+ optional FAAL)
│
├─ utils/
│  ├─ carlini_wagner.py       # CW margin losses
│  ├─ data_process.py         # dataset transforms & dataloaders (edit data_dir)
│  ├─ plot_visual_metrics.py  # heatmaps + exemplar strips
│  └─ __init__.py
│
├─ README.md
├─ LICENSE.md
└─ requirements.txt
```
---

## Installation

conda create -n syfar python -y
conda activate syfar
pip install -r requirements.txt

**Dataset**

* We use an ImageFolder-style layout. Update `data_dir` in `utils/data_process.py`:

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
python training_schemes/standard.py \
  --batch-size 8 --epochs 25 --lr 1e-3 --opt sgd --step-size 7 --gamma 0.1
```

### 2) Sy-FAR training (ours)

```bash
python training_schemes/syfar.py \
  --batch-size 8 --epochs 5 --lr 1e-3 \
  --clean-weight 0.1 --adv-weight 10 --sym-weight 10 --epsilon 0.1 \
  --alpha 20 --iters 100 --width 70 --height 70 --xskip 10 --yskip 10 \
  --out-dir runs/syfar --tag pubfig_vgg16
```

*Notes:*

* The symmetry penalty is computed from the adversarial confusion matrix within each batch.
* FAAL (KL-DRO) class reweighting is available in the codebase; enable it if your run script exposes the flags.

### 3) Adversarial training via ROA (baseline)

```bash
python training_schemes/adversarial.py \
  --batch-size 8 --epochs 5 --lr 1e-3 \
  --alpha 20 --iters 100 --width 70 --height 70 --xskip 10 --yskip 10
```

### 4) Evaluate clean accuracy

```bash
python evaluation/test_clean_accuracy.py \
  --model-path path/to/model.pt
```

### 5) AutoAttack (CIFAR-10/100)

```bash
python attacks/autoattack.py \
  --data-dir ./cifar-data --batch-size 200 \
  --model WRN --pre-trained MART --model-name best \
  --epsilon 8 --normalization 01
```

### 6) Randomized smoothing vs glasses/patch

```bash
python attacks/smooth_glassattack.py gaussian_model.pt -sigma 1.0 -outfile out_glass.txt --batch 32 --N 1000 --alpha 0.001
python attacks/smooth_patch.py gaussian_model.pt -sigma 1.0
```

### 7) Visualize fairness/robustness

```bash
python utils/plot_visual_metrics.py
# writes: *_full_heatmap.png, *_upper_triangle_diff.png, *_upper_triangle_absdiff.png
```

---

## Backbones

* **VGG-16** (`models/vgg16.py`): VGG-Face conv stacks + custom FC head(s).
  Options include:

  * training mode: end-to-end or head-only,
  * several head initializations (xavier/kaiming/trunc. normal, etc.).
* **ResNet** (`models/resnet.py`): ResNet18/34/50 variants.
* **ViT** (`models/vit.py`): `timm.create_model('vit_base_patch16_224', pretrained=True)`.

---

## Metrics & fairness

We provide:

* **Robust accuracy** (overall and per-class),
* **Subgroup Robust Accuracy (SRA)** and **Robustness Disparity (RD)**,
* **Target-side misclassification shares** and gaps,
* **Max asymmetry gap:** $\max_{i<j} |\mathrm{cm}_{i,j} - \mathrm{cm}_{j,i}|$,
* The **symmetry penalty** used during training.

See `evaluation/metrics_report.py` and `utils/plot_visual_metrics.py`.

---

## Pretrained Models (quick use)

**Provided:**
- `pretrained_models/vgg16_pubfig.pt`
- `pretrained_models/vgg16_pubfig_siblings.pt`

**Evaluate clean accuracy:**
```bash
python evaluation/test_clean_accuracy.py --model-path pretrained_models/vgg16_pubfig.pt
# or
python evaluation/test_clean_accuracy.py --model-path pretrained_models/vgg16_pubfig_siblings.pt
````

**Evaluate under glasses attack:**

```bash
python attacks/glass_attack.py --model-path pretrained_models/vgg16_pubfig.pt
# or
python attacks/glass_attack.py --model-path pretrained_models/vgg16_pubfig_siblings.pt
```

---

## Reproducibility

* Seeds are set and deterministic options enabled where feasible.
* Some CUDA/cuDNN ops may remain nondeterministic; report seeds and versions in experiments.

---

## Requirements (high level)

Pinned in `requirements.txt`. Additional packages used by this repo include:

* `timm` (ViT), `autoattack`, `robustbench`, `torchattacks`,
* `cvxpy` + solvers `ecos`, `scs` (and optional `mosek` if you have a license)

## Contact

Questions, issues, or contributions are welcome—please open a GitHub issue or pull request.
