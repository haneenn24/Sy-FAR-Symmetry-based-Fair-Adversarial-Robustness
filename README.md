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
│  ├─ mask                    # Include physical glasses, stickers, masks, etc.
│  ├─ autoattack.py           # AutoAttack evaluation (CIFAR-10/100)
│  ├─ facemask_attack.py      # grid level facemask-frame attack
│  ├─ glass_attack.py         # eyeglass-frame attack
│  └─ sticker_attack.py       # ROA as a standalone attack script
│
|─ datasets/                  # pubfig preprocessed dataset
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
├─ pretrained_models/         # standard training, ROA-based adversarial training, FAAL, SpecNorm, and our Sy-FAR models
│
├─ training_schemes/
│  ├─ baselines/              # faal and specnorm
│  ├─ adversarial.py          # adversarial training (ROA baseline)
│  ├─ standard.py             # standard ERM training
│  └─ syfar.py                # Sy-FAR: clean + ROA + symmetry penalty (+ optional FAAL)
│
├─ utils/
│  ├─ carlini_wagner.py       # CW margin losses
│  ├─ data_process.py         # dataset transforms & dataloaders (edit data_dir)
│  ├─ plot_visual_metrics.py  # heatmaps + exemplar strips
│  ├─ split_dataset.py        # split a dataset into train/val/test subsets in ImageFolder format.
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

## Quickstart

### 1) Standard training (clean ERM)

```bash
python training_schemes/standard.py \
  --batch-size 8 --epochs 25 --lr 1e-3 --opt sgd --step-size 7 --gamma 0.1
```

### 2) Adversarial training via ROA (baseline)

```bash
python training_schemes/adversarial.py \
  --batch-size 8 --epochs 5 --lr 1e-3 \
  --alpha 20 --iters 100 --width 70 --height 70 --xskip 10 --yskip 10
```

### 3) Sy-FAR training (ours)

```bash
python training_schemes/syfar.py \
  --batch-size 8 --epochs 5 --lr 1e-3 \
  --clean-weight 0.1 --adv-weight 10 --sym-weight 10 --epsilon 0.1 \
  --alpha 20 --iters 100 --width 70 --height 70 --xskip 10 --yskip 10 \
  --out-dir runs/syfar --tag pubfig_vgg16
```

*Notes:*
* The symmetry penalty is computed from the adversarial confusion matrix within each batch.
* Provided other baselines training schemes (FAAL and SpecNorm).

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

## 6) Glass Attack (Eyeglass Frame)

### Untargeted Glass Attack
python glass_attack.py \
  --model-checkpoint /path/to/model.pt \
  --glass-mask-path ./attacks/mask/eyeglass.png \
  --batch-size 64 \
  --alpha 20 \
  --iters 1 10 50 100 300 \
  --restarts 1 \
  --num-classes 8 \
  --save-images \
  --save-dir ./attack_outputs/glass_untargeted

### Targeted Glass Attack
python glass_attack.py \
  --model-checkpoint /path/to/model.pt \
  --glass-mask-path ./attacks/mask/eyeglass.png \
  --targeted \
  --target-class 5 \
  --batch-size 64 \
  --alpha 20 \
  --iters 100 \
  --restarts 1 \
  --num-classes 8 \
  --save-images \
  --save-dir ./attack_outputs/glass_targeted_class5

---

## 7) Face Mask Attack (Grid-Level δ)

### Untargeted Face Mask Attack
python mask_attack.py \
  --model-checkpoint /path/to/model.pt \
  --mask-path ./attacks/mask/facemask.png \
  --batch-size 64 \
  --alpha 20 \
  --iters 50 100 200 \
  --restarts 1 \
  --num-classes 8 \
  --save-images \
  --save-dir ./attack_outputs/mask_untargeted

### Targeted Face Mask Attack
python mask_attack.py \
  --model-checkpoint /path/to/model.pt \
  --mask-path ./attacks/mask/facemask.png \
  --targeted \
  --target-class 3 \
  --batch-size 64 \
  --alpha 20 \
  --iters 200 \
  --restarts 1 \
  --num-classes 8 \
  --save-images \
  --save-dir ./attack_outputs/mask_targeted_class3


## Backbones

* **VGG-16** (`models/vgg16.py`): VGG-Face conv stacks + custom FC head(s).
  Options include:

  * training mode: end-to-end or head-only,
  * several head initializations (xavier/kaiming/trunc. normal, etc.).
* **ResNet** (`models/resnet.py`): ResNet18/34/50 variants.
* **ViT** (`models/vit.py`)

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

## Requirements (High-Level Overview)

All dependencies are pinned in `requirements.txt`.  
This repository additionally uses several commonly adopted packages for vision models, adversarial evaluation, and robust optimization:

- **autoattack**, **robustbench**, **torchattacks** — standard adversarial evaluation libraries  
- **cvxpy** with open-source solvers **ECOS** and **SCS** for the FAAL reweighting module  

All components are installable directly via `pip`, and no proprietary solvers are required.

---
## Dataset (Preprocessed & Cropped)

This artifact includes a **PubFig dataset with 12 identities** (10 PubFig + 2 siblings).  
**All images are already preprocessed and cropped using FaceX-Zoo** (detect → align → crop).  
All dataset images in this artifact were generated using the above FaceX-Zoo preprocessing pipeline.

### FaceX-Zoo (used for preprocessing)

git clone https://github.com/JDAI-CV/FaceX-Zoo.git  
export PYTHONPATH="$PWD/FaceX-Zoo:$PYTHONPATH"  
git checkout 16b793a7564a4b9308cf94e62bdb2ffacb3a725a  

### FaceX-Zoo API examples (used during preprocessing)

python api_usage/face_detect.py  
python api_usage/face_alignment.py  
python api_usage/face_crop.py  
python api_usage/face_feature.py  
python api_usage/face_pipline.py  
python api_usage/face_parsing.py

## Downloading Full Dataset: PubFig or VGGFace:

### PubFig (URL-based dataset)
PubFig provides URL lists instead of images. Download from:
- https://www.cs.columbia.edu/CAVE/databases/pubfig/
- https://www.cs.columbia.edu/CAVE/databases/pubfig/download/

Optional image fetcher:
- https://github.com/dimatura/getpubfig

Example:
git clone https://github.com/dimatura/getpubfig.git
cd getpubfig
# place dev_urls.txt / eval_urls.txt here
python getpubfig.py

### VGGFace / VGGFace2 (for sibling identities)
If adding sibling pairs, obtain images from:
- https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/
- https://www.robots.ox.ac.uk/~vgg/software/vgg_face/

Download per dataset terms, then run the same preprocessing (detect → align → crop).


### Split & Layout (ImageFolder)

If you want to create a train/val/test ImageFolder structure from the preprocessed dataset, run:

python utils/split_dataset.py

This generates train, val and test dirs.


### Normalization

We use VGG-Face–style normalization:

```python
# utils/data_process.py
mean = [0.367035294117647, 0.41083294117647057, 0.5066129411764705]
std  = [1/255, 1/255, 1/255]
transforms.Normalize(mean, std)
```


Some attack scripts also subtract classic VGG-Face **BGR** means in pixel space:
`[129.1863, 104.7624, 93.5940]`.

---

## Pretrained models

We also include pre-trained models for all training schemes—standard training, ROA-based adversarial training, FAAL, SpecNorm, and our Sy-FAR method so users can directly evaluate clean and adversarial performance without retraining models from scratch.

---

## Contact

Questions, issues, or contributions are welcome—please open a GitHub issue or pull request.

---

## Citation

If you find this repository useful in your research, please cite:

```bibtex
@inproceedings{najjar2025syfar,
    author  = {Najjar, Haneen and Ronen, Eyal and Sharif, Mahmood},
    title   = {{Sy-FAR}: {S}ymmetry-based Fair Adversarial Robustness},
    year    = {2026},
    booktitle = {USENIX Security Symposium}
}
