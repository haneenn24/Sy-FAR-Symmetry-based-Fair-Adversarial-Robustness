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

## Evaluate

**Clean accuracy**
```bash
python evaluation/test_clean_accuracy.py --model-path <PATH_TO_YOUR_MODEL>.pt
````

**Glasses attack**

```bash
python attacks/glass_attack.py --model-path <PATH_TO_YOUR_MODEL>.pt
```

---

## Requirements (High-Level Overview)

All dependencies are pinned in `requirements.txt`.  
This repository additionally uses several commonly adopted packages for vision models, adversarial evaluation, and robust optimization:

- **timm** — Vision Transformer (ViT) models  
- **autoattack**, **robustbench**, **torchattacks** — standard adversarial evaluation libraries  
- **cvxpy** with open-source solvers **ECOS** and **SCS** for the FAAL reweighting module  

All components are installable directly via `pip`, and no proprietary solvers are required.

---
## Dataset: PubFig (setup & preprocessing)

### 1) Download PubFig
PubFig is released as **URL lists** (not images). Get the official files from Columbia CAVE:
- Homepage: <https://www.cs.columbia.edu/CAVE/databases/pubfig/>
- Download page (URL lists + metadata): <https://www.cs.columbia.edu/CAVE/databases/pubfig/download/>

Optional helper to fetch images from the URL lists:
- `getpubfig`: <https://github.com/dimatura/getpubfig>

```bash
# Example using getpubfig
git clone https://github.com/dimatura/getpubfig.git
cd getpubfig
# place the official dev_urls.txt / eval_urls.txt in this folder
python getpubfig.py      # downloads into ./images by default
````

> If you curate a **PubFig-10** subset, create `classes.txt` (10 identities) and only keep those folders.

### PubFig-10 class list (used in this repo)

Save as `classes.txt` (one per line) and use to subset your dataset:

**Subnote:** “siblings” identities (optional)

If you want sibling pairs, you can source additional identities from VGGFace/VGGFace2 and then run the same preprocessing:

VGGFace2 info page: https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/

Original VGG Face page: https://www.robots.ox.ac.uk/~vgg/software/vgg_face/

(Obtain images per the dataset’s instructions/terms, align & crop with your pipeline, and add the classes to your classes.txt.)

---

### 2) Preprocess (detect → align → crop)


Use **FaceX-Zoo** for ArcFace-style alignment/cropping:

```bash
# Clone the SDK once
git clone https://github.com/JDAI-CV/FaceX-Zoo.git
export PYTHONPATH="$PWD/FaceX-Zoo:$PYTHONPATH"   # make SDK importable
git checkout 16b793a7564a4b9308cf94e62bdb2ffacb3a725a
````

Now run our wrapper:

```bash
# A) Direct Python call
python preprocess/crop_lfw_edited_by_arcface.py \
  --lfw_edited_root /path/to/pubfig/images \
  --lfw_lms_file   /path/to/landmarks.txt \
  --target_folder  /path/to/PubFig_cropped
```

```bash
# B) Shell wrapper
bash preprocess/crop_images.sh \
  RAW_IMG_DIR=/path/to/pubfig/images \
  LANDMARKS=/path/to/landmarks.txt \
  OUT_DIR=/path/to/PubFig_cropped
```

---

### 3) Split & layout (ImageFolder)

Organize cropped images as:

```
<DATA_DIR>/
  train/<class_name>/*.jpg
  val/<class_name>/*.jpg
  test/<class_name>/*.jpg
```

> Set this path in `utils/data_process.py` (`data_dir`).

---

### 4) Normalization (match our code)

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

## Contact

Questions, issues, or contributions are welcome—please open a GitHub issue or pull request.



---

## Citation

If you find this repository useful in your research, please cite:

```bibtex
@article{najjar2025syfar,
    author  = {Najjar, Haneen and Ronen, Eyal and Sharif, Mahmood},
    title   = {{Sy-FAR}: {S}ymmetry-based Fair Adversarial Robustness},
    year    = {2025},
    journal = {arXiv preprint}
}

