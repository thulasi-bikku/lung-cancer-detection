# Lightweight Hybrid CNN–Transformer for Three-Class Lung CT Classification

Reference implementation accompanying the manuscript *"Lightweight Hybrid CNN–Transformer for Three-Class
Lung CT Classification: A Controlled Ablation and Validity Analysis on the IQ-OTH/NCCD Dataset"*.

An ImageNet-pre-trained **ResNet18** is followed by a **1×1 projection** (512→256), a learnable
**positional embedding** on the 7×7 = 49 spatial tokens, a **two-layer Transformer encoder**
(heads, feed-forward width and dropout are set in `configs/default.yaml`), token averaging, and a linear classifier with three outputs
(Benign, Malignant, Normal).

```
CT slice → ResNet18 → 1×1 conv → flatten + E_pos → Transformer ×2 → mean over tokens → linear → class
```

## Repository layout

```
src/                 model, data pipeline, training, evaluation, Grad-CAM, complexity, statistics
configs/             default.yaml (manuscript settings) and the ablation variants
scripts/             training-curve and confusion-matrix plots from the logged runs
paper_figures/       script that redraws the manuscript figures from results/*.csv
results/             numbers reported in the manuscript (fold metrics, baselines, ablation, efficiency)
tests/               unit tests
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python ≥ 3.9 and PyTorch ≥ 2.1 are required. A CUDA GPU or Apple-silicon GPU (MPS) is used when available;
the model also runs on CPU.

## Data

Download the public IQ-OTH/NCCD image set from
[Kaggle](https://www.kaggle.com/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset) or
[Mendeley Data](https://data.mendeley.com/datasets/bhmdr45bh2/4) (doi:10.17632/bhmdr45bh2.4) and unzip it.
The loader searches recursively for the three class folders (`Bengin cases`, `Malignant cases`,
`Normal cases`; the spelling `Benign cases` is accepted as well). The distributed image set contains 1,097 slices
(416 Normal, 120 Benign, 561 Malignant). Images are not redistributed in this repository.

## Training and evaluation

Stratified five-fold cross-validation with the manuscript settings:

```bash
python -m src.train --data "/path/to/The IQ-OTHNCCD lung cancer dataset" \
    --config configs/default.yaml --out runs/hybrid
python -m src.evaluate --run runs/hybrid
```

Outputs per fold: `best.pt`, `history.json` (loss/accuracy per epoch), `predictions.csv` (per-slice
class probabilities); and `fold_metrics.csv` for the whole run. `fold_metrics.csv` reports macro F1 both as the
mean of class-wise F1 (`macro_f1`, scikit-learn convention) and as the harmonic mean of macro precision and
macro recall (`macro_f1_harmonic`).

Ablation variants (Table 11 of the manuscript):

| Variant | Config |
|---|---|
| ResNet18 + projection (no encoder) | `configs/ablation_no_encoder.yaml` |
| Encoder without projection (d = 512) | `configs/ablation_no_projection.yaml` |
| 1 Transformer layer | `configs/ablation_1_layer.yaml` |
| 4 Transformer layers | `configs/ablation_4_layers.yaml` |
| Without positional embedding | `configs/ablation_no_pos_embed.yaml` |

The plain ResNet18 baseline and the other baselines (DenseNet-121, EfficientNet-B3, ViT-B/16, Swin-T) are
standard `torchvision` models trained with the same partitions, augmentation, optimiser, scheduler and stopping rule.

### Stricter evaluation options

These options are provided so that the validity threats discussed in the manuscript can be removed as soon as the
required information is available; they were **not** part of the reported experiments.

```bash
# case-level (grouped) folds: needs a CSV with columns  filename,case_id
python -m src.train --data ... --groups case_ids.csv --out runs/hybrid_grouped
# repeated seeds
python -m src.train --data ... --seeds 42 43 44 45 46 --out runs/hybrid_5seeds
```

`src/stats_tests.py` provides an exact McNemar test for two classifiers on the same slices and a
percentile bootstrap interval (resampling slices, or cases when groups are given).

## Inference and Grad-CAM

```bash
python -m src.predict --weights runs/hybrid/seed42/fold1/best.pt --image slice.jpg --cam cam.png
```

Grad-CAM is computed on the final 7×7 backbone feature map and upsampled to the input size.

## Model complexity

```bash
python -m src.complexity --config configs/default.yaml
```

prints the parameter count, fp32 model size, CPU/GPU latency (batch size 1, 100 passes after warm-up) and
FLOPs when `fvcore` is installed. Timings depend on the hardware; the manuscript values were measured on an Apple M4.

## Figures

```bash
python paper_figures/make_figures.py --results results --out paper_figures/output \
    --assets paper_figures/ct_panels
python scripts/plot_training_curves.py --run runs/hybrid --seed 42 --fold 1 --out accuracy_vs_epoch
python scripts/plot_confusion_matrix.py --run runs/hybrid --seed 42 --fold 1 --out confusion_fold1
```

Vector PDFs and 600 dpi PNGs are written. The chart figures use only the numbers in `results/`.

## Tests

```bash
pytest -q
```

`tests/test_stats.py` checks the metric and statistics code against the reported fold table.
`tests/test_model.py` checks tensor shapes and runs only when PyTorch is installed.

## Limitations (summary)

The public image files carry no case identifiers, so folds are formed at slice level and slices of one patient may
appear on both sides of a split; checkpoint selection used the reported validation folds; one random seed was used;
no external cohort was evaluated. Reported figures are internal estimates and are not evidence of clinical
performance. See Section 5 of the manuscript.

## Citation

See `CITATION.cff`. Released under the MIT License.
