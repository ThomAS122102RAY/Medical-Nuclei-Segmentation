# Pan-Cancer Nuclei Segmentation (PanNuke)

End‑to‑end pipeline for nuclei segmentation and type prediction on the PanNuke dataset.  
Includes dataset split, training (TransUNet-based), prediction, and evaluation.

## 1) Environment

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```
NOTICE: PUT THE FOLDER IN DESKTOP FOLDER!

Tested with Python 3.10+ and PyTorch 2.1+. For CUDA, install the matching `torch` build from the official instructions.

## 2) Project Structure

```
pannuke_project/
├─ main_pipeline.py
├─ segmentor/
│  ├─ transunet.py
│  └─ train_segmentor.py
├─ preprocess/
│  ├─ data_loader.py
│  └─ shuffle_and_split_pannuke.py
├─ evaluate/
│  ├─ evaluate.py
│  └─ metrics.py
├─ data/
│  ├─ images/            # *_images.npy (raw bundles)
│  ├─ masks/             # *_masks.npy (raw bundles, optional for SEG eval)
│  ├─ types/             # *_types.npy (raw bundles, optional for TYPE eval)
│  ├─ train/             # auto/generated split
│  └─ test/              # auto/generated split
├─ checkpoints/
│  ├─ dino/
│  └─ segmentor/
└─ predictions/
```

> **Important**  
> The pipeline uses relative paths. Keep `TEST_ROOT = PANNUKE_ROOT / 'test'` (no hard‑coded absolute paths).

## 3) Data Setup

Place your PanNuke bundles under `data/`:

- `data/images/*_images.npy`
- `data/masks/*_masks.npy` (optional but required for SEG evaluation)
- `data/types/*_types.npy` (optional but required for TYPE evaluation)

### Split options

**A. Auto‑split (built into `main_pipeline.py`)**  
If `data/train/images.npy` is missing, the pipeline will:
- find `*_images.npy` (and matching masks/types if present),
- concatenate, and
- create `data/train/` and `data/test/` according to `SPLIT_RATIO` (default 0.8).

**B. Manual split script**
```bash
python preprocess/shuffle_and_split_pannuke.py --data_root data --train_ratio 0.8 --seed 42
```

## 4) Training & Prediction

Run the full pipeline (ensure DINO weights are auto‑downloaded if missing):

```bash
python main_pipeline.py
```

This will:
1. Ensure pretrained DINO weights (download if needed)
2. Ensure dataset split (auto or existing)
3. Train `segmentor/train_segmentor.py`
4. Save checkpoints to `checkpoints/segmentor/`
5. Predict to `predictions/`:
   - `{split}_pred_types.npy`
   - `{split}_pred_seg.npy`
   - `single_masks/` (per‑image exports)

### Environment variables

- `ON_EXIST` = `ask` | `retrain` | `predict`  
  Behavior when `model_final.pth` exists.
- `SEG_BSZ` (default `12`)  
  Batch size for training/prediction subprocesses.
- `SEG_WORKERS` (default `18`)  
  DataLoader workers.
- `W_ORG` (default `0.2`)  
  Loss weight for organ branch (if used).
- `SPLIT_RATIO` (default `0.8`)  
  Auto‑split ratio when building `train/test`.
- `SPLIT_SEED` (default `42`)  
  RNG seed for auto‑split.
- `CUDA_VISIBLE_DEVICES`  
  Specify GPUs or set empty to force CPU.

Examples:
```bash
ON_EXIST=retrain SEG_BSZ=8 SEG_WORKERS=8 python main_pipeline.py
```

## 5) Evaluation

### Built‑in (from `main_pipeline.py`)
After prediction, the pipeline prints:
- TYPE: Accuracy, Macro‑F1, Mean IoU, per‑class IoU (when pixel‑level types exist)
- SEG: Dice and IoU (requires `data/{split}/masks/*.npy`)

If you see:
```
[WARN] test SEG: masks/  pixel-level types SEG
```
your `data/test/masks/` is missing or empty.

### Standalone script
Pairwise evaluation using folder‑level predictions vs ground truth:

```bash
python evaluate/evaluate.py   --pred_dir predictions/single_masks   --gt_dir   data/test/masks   --num_classes 2   --ignore_index 0
```

> Adjust `--num_classes` depending on your labels (e.g., 2 for binary SEG, 6 for TYPE).

## 6) Data‑Leak Check (optional)

A quick checker is provided to detect simple split leakage (same stems or duplicate file content). Place it in project root and run:

```bash
python leak_check.py
```

Output includes:
- `overlapping_stems` (train vs test)
- quick duplicate hashes
- shapes of `train/test/images.npy` if present

## 7) Common Issues

- **Hard‑coded paths**: Ensure `TEST_ROOT = PANNUKE_ROOT / 'test'`.
- **Missing masks**: If SEG metrics warn about missing masks, verify `data/test/masks/*.npy` exists and matches `images` stems.
- **Mixed shapes/logits**: Some `.npy` predictions store logits. The evaluation scripts handle common shapes via `argmax`; confirm your model’s output format if results look off.
- **CUDA errors**: Set `CUDA_VISIBLE_DEVICES=` to force CPU, or install a CUDA‑enabled PyTorch build.

## 8) License & Dataset

This project trains on the PanNuke dataset. Please follow the dataset’s original license and citation requirements.
