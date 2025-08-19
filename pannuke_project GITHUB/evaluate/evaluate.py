import os, sys, argparse
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, jaccard_score

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate segmentation/class masks")
    p.add_argument('--pred_dir', required=True, help='Predictions directory')
    p.add_argument('--gt_dir', required=True, help='Ground truth directory')
    p.add_argument('--num_classes', type=int, default=6)
    p.add_argument('--ignore_index', type=int, default=0)
    p.add_argument('--bin_threshold', type=float, default=0.5)
    return p.parse_args()

def is_npy(x): return x.lower().endswith('.npy')
def load_any(path):
    if is_npy(path):
        arr = np.load(path, allow_pickle=True, mmap_mode=None)
    else:
        arr = np.array(Image.open(path).convert('L'))
    return arr

def to_label_map(x, K, thr=0.5):
    if x.ndim == 4 and x.shape[-1] == K: return np.argmax(x, axis=-1)
    if x.ndim == 4 and x.shape[1] == K:  return np.argmax(x, axis=1)
    if x.ndim == 3 and x.shape[-1] == 1: return (x[...,0] > thr).astype(np.int64)
    if x.ndim == 3 and x.shape[0] == 1:  return (x[0] > thr).astype(np.int64)
    if x.ndim == 3 and x.shape[-1] == 2: return np.argmax(x, axis=-1)
    if x.ndim == 3 and x.shape[0] == 2:  return np.argmax(x, axis=0)
    if x.ndim == 3 and x.shape[-1] > 2:  return np.argmax(x, axis=-1)
    if x.ndim == 3 and x.shape[0] > 2:   return np.argmax(x, axis=0)
    if x.ndim == 2: return x.astype(np.int64, copy=False)
    if x.ndim == 1: return x.astype(np.int64, copy=False)
    return x.squeeze().astype(np.int64, copy=False)

def normalize_binary_labels(x):
    u = np.unique(x)
    if set(u.tolist()) <= {0,1}: return x
    if set(u.tolist()) <= {0,255}: return (x==255).astype(np.int64)
    return x

def list_files_by_stem(d):
    fs = [f for f in os.listdir(d) if f.lower().endswith(('.npy','.png','.jpg','.jpeg','.tif','.tiff'))]
    m = {}
    for f in fs:
        s = os.path.splitext(f)[0]
        s = s.replace('_pred','').replace('_gt','')
        s = s.replace('_images','').replace('_masks','').replace('_types','')
        m[s] = f
    return m

def main():
    args = parse_args()
    if not os.path.isdir(args.pred_dir):
        print(f"Error: missing pred_dir {args.pred_dir}", file=sys.stderr); sys.exit(1)
    if not os.path.isdir(args.gt_dir):
        print(f"Error: missing gt_dir {args.gt_dir}", file=sys.stderr); sys.exit(1)

    pred_map = list_files_by_stem(args.pred_dir)
    gt_map   = list_files_by_stem(args.gt_dir)
    common = sorted(set(pred_map.keys()) & set(gt_map.keys()))
    if not common:
        print("Error: no matching stems between pred_dir and gt_dir", file=sys.stderr); sys.exit(1)

    flats_pred = []
    flats_gt   = []

    for stem in common:
        p_path = os.path.join(args.pred_dir, pred_map[stem])
        g_path = os.path.join(args.gt_dir,   gt_map[stem])
        p = load_any(p_path)
        g = load_any(g_path)

        p = to_label_map(p, args.num_classes, args.bin_threshold)
        g = to_label_map(g, args.num_classes, args.bin_threshold)

        if args.num_classes == 2:
            p = normalize_binary_labels(p)
            g = normalize_binary_labels(g)

        if p.ndim != 2 or g.ndim != 2:
            p = p.squeeze()
            g = g.squeeze()
        if p.ndim != 2 or g.ndim != 2:
            print(f"Warning: {stem}: unsupported shapes {p.shape} vs {g.shape}", file=sys.stderr); continue

        h = min(p.shape[0], g.shape[0])
        w = min(p.shape[1], g.shape[1])
        p = p[:h,:w]
        g = g[:h,:w]

        mask = np.ones((h,w), dtype=bool)
        if args.ignore_index >= 0:
            mask &= (g != args.ignore_index)

        if mask.sum() == 0:
            continue

        flats_pred.append(p[mask].ravel())
        flats_gt.append(g[mask].ravel())

    if not flats_pred or not flats_gt:
        print("Error: no valid pairs after preprocessing", file=sys.stderr); sys.exit(1)

    flat_pred = np.concatenate(flats_pred)
    flat_gt   = np.concatenate(flats_gt)

    labels = [l for l in range(args.num_classes) if l != args.ignore_index]
    acc = accuracy_score(flat_gt, flat_pred)
    f1_macro = f1_score(flat_gt, flat_pred, labels=labels, average='macro', zero_division=0)
    iou_macro = jaccard_score(flat_gt, flat_pred, labels=labels, average='macro', zero_division=0)

    per_iou = []
    for c in range(args.num_classes):
        if c == args.ignore_index:
            per_iou.append(float('nan')); continue
        pc = (flat_pred == c)
        tc = (flat_gt == c)
        inter = (pc & tc).sum()
        union = int(pc.sum()) + int(tc.sum()) - int(inter)
        per_iou.append(float('nan') if union == 0 else (inter / union))

    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1 (ignore={args.ignore_index}): {f1_macro:.4f}")
    print(f"Mean IoU (ignore={args.ignore_index}): {iou_macro:.4f}")
    print("Per-class IoU:")
    for c, v in enumerate(per_iou):
        s = "nan" if (v!=v) else f"{float(v):.4f}"
        print(f"  class {c}: {s}")

if __name__ == "__main__":
    main()
