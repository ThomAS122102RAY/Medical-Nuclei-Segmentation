
import os, sys


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

PREPROCESS_DIR = os.path.join(PROJECT_ROOT, 'preprocess')
if PREPROCESS_DIR not in sys.path:
    sys.path.insert(0, PREPROCESS_DIR)
import subprocess
import logging
import gc
from pathlib import Path

def should_retrain_if_model_exists(final_ckpt: Path) -> bool:
    mode = os.getenv('ON_EXIST', 'ask').lower()
    if not final_ckpt.exists():
        return True
    if mode == 'retrain':
        return True
    if mode == 'predict':
        return False
    try:
        ans = input(f"{final_ckpt}\n [r] to retrain / [p] to predict trained model: ").strip().lower()
    except Exception:
        ans = 'p'
    return ans.startswith('r') or ans == 'r'
import importlib
import importlib.util
import shutil

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.backends.cudnn as cudnn

from segmentor.train_segmentor import pad_collate


import segmentor.transunet
importlib.reload(segmentor.transunet)
from segmentor.transunet import TransUNet


DATA_LOADER_PATH = Path(__file__).resolve().parent / 'preprocess' / 'data_loader.py'
spec_loader = importlib.util.spec_from_file_location('data_loader', str(DATA_LOADER_PATH))
data_loader_mod = importlib.util.module_from_spec(spec_loader)
spec_loader.loader.exec_module(data_loader_mod)
PanNukeDataset = data_loader_mod.PanNuke

PROJECT_ROOT = Path(__file__).resolve().parent

sys.path.append(str(PROJECT_ROOT / "evaluate"))

from evaluate.metrics import per_class_iou_mc, mean_iou_mc, macro_f1_mc

PROJECT_ROOT       = Path(__file__).resolve().parent
PANNUKE_ROOT       = PROJECT_ROOT / 'data'
TRAIN_ROOT         = PANNUKE_ROOT / 'train'     
TEST_ROOT          = PANNUKE_ROOT / 'test'
PRED_ROOT          = PROJECT_ROOT / 'predictions' 
CHECKPOINTS_DIR    = PROJECT_ROOT / 'checkpoints'
SEG_CKPT_DIR       = CHECKPOINTS_DIR / 'segmentor'
DINO_PRETRAIN_PATH = CHECKPOINTS_DIR / 'dino' / 'dino_deitsmall16_pretrain_full_checkpoint.pth'
DINO_PRETRAIN_URL  = 'https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain_full_checkpoint.pth'

FOLDS = {
    'train': TRAIN_ROOT,
    'test':  TEST_ROOT,
}

logging.basicConfig(
    filename=PROJECT_ROOT / 'pipeline.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    encoding='utf-8'
)

DATA_LOADER_PATH = PROJECT_ROOT / 'preprocess' / 'data_loader.py'
if not DATA_LOADER_PATH.is_file():
    raise FileNotFoundError(f' data_loader.py{DATA_LOADER_PATH}')
spec = importlib.util.spec_from_file_location('data_loader', str(DATA_LOADER_PATH))
data_loader_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_loader_mod)
PanNukeDataset = data_loader_mod.PanNuke

TRANSUNET_PATH = PROJECT_ROOT / 'segmentor' / 'transunet.py'
if not TRANSUNET_PATH.is_file():
    raise FileNotFoundError(f' transunet.py{TRANSUNET_PATH}')
spec2 = importlib.util.spec_from_file_location('transunet', str(TRANSUNET_PATH))
transunet_mod = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(transunet_mod)
TransUNet = transunet_mod.TransUNet

def safe_run(cmd, cwd=None, env=None, desc=''):
    """/ raise stderrUTF-8 """
    child_env = os.environ.copy()
    if env:
        child_env.update(env)
    child_env.setdefault('PYTHONUNBUFFERED','1')
    child_env.setdefault('PYTHONIOENCODING','utf-8')
    logging.info(f'>>> {cmd} {desc}')
    proc = subprocess.Popen(
        cmd,
        cwd=cwd or PROJECT_ROOT,
        env=child_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        encoding='utf-8',
        errors='replace',
    )
    out_lines = []
    for line in proc.stdout:
        print(line, end=''); out_lines.append(line)
    proc.wait()
    stdout = ''.join(out_lines)
    if proc.returncode != 0:
        logging.error(stdout)
        raise RuntimeError(
            f' ({proc.returncode}): {cmd}\n'
            f'--- STDOUT ---\n{stdout}\n'
        )
    return stdout

def ensure_pretrained():
    """ DINO """
    if DINO_PRETRAIN_PATH.is_file():
        return
    DINO_PRETRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    safe_run(
        ['curl', '-L', DINO_PRETRAIN_URL, '-o', str(DINO_PRETRAIN_PATH)],
        desc='download DINO'
    )

def check_gpu(min_gb=8):
    """ GPU  CUDA"""
    if not torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        return
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if total_gb < min_gb:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

def ensure_split():
    """ data/train  data/test"""
    if (TRAIN_ROOT / 'images.npy').is_file():
        return
    shuffle = PROJECT_ROOT / 'preprocess' / 'shuffle_and_split_pannuke.py'
    if not shuffle.is_file():
        raise FileNotFoundError(shuffle)
    safe_run([sys.executable, '-u', str(shuffle)], desc='split')
    merge = PROJECT_ROOT / 'preprocess' / 'merge_split_npy.py'
    if merge.is_file():
        safe_run([sys.executable, '-u', str(merge)], desc='merge')


def newest_pth(d: Path):
    files = sorted(Path(d).glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None

def ensure_final_checkpoint():
    SEG_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    final_ckpt = SEG_CKPT_DIR / 'model_final.pth'
    if final_ckpt.is_file():

        legacy_ckpt = CHECKPOINTS_DIR / 'model_final.pth'
        try:
            if not legacy_ckpt.exists() or legacy_ckpt.stat().st_mtime < final_ckpt.stat().st_mtime:
                shutil.copy2(final_ckpt, legacy_ckpt)
        except Exception as e:
            print(f"[WARN]  {legacy_ckpt}: {e}")
        return final_ckpt
    cand = newest_pth(SEG_CKPT_DIR)
    if cand is None:
        raise FileNotFoundError(f' .pth{SEG_CKPT_DIR}')
    shutil.copy2(cand, final_ckpt)
    print(f"[INFO]  {final_ckpt.name}  {cand.name}")
    return final_ckpt

def train_segmentor(dummy_model, dummy_loader, cpu_device):
    SEG_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    child_env = os.environ.copy()
    child_env['SEG_BSZ'] = os.getenv('SEG_BSZ', '12')
    final_ckpt = SEG_CKPT_DIR / 'model_final.pth'
    if not should_retrain_if_model_exists(final_ckpt):
        print(f"[INFO]  PREDICT {final_ckpt}")
    else:
        safe_run([
            sys.executable, '-u', '-Xfrozen_modules=off',
            'segmentor/train_segmentor.py',
            '--data_root',      str(TRAIN_ROOT),
            '--checkpoint_dir', str(SEG_CKPT_DIR),
            '--batch_size',     child_env.get('SEG_BSZ', '16'),
            '--num_workers',    child_env.get('SEG_WORKERS', '18'),
            '--pin_memory',
            '--num_organs',     '19',
            '--w_org',          os.getenv('W_ORG', '0.2'),
        ], env=child_env)
        ensure_final_checkpoint()


def predict_segmentor():
    final_ckpt = ensure_final_checkpoint()
    single_dir = PRED_ROOT / 'single_masks'
    single_dir.mkdir(parents=True, exist_ok=True)
    child_env = os.environ.copy()
    child_env['SEG_BSZ'] = os.getenv('SEG_BSZ', '12')
    for split, folder in FOLDS.items():
        out_npy = PRED_ROOT / f'{split}_pred_types.npy'
        seg_npy = PRED_ROOT / f'{split}_pred_seg.npy'   
        if out_npy.is_file() and seg_npy.is_file():    
            continue
        safe_run([
            sys.executable, '-u',
            'segmentor/predict_segmentor.py',
            '--checkpoint',      str(final_ckpt),
            '--data_root',       str(folder),
            '--input_dir',       str(folder),
            '--output_dir',      str(single_dir),
            '--output_pred_npy', str(out_npy),
            '--output_pred_seg_npy', str(seg_npy),
            '--batch_size',      child_env['SEG_BSZ'],
        ], env=child_env, cwd=str(PROJECT_ROOT), desc=split)





def evaluate_segmentation(num_classes=6, ignore_index=0):
    import numpy as np, torch, importlib.util
    from pathlib import Path

    DATA_LOADER_PATH = Path(__file__).resolve().parent / 'preprocess' / 'data_loader.py'
    spec = importlib.util.spec_from_file_location('data_loader', str(DATA_LOADER_PATH))
    data_loader_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(data_loader_mod)
    PanNukeDataset = data_loader_mod.PanNuke

    def _to_label_map(x: np.ndarray, K: int) -> np.ndarray:
        if x.ndim == 4 and x.shape[-1] == K: return np.argmax(x, axis=-1).astype(np.int64, copy=False)
        if x.ndim == 4 and x.shape[1] == K:  return np.argmax(x, axis=1).astype(np.int64, copy=False)
        if x.ndim == 3: return x.astype(np.int64, copy=False)
        if x.ndim == 2: return x[None, ...].astype(np.int64, copy=False)
        if x.ndim == 1: return x[:, None, None].astype(np.int64, copy=False)
        raise ValueError(f"{x.shape}")

    def _load_concat(folder: Path, sub: str):
        d = folder / sub
        files = sorted(d.glob("*.npy")) if d.is_dir() else []
        if not files: return None
        arrs = []
        for f in files:
            a = np.load(f, allow_pickle=True, mmap_mode='r')
            if a.ndim >= 3: pass
            elif a.ndim == 2: a = a[None, ...]
            elif a.ndim == 1: a = a[:, None, None]
            else: a = a[None, ...]
            arrs.append(a)
        return np.concatenate(arrs, axis=0) if len(arrs) > 1 else arrs[0]

    for split, folder in FOLDS.items():
        pred_path = PRED_ROOT / f'{split}_pred_types.npy'
        if not pred_path.is_file():
            print(f'[WARN] {split}:  {pred_path} TYPE/SEG/ORGAN '); continue
        pred_np = np.load(pred_path, allow_pickle=True, mmap_mode='r')

        types_np = _load_concat(folder, 'types')
        type_is_pixel = False; gt_lbl = None
        if types_np is not None and types_np.dtype.kind not in ('U','S','O'):
            try:
                gt_lbl = _to_label_map(types_np, num_classes); type_is_pixel = True
            except Exception:
                type_is_pixel = False

        if type_is_pixel:
            pred_lbl = _to_label_map(pred_np, num_classes)
            def _nhw(x):
                if x.ndim == 1: return x[:,None,None]
                if x.ndim == 2: return x[:,:,None]
                return x
            gt_lbl = _nhw(gt_lbl); pred_lbl = _nhw(pred_lbl)
            h = min(gt_lbl.shape[-2], pred_lbl.shape[-2]); w = min(gt_lbl.shape[-1], pred_lbl.shape[-1])
            gt_lbl = gt_lbl[...,:h,:w]; pred_lbl = pred_lbl[...,:h,:w]
            gt_t = torch.from_numpy(gt_lbl.copy()).long(); pred_t = torch.from_numpy(pred_lbl.copy()).long()
            acc = (pred_t == gt_t).float().mean().item()
            mf1  = macro_f1_mc(pred_t, gt_t, num_classes=num_classes, ignore_index=ignore_index)
            miou = mean_iou_mc(pred_t, gt_t, num_classes=num_classes, ignore_index=ignore_index)
            ious = per_class_iou_mc(pred_t, gt_t, num_classes=num_classes, ignore_index=ignore_index)
            print(f'{split} TYPE Accuracy: {acc:.4f}')
            print(f'{split} TYPE Macro-F1 (ignore={ignore_index}): {mf1:.4f}')
            print(f'{split} TYPE Mean IoU (ignore={ignore_index}): {miou:.4f}')
            for c,v in enumerate(ious):
                s = "nan" if (float(v)!=float(v)) else f"{float(v):.4f}"
                print(f'{split} TYPE class {c} IoU: {s}')
        else:
            print(f'{split} TYPE: N/Atypes/ ')

        if type_is_pixel:
            seg_gt = (gt_lbl > 0).astype(np.uint8)
        else:
            masks_np = _load_concat(folder, 'masks')
            if masks_np is None:
                print(f'[WARN] {split} SEG:  masks/  pixel-level types SEG'); continue
            if masks_np.ndim == 4 and masks_np.shape[-1] >= 2:
                seg_gt = (np.argmax(masks_np, axis=-1) > 0).astype(np.uint8)
            elif masks_np.ndim == 4 and masks_np.shape[1] >= 2:
                seg_gt = (np.argmax(masks_np, axis=1) > 0).astype(np.uint8)
            elif masks_np.ndim == 3:
                seg_gt = (masks_np > 0).astype(np.uint8)
            elif masks_np.ndim == 2:
                seg_gt = (masks_np[None, ...] > 0).astype(np.uint8)
            else:
                print(f'[WARN] {split} SEG:  masks/ '); continue

        seg_pred_path = PRED_ROOT / f'{split}_pred_seg.npy'
        if seg_pred_path.is_file():
            seg_np = np.load(seg_pred_path, mmap_mode='r')
            if seg_np.ndim == 4:
                if seg_np.shape[1] == 1: seg_np = seg_np[:,0]
                elif seg_np.shape[-1] == 1: seg_np = seg_np[...,0]
                elif seg_np.shape[1] == 2: seg_np = seg_np.argmax(1)
                elif seg_np.shape[-1] == 2: seg_np = seg_np.argmax(-1)
            elif seg_np.ndim == 2: seg_np = seg_np[None, ...]
            elif seg_np.ndim == 1: seg_np = seg_np[:, None, None]
            seg_pred = (seg_np > 0).astype(np.uint8)
        else:
            pred_lbl = _to_label_map(pred_np, num_classes)
            seg_pred = (pred_lbl > 0).astype(np.uint8)

        hh = min(seg_gt.shape[-2], seg_pred.shape[-2]); ww = min(seg_gt.shape[-1], seg_pred.shape[-1])
        seg_gt = seg_gt[...,:hh,:ww]; seg_pred = seg_pred[...,:hh,:ww]
        tp = ((seg_gt==1) & (seg_pred==1)).sum(); fp = ((seg_gt==0) & (seg_pred==1)).sum(); fn = ((seg_gt==1) & (seg_pred==0)).sum()
        dice = (2*tp / (2*tp + fp + fn)) if (2*tp + fp + fn) else float('nan')
        union = ((seg_gt==1) | (seg_pred==1)).sum(); iou = (tp/union) if union else float('nan')
        print(f'{split} SEG Dice: {float(dice):.4f}')
        print(f'{split} SEG IoU : {float(iou):.4f}')

        org_pred_path = PRED_ROOT / f'{split}_pred_organs.npy'
        if org_pred_path.is_file():
            org_pred = np.load(org_pred_path)  

            ds = PanNukeDataset(str(folder))
            gt = []
            for i in range(len(ds)):
                item = ds[i]
                if isinstance(item, (list, tuple)) and len(item) >= 4:
                    try:
                        gt.append(int(item[3]))
                    except Exception:
                        pass
            if gt and len(gt) >= len(org_pred):
                gt = np.array(gt[:len(org_pred)], dtype=np.int64)
                top1 = float((gt == org_pred).mean())
                print(f'{split} ORGAN Top-1: {top1:.4f}')
            else:
                print(f'{split} ORGAN Top-1: N/A')
        else:
            print(f'{split} ORGAN Top-1: N/A')

def clear_pycache(root_dir):
    """ __pycache__  .pyc """
    for dirpath, dirnames, filenames in os.walk(root_dir):

        if "__pycache__" in dirnames:
            cache_path = os.path.join(dirpath, "__pycache__")
            shutil.rmtree(cache_path, ignore_errors=True)

        for f in filenames:
            if f.endswith(".pyc"):
                try:
                    os.remove(os.path.join(dirpath, f))
                except:
                    pass



    class CUDAPrefetcher:
        def __init__(self, loader, device):
            self.loader = iter(loader)
            self.device = device
            self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None
            self.next_batch = None
            self.preload()
        def preload(self):
            try:
                batch = next(self.loader)
            except StopIteration:
                self.next_batch = None
                return
            if self.stream is None:
                if isinstance(batch, (list, tuple)) and len(batch) == 4:
                    imgs, masks, types, organs = batch
                else:
                    imgs, masks, types = batch[:3]
                    organs = None
                items = [
                    imgs.to(self.device, non_blocking=True),
                    masks.to(self.device, non_blocking=True),
                    types.to(self.device, non_blocking=True),
                ]
                if organs is not None:
                    items.append(organs.to(self.device, non_blocking=True))
                self.next_batch = tuple(items)
                return
            with torch.cuda.stream(self.stream):
                if isinstance(batch, (list, tuple)) and len(batch) == 4:
                    imgs, masks, types, organs = batch
                else:
                    imgs, masks, types = batch[:3]
                    organs = None
                items = [
                    imgs.to(self.device, non_blocking=True),
                    masks.to(self.device, non_blocking=True),
                    types.to(self.device, non_blocking=True),
                ]
                if organs is not None:
                    items.append(organs.to(self.device, non_blocking=True))
                self.next_batch = tuple(items)
        def next(self):
            if self.stream is not None:
                torch.cuda.current_stream().wait_stream(self.stream)
            batch = self.next_batch
            if batch is not None:
                self.preload()
            return batch


def main():
    PROJECT_ROOT = os.path.dirname(__file__)
    clear_pycache(PROJECT_ROOT)

    cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass

    check_gpu()
    ensure_pretrained()
    ensure_split()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f" GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(" CPU")

        
    train_dataset = PanNukeDataset(TRAIN_ROOT)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(os.getenv("SEG_BSZ", "12")),
        shuffle=True,
        num_workers=int(os.getenv("SEG_WORKERS", "4")),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        collate_fn=pad_collate,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransUNet(in_chans=3, num_classes=6, num_organs=19).to(device)
    try:
        model = model.to(memory_format=torch.channels_last)
    except Exception:
        pass

    dummy = torch.randn(1, 3, 256, 256).to(next(model.parameters()).device)
    try:
        _ = model.forward_features(dummy)
    except Exception:
        try:
            _ = model(dummy)
        except Exception:
            pass

    train_segmentor(model, train_loader, torch.device)

    ensure_final_checkpoint()
    predict_segmentor()
    evaluate_segmentation()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

if __name__=='__main__':
    main()
