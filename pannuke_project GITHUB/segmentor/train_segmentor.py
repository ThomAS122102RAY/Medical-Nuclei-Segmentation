
import os, sys, platform, argparse, logging
from pathlib import Path
import importlib.util
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

def pad_collate(batch):

    has_organs = len(batch[0]) == 4
    imgs  = [b[0] for b in batch]
    masks = [b[1] for b in batch]
    types = [b[2] for b in batch]
    organs = [b[3] for b in batch] if has_organs else None

    max_h = max(int(x.shape[1]) for x in imgs)
    max_w = max(int(x.shape[2]) for x in imgs)

    def pad_img(img: torch.Tensor) -> torch.Tensor:
        C, H, W = img.shape
        out = torch.zeros((C, max_h, max_w), dtype=img.dtype)
        out[:, :H, :W] = img
        return out

    def pad_map(t: torch.Tensor, img_h: int, img_w: int) -> torch.Tensor:

        if t.ndim == 2:
            H, W = t.shape
            out = torch.zeros((max_h, max_w), dtype=t.dtype)
            out[:H, :W] = t
            return out

        if t.ndim == 3:

            if t.shape[1] == img_h and t.shape[2] == img_w and t.shape[0] <= 64:
                K, H, W = int(t.shape[0]), int(t.shape[1]), int(t.shape[2])
                out = torch.zeros((K, max_h, max_w), dtype=t.dtype)
                out[:, :H, :W] = t
                return out
            else:
                H, W, K = int(t.shape[0]), int(t.shape[1]), int(t.shape[2])
                out = torch.zeros((max_h, max_w, K), dtype=t.dtype)
                out[:H, :W, :] = t
                return out

        return t

    imgs_p  = [pad_img(i) for i in imgs]
    masks_p = [pad_map(m, int(i.shape[1]), int(i.shape[2])) for m, i in zip(masks, imgs)]
    types_p = [pad_map(t, int(i.shape[1]), int(i.shape[2])) for t, i in zip(types, imgs)]

    def homogenize(lst):
        nds = [x.ndim for x in lst]
        if max(nds) == 3 and min(nds) == 2:
            out = []
            for x in lst:
                if x.ndim == 2:
                    out.append(x.unsqueeze(-1))
                else:
                    out.append(x)
            return out
        return lst

    masks_p = homogenize(masks_p)
    types_p = homogenize(types_p)

    batch_imgs  = torch.stack(imgs_p, 0)
    batch_masks = torch.stack(masks_p, 0)
    batch_types = torch.stack(types_p, 0)

    if has_organs:
        organs_t = torch.stack([o if isinstance(o, torch.Tensor) and o.ndim==0 else torch.as_tensor(o) for o in organs]).long()
        return batch_imgs, batch_masks, batch_types, organs_t
    else:
        return batch_imgs, batch_masks, batch_types

_PRJ = Path(__file__).resolve().parent.parent
if str(_PRJ) not in sys.path:
    sys.path.insert(0, str(_PRJ))

IS_WINDOWS = platform.system().lower().startswith('win')

from preprocess.data_loader import PanNuke as PanNukeDataset

_SEG_DIR = Path(__file__).resolve().parent
_TU_PATH = _SEG_DIR / 'transunet.py'
spec = importlib.util.spec_from_file_location('transunet', str(_TU_PATH))
tu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tu)
TransUNet = tu.TransUNet

_AUG_FN = None
try:
    _AUG_PATH = _PRJ / 'src' / 'augment.py'
    if _AUG_PATH.exists():
        _spec_aug = importlib.util.spec_from_file_location('augment', str(_AUG_PATH))
        _augmod = importlib.util.module_from_spec(_spec_aug)
        _spec_aug.loader.exec_module(_augmod)
        if hasattr(_augmod, 'augment'):
            _AUG_FN = _augmod.augment
except Exception:
    _AUG_FN = None

class AugWrap:
    def __init__(self, base):
        self.base = base
    def __len__(self):
        return len(self.base)
    def __getitem__(self, i):
        out = self.base[i]
        if _AUG_FN is None:
            return out
        if len(out) == 4:
            img, masks, types, organs = out
            img2, masks2, types2 = _AUG_FN(img, masks, types)
            return img2, masks2, types2, organs
        else:
            img, masks, types = out
            img2, masks2, types2 = _AUG_FN(img, masks, types)
            return img2, masks2, types

def auto_data_root(given: Optional[str]) -> Path:
    if given:
        p = Path(given)
    else:
        env = os.environ.get('DATA_ROOT')
        if env:
            p = Path(env)
        else:
            p = _PRJ / 'data'
    if not p.exists():
        raise SystemExit(f"[ERR] data_root {p}")
    return p

def choose_num_workers(requested: Optional[int], safe_io: bool) -> int:
    if safe_io:
        return 0
    if requested is None:
        requested = 0
    if requested > 0:
        return int(requested)
    cpu = os.cpu_count() or 0
    return max(0, cpu - 2)

def as_label_map(t: torch.Tensor) -> torch.Tensor:

    if t.ndim == 3:
        return t.long()
    if t.ndim == 4:

        if t.shape[1] <= 32:
            return t.argmax(1).long()
        else:
            return t.argmax(-1).long()
    return t.long()

@torch.no_grad()
def evaluate_type_miou(model, loader, device, num_classes:int) -> Tuple[float, float]:
    model.eval()
    inter = torch.zeros(num_classes, dtype=torch.float64)
    union = torch.zeros(num_classes, dtype=torch.float64)
    tp = torch.zeros(num_classes, dtype=torch.float64)
    fp = torch.zeros(num_classes, dtype=torch.float64)
    fn = torch.zeros(num_classes, dtype=torch.float64)
    for batch in loader:
        imgs, _, types = batch[:3]
        imgs = imgs.to(device, non_blocking=True)
        types = as_label_map(types.to(device, non_blocking=True))
        with torch.no_grad():
            outputs = model(imgs)
            if isinstance(outputs, (tuple, list)) and len(outputs)>=2:
                type_logits = outputs[1]
            else:
                continue

            h = min(type_logits.size(2), types.size(1))
            w = min(type_logits.size(3), types.size(2))
            type_logits = type_logits[:, :, :h, :w]
            tgt = types[:, :h, :w]
            pred = type_logits.argmax(1)
            for c in range(num_classes):
                pi = (pred==c)
                ti = (tgt==c)
                inter[c] += (pi & ti).sum().item()
                union[c] += (pi | ti).sum().item()
                tp[c]    += (pi & ti).sum().item()
                fp[c]    += (pi & (~ti)).sum().item()
                fn[c]    += ((~pi) & ti).sum().item()
    miou = torch.mean(torch.where(union>0, inter/union, torch.zeros_like(union))).item()

    denom = 2*tp + fp + fn
    macro_f1 = torch.mean(torch.where(denom>0, 2*tp/denom, torch.zeros_like(denom))).item()
    return miou, macro_f1

@torch.no_grad()
def evaluate_organ_top1(model, loader, device, num_organs:int) -> float:
    model.eval()
    total, correct = 0, 0
    for batch in loader:
        if len(batch) < 4: 
            continue
        imgs, _, _, organs = batch
        imgs = imgs.to(device, non_blocking=True)
        organs = organs.to(device, non_blocking=True)
        with torch.no_grad():
            outputs = model(imgs)
            organ_logits = None
            if isinstance(outputs, (tuple, list)):
                if len(outputs) == 3:
                    organ_logits = outputs[2]
            if organ_logits is None:
                continue
            pred = organ_logits.argmax(1)
            valid = (organs >= 0) & (organs < num_organs)
            if valid.any():
                total += int(valid.sum().item())
                correct += int((pred[valid]==organs[valid]).sum().item())
    return (correct/total) if total>0 else 0.0

def parse_args():
    p = argparse.ArgumentParser(description='PanNuke Multi-Task Training (seg+type[+organ])')
    p.add_argument('--data_root', default=None, help='data root <project>/data  DATA_ROOT')
    p.add_argument('--checkpoint_dir', default=str(_PRJ/'checkpoints'/'segmentor'), help='checkpoint dir')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--val_split', type=float, default=0.1, help='0~1 >1 ')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--pin_memory', action='store_true')
    p.add_argument('--feat_chans', type=int, default=64)
    p.add_argument('--num_type_classes', type=int, default=6)
    p.add_argument('--num_organs', type=int, default=19)
    p.add_argument('--w_seg', type=float, default=1.0)
    p.add_argument('--w_type', type=float, default=1.0)
    p.add_argument('--w_org', type=float, default=0.0)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--resume', default=None)
    p.add_argument('--accum_steps', type=int, default=1)
    p.add_argument('--grad_clip', type=float, default=0.0)
    p.add_argument('--save_best', action='store_true')
    p.add_argument('--stream', action='store_true', help=' PN_STREAM_MODE=1 RAM')
    return p.parse_args()

def main():
    args = parse_args()
    if args.stream:
        os.environ['PN_STREAM_MODE'] = '1'

    data_root = auto_data_root(args.data_root)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    logging.basicConfig(
        filename=str(Path(args.checkpoint_dir)/'train.log'),
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        encoding='utf-8'
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    safe_io = IS_WINDOWS or os.environ.get('PN_SAFE_IO','0') == '1'
    nw = choose_num_workers(args.num_workers, safe_io)
    pin_mem = (not safe_io) and (device.type=='cuda' or args.pin_memory)
    persistent = False if safe_io else (nw > 0)
    prefetch = None if safe_io else (4 if nw>0 else None)

    full = PanNukeDataset(str(data_root))
    n = len(full)
    val_n = int(args.val_split*n) if 0 < args.val_split < 1 else int(args.val_split)
    val_n = min(max(val_n, 0), n//5)  
    idx = torch.randperm(n)
    val_idx = idx[:val_n]
    tr_idx  = idx[val_n:]

    train_ds = Subset(full, tr_idx.tolist())
    train_ds = AugWrap(train_ds) if _AUG_FN is not None else train_ds
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=nw,
        pin_memory=pin_mem,
        persistent_workers=persistent,
        prefetch_factor=prefetch
    )
    val_loader = None
    if val_n > 0:
        val_loader = DataLoader(
            Subset(full, val_idx.tolist()),
            batch_size=max(1, args.batch_size//2),
            shuffle=False,
            num_workers=0, 
            pin_memory=False
        )

    model = TransUNet(in_chans=3, num_classes=args.num_type_classes, num_organs=args.num_organs).to(device)

    seg_ce = nn.CrossEntropyLoss()
    seg_bce = nn.BCEWithLogitsLoss()
    type_ce = nn.CrossEntropyLoss()
    organ_ce = nn.CrossEntropyLoss(ignore_index=-1)

    optim = Adam(model.parameters(), lr=args.lr)

    
    scheduler = ReduceLROnPlateau(optim, mode='max', factor=0.5, patience=2, threshold=1e-3, min_lr=1e-6)
    start_ep = 0
    if args.resume:
        cp = Path(args.resume)
        if not cp.is_absolute():
            cp = Path(args.checkpoint_dir) / cp
        if cp.is_file():
            ck = torch.load(str(cp), map_location=device)
            model.load_state_dict(ck.get('model_state', ck), strict=False)
            if 'optimizer' in ck:
                optim.load_state_dict(ck['optimizer'])
            start_ep = ck.get('epoch', 0)

    best_miou = -1.0
    last_ckpt = None

    for epoch in range(start_ep, args.epochs):
        model.train()
        total_seg = total_type = total_org = 0.0
        total_cnt = 0

        pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch", ascii=True, dynamic_ncols=True)

        optim.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            if len(batch) == 3:
                imgs, masks, types = batch
                organs = None
            else:
                imgs, masks, types, organs = batch

            imgs  = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            types = types.to(device, non_blocking=True)

            outputs = model(imgs)
            if isinstance(outputs, (tuple, list)):
                if len(outputs) == 3:
                    seg_logits, type_logits, organ_logits = outputs
                else:
                    seg_logits, type_logits = outputs[0], outputs[1]
                    organ_logits = None
            else:
                raise RuntimeError("Model must return (seg_logits, type_logits[, organ_logits])")


            if masks.ndim == 4:
                if masks.shape[1] <= 32:   
                    m_label = masks.argmax(1)
                else:                      
                    m_label = masks.argmax(-1)
            else:
                m_label = masks.long()

            t_label = as_label_map(types)

            h = min(seg_logits.size(-2), m_label.size(-2))
            w = min(seg_logits.size(-1), m_label.size(-1))
            seg_logits = seg_logits[..., :h, :w]
            m_label    = m_label[..., :h, :w]

            h2 = min(type_logits.size(-2), t_label.size(-2))
            w2 = min(type_logits.size(-1), t_label.size(-1))
            type_logits = type_logits[..., :h2, :w2]
            t_label     = t_label[..., :h2, :w2]

            m_bin = (m_label > 0).long()

            if seg_logits.size(1) == 1:

                loss_seg = seg_bce(seg_logits.squeeze(1), m_bin.float())
            else:

                loss_seg = seg_ce(seg_logits, m_bin)

            ce_map = F.cross_entropy(type_logits, t_label, reduction='none')
            valid = (m_label > 0).float()
            loss_type = (ce_map * valid).sum() / valid.sum().clamp_min(1.0)

            loss_org = torch.tensor(0.0, device=device)
            if args.w_org > 0 and organ_logits is not None and organs is not None:

                org = organs.detach()
                if org.is_cuda: org = org.cpu()
                org = org.to(torch.float32)
                org[torch.isnan(org)] = -1
                org = org.long()
                num_org = organ_logits.size(1)
                org[(org < 0) | (org >= num_org)] = -1
                org = org.to(device, non_blocking=True)

                valid = (org >= 0) & (org < num_org)
                if valid.any():
                    loss_org = organ_ce(organ_logits, org)

            loss = args.w_seg*loss_seg + args.w_type*loss_type + args.w_org*loss_org
            (loss/ max(1, args.accum_steps)).backward()

            do_step = ((step + 1) % max(1, args.accum_steps) == 0) or (step == len(train_loader)-1)
            if do_step:
                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optim.step()
                optim.zero_grad(set_to_none=True)

            bs = imgs.size(0)
            total_seg  += loss_seg.item()  * bs
            total_type += loss_type.item() * bs
            total_org  += (loss_org.item() if isinstance(loss_org, torch.Tensor) else 0.0) * bs
            total_cnt  += bs

            pbar.update(1)
            pbar.set_postfix(seg=f"{loss_seg.item():.3f}", typ=f"{loss_type.item():.3f}", org=(f"{loss_org.item():.3f}" if isinstance(loss_org, torch.Tensor) else "-"), lr=f"{optim.param_groups[0]['lr']:.1e}")
        pbar.close()

        ckpt = Path(args.checkpoint_dir)/f'ckpt_ep{epoch+1}.pth'
        torch.save({'epoch': epoch+1, 'model_state': model.state_dict(), 'optimizer': optim.state_dict()}, str(ckpt))
        last_ckpt = ckpt

        if val_loader is not None:
            miou, mf1 = evaluate_type_miou(model, val_loader, device, args.num_type_classes)
            oacc = evaluate_organ_top1(model, val_loader, device, args.num_organs) if args.w_org>0 else 0.0
            print(f"[VAL] epoch {epoch+1}: type mIoU={miou:.4f} macroF1={mf1:.4f} organ@1={oacc:.4f}")
            scheduler.step(miou)
            if args.save_best and miou > best_miou:
                best_miou = miou
                best_path = Path(args.checkpoint_dir)/'best.pth'
                torch.save({'epoch': epoch+1, 'model_state': model.state_dict(), 'optimizer': optim.state_dict(), 'best_miou': best_miou}, str(best_path))
                print(f"[SAVE] best -> {best_path} (mIoU={best_miou:.4f})")

    if last_ckpt is not None:
        final_path = Path(args.checkpoint_dir)/'model_final.pth'
        if final_path.exists(): final_path.unlink()
        os.replace(last_ckpt, final_path)
        print(f"[SAVE] {final_path}")

if __name__ == '__main__':
    main()
