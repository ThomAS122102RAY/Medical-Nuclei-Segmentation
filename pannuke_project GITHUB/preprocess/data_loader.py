
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


TYPE_TARGET_FORMAT = os.getenv("PN_TYPE_FMT", "label")

NUM_TYPES = int(os.getenv("PN_NUM_TYPES", "6"))

STREAM_MODE = os.getenv("PN_STREAM_MODE", os.getenv("PN_STREAM_MODE".upper(), "0")) in ("1","true","True")

LOAD_TO_RAM = os.getenv("PN_LOAD_TO_RAM", "1") == "1"

_ORGAN_LIST = [
    "Adrenal_gland","Bile-duct","Bladder","Breast","Cervix","Colon","Esophagus",
    "HeadNeck","Kidney","Liver","Lung","Ovarian","Pancreas","Prostate",
    "Skin","Stomach","Testis","Thyroid","Uterus"
]
_ORGAN2ID: Dict[str,int] = {k.lower(): i for i,k in enumerate(_ORGAN_LIST)}

def _norm_str(x) -> str:
    s = str(x)
    return s.strip()

def _to_organ_id(v) -> int:
    if v is None:
        return -1
    s = _norm_str(v).lower().replace(" ", "").replace("_","").replace("-","")

    if s in _ORGAN2ID:
        return _ORGAN2ID[s]

    for k,i in _ORGAN2ID.items():
        if s == k.lower().replace(" ", "").replace("_","").replace("-",""):
            return i
    return -1

def _find_shards(root: Path, kind: str) -> List[Path]:

    candidates = []

    sub = root / kind
    patterns = [f"*_{kind}.npy", f"*{kind}.npy", f"{kind}.npy"]
    search_dirs = [sub, root] if sub.exists() else [root]
    for d in search_dirs:
        for pat in patterns:
            candidates += sorted(d.rglob(pat))

    seen = set()
    shards = []
    for p in candidates:
        if p.exists():
            key = str(p.resolve())
            if key not in seen:
                seen.add(key)
                shards.append(p)
    return shards

class _ShardAccessor:
    """Keeps shard file list and returns sample i across shards."""
    def __init__(self, shards: List[Path], load_to_ram: bool, mmap: bool):
        self.shards = shards
        self.load_to_ram = load_to_ram
        self.mmap = mmap
        self._arrays: List[Optional[np.ndarray]] = [None]*len(shards)
        self._lengths: List[int] = []
        self._cum: List[int] = []
        off = 0
        for idx,p in enumerate(shards):
            arr = np.load(p, mmap_mode='r') if (not load_to_ram and mmap) else np.load(p)
            n = int(arr.shape[0])
            self._lengths.append(n)
            self._cum.append(off)
            off += n

            if load_to_ram:
                self._arrays[idx] = arr
            else:

                self._arrays[idx] = arr
        self.total = off
        self._last_idx = -1  

    def __len__(self):
        return self.total

    def get(self, i: int) -> Tuple[np.ndarray, int]:

        if i < 0 or i >= self.total:
            raise IndexError(i)



        shard_idx = 0
        for s in range(len(self._cum)-1, -1, -1):
            if i >= self._cum[s]:
                shard_idx = s
                break
        local_i = i - self._cum[shard_idx]
        arr = self._arrays[shard_idx]
        if arr is None:

            arr = np.load(self.shards[shard_idx], mmap_mode='r')
            self._arrays[shard_idx] = arr
        return arr[local_i], i

class PanNuke(Dataset):
    """PanNuke dataset: returns (img[C,H,W], mask, types_target, organ_id)"""
    def __init__(self, data_root: str):
        super().__init__()
        self.root = Path(data_root)
        if not self.root.exists():
            raise FileNotFoundError(f"data_root not found: {data_root}")

        img_shards = _find_shards(self.root, 'images')
        if not img_shards:
            raise FileNotFoundError(" images  .npy *_images.npy  images.npy")
        mask_shards = _find_shards(self.root, 'masks')
        if not mask_shards:
            raise FileNotFoundError(" masks  .npy *_masks.npy  masks.npy")
        type_meta_shards = _find_shards(self.root, 'types') 

        self.imgs = _ShardAccessor(img_shards, load_to_ram=LOAD_TO_RAM and not STREAM_MODE, mmap=not LOAD_TO_RAM)
        self.masks = _ShardAccessor(mask_shards, load_to_ram=LOAD_TO_RAM and not STREAM_MODE, mmap=not LOAD_TO_RAM)
        self.types_meta = _ShardAccessor(type_meta_shards, load_to_ram=LOAD_TO_RAM and not STREAM_MODE, mmap=not LOAD_TO_RAM) if type_meta_shards else None

        if len(self.imgs) != len(self.masks):
            raise ValueError(f"images  masks {len(self.imgs)} vs {len(self.masks)}")
        self.N = len(self.imgs)

    def __len__(self):
        return self.N

    def _img_to_tensor(self, img_np: np.ndarray) -> torch.Tensor:

        if img_np.ndim == 3 and img_np.shape[-1] == 3:
            img = torch.from_numpy(img_np).permute(2,0,1).contiguous()
        elif img_np.ndim == 3 and img_np.shape[0] == 3:
            img = torch.from_numpy(img_np).contiguous()
        else:

            if img_np.ndim == 2:
                img_np = np.repeat(img_np[...,None], 3, axis=-1)
            img = torch.from_numpy(img_np).permute(2,0,1).contiguous()
        img = img.float()

        if img.max() > 1.5:
            img = img / 255.0
        return img

    def _mask_to_tensor(self, mask_np: np.ndarray) -> torch.Tensor:

        return torch.from_numpy(mask_np.copy())

    def _types_from_mask_label(self, mask_np: np.ndarray) -> np.ndarray:

        if mask_np.ndim == 3:
            if mask_np.shape[-1] == NUM_TYPES:
                t = np.argmax(mask_np, axis=-1).astype(np.int64)
                return t
            if mask_np.shape[0] == NUM_TYPES:
                t = np.argmax(mask_np, axis=0).astype(np.int64)
                return t

            if mask_np.shape[-1] == 2:
                return (np.argmax(mask_np, axis=-1)).astype(np.int64)
            if mask_np.shape[0] == 2:
                return (np.argmax(mask_np, axis=0)).astype(np.int64)
        elif mask_np.ndim == 2:

            return mask_np.astype(np.int64, copy=False)

        return np.zeros(mask_np.shape[:2], dtype=np.int64)

    def __getitem__(self, i: int):
        img_np, _ = self.imgs.get(i)
        mask_np, _ = self.masks.get(i)
        img = self._img_to_tensor(img_np)
        mask_t = self._mask_to_tensor(mask_np)

        tmap = self._types_from_mask_label(mask_np)
        if TYPE_TARGET_FORMAT == 'onehot':
            t = torch.from_numpy(tmap).long()
            types_t = F.one_hot(t, num_classes=NUM_TYPES).permute(2,0,1).float()
        else:
            types_t = torch.from_numpy(tmap).long()

        organ_id = -1
        if self.types_meta is not None and len(self.types_meta) == self.N:
            meta, _ = self.types_meta.get(i)

            if np.asarray(meta).ndim == 0:
                organ_id = _to_organ_id(meta.item() if hasattr(meta, 'item') else meta)
            else:

                organ_id = _to_organ_id(meta[0])

        return img, mask_t, types_t, torch.tensor(organ_id, dtype=torch.long)


