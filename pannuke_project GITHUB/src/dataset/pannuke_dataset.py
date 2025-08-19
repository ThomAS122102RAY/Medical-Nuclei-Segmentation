import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

NUM_TYPES = 6
STRICT_TYPES = False 

ORGANS = ['Adrenal_gland','Bile-duct','Bladder','Breast','Cervix','Colon','Esophagus','HeadNeck','Kidney','Liver','Lung','Ovarian','Pancreatic','Prostate','Skin','Stomach','Testis','Thyroid','Uterus']

def _norm_organ_key(s: str):
    s = str(s).strip().lower().replace(' ', '').replace('_', '').replace('-', '')
    s = s.replace('ovary', 'ovarian')
    s = s.replace('headandneck', 'headneck')
    return s

ORG2ID = { _norm_organ_key(k): i for i,k in enumerate(ORGANS) }

def _organ_id_from_any(x):
    if isinstance(x, (bytes, bytearray)):
        x = x.decode('utf-8', 'ignore')
    try:
        i = int(x)
        if 0 <= i < len(ORGANS):
            return i
    except Exception:
        pass
    key = _norm_organ_key(x)
    return ORG2ID.get(key, None)

def _to_type_map(raw, inst_map, warn_once_flag: dict):

    arr = np.asarray(raw)
    if arr.ndim == 2 and np.issubdtype(arr.dtype, np.number):
        return arr.astype(np.int64)
    if arr.ndim == 3 and np.issubdtype(arr.dtype, np.number):
        if arr.shape[-1] == NUM_TYPES:
            return np.argmax(arr, axis=-1).astype(np.int64)
        if arr.shape[0] == NUM_TYPES:
            return np.argmax(arr, axis=0).astype(np.int64)

    if not warn_once_flag.get('types', False):
        print('[WARN] pannuke_dataset:  types  0 ')
        warn_once_flag['types'] = True
    return np.zeros_like(inst_map, dtype=np.int64)

class PannukeDataset(Dataset):
    def __init__(self, data_root, aug=None):
        self.aug = aug
        self._warn_once = {}

        img_shards = sorted(glob.glob(os.path.join(data_root, "*_images.npy")))
        if not img_shards:
            single = os.path.join(data_root, "images.npy")
            if os.path.isfile(single):
                img_shards = [single]
        if not img_shards:
            print(f"[Warning]  {data_root}  imagesDataset ")
            self.images = np.zeros((0,))
        else:
            self.images = np.concatenate([np.load(p, mmap_mode='r') for p in img_shards], axis=0)

        typ_shards = sorted(glob.glob(os.path.join(data_root, "*_types.npy")))
        if not typ_shards:
            single = os.path.join(data_root, "types.npy")
            if os.path.isfile(single):
                typ_shards = [single]
        if not typ_shards:
            print(f"[Warning]  {data_root}  types HW ")

            self.types = None
        else:

            arrs = [np.load(p, allow_pickle=True, mmap_mode='r') for p in typ_shards]
            try:
                self.types = np.concatenate(arrs, axis=0)
            except Exception:

                self.types = arrs

        mask_shards = sorted(glob.glob(os.path.join(data_root, "*_masks.npy")))
        if not mask_shards:
            single = os.path.join(data_root, "masks.npy")
            if os.path.isfile(single):
                mask_shards = [single]
        if not mask_shards:
            print(f"[Warning]  {data_root}  masks")
            self.masks = np.zeros_like(self.images)
        else:
            self.masks = np.concatenate([np.load(p, mmap_mode='r') for p in mask_shards], axis=0)

        assert len(self.images) == len(self.masks), "images  masks "
        if isinstance(self.types, np.ndarray):
            assert len(self.images) == len(self.types), "images  types "

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        msk = self.masks[idx]
        typ = None if self.types is None else (self.types[idx] if isinstance(self.types, np.ndarray) else self.types[0][idx] if isinstance(self.types, list) and self.types and hasattr(self.types[0], 'shape') and self.types[0].shape[0] == len(self.images) else self.types)

        if img.ndim == 3 and img.shape[-1] == 3:
            pass
        else:
            raise RuntimeError(f': idx={idx}, shape={img.shape}')
        if msk.ndim == 3 and msk.shape[-1] > 1:
            msk = np.argmax(msk, axis=-1)
        if msk.ndim != 2:
            raise RuntimeError(f'mask : idx={idx}, shape={msk.shape}')

        organ_id = None
        if typ is not None:

            arr = np.asarray(typ)
            if arr.ndim <= 1 or (hasattr(arr, 'dtype') and arr.dtype.kind in ('U','S','O')):

                val = typ
                try:
                    if isinstance(arr, np.ndarray) and arr.ndim == 1 and arr.size > 0:
                        val = arr[0]
                except Exception:
                    pass
                organ_id = _organ_id_from_any(val)
                if organ_id is None and not self._warn_once.get('organ', False):
                    print(f"[WARN] pannuke_dataset: idx={idx}={val}organ_id  -1 ")
                    self._warn_once['organ'] = True
        if organ_id is None:
            organ_id = -1  

        tmap = _to_type_map(typ if typ is not None else np.array(0), msk, self._warn_once)

        if self.aug:
            try:
                aug = self.aug(image=img, mask=msk)
                img, msk = aug["image"], aug["mask"]
            except Exception:
                pass

        x = torch.from_numpy(img).permute(2,0,1).float()
        if x.max() > 1.5:
            x = x / 255.0
        y_mask  = torch.from_numpy(msk).long()
        y_types = torch.from_numpy(tmap).long()
        y_organ = torch.tensor(int(organ_id), dtype=torch.long)

        return x, y_mask, y_types, y_organ
