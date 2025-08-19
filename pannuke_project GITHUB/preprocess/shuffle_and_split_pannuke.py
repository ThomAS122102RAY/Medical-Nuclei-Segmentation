import argparse,random,shutil
from pathlib import Path

SUFFIX_MAP={
    'images':'_images.npy',
    'masks':'_masks.npy',
    'types':'_types.npy'
}


def gather_images(img_dir:Path):
    return sorted(img_dir.glob('*_images.npy'))


def split_files(files,train_ratio):
    random.shuffle(files)
    k=int(len(files)*train_ratio)
    return files[:k],files[k:]


def copy_group(group,dst_root:Path,src_dirs:dict):
    for img_path in group:
        stem=img_path.stem.replace('_images','')
        for key,suf in SUFFIX_MAP.items():
            src=src_dirs[key]/f'{stem}{suf}'
            if not src.exists():
                continue
            dst=(dst_root/key)/src.name
            dst.parent.mkdir(parents=True,exist_ok=True)
            if not dst.exists():
                shutil.copy2(src,dst)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data_root',type=Path,default=Path(__file__).resolve().parent.parent/'data')
    ap.add_argument('--train_ratio',type=float,default=0.8)
    ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args()

    random.seed(args.seed)
    data_root=args.data_root
    img_dir=data_root/'images'
    mask_dir=data_root/'masks'
    type_dir=data_root/'types'

    if not img_dir.is_dir():
        raise FileNotFoundError(img_dir)
    src_dirs={'images':img_dir,'masks':mask_dir,'types':type_dir}

    imgs=gather_images(img_dir)
    if not imgs:
        raise RuntimeError('No *_images.npy found under images/')

    train,test=split_files(imgs,args.train_ratio)
    for split_name,grp in (('train',train),('test',test)):
        dst_root=data_root/split_name
        copy_group(grp,dst_root,src_dirs)
        print(f'{split_name}: {len(grp)} slides')

if __name__=='__main__':
    main()
