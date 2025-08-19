import os, sys, argparse
import torch, numpy as np
from PIL import Image
from torch.utils.data import DataLoader

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from segmentor.transunet import TransUNet
from preprocess.data_loader import PanNuke

def types_to_label(t):

    if t.ndim == 3:
        return t.long()
    if t.ndim == 4:

        if t.shape[1] <= 64:   
            return t.argmax(1).long()
        else:                  
            return t.argmax(-1).long()
    return t.long()

def parse_args():
    p = argparse.ArgumentParser(description="PanNuke  main_pipeline ")
    p.add_argument('--weights',   default=os.path.join(PROJECT_ROOT, 'checkpoints', 'segmentor', 'model_final.pth'))
    p.add_argument('--checkpoint', default=None, help=' --weights main_pipeline ')
    p.add_argument('--input_dir', default=os.path.join(PROJECT_ROOT, 'data', 'test'))
    p.add_argument('--data_root', default=None, help=' --input_dir main_pipeline ')
    p.add_argument('--output_dir', default=os.path.join(PROJECT_ROOT, 'results', 'single_masks'))
    p.add_argument('--output_npy', default=os.path.join(PROJECT_ROOT, 'results', 'all_masks.npy'))
    p.add_argument('--output_pred_npy', default=None, help='main_pipeline ')
    p.add_argument('--output_pred_seg_npy', default=None)
    p.add_argument('--batch_size', type=int, default=12)
    args, _ = p.parse_known_args()    
    if args.checkpoint and not args.weights:
        args.weights = args.checkpoint
    if args.data_root:
        args.input_dir = args.data_root
    if args.output_pred_npy:        
        args.output_npy = args.output_pred_npy
    return args

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = PanNuke(args.input_dir)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,              
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sample_img = dataset[0][0] 
    in_chans = int(sample_img.shape[0])
    model = TransUNet(in_chans=in_chans, num_classes=6, num_organs=19).to(device)

    state = torch.load(args.weights, map_location=device)
    state_dict = state.get('model_state', state) if isinstance(state, dict) else state
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    all_preds = []
    all_org_idx = []
    all_org_logits = [] 
    all_seg   = [] 
    all_org_idx = []
    all_org_logits = []
    inter = union = None
    n_cls = None

    with torch.no_grad():
        for batch in loader:

            imgs, masks, types = batch[:3]
            imgs  = imgs.to(device, non_blocking=True).float()
            t_lab = types_to_label(types)            

            outputs = model(imgs)
            if isinstance(outputs, (tuple, list)):

                seg_logits  = outputs[0]

                type_logits = outputs[1]
                organ_logits = outputs[2] if len(outputs) >= 3 else None            
                type_logits = outputs[1]
            else:

                seg_logits  = None
                type_logits = outputs

            h = min(type_logits.size(-2), t_lab.size(-2))
            w = min(type_logits.size(-1), t_lab.size(-1))
            type_logits = type_logits[..., :h, :w]
            t_lab = t_lab[..., :h, :w]

            preds = type_logits.argmax(1).cpu().numpy() 
            gts   = t_lab.cpu().numpy()

            if seg_logits is not None:
                h3 = min(seg_logits.size(-2), imgs.size(-2))
                w3 = min(seg_logits.size(-1), imgs.size(-1))
                seg_logits = seg_logits[..., :h3, :w3]
                if seg_logits.size(1) == 1:
                    seg_pred = (seg_logits.sigmoid() > 0.5).long().cpu().numpy()[:, 0]
                else:
                    seg_pred = seg_logits.argmax(1).cpu().numpy()
            else:
                seg_pred = None

            if n_cls is None:
                n_cls = int(type_logits.shape[1])
                print(f"[INFO]  n_cls = {n_cls}")
                inter = np.zeros(n_cls, dtype=np.int64)
                union = np.zeros(n_cls, dtype=np.int64)

            for c in range(n_cls):
                inter[c] += np.logical_and(gts==c, preds==c).sum()
                union[c] += np.logical_or(gts==c, preds==c).sum()

            for i, pmask in enumerate(preds):
                idx = len(all_preds)
                all_preds.append(pmask)
                np.save(os.path.join(args.output_dir, f'pred_{idx:05d}.npy'), pmask)
                Image.fromarray((pmask * (255//max(1,n_cls-1))).astype(np.uint8)).save(
                    os.path.join(args.output_dir, f'pred_{idx:05d}.png')
                )

                if seg_pred is not None:
                    all_seg.append(seg_pred[i].astype(np.uint8))

            if 'organ_logits' in locals() and organ_logits is not None:
                org_idx = organ_logits.argmax(1).detach().cpu().numpy()
                all_org_idx.extend(org_idx.tolist())
                try:
                    all_org_logits.append(organ_logits.float().detach().cpu().numpy())
                except Exception:
                    pass

    if all_preds:
        merged = np.stack(all_preds, axis=0)
        os.makedirs(os.path.dirname(args.output_npy), exist_ok=True)
        np.save(args.output_npy, merged)
        print(f"[] {args.output_npy}")
        ious = np.divide(inter, np.maximum(union, 1), where=union>0)
        print('[IoU]', ' '.join(f'c{c}:{iou:.4f}' for c, iou in enumerate(ious)), f'mIoU:{np.nanmean(ious):.4f}')

        if all_seg:
            seg_merged = np.stack(all_seg, axis=0)
            if args.output_pred_seg_npy:
                seg_out = args.output_pred_seg_npy
            elif args.output_npy.endswith('_pred_types.npy'):
                seg_out = args.output_npy.replace('_pred_types.npy', '_pred_seg.npy')
            else:
                base, ext = os.path.splitext(args.output_npy)
                seg_out = base + '_seg.npy'
            os.makedirs(os.path.dirname(seg_out), exist_ok=True)
            np.save(seg_out, seg_merged)
            print(f"[] SEG {seg_out}")

        if len(all_org_idx) > 0:
            if args.output_npy.endswith('_pred_types.npy'):
                out_org = args.output_npy.replace('_pred_types.npy', '_pred_organs.npy')
                out_log = args.output_npy.replace('_pred_types.npy', '_pred_organ_logits.npy')
            else:
                base, ext = os.path.splitext(args.output_npy)
                out_org = base + '_organs.npy'
                out_log = base + '_organ_logits.npy'
            np.save(out_org, np.array(all_org_idx, dtype=np.int64))
            if len(all_org_logits) > 0:
                np.save(out_log, np.concatenate(all_org_logits, axis=0).astype(np.float32))
            print(f"[] ORGAN {out_org}")
    else:
        print("[] ")


if __name__ == '__main__':
    main()
