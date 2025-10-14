import numpy as np
from tqdm import tqdm
import torch

import os
import os.path as osp
from argparse import ArgumentParser
from mmcv import Config
from models import MODELS
from dataloaders import build_dataset
from torch.utils.data import DataLoader

from models.metrics.eval_metric import compute_depth_errors
from pytorch_lightning import seed_everything

from utils.visualization import *
import cv2


def parse_args():
    parser = ArgumentParser()

    # configure file
    parser.add_argument('--config', help='config file path', required=True)
    parser.add_argument('--test_env', type=str, default='test_day')  # test_night, test_rain
    parser.add_argument('--save_dir', type=str, default=' ')
    parser.add_argument('--modality', type=str, required=True)
    parser.add_argument('--seed', type=int, default=1024)

    # NEW: multiple checkpoints (space-separated)
    parser.add_argument(
        '--ckpt_paths',
        type=str,
        nargs='+',
        required=True,
        help='List of pretrained checkpoint paths, one per model'
    )

    # Optional labels to name models in visuals/printouts; if absent, uses basename of ckpt
    parser.add_argument(
        '--model_labels',
        type=str,
        nargs='+',
        default=None,
        help='Optional list of short labels to tag models in outputs'
    )

    # Optional: force image save cadence (every k frames)
    parser.add_argument('--save_every', type=int, default=10)

    return parser.parse_args()


@torch.no_grad()
def main():
    # parse args
    args = parse_args()

    # parse cfg
    cfg = Config.fromfile(osp.join(args.config))

    # show information
    print(f'Now evaluating with {args.config}...')
    print(f'Number of models: {len(args.ckpt_paths)}')

    # configure seed
    seed_everything(args.seed)

    # prepare data loader
    dataset_name = cfg.dataset['list'][0]
    cfg.dataset[dataset_name].test_env = args.test_env
    cfg.dataset[dataset_name].test.modality = args.modality
    dataset = build_dataset(cfg.dataset, eval_mode='depth', split='test')

    test_loader = DataLoader(
        dataset['test']['depth'],
        batch_size=1,
        shuffle=False,
        num_workers=cfg.workers_per_gpu,
        drop_last=False
    )

    print('{} samples found for evaluation'.format(len(test_loader)))

    # ---------------------------------------------------------------------
    # Build N models (all share the same cfg.model definition)
    # ---------------------------------------------------------------------
    ckpts = args.ckpt_paths
    N = len(ckpts)

    if args.model_labels is not None:
        if len(args.model_labels) != N:
            raise ValueError("--model_labels length must match --ckpt_paths")
        model_labels = args.model_labels
    else:
        # default to checkpoint basenames
        model_labels = [osp.splitext(osp.basename(p))[0] for p in ckpts]

    models = []
    is_midas_like = []  # track models that need GT-based scale/shift
    for p in ckpts:
        m = MODELS.build(name=cfg.model.name, option=cfg)
        if p is not None:
            print(f'Load pre-trained model from {p}')
            state = torch.load(p, map_location='cpu')
            # handle common wrapping
            if 'state_dict' in state:
                m.load_state_dict(state['state_dict'], strict=False)
            else:
                m.load_state_dict(state, strict=False)
        m.cuda()
        m.eval()
        models.append(m)
        midas_flag = ('Midas' in p) or ('DPT' in p)
        is_midas_like.append(midas_flag)

    # ---------------------------------------------------------------------
    # Prepare save dirs
    # save_dir structure:
    #   save_dir/
    #     <seq_name>/        # tries to use batch keys; falls back to "seq_default"
    #       1/               # model 1 per-model panels
    #       2/
    #       ...
    #       N/
    #       N+1/             # combined multi-model panels
    # ---------------------------------------------------------------------
    root_save = None
    if args.save_dir != ' ':
        root_save = osp.abspath(args.save_dir)
        os.makedirs(root_save, exist_ok=True)

    # ---------------------------------------------------------------------
    # Metrics: collect per-model errors
    # all_errs[k] is list over frames; each item is np.array(errors_for_that_frame)
    # ---------------------------------------------------------------------
    all_errs = [[] for _ in range(N)]

    # helper: fetch optional sequence/frame names from batch
    def _get_seq_and_frame(batch, default_idx):
        # Try common keys if present in your dataset; else fallback to defaults
        seq_name = batch.get('seq_name', ['seq_default'])[0] if isinstance(batch.get('seq_name', None), list) else batch.get('seq_name', 'seq_default')
        frame_id = batch.get('frame_id', [f'{default_idx:05d}'])[0] if isinstance(batch.get('frame_id', None), list) else batch.get('frame_id', f'{default_idx:05d}')
        # Make filesystem-safe
        seq_name = str(seq_name).replace('/', '_')
        frame_id = str(frame_id)
        return seq_name, frame_id

    # ---------------------------------------------------------------------
    # Loop
    # ---------------------------------------------------------------------
    for i, batch in enumerate(tqdm(test_loader)):
        tgt_img = batch['tgt_image']            # [B, C, H, W]
        gt_depth = batch['tgt_depth_gt']        # [B, H, W], sparse

        # If present but unused in the original loop, keep access (won't break)
        tgt_eh_img = batch.get('tgt_image_eh', None)

        # Build predictions for each model
        preds = []  # list of [B, H, W] (float tensors, CUDA->CPU later only for viz)
        # Inference
        for k, m in enumerate(models):
            if is_midas_like[k]:
                # Models that require GT scale/shift during inference
                pred_depth = m.inference_depth(tgt_img.cuda(), gt_depth.cuda())
            else:
                pred_depth = m.inference_depth(tgt_img.cuda())

            # reshape to [B,H,W]
            if pred_depth.ndim == 4:
                pred_depth = pred_depth.squeeze(1)
            elif pred_depth.ndim == 2:
                pred_depth = pred_depth.unsqueeze(0)  # assume [H,W] -> [1,H,W]

            # resize to GT size if needed
            B, H, W = gt_depth.size()
            if pred_depth.nelement() != gt_depth.nelement():
                pred_depth = torch.nn.functional.interpolate(
                    pred_depth.unsqueeze(1), [H, W], mode='bilinear', align_corners=False
                ).squeeze(1)

            # compute metrics for this model on this frame
            if is_midas_like[k]:
                errs = compute_depth_errors(gt_depth.cuda(), pred_depth, align=False)
            else:
                errs = compute_depth_errors(gt_depth.cuda(), pred_depth)
            all_errs[k].append(np.array(errs))

            preds.append(pred_depth)

        # -----------------------------------------------------------------
        # Visualizations (save every args.save_every frames by default)
        # -----------------------------------------------------------------
        if root_save is not None and (i % args.save_every == 0):
            seq_name, frame_id = _get_seq_and_frame(batch, i)
            seq_dir = osp.join(root_save, seq_name)
            # create per-model dirs: 1..N
            for idx in range(1, N + 2):  # include N+1
                os.makedirs(osp.join(seq_dir, f'{idx}'), exist_ok=True)

            # Prepare base RGB image at GT resolution
            # tgt_img: [1,C,H,W]
            _, H, W = gt_depth.size()
            if tgt_img.nelement() != gt_depth.nelement():
                rgb_for_viz = torch.nn.functional.interpolate(
                    tgt_img, [H, W], mode='bilinear', align_corners=False
                )[0]  # [C,H,W]
            else:
                rgb_for_viz = tgt_img[0]  # [C,H,W]
            img_vis = visualize_image(rgb_for_viz, flag_np=True).transpose(1, 2, 0)  # HxWx3 in RGB

            # Build GT viz (sparse) and compute vmax from it for consistent scaling
            gt = gt_depth[0]  # [H,W]
            # 'visualize_depth_as_numpy' returns (viz, cm, vmax) if 3 returns else (viz, cm)
            gt_vis, gt_valid, vmax = visualize_depth_as_numpy(gt, 'jet', is_sparse=True)

            # Also compute GT mask (valid pixels)
            # gt_valid = (gt > 0)
            gt_mask_rgb = (gt_valid.astype(np.uint8) * 255).reshape(H, W, 1).repeat(3, axis=2)

            # -------------------------------------------------------------
            # Per-model panels in folders 1..N:
            # each panel: [tgt_image, GT_sparse_depth, dense_pred, sparse_pred]
            # -------------------------------------------------------------
            for k in range(N):
                pred = preds[k][0]  # [H,W], CUDA tensor
                pred_cpu = pred.detach().cpu()

                # Dense pred (full)
                pred_dense_vis, _, _ = visualize_depth_as_numpy(pred_cpu, 'jet', is_sparse=False, vmax=vmax)

                # Sparse pred: zero out where GT invalid, then colorize with same vmax
                pred_sparse = pred_cpu.clone()
                pred_sparse[gt_valid] = 0.0
                pred_sparse_vis, _, _ = visualize_depth_as_numpy(pred_sparse, 'jet', is_sparse=True, vmax=vmax)

                # Stack vertically as in original code
                # Convert everything to HxWx3 RGB uint8
                panel = np.concatenate(
                    (img_vis, gt_vis, pred_dense_vis, pred_sparse_vis),
                    axis=0
                )
                panel_bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
                out_path = osp.join(seq_dir, f'{k+1}', f'{frame_id}.png')
                cv2.imwrite(out_path, panel_bgr)

            # -------------------------------------------------------------
            # Combined panel in folder N+1:
            # [tgt_image, GT_sparse_depth, dense_pred_m1, sparse_pred_m1, ..., dense_pred_mN, sparse_pred_mN]
            # -------------------------------------------------------------
            combined_rows = [img_vis, gt_vis]
            for k in range(N):
                pred = preds[k][0]
                pred_cpu = pred.detach().cpu()

                pred_dense_vis, _, _ = visualize_depth_as_numpy(pred_cpu, 'jet', is_sparse=False, vmax=vmax)
                pred_sparse = pred_cpu.clone()
                pred_sparse[gt_valid] = 0.0
                pred_sparse_vis, _, _ = visualize_depth_as_numpy(pred_sparse, 'jet', is_sparse=True, vmax=vmax)

                combined_rows.extend([pred_dense_vis, pred_sparse_vis])

            combined = np.concatenate(combined_rows, axis=0)
            combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
            out_path = osp.join(seq_dir, f'{N+1}', f'{frame_id}.png')
            cv2.imwrite(out_path, combined_bgr)

    # ---------------------------------------------------------------------
    # Print per-model metrics
    # ---------------------------------------------------------------------
    print('test set: {}, len: {}'.format(args.test_env, len(test_loader)))
    header = "  " + ("{:>10} | " * 9).format("Model", "abs_diff", "abs_rel",
                                             "sq_rel", "log10", "rmse", "rmse_log", "a1", "a2", "a3")
    print(header)
    print("-" * len(header))

    for k in range(N):
        errs_np = np.stack(all_errs[k])  # [num_frames, 9]
        mean_errs = np.mean(errs_np, axis=0)
        line = ("{:>10} | " + ("{: 8.7f} | " * 9)).format(model_labels[k], *mean_errs.tolist())
        print(line)


if __name__ == '__main__':
    main()
