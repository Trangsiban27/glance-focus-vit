"""Generate Grad-CAM visualizations for five models on internal-test images.

Default: seed 42, first five rows of data/Datos/internal_test.csv, all five
models (best_v2, best_vit, ResNet-50, DenseNet-121 and Swin-Tiny).

Example:
    python gradcam_internal.py
    python gradcam_internal.py --seed 46 --models best_v2 resnet50

Grad-CAM uses each model's predicted class. For Glance-Focus models with two
branches, the global CAM and the local ROI CAM are mapped to the original image
and averaged into one visualization. Outputs are written under results/gradcam/.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    if exc.name != 'matplotlib':
        raise
    raise ImportError('Grad-CAM figures need matplotlib; install it with `pip install matplotlib`.') from exc

from dataset import ImageManifest
from train import device_for, load_checkpoint, seed_all


MODEL_NAMES = ('best_v2', 'best_vit', 'resnet50', 'densenet121', 'swin_tiny')
CHECKPOINTS = {
    42: {
        'best_v2': 'seed 42/best_v2.pt',
        'best_vit': 'seed 42/best_vit.pt',
        'resnet50': 'seed 42/resnet-42.pt',
        'densenet121': 'seed 42/densenet42.pt',
        'swin_tiny': 'seed 42/swin_tiny_42.pt',
    },
    46: {
        'best_v2': 'seed 46/best_v2_46.pt',
        'best_vit': 'seed 46/best_vit46.pt',
        'resnet50': 'seed 46/resnet-46.pt',
        'densenet121': 'seed 46/densenet46.pt',
        'swin_tiny': 'seed 46/swin_tiny_46.pt',
    },
    49: {
        'best_v2': 'seed 49/best_v2_49.pt',
        'best_vit': 'seed 49/best_vit49.pt',
        'resnet50': 'seed 49/resnet-49.pt',
        'densenet121': 'seed 49/densenet49.pt',
        'swin_tiny': 'seed 49/swin-tiny-49.pt',
    },
}


def target_layer(model):
    """Return a spatial feature layer and its tensor-layout description."""
    if hasattr(model, 'vit'):
        return model.vit.blocks[-1].norm1, 'vit_tokens'
    model_type = model.config.get('model_type') if hasattr(model, 'config') else None
    if model_type == 'resnet50':
        return model.backbone.layer4[-1], 'nchw'
    if model_type == 'densenet121':
        return model.backbone.features.denseblock4, 'nchw'
    if model_type == 'swin_tiny':
        return model.swin.layers[-1].blocks[-1], 'nhwc'
    raise ValueError(f'Unsupported model class for Grad-CAM: {type(model).__name__}')


def _activation_cam(activation, gradient, layout, model):
    """Convert a target-layer activation and its gradient into a 2D CAM."""
    if layout == 'vit_tokens':
        prefix = int(model.vit.num_prefix_tokens)
        features = activation[:, prefix:, :]
        gradients = gradient[:, prefix:, :]
        token_count = features.shape[1]
        side = math.isqrt(token_count)
        if side * side != token_count:
            raise ValueError(f'ViT patch token count is not square: {token_count}')
        channel_weights = gradients.mean(dim=1, keepdim=True)
        cam = (features * channel_weights).sum(dim=-1).relu()
        cam = cam.reshape(cam.shape[0], 1, side, side)
    elif layout == 'nhwc':
        features = activation.permute(0, 3, 1, 2)
        gradients = gradient.permute(0, 3, 1, 2)
        channel_weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (features * channel_weights).sum(dim=1, keepdim=True).relu()
    else:
        channel_weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = (activation * channel_weights).sum(dim=1, keepdim=True).relu()
    cam = cam.detach()
    maximum = cam.amax(dim=(2, 3), keepdim=True)
    return cam / maximum.clamp_min(1e-8)


def _to_rgb_tensor(image):
    return image[0].detach().float().cpu().permute(1, 2, 0).numpy().clip(0, 1)


def _overlay(axis, image, cam, title):
    axis.imshow(image)
    axis.imshow(cam, cmap='jet', alpha=0.43, vmin=0, vmax=1)
    axis.set_title(title)
    axis.axis('off')


def gradcam_for_image(model, hi, mask, selector=None):
    """Return per-view CAMs, prediction info, and the ROI box if present."""
    layer, layout = target_layer(model)
    activations = []

    def capture(_module, _inputs, output):
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not torch.is_tensor(output):
            raise TypeError('Target layer did not return a tensor')
        activations.append(output)

    hook = layer.register_forward_hook(capture)
    try:
        model.eval()
        model_input = hi.detach().requires_grad_(True)
        with torch.enable_grad():
            output = model(model_input, mask, selector)
            logits = output['logits']
            probabilities = logits.softmax(-1)[0].detach().cpu().numpy()
            prediction = int(logits.argmax(-1).item())
            score = logits[0, prediction]
            if not activations:
                raise RuntimeError('Target layer was not executed during model forward')
            gradients = torch.autograd.grad(score, tuple(activations), allow_unused=True)
        cams = []
        for activation, gradient in zip(activations, gradients):
            if gradient is None:
                cams.append(None)
            else:
                cams.append(_activation_cam(activation, gradient, layout, model))
        box = output['boxes'][0].detach() if output.get('boxes') is not None else None
        return cams, prediction, probabilities, box
    finally:
        hook.remove()


def _combine_branch_cams(global_cam, local_cam, box):
    """Map the local ROI CAM onto the full image and average both branch maps."""
    if global_cam is None:
        return local_cam
    if local_cam is None or box is None:
        return global_cam

    height, width = global_cam.shape
    local = torch.as_tensor(local_cam, dtype=torch.float32)[None, None]
    box = torch.as_tensor(box, dtype=torch.float32)
    x = torch.arange(width, dtype=torch.float32) + 0.5
    y = torch.arange(height, dtype=torch.float32) + 0.5
    grid_x = (x - box[0]) / (box[2] - box[0]).clamp_min(1e-8)
    grid_y = (y - box[1]) / (box[3] - box[1]).clamp_min(1e-8)
    grid = torch.stack(torch.meshgrid(2 * grid_x - 1, 2 * grid_y - 1, indexing='xy'), dim=-1)[None]
    local_full = F.grid_sample(local, grid, mode='bilinear', padding_mode='zeros',
                               align_corners=False)[0, 0].numpy()
    inside_x = (x >= box[0]) & (x < box[2])
    inside_y = (y >= box[1]) & (y < box[3])
    local_full *= (inside_y[:, None] & inside_x[None, :]).numpy()
    combined = 0.5 * (global_cam + local_full)
    return combined / max(float(combined.max()), 1e-8)


def _make_figure(image, cam, title, path, cam_title):
    figure, axes = plt.subplots(1, 2, figsize=(9, 4.4))
    axes[0].imshow(image)
    axes[0].set_title(title)
    axes[0].axis('off')
    if cam is not None:
        _overlay(axes[1], image, cam, cam_title)
    else:
        axes[1].axis('off')
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(figure)


def _save_overlay(image, cam, path, cam_title):
    figure, axis = plt.subplots(figsize=(6, 6))
    _overlay(axis, image, cam, cam_title)
    figure.savefig(path, dpi=160, bbox_inches='tight', pad_inches=0.05)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=sorted(CHECKPOINTS), default=42)
    parser.add_argument('--models', nargs='+', choices=(*MODEL_NAMES, 'all'), default=['all'])
    parser.add_argument('--csv', default='data/Datos/internal_test.csv')
    parser.add_argument('--data-root')
    parser.add_argument('--checkpoint-root', default='checkpoints')
    parser.add_argument('--checkpoint',
                        help='Override checkpoint path for one selected model')
    parser.add_argument('--output', default='results/gradcam_internal')
    parser.add_argument('--num-images', type=int, default=5)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    if args.num_images < 1:
        parser.error('--num-images must be >= 1')
    models = list(MODEL_NAMES) if 'all' in args.models else args.models
    if args.checkpoint and len(models) != 1:
        parser.error('--checkpoint can only be used with exactly one model in --models')
    seed_all(42)
    device = device_for(args.device)
    checkpoint_root = Path(args.checkpoint_root)
    output_root = Path(args.output) / f'seed{args.seed}'
    output_root.mkdir(parents=True, exist_ok=True)
    image_metadata = None

    for model_name in models:
        checkpoint_path = (Path(args.checkpoint).expanduser().resolve() if args.checkpoint else
                           checkpoint_root / CHECKPOINTS[args.seed][model_name])
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
        model, checkpoint = load_checkpoint(checkpoint_path, device)
        if checkpoint.get('seed') != args.seed:
            raise ValueError(f'{checkpoint_path} records seed {checkpoint.get("seed")}, expected {args.seed}')
        dataset = ImageManifest(
            args.csv, args.data_root, classes=checkpoint['classes'],
            hi_size=checkpoint['hi_size'], use_fov=model.use_fov,
            task=checkpoint.get('task', 'multiclass'),
            front_end=checkpoint.get('front_end', False),
        )
        selector = None
        if (getattr(model, 'attention_method', None) in {'gradcam', 'gradrollout'}
                and model.fusion != 'global'):
            # Keep the selector separate so its ROI-gradient pass is not
            # captured as the target model's global-branch activation.
            selector = copy.deepcopy(model).to(device).eval().requires_grad_(False)
            selector.attention_method = model.attention_method
        if args.num_images > len(dataset):
            raise ValueError(f'Requested {args.num_images} images, but CSV has {len(dataset)}')

        model_dir = output_root / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        source_dir = model_dir / 'source_images'
        input_dir = model_dir / 'input_images'
        heatmap_dir = model_dir / 'heatmaps'
        overlay_dir = model_dir / 'overlays'
        for directory in (source_dir, input_dir, heatmap_dir, overlay_dir):
            directory.mkdir(parents=True, exist_ok=True)
        # Remove branch-specific CAM files from older runs; each image now has
        # one combined CAM artifact instead.
        for old_cam in (*model_dir.glob('image_*_global_cam.npy'),
                        *model_dir.glob('image_*_local_cam.npy')):
            old_cam.unlink()
        labels = checkpoint['classes']
        print(f'Grad-CAM: {model_name}, seed {args.seed}, device {device}', flush=True)
        model_rows = []
        sheet_data = []
        for index in range(args.num_images):
            hi, mask, target, image_path = dataset[index]
            hi = hi.unsqueeze(0).to(device)
            mask = mask.unsqueeze(0).to(device)
            cams, prediction, probabilities, box = gradcam_for_image(model, hi, mask, selector)
            raw_image = _to_rgb_tensor(hi)
            global_cam = None
            local_cam = None
            if cams:
                global_map = cams[0]
                if global_map is not None:
                    global_cam = F.interpolate(global_map, size=hi.shape[-2:],
                                               mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
            if box is not None and len(cams) > 1 and cams[-1] is not None:
                local_cam = F.interpolate(cams[-1], size=(224, 224),
                                          mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
            combined_cam = _combine_branch_cams(global_cam, local_cam, box)
            cam_title = ('Grad-CAM · global + local ROI' if local_cam is not None
                         else 'Grad-CAM · model output')

            true_name = labels[int(target)]
            pred_name = labels[prediction]
            confidence = float(probabilities[prediction])
            filename = f'image_{index + 1:02d}_true-{true_name}_pred-{pred_name}.png'
            image_stem = f'image_{index + 1:02d}'
            title = f'{index + 1}. True: {true_name} | Pred: {pred_name} ({confidence:.1%})'
            _make_figure(raw_image, combined_cam, title, model_dir / filename, cam_title)
            plt.imsave(input_dir / f'{image_stem}.png', raw_image)
            with Image.open(image_path) as source:
                source.convert('RGB').save(source_dir / f'{image_stem}.png')
            if combined_cam is not None:
                np.save(model_dir / f'image_{index + 1:02d}_fused_cam.npy', combined_cam)
                heatmap_path = heatmap_dir / f'{image_stem}_gradcam.png'
                overlay_path = overlay_dir / f'{image_stem}_gradcam_overlay.png'
                plt.imsave(heatmap_path, combined_cam, cmap='jet', vmin=0, vmax=1)
                _save_overlay(raw_image, combined_cam, overlay_path, cam_title)
            else:
                heatmap_path = overlay_path = None

            model_rows.append({
                'index_in_csv': index,
                'path': image_path,
                'true_label': int(target),
                'true_class': true_name,
                'prediction': prediction,
                'predicted_class': pred_name,
                'probabilities': probabilities.tolist(),
                'box_hi_xyxy': box.cpu().tolist() if box is not None else None,
                'cam_fusion': ('mean(global_cam, local_roi_cam_mapped_to_full_image)'
                               if local_cam is not None else 'global_cam'),
                'source_image_file': str(source_dir / f'{image_stem}.png'),
                'model_input_file': str(input_dir / f'{image_stem}.png'),
                'heatmap_file': str(heatmap_path) if heatmap_path else None,
                'overlay_file': str(overlay_path) if overlay_path else None,
                'visualization': filename,
            })
            sheet_data.append((raw_image, combined_cam, cam_title, title))
            print(f'  {index + 1}/{args.num_images}: {Path(image_path).name} '
                  f'true={true_name}, pred={pred_name}', flush=True)

        (model_dir / 'predictions.json').write_text(
            json.dumps(model_rows, indent=2, ensure_ascii=False), encoding='utf-8')
        figure, axes = plt.subplots(args.num_images, 2,
                                    figsize=(9, max(4.2, 3.4 * args.num_images)),
                                    squeeze=False)
        for row_index, (image, combined_cam, cam_title, title) in enumerate(sheet_data):
            axes[row_index, 0].imshow(image)
            axes[row_index, 0].set_title(title, fontsize=9)
            axes[row_index, 0].axis('off')
            if combined_cam is not None:
                _overlay(axes[row_index, 1], image, combined_cam, cam_title)
            else:
                axes[row_index, 1].axis('off')
        figure.tight_layout()
        figure.savefig(model_dir / 'contact_sheet.png', dpi=160, bbox_inches='tight')
        plt.close(figure)
        del selector
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if image_metadata is None:
            image_metadata = [{'index_in_csv': row['index_in_csv'], 'path': row['path'],
                               'true_class': row['true_class']} for row in model_rows]

    (output_root / 'selected_images.json').write_text(
        json.dumps(image_metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Grad-CAM images saved under {output_root.resolve()}', flush=True)


if __name__ == '__main__':
    main()
