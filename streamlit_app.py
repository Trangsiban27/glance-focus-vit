"""Streamlit image-classification demo for Glance-Focus ViT v2/v3 checkpoints.

Run from the project root with:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import copy
import io
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F

from processing import ImageProcessor
from train import device_for, load_checkpoint


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PROJECT_DIR / 'checkpoints' / 'seed 42' / 'best_v2.pt'
GRAD_METHODS = {'gradcam', 'gradrollout'}
LOCAL_SIZE = 224
BOX_COLOR = (255, 214, 0)
NEAR_THRESHOLD = .6


@st.cache_resource(show_spinner=False)
def load_model(checkpoint_path, checkpoint_mtime_ns, checkpoint_size, device_name):
    """Cache model resources; file metadata invalidates the cache if replaced."""
    del checkpoint_mtime_ns, checkpoint_size
    device = device_for(device_name)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    if checkpoint.get('model_config', {}).get('roi_version') not in {'v2', 'v3'}:
        raise ValueError('Checkpoint này không phải Glance-Focus ViT v2/v3.')
    model.eval().requires_grad_(False)

    selector = None
    if (getattr(model, 'attention_method', None) in GRAD_METHODS
            and model.fusion != 'global'):
        # Keep selector Grad-CAM hooks separate from the two classifier views.
        selector = copy.deepcopy(model).to(device).eval().requires_grad_(False)
        selector.attention_method = model.attention_method
    return model, checkpoint, selector, device


# --------------------------------------------------------------------------
# Map helpers
# --------------------------------------------------------------------------

def target_layer(model):
    """Return the final ViT block's pre-attention norm, unwrapping adapters."""
    block = model.vit.blocks[-1]
    while hasattr(block, 'block') and hasattr(block.block, 'norm1'):
        block = block.block
    return block.norm1


def activation_cam(activation, gradient, model):
    prefix = int(model.vit.num_prefix_tokens)
    features = activation[:, prefix:, :]
    gradients = gradient[:, prefix:, :]
    token_count = features.shape[1]
    side = math.isqrt(token_count)
    if side * side != token_count:
        raise ValueError(f'ViT patch tokens do not form a square grid: {token_count}')
    weights = gradients.mean(dim=1, keepdim=True)
    cam = (features * weights).sum(dim=-1).relu().reshape(-1, 1, side, side)
    return cam.detach()


def normalize(arr):
    """Scale a non-negative map to [0, 1]; an all-zero map stays zero."""
    arr = np.clip(np.asarray(arr, dtype=np.float32), 0, None)
    peak = float(arr.max()) if arr.size else 0.
    return arr / peak if peak > 1e-8 else arr


def upsample(grid_map, height, width):
    """Bilinearly resize a [1,1,g,g] tensor map to a [H,W] numpy array."""
    return F.interpolate(grid_map.float(), (height, width), mode='bilinear',
                         align_corners=False)[0, 0].cpu().numpy()


def crop_view(tensor, box, size=LOCAL_SIZE):
    """Sample box from a [C,H,W] tensor exactly like GlanceFocusViTV2.forward."""
    tensor = torch.as_tensor(tensor, dtype=torch.float32)
    if tensor.ndim == 2:
        tensor = tensor[None]
    height, width = tensor.shape[-2:]
    x0, y0, x1, y1 = [float(v) for v in box]
    t = (torch.arange(size, dtype=torch.float32) + .5) / size
    xs = x0 + (x1 - x0) * t
    ys = y0 + (y1 - y0) * t
    gx = (2 * xs / width - 1)[None, :].expand(size, -1)
    gy = (2 * ys / height - 1)[:, None].expand(-1, size)
    grid = torch.stack([gx, gy], -1)[None]
    return F.grid_sample(tensor[None], grid, align_corners=False,
                         padding_mode='border')[0]


def to_pil(tensor):
    """[3,H,W] tensor in [0,1] → PIL image."""
    arr = tensor.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))


# --------------------------------------------------------------------------
# Inference + explanation signals
# --------------------------------------------------------------------------

def predict_and_explain(model, hi, mask, selector, device):
    """Predict, then collect every signal used to explain the decision.

    Returns a dict with probabilities, the selected box, separate Grad-CAMs for
    the global view and the zoomed local view (no stitching), the selector map
    that chose the box, per-branch probabilities from the auxiliary heads and
    the gate statistics.
    """
    activations, features = [], []

    def capture_activation(_module, _inputs, output):
        if isinstance(output, (tuple, list)):
            output = output[0]
        activations.append(output)

    def capture_feature(_module, _inputs, output):
        if isinstance(output, (tuple, list)):
            output = output[0]
        features.append(output)

    two_views = model.fusion in {'concat', 'gate'}
    has_aux = two_views and hasattr(model, 'aux_g') and hasattr(model, 'aux_l')
    view_embed = getattr(model, 'view_embed', False)
    hooks = [target_layer(model).register_forward_hook(capture_activation),
             model.vit.register_forward_hook(capture_feature)]
    local_aux_gradient = None
    try:
        model_input = hi.unsqueeze(0).to(device).detach().requires_grad_(True)
        batch_mask = mask.unsqueeze(0).to(device)
        with torch.enable_grad():
            output = model(model_input, batch_mask, selector)
            logits = output['logits']
            probabilities = logits.softmax(-1)[0].detach().cpu().numpy()
            prediction = int(logits.argmax(-1).item())
            score = logits[0, prediction]
            if not activations:
                raise RuntimeError('Grad-CAM hook did not capture ViT features.')
            gradients = torch.autograd.grad(score, tuple(activations), allow_unused=True,
                                            retain_graph=has_aux)
            if has_aux and len(features) >= 2 and len(activations) >= 2:
                # Explain the zoomed view by its own auxiliary head: when the
                # learned fusion scale s is ~0 the final score barely depends on
                # the local view, so its final-score gradient would be noise.
                zl_aux = features[-1] + model.view_l if view_embed else features[-1]
                aux_score = model.aux_l(zl_aux)[0, prediction]
                local_aux_gradient, = torch.autograd.grad(aux_score, activations[-1],
                                                          allow_unused=True)
    finally:
        for hook in hooks:
            hook.remove()
    features = [f.detach() for f in features]

    cams = [activation_cam(a, g, model) if g is not None else None
            for a, g in zip(activations, gradients)]
    if local_aux_gradient is not None:
        cams[-1] = activation_cam(activations[-1], local_aux_gradient, model)
    height, width = hi.shape[-2:]
    mask_np = mask[0].cpu().numpy()

    # The classifier calls the ViT once per view: [global, local] (or one of them).
    global_raw = local_raw = None
    if model.fusion == 'local':
        local_raw = cams[-1] if cams else None
    elif model.fusion == 'global':
        global_raw = cams[-1] if cams else None
    elif len(cams) >= 2:
        global_raw, local_raw = cams[-2], cams[-1]

    box = output['boxes'][0].detach().cpu().numpy() if output.get('boxes') is not None else None

    global_cam = normalize(upsample(global_raw, height, width) * mask_np) if global_raw is not None else None

    local_cam = local_mask = local_image = None
    if box is not None:
        local_mask = crop_view(mask.cpu(), box)[0].numpy()
        local_image = to_pil(crop_view(hi.cpu(), box))
        if local_raw is not None:
            local_cam = normalize(upsample(local_raw, LOCAL_SIZE, LOCAL_SIZE) * local_mask)

    selector_map = None
    if output.get('attention') is not None:
        selector_map = normalize(upsample(output['attention'][:1].detach(), height, width) * mask_np)

    branch_probs = no_local_probs = local_share = None
    gate = output.get('gate')
    gate_mean = float(gate.detach().mean()) if gate is not None else None
    gate_scale = float(model.s.detach().reshape(-1)[0]) if hasattr(model, 's') else None
    if two_views and len(features) >= 2:
        zg, zl = features[-2], features[-1]
        if view_embed:
            zg, zl = zg + model.view_g, zl + model.view_l
        with torch.no_grad():
            if has_aux:
                branch_probs = {
                    'global': model.aux_g(zg).softmax(-1)[0].cpu().numpy(),
                    'local': model.aux_l(zl).softmax(-1)[0].cpu().numpy(),
                }
            if model.fusion == 'gate' and gate is not None and gate_scale is not None:
                # Counterfactual: the same head applied to z_g alone, i.e. the
                # local branch switched off. This is the honest measure of how
                # much the zoomed view changed the final decision.
                local_term = model.s * gate.detach() * zl
                local_share = float(local_term.norm() / zg.norm().clamp_min(1e-8))
                no_local_probs = model.head(zg).softmax(-1)[0].cpu().numpy()

    return dict(prediction=prediction, probabilities=probabilities, box=box,
                mask=mask_np, global_cam=global_cam, local_cam=local_cam,
                local_mask=local_mask, local_image=local_image,
                selector_map=selector_map, branch_probs=branch_probs,
                no_local_probs=no_local_probs, local_share=local_share,
                local_cam_from_aux=local_aux_gradient is not None,
                gate_mean=gate_mean, gate_scale=gate_scale)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def overlay(base, cam, valid=None, alpha=.45):
    """Blend a jet heatmap over base, only inside the valid (FoV) region."""
    base_arr = np.asarray(base.convert('RGB'), dtype=np.float32) / 255
    heat = plt.get_cmap('jet')(np.clip(cam, 0, 1))[..., :3]
    weight = alpha * (valid if valid is not None else np.ones(cam.shape))[..., None]
    out = (1 - weight) * base_arr + weight * heat
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))


def draw_box(image, box, width=3):
    image = image.copy()
    if box is not None:
        ImageDraw.Draw(image).rectangle([float(v) for v in box], outline=BOX_COLOR, width=width)
    return image


def png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()


def box_position(box, mask_np):
    """Describe where the box sits inside the FoV and how much of it it covers."""
    ys, xs = np.where(mask_np > .5)
    if xs.size == 0:
        height, width = mask_np.shape
        fx0, fy0, fx1, fy1 = 0, 0, width, height
        fov_area = float(width * height)
    else:
        fx0, fy0, fx1, fy1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        fov_area = float((mask_np > .5).sum())
    cx = ((box[0] + box[2]) / 2 - fx0) / max(fx1 - fx0, 1)
    cy = ((box[1] + box[3]) / 2 - fy0) / max(fy1 - fy0, 1)
    vertical = ('phía trên', 'giữa', 'phía dưới')[min(max(int(cy * 3), 0), 2)]
    horizontal = ('bên trái', 'giữa', 'bên phải')[min(max(int(cx * 3), 0), 2)]
    if vertical == horizontal == 'giữa':
        where = 'chính giữa khung nội soi'
    elif vertical == 'giữa':
        where = f'{horizontal} khung nội soi'
    elif horizontal == 'giữa':
        where = f'{vertical} khung nội soi'
    else:
        where = f'{vertical} {horizontal} khung nội soi'
    covered = float(((mask_np[int(box[1]):int(math.ceil(box[3])),
                              int(box[0]):int(math.ceil(box[2]))]) > .5).sum())
    return where, covered / max(fov_area, 1.)


def explanation_text(result, classes):
    """Turn the model's internal signals into a short Vietnamese explanation."""
    pred = result['prediction']
    name = classes[pred]
    prob = float(result['probabilities'][pred])
    sentences = [f'Mô hình dự đoán **{name}** với xác suất **{prob:.0%}**.']

    if result['box'] is not None:
        where, coverage = box_position(result['box'], result['mask'])
        sentences.append(f'Vùng được chọn để phóng to nằm ở {where}, '
                         f'chiếm khoảng {coverage:.0%} vùng ảnh hợp lệ.')

    no_local = result['no_local_probs']
    if no_local is not None:
        p_without = float(no_local[pred])
        change = prob - p_without
        if abs(change) < .01:
            sentences.append(f'Vùng phóng to gần như không làm thay đổi dự đoán cuối: tắt nhánh này đi, '
                             f'xác suất vẫn là {p_without:.0%}. Quyết định do nhánh toàn cục đưa ra.')
        else:
            verb = 'tăng' if change > 0 else 'giảm'
            to = 'lên' if change > 0 else 'xuống'
            sentences.append(f'Vùng phóng to làm xác suất {verb} từ {p_without:.0%} '
                             f'(chỉ nhánh toàn cục) {to} {prob:.0%}.')

    branch = result['branch_probs']
    if branch is not None:
        p_global, p_local = float(branch['global'][pred]), float(branch['local'][pred])
        sentences.append(f'Xét riêng từng nhánh qua head phụ: toàn ảnh cho {p_global:.0%}, '
                         f'vùng phóng to cho {p_local:.0%}.')
        if (p_global >= .5) != (p_local >= .5):
            sentences.append('Lưu ý: hai nhánh không đồng ý với nhau về nhãn.')

    if prob < NEAR_THRESHOLD:
        sentences.append('Xác suất gần ngưỡng 0,5 nên kết quả kém chắc chắn.')
    return ' '.join(sentences)


def main():
    st.set_page_config(page_title='Glance-Focus ViT v2', page_icon='🔎', layout='wide')
    st.title('Glance-Focus ViT v2')
    st.caption('Tải ảnh nội soi lên để nhận dự đoán, vùng mô hình chọn phóng to và phần giải thích.')
    st.info('Kết quả chỉ hỗ trợ nghiên cứu/tham khảo, không thay thế đánh giá của bác sĩ.')

    with st.sidebar:
        st.header('Cấu hình mô hình')
        checkpoint_text = st.text_input('Checkpoint', value=str(DEFAULT_CHECKPOINT))
        device_name = st.selectbox('Thiết bị', ('auto', 'cuda', 'mps', 'cpu'), index=0)

    with st.form('image_diagnosis'):
        uploaded_file = st.file_uploader('Chọn ảnh đầu vào', type=('jpg', 'jpeg', 'png', 'webp'))
        submitted = st.form_submit_button('Phân tích ảnh', type='primary')

    if not submitted:
        return
    if uploaded_file is None:
        st.warning('Vui lòng tải lên một ảnh trước khi phân tích.')
        return

    checkpoint_path = Path(checkpoint_text).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = (PROJECT_DIR / checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        st.error(f'Không tìm thấy checkpoint: {checkpoint_path}')
        return

    try:
        uploaded_image = Image.open(uploaded_file).convert('RGB')
        with st.spinner('Đang nạp mô hình và phân tích ảnh…'):
            model, checkpoint, selector, device = load_model(
                str(checkpoint_path), checkpoint_path.stat().st_mtime_ns,
                checkpoint_path.stat().st_size, device_name)
            processor = ImageProcessor(
                hi_size=int(checkpoint['hi_size']), training=False,
                use_fov=model.use_fov, front_end=checkpoint.get('front_end', False))
            hi, mask = processor(uploaded_image)
            result = predict_and_explain(model, hi, mask, selector, device)
    except Exception as exc:
        st.error(f'Không thể phân tích ảnh: {exc}')
        return

    classes = list(checkpoint['classes'])
    prediction = result['prediction']
    probabilities = result['probabilities']
    predicted_class = classes[prediction]
    probability = float(probabilities[prediction])

    # ---- 1. Kết luận ----------------------------------------------------
    st.subheader('Kết quả dự đoán')
    result_col, prob_col = st.columns(2)
    result_col.metric('Lớp dự đoán', predicted_class)
    prob_col.metric(f'Xác suất {predicted_class}', f'{probability:.1%}')
    if probability < NEAR_THRESHOLD:
        st.warning('Xác suất gần ngưỡng 0,5: kết quả kém chắc chắn, nên xem kỹ ảnh và phần giải thích.')
    with st.expander('Xác suất theo từng lớp', expanded=False):
        for class_name, class_prob in zip(classes, probabilities):
            st.write(f'{class_name}: {class_prob:.1%}')
            st.progress(float(class_prob))
        st.caption('Xác suất softmax từ cosine head, chưa được hiệu chỉnh (calibration).')

    # ---- 2. Giải thích bằng lời ----------------------------------------
    st.subheader('Giải thích bằng lời')
    st.markdown(explanation_text(result, classes))

    # ---- 3. Đóng góp của hai nhánh ---------------------------------------
    branch = result['branch_probs']
    no_local = result['no_local_probs']
    if branch is not None or no_local is not None:
        st.subheader('Đóng góp của hai nhánh')
        metrics = []
        if no_local is not None:
            p_without = float(no_local[prediction])
            metrics.append(('Tắt nhánh vùng phóng to', f'{p_without:.1%}'))
        metrics.append(('Sau hợp nhất (đầu ra cuối)', f'{probability:.1%}'))
        if branch is not None:
            metrics.append(('Head phụ · toàn ảnh', f"{float(branch['global'][prediction]):.1%}"))
            metrics.append(('Head phụ · vùng phóng to', f"{float(branch['local'][prediction]):.1%}"))
        for column, (label, value) in zip(st.columns(len(metrics)), metrics):
            column.metric(label, value)

    # ---- 4. Hình minh họa ----------------------------------------------
    st.subheader('Mô hình đã nhìn vào đâu')
    model_input = to_pil(hi)
    box = result['box']
    mask_np = result['mask']
    panels = [(draw_box(model_input, box), 'Ảnh sau tiền xử lý · khung vàng là vùng được phóng to')]
    if result['selector_map'] is not None:
        method = getattr(model, 'attention_method', 'attention')
        panels.append((draw_box(overlay(model_input, result['selector_map'], mask_np), box),
                       f'Bản đồ dùng để chọn vùng ({method})'))
    global_overlay = None
    if result['global_cam'] is not None:
        global_overlay = draw_box(overlay(model_input, result['global_cam'], mask_np), box)
        panels.append((global_overlay, 'Grad-CAM · nhánh toàn cục'))
    local_overlay = None
    if result['local_cam'] is not None and result['local_image'] is not None:
        local_overlay = overlay(result['local_image'], result['local_cam'], result['local_mask'])
        local_caption = ('Grad-CAM · vùng phóng to (theo head phụ của nhánh)'
                         if result['local_cam_from_aux'] else 'Grad-CAM · vùng phóng to')
        panels.append((local_overlay, local_caption))
    elif result['local_image'] is not None:
        panels.append((result['local_image'], 'Vùng phóng to'))

    for column, (image, caption) in zip(st.columns(len(panels)), panels):
        column.image(image, caption=caption, use_container_width=True)
    if box is not None:
        st.caption('Toạ độ khung trên ảnh đã tiền xử lý: '
                   f'x1={box[0]:.0f}, y1={box[1]:.0f}, x2={box[2]:.0f}, y2={box[3]:.0f} px.')
    if result['global_cam'] is None and result['local_cam'] is None:
        st.warning('Mô hình không tạo được heatmap cho ảnh này.')

    with st.expander('Ảnh gốc đã tải lên'):
        st.image(uploaded_image, use_container_width=True)

    download_cols = st.columns(2)
    if global_overlay is not None:
        download_cols[0].download_button('Tải Grad-CAM nhánh toàn cục', data=png_bytes(global_overlay),
                                         file_name='gradcam_global.png', mime='image/png')
    if local_overlay is not None:
        download_cols[1].download_button('Tải Grad-CAM vùng phóng to', data=png_bytes(local_overlay),
                                         file_name='gradcam_local.png', mime='image/png')


if __name__ == '__main__':
    main()
