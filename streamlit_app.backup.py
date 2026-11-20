"""Streamlit image-classification demo for the Glance-Focus ViT v2 checkpoint.

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
from PIL import Image
from torch.nn import functional as F

from processing import ImageProcessor
from train import device_for, load_checkpoint


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PROJECT_DIR / 'checkpoints' / 'seed 42' / 'best_v2.pt'
GRAD_METHODS = {'gradcam', 'gradrollout'}


@st.cache_resource(show_spinner=False)
def load_model(checkpoint_path, checkpoint_mtime_ns, checkpoint_size, device_name):
    """Cache model resources; file metadata invalidates the cache if replaced."""
    del checkpoint_mtime_ns, checkpoint_size
    device = device_for(device_name)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    if checkpoint.get('model_config', {}).get('roi_version') != 'v2':
        raise ValueError('Checkpoint này không phải Glance-Focus ViT v2.')
    model.eval().requires_grad_(False)

    selector = None
    if (getattr(model, 'attention_method', None) in GRAD_METHODS
            and model.fusion != 'global'):
        # Keep selector Grad-CAM hooks separate from the two classifier views.
        selector = copy.deepcopy(model).to(device).eval().requires_grad_(False)
        selector.attention_method = model.attention_method
    return model, checkpoint, selector, device


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
    cam = cam.detach()
    return cam / cam.amax(dim=(2, 3), keepdim=True).clamp_min(1e-8)


def map_local_cam(local_cam, box, height, width):
    """Place a 224x224 local CAM into its ROI coordinates on the full image."""
    local = torch.as_tensor(local_cam, dtype=torch.float32)[None, None]
    box = torch.as_tensor(box, dtype=torch.float32)
    x = torch.arange(width, dtype=torch.float32) + 0.5
    y = torch.arange(height, dtype=torch.float32) + 0.5
    gx = (x - box[0]) / (box[2] - box[0]).clamp_min(1e-8)
    gy = (y - box[1]) / (box[3] - box[1]).clamp_min(1e-8)
    grid = torch.stack(torch.meshgrid(2 * gx - 1, 2 * gy - 1, indexing='xy'), dim=-1)[None]
    full = F.grid_sample(local, grid, mode='bilinear', padding_mode='zeros',
                         align_corners=False)[0, 0].numpy()
    inside = ((y[:, None] >= box[1]) & (y[:, None] < box[3])
              & (x[None, :] >= box[0]) & (x[None, :] < box[2])).numpy()
    return full * inside


def combine_cams(global_cam, local_cam, box, height, width):
    local_full = map_local_cam(local_cam, box, height, width) if local_cam is not None and box is not None else None
    if global_cam is None:
        combined = local_full
    elif local_full is None:
        combined = global_cam
    else:
        combined = 0.5 * (global_cam + local_full)
    if combined is None:
        return None
    return combined / max(float(np.max(combined)), 1e-8)


def predict_with_gradcam(model, hi, mask, selector, device):
    """Predict the class and return its Grad-CAM, fused across model views."""
    activations = []

    def capture(_module, _inputs, output):
        if isinstance(output, (tuple, list)):
            output = output[0]
        activations.append(output)

    hook = target_layer(model).register_forward_hook(capture)
    try:
        model_input = hi.unsqueeze(0).to(device).detach().requires_grad_(True)
        mask = mask.unsqueeze(0).to(device)
        with torch.enable_grad():
            output = model(model_input, mask, selector)
            logits = output['logits']
            probabilities = logits.softmax(-1)[0].detach().cpu().numpy()
            prediction = int(logits.argmax(-1).item())
            score = logits[0, prediction]
            if not activations:
                raise RuntimeError('Grad-CAM hook did not capture ViT features.')
            gradients = torch.autograd.grad(score, tuple(activations), allow_unused=True)

        cams = [activation_cam(a, g, model) if g is not None else None
                for a, g in zip(activations, gradients)]
        height, width = hi.shape[-2:]
        global_cam = local_cam = None
        if model.fusion == 'local':
            if cams and cams[0] is not None:
                local_cam = F.interpolate(cams[0], (224, 224), mode='bilinear',
                                          align_corners=False)[0, 0].cpu().numpy()
        elif cams and cams[0] is not None:
            global_cam = F.interpolate(cams[0], (height, width), mode='bilinear',
                                       align_corners=False)[0, 0].cpu().numpy()
            if len(cams) > 1 and cams[1] is not None:
                local_cam = F.interpolate(cams[1], (224, 224), mode='bilinear',
                                          align_corners=False)[0, 0].cpu().numpy()

        box = output['boxes'][0].detach().cpu().numpy() if output.get('boxes') is not None else None
        cam = combine_cams(global_cam, local_cam, box, height, width)
        return prediction, probabilities, cam, box
    finally:
        hook.remove()


def cam_images(model_input, cam):
    """Create standalone heatmap and overlay PIL images."""
    heat = (plt.get_cmap('jet')(np.clip(cam, 0, 1))[..., :3] * 255).astype(np.uint8)
    heat_image = Image.fromarray(heat)
    base = np.asarray(model_input.convert('RGB'), dtype=np.float32) / 255
    overlay = np.clip(0.57 * base + 0.43 * (heat.astype(np.float32) / 255), 0, 1)
    overlay_image = Image.fromarray((overlay * 255).astype(np.uint8))
    return heat_image, overlay_image


def main():
    st.set_page_config(page_title='Glance-Focus ViT v2', page_icon='🔎', layout='wide')
    st.title('Glance-Focus ViT v2')
    st.caption('Tải ảnh nội soi lên để nhận dự đoán và phần giải thích Grad-CAM.')
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
            prediction, probabilities, cam, box = predict_with_gradcam(
                model, hi, mask, selector, device)
    except Exception as exc:
        st.error(f'Không thể phân tích ảnh: {exc}')
        return

    classes = list(checkpoint['classes'])
    predicted_class = classes[prediction]
    confidence = float(probabilities[prediction])
    st.subheader('Kết quả dự đoán')
    result_col, confidence_col = st.columns(2)
    result_col.metric('Lớp dự đoán', predicted_class)
    confidence_col.metric('Độ tin cậy của mô hình', f'{confidence:.1%}')

    with st.expander('Xác suất theo từng lớp', expanded=True):
        for class_name, probability in zip(classes, probabilities):
            st.write(f'{class_name}: {probability:.1%}')
            st.progress(float(probability))

    model_input = Image.fromarray((hi.permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8))
    if cam is None:
        st.warning('Mô hình không tạo được heatmap cho ảnh này.')
        st.image(uploaded_image, caption='Ảnh đã tải lên', use_container_width=True)
        return

    heat_image, overlay_image = cam_images(model_input, cam)
    st.subheader('Giải thích Grad-CAM')
    st.caption('Heatmap được nội suy về kích thước ảnh đầu vào; vùng nóng cho biết vùng ảnh có ảnh hưởng tới dự đoán. '
               'Với best_v2, heatmap global và local ROI được ghép thành một kết quả.')
    image_col, overlay_col, heat_col = st.columns(3)
    image_col.image(uploaded_image, caption='Ảnh tải lên', use_container_width=True)
    overlay_col.image(overlay_image, caption='Grad-CAM trên ảnh đã tiền xử lý', use_container_width=True)
    heat_col.image(heat_image, caption='Heatmap Grad-CAM', use_container_width=True)
    if box is not None:
        st.caption('ROI local trên ảnh đã tiền xử lý: '
                   f'x1={box[0]:.0f}, y1={box[1]:.0f}, x2={box[2]:.0f}, y2={box[3]:.0f} px.')

    overlay_buffer = io.BytesIO()
    overlay_image.save(overlay_buffer, format='PNG')
    st.download_button('Tải ảnh Grad-CAM overlay', data=overlay_buffer.getvalue(),
                       file_name='gradcam_overlay.png', mime='image/png')


if __name__ == '__main__':
    main()
