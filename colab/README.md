# Train Glance–Focus ViT trên Google Colab

`train.py` là bản tự chứa để copy/paste: model, preprocessing, dataset và DataLoader đều nằm trong cùng file. Không cần mở hoặc chạy notebook `.ipynb`, cũng không cần import các file `.py` khác.

1. Giải nén dữ liệu để có `/content/data/Datos/train.csv` và `/content/data/Datos/valid.csv` cùng các thư mục ảnh tương ứng.
2. Chọn GPU trong **Runtime → Change runtime type → GPU**.
3. Mở `colab/train.py`, copy toàn bộ file vào một cell Colab và chạy.

Các thiết lập nằm đầu file. Mặc định dùng:

- `TRAIN_CSV = /content/data/Datos/train.csv`
- `VAL_CSV = /content/data/Datos/valid.csv`
- `TASK = binary` (Normal vs Abnormal), `FUSION = gate` (E4), `BATCH_SIZE = 2`, `EPOCHS = 30`, patience `5`
- Checkpoint và lịch sử train lưu tại `/content/runs/glance_focus_gate/`

Mọi nhãn trong CSV khác `Normal` được gộp thành `Abnormal` trước khi tính loss. Nếu cấu trúc dữ liệu khác, sửa `TRAIN_CSV` và `VAL_CSV` trước khi chạy. Đổi `OUTPUT_DIR` nếu thư mục kết quả đã tồn tại. Để chạy E2, đặt `FUSION = 'local'` và điền `SELECTOR_CHECKPOINT` đến checkpoint E1.

Script tự cài `timm`, `scipy`, `scikit-learn` khi thiếu; Colab cung cấp sẵn `torch` và `torchvision`. Cần Internet ở lần đầu để tải trọng số pretrained ViT-B/16. Nếu GPU hết bộ nhớ, giảm `BATCH_SIZE`.

Các file modular như `evaluate.py`, `model.py`, `dataset.py`, `dataloader.py`, `processing.py` và `label_mapping.py` vẫn có sẵn để dùng riêng; bản `train.py` copy/paste không phụ thuộc vào chúng.
