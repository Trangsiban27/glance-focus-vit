# ĐẶC TẢ KỸ THUẬT — GF-ViT LEVEL 1 (A1)

**Tên giai đoạn:** Global–Local Diagnostic & Complementarity Analysis  
**Dự án:** Glance–Focus Vision Transformer cho phân loại ảnh nội soi tai  
**Phiên bản:** 1.0 — 14/11/2026  
**Trạng thái:** Đặc tả triển khai và đánh giá; **không phải** kết quả thực nghiệm đã chạy  
**Phạm vi:** Chỉ Level 1 (A1); **không** thay ROI selector, fusion module hoặc mở rộng multi-ROI trong giai đoạn này.

---

## 1. Mục tiêu và câu hỏi nghiên cứu

Level 1 trả lời ba câu hỏi trước khi thay đổi kiến trúc:

1. **Global ViT** đang giải quyết tốt và thất bại ở những ảnh nào?
2. **Local ViT** có nhận diện đúng những ảnh mà Global ViT dự đoán sai không (tính *bổ sung*)?
3. **Gate Fusion** có tận dụng được những trường hợp Local đúng / Global sai, hay đang làm mất thông tin hữu ích của Local?

**Mục tiêu chính của A1 là chẩn đoán điểm nghẽn**, không nhằm bảo đảm chỉ số cao hơn. Kết quả quyết định ưu tiên của A2 (ROI Selection) hay A3 (Fusion).

### 1.1. Giả thuyết cần kiểm tra

- **H1 — Local hữu ích:** tồn tại một tỷ lệ đáng kể ảnh mà Global sai nhưng Local đúng.
- **H2 — ROI hạn chế:** nếu Global và Local cùng sai nhiều ảnh, vùng Local hiện tại có thể không bổ sung thông tin; đây mới là giả thuyết, cần kiểm tra ROI trước khi kết luận.
- **H3 — Fusion hạn chế:** nếu Local đúng khi Global sai nhưng Fusion vẫn sai, phương pháp hợp nhất hoặc quá trình học fusion cần được xem xét.
- **H4 — Local bị bỏ qua:** thành phần bổ sung từ Local vào vector fusion có cường độ rất nhỏ hoặc gần như không ảnh hưởng đến logits.

Các giả thuyết này **không thể kết luận chỉ từ Accuracy hoặc Macro-F1**.

---

## 2. Cơ sở hiện có và ranh giới kết luận

### 2.1. Tài liệu đầu vào

| Tài liệu | Vai trò |
|---|---|
| `Glance_Focus_ViT_model_spec.md` | Nguồn đặc tả kiến trúc, ROI attention rollout, gate, loss và huấn luyện |
| `report.md` | Báo cáo 3 seed × internal/external; mốc đối chiếu, **không phải** bằng chứng trực tiếp cho Glance–Focus đầy đủ |
| `preprocess_data.py` | Đối chiếu interface manifest/dữ liệu; **không** chỉnh sửa preprocessing trong A1 |

**Lưu ý quan trọng:** `report.md` chỉ định `best.pt` là mô hình ViT cải tiến có *gate fusion*; tài liệu chưa xác nhận checkpoint này đã triển khai đủ cơ chế *attention-rollout ROI → local crop → shared ViT* theo bản đặc tả Glance–Focus. Vì vậy phải xác minh cấu hình từ checkpoint/source trước khi coi nó là A1.3.

### 2.2. Mốc tham chiếu đã quan sát (chưa phải A1)

| Metric | ViT gốc, External mean | ViT gate cải tiến, External mean |
|---|---:|---:|
| ROC-AUC | 66,29% | 73,50% |
| Sensitivity (Abnormal) | 19,66% | 35,90% |
| Macro-F1 | 59,01% | 62,34% |

Các số trên chỉ để tham chiếu. **Không** sử dụng chúng làm ngưỡng bắt buộc hoặc để tinh chỉnh A1 trên external test.

---

## 3. Phạm vi và cấu hình phải giữ cố định

### 3.1. Kiến trúc nền

- Pretrained backbone: `vit_base_patch16_224.augreg2_in21k_ft_in1k`.
- Global view `x_g`: 224 × 224, encoder đầu ra `z_g` là CLS 768 chiều.
- Local view `x_l`: ROI từ bản ảnh gốc/độ phân giải cao, crop sau đó resize 224 × 224, encoder đầu ra `z_l` 768 chiều.
- ROI selector mặc định: attention rollout không phụ thuộc lớp, nhân FoV mask, làm mượt, cửa sổ vuông với tỷ lệ `r` **được chốt từ validation** theo đặc tả gốc; `stop-gradient` qua bước chọn hộp.
- Fusion A1.3: `g = sigmoid(MLP([LN(z_g); LN(z_l_used)]))` và `z_fused = z_g + s * g ⊙ z_l_used`.
- Gate hidden = 256; `s` là scalar học được, khởi tạo 0; Local-branch dropout `p=0.2` trong train, tắt lúc eval.
- Auxiliary CE của A1.3: `L_total = L_fusion + 0.3*(CE(aux_g, y) + CE(aux_l, y))`.
- `view_embed=False` (mặc định). Không thêm cross-attention, adapter, multiple ROI, trainable selector trong A1.

**Không suy đoán siêu tham số khác:** learning rate, batch size, epoch, weight decay, scheduler, classifier Linear/cosine, label smoothing, class weights, normalization, augmentation phải được **đọc từ cấu hình pipeline thực tế**, ghi vào `resolved_config.yaml` và giữ nhất quán trong những so sánh có thể đối chiếu. Những tham số không có trong tài liệu là *chưa xác nhận*, không tự nhận là thông số đã sử dụng.

### 3.2. Điều kiện công bằng

- Giữ nguyên các manifest và cách ánh xạ nhãn; `Abnormal` là lớp dương.
- Chạy 3 seed **42, 46, 49**; mỗi seed khởi tạo ngẫu nhiên của head/optimization độc lập nhưng cùng nguồn pretrained.
- Chọn best checkpoint bằng **validation Macro-F1**, cố định quy tắc hòa điểm trước khi chạy (ví dụ ROC-AUC validation, sau đó epoch sớm hơn).
- Cùng preprocessing, phương pháp augment, optimizer, schedule, số epoch tối đa, early stopping và tiêu chí lưu EMA (nếu dùng).
- Không dùng nhãn test để chọn ROI, chọn checkpoint, chỉnh threshold hay điều chỉnh các mức dropout/loss.
- Ngưỡng dự đoán nhị phân (mặc định 0,5 nếu dùng softmax/sigmoid chuẩn) được **khóa trước khi xem kết quả test**. Nếu điều chỉnh threshold, chỉ chọn bằng validation và ghi rõ.

---

## 4. Ba thí nghiệm bắt buộc

| ID | Tên | Input chính | Trainable model | Sản phẩm chính |
|---|---|---|---|---|
| **A1.1** | Global Only | Toàn ảnh 224 | ViT + classifier | `p_global`, logits, CLS |
| **A1.2** | Local Only | ROI 224 từ selector A1.1 **frozen** | ViT + classifier Local (selector đóng băng) | `p_local`, logits, ROI |
| **A1.3** | Global + Local + Gate | Toàn ảnh + ROI **online** | Shared ViT + gate + 3 heads | `p_fused`, `p_aux_global`, `p_aux_local`, gate, `s`, ROI |

> **Khác biệt bắt buộc phải báo cáo:** A1.2 dùng selector A1.1 đóng băng như E2 trong đặc tả gốc; A1.3 dùng selector online theo 1-a như E4. Vì thế A1.1/A1.2/A1.3 là đối chứng giá trị nhánh nhưng **không hoàn toàn kiểm soát cách sinh ROI**. Để chẩn đoán fusion một cách công bằng, phải so sánh `aux_global`, `aux_local`, `fused` **trên cùng checkpoint A1.3, cùng ROI và cùng ảnh** (xem mục 7). Một thí nghiệm `A1.3-F` dùng frozen selector là đối chứng bổ sung nếu cần xác định tác động của cơ chế online/frozen; không thuộc ba mô hình bắt buộc.

### 4.1. A1.1 — Global Only

**Kiến trúc**

```text
Ảnh → Global 224×224 → ViT-B/16 → CLS z_g → Head → logits_global
```

**Yêu cầu**

1. Train E1 với các seed tương ứng, chọn checkpoint theo validation.
2. Lưu checkpoint, model config, seed và danh sách image IDs.
3. Để phục vụ A1.2, đóng băng một bản checkpoint A1.1 làm selector cho chính seed đó.
4. Trích xuất logits và xác suất lớp Abnormal trên từng ảnh theo đúng thứ tự image ID.

**Đầu ra:** `a1_1_global_predictions_{split}_seed{seed}.csv` và `selector_seed{seed}.pt`.

### 4.2. A1.2 — Local Only

**Kiến trúc**

```text
Ảnh ─► Selector A1.1 đã đóng băng ─► Attention rollout ─► ROI
Ảnh gốc + ROI ─► Local crop 224×224 ─► ViT-B/16 ─► Head ─► logits_local
```

**Yêu cầu**

1. Selector chỉ được train từ train split, checkpoint chọn bằng validation; tất cả trọng số selector `eval()` + `requires_grad=False`.
2. Sinh ROI cho train/validation bằng selector frozen; test chỉ sinh sau khi đã khóa cấu hình; lưu toạ độ ROI, tên selector và mã image ID.
3. Train một **classifier Local riêng** trên crop, khởi tạo cùng pretrained ViT như A1.1; không truyền gradient vào selector.
4. Nếu dùng ROI jitter cho train theo đặc tả, khóa cùng cấu hình jitter; validation/test không jitter.
5. Trình bày rõ nguy cơ selector đã được train trên chính ảnh train rồi tạo ROI train (in-sample boxes). Có thể thử cross-fitting trên train sau A1 nếu kết quả cho thấy selection bias; **không** thay đổi giữa chừng trong thí nghiệm chính.

**Chi phí:** Với selector frozen chạy lúc suy luận trực tiếp cần một lượt Global cho chọn ROI + một lượt Local; nếu ROI đã được cache, phép đo thời gian phải báo rõ *đang tính có/không có selector*. Không được báo latency chỉ cho Local encoder như latency end-to-end.

**Đầu ra:** `a1_2_local_predictions_{split}_seed{seed}.csv`, `boxes_{split}_seed{seed}.json`, ROI overlays.

### 4.3. A1.3 — Global + Local + Gate (Glance–Focus dự kiến)

**Kiến trúc**

```text
                     ┌──────── z_g ───────────────────────────┐
Ảnh → Global 224 → Shared ViT ──► rollout → ROI               │
Ảnh gốc + ROI → Local 224 → Shared ViT → z_l                   │
                     │               │                        │
                     │             Gate([z_g,z_l])            │
                     └──────────────────┬─────────────────────┘
                                    z_fused → Head → p_fused
                           (aux_g trên z_g; aux_l trên z_l)
```

**Yêu cầu**

1. Dùng shared ViT đúng đặc tả 1-a, chọn ROI trực tuyến mỗi lần forward bằng attention Global và `stop-gradient` trên chọn hộp.
2. Train với auxiliary loss weight `0.3`, branch dropout `0.2` theo bản thiết kế; không thay đổi trong A1.
3. Khi eval, `model.eval()`; **tắt** branch dropout, ROI jitter và stochastic augmentation.
4. Bổ sung **chế độ phân tích** (không thay logits fusion) để xuất đồng thời:
   - `logits_fused` / `prob_fused`;
   - `logits_aux_global` / `prob_aux_global`;
   - `logits_aux_local` / `prob_aux_local`;
   - `z_global`, `z_local` hoặc các thống kê norm cần thiết;
   - `g` 768 chiều, `s`, ROI B.
5. Chú ý trong pseudocode đặc tả gốc, `aux_g` và `aux_l` chỉ xuất khi `self.training`. **Sửa chế độ phân tích eval để tính các auxiliary logits nhưng không tính training loss**, tuyệt đối không bật `model.train()` để lấy logits phụ.
6. Ghi rõ classifier của head chính và head phụ (nếu Linear/cosine khác nhau); chênh lệch giữa các head còn có thể đến từ thiết kế head, không hoàn toàn do ROI.

**Đầu ra:** `a1_3_fusion_predictions_{split}_seed{seed}.csv`, `gate_stats_{split}_seed{seed}.csv`, ROI overlays, checkpoint và config.

---

## 5. Quy trình triển khai tuần tự

### Phase 0 — Kiểm tra hiện trạng (bắt buộc)

- [ ] Xác minh `best.pt` có thực sự là shared-backbone Glance–Focus + attention rollout + local crop hay là một kiến trúc gate khác; đọc checkpoint keys và model config.
- [ ] Xác minh đường dẫn train/valid/internal/external và cột mapping nhãn trong CSV.
- [ ] Cố định `resolved_config.yaml`, phiên bản thư viện và Git commit (nếu có).
- [ ] Xác nhận đầu vào Global/Local sau augment là hai view của **cùng ảnh**, cùng hệ tọa độ.
- [ ] Kiểm tra đơn vị tọa độ ROI: ảnh gốc, ảnh high-res và `roi_align` khớp nhau.
- [ ] Kiểm tra attention hook khi `timm` dùng fused attention; rollout có tensor đúng `[B, 14, 14]`.

### Phase 1 — Baseline và selector

- [ ] Train/khôi phục A1.1 theo 3 seed.
- [ ] Xuất dự đoán Validation; lưu `selector_seed*.pt`.
- [ ] Đóng băng selector, sinh ROI/overlay cho train/valid theo mỗi seed.
- [ ] Kiểm tra ROI nằm trong ảnh, không rỗng, vị trí crop hợp lý (chỉ kiểm tra kỹ thuật, không chỉnh bằng nhãn test).

### Phase 2 — Local Only

- [ ] Train A1.2 theo 3 seed với đúng selector cùng seed.
- [ ] Kiểm tra forward khi không có nhánh Global classifier tham gia loss.
- [ ] Xuất dự đoán và đối chiếu từng image ID với A1.1.

### Phase 3 — Online Gate

- [ ] Train A1.3 theo 3 seed.
- [ ] Xuất logits phụ trong `eval()` qua diagnostic mode.
- [ ] Log giá trị `s`, phân bố gate, tỷ lệ norm đóng góp Local theo epoch và theo ảnh.
- [ ] Xuất dự đoán và ROI theo image ID.

### Phase 4 — Phân tích và khóa Level 1

- [ ] Thực hiện các bảng 2×2; phân tích trường hợp Local cứu Global và Fusion bỏ lỡ.
- [ ] Tổng hợp mean ± sample SD qua 3 seed.
- [ ] Đối chiếu hiệu quả validation trước khi đưa ra nhánh công việc cho A2/A3.
- [ ] Khóa cấu hình rồi mới chạy test theo protocol mục 9.
- [ ] Viết `A1_FINDINGS.md` gồm bằng chứng, kết luận, hạn chế và đề xuất Level kế tiếp.

---

## 6. Quy tắc xuất dự đoán và nhận dạng ảnh

**Mỗi dòng tương ứng một ảnh gốc ở một split, một seed và một model.** Không nối bảng bằng index sau DataLoader vì có thể shuffle; phải join bằng image ID ổn định.

### 6.1. CSV chung

```csv
image_id,split,seed,model,y_true,prob_abnormal,y_pred,correct,checkpoint_id
img_0001,valid,42,A1.1,1,0.812,1,1,a1_1_seed42
```

Đối với A1.3, tạo thêm:

```csv
image_id,split,seed,y_true,p_global_aux,p_local_aux,p_fused,pred_global_aux,pred_local_aux,pred_fused,roi_x1,roi_y1,roi_x2,roi_y2,s,gate_mean,gate_std,local_contrib_ratio
img_0001,valid,42,1,0.74,0.93,0.88,1,1,1,42,58,242,258,0.12,0.48,0.11,0.07
```

**Các con số trong hai dòng CSV là ví dụ minh họa cấu trúc, không phải kết quả thực tế.** Bảo đảm `image_id`, `split`, `seed`, `y_true` là duy nhất/nhất quán trong từng file; trường `p_*` là xác suất lớp Abnormal, không phải confidence của lớp được dự đoán.

### 6.2. ROI metadata

- Dùng pixel coordinates nhất quán `x1,y1,x2,y2` trong **hệ tọa độ x_hi**; ghi thêm `input_width`, `input_height`.
- Nếu export boxes dùng pixel gốc thì phải có trường `coordinate_space="original"` và cơ chế transform đúng sang `x_hi`.
- Lưu `selector_checkpoint_id`, `crop_ratio`, `fov_enabled`, `rollout_method`, `roi_jitter_applied`.
- Validation/test luôn `roi_jitter_applied=false`.
- ROI overlay nên kèm ảnh toàn cảnh, map rollout, hộp crop, crop cuối cùng và nhãn/dự đoán (nhãn chỉ dùng **để phân tích sau inference**, không làm đầu vào selector).

---

## 7. Phân tích A1 — bắt buộc để tìm điểm nghẽn

### 7.1. Ma trận 2×2 Global × Local

Dùng cặp dự đoán **A1.1 và A1.2** trên cùng ảnh/split/seed; tính thêm riêng ma trận cặp **A1.3 `aux_global` và `aux_local`** để tránh nhiễu do selector khác nhau.

| | Local đúng | Local sai |
|---|---:|---:|
| **Global đúng** | `n_G1_L1` | `n_G1_L0` |
| **Global sai** | `n_G0_L1` — Local cứu Global | `n_G0_L0` — Cả hai sai |

Định nghĩa:

- `local_rescue_rate = n_G0_L1 / (n_G0_L1 + n_G0_L0)` — tỷ lệ lỗi Global có khả năng được Local sửa; chỉ tính khi mẫu số > 0.
- `local_disagreement_rate = (n_G0_L1 + n_G1_L0) / N` — mức bất đồng.
- `both_wrong_rate = n_G0_L0 / N` — lỗi mà cả hai nhánh đều không giải quyết.

**Diễn giải:** `local_rescue_rate` cao chứng minh có tính bổ sung ở mức **dự đoán**, nhưng chưa tự chứng minh ROI attention tốt hơn một ROI khác; câu hỏi ROI thuộc A2.

### 7.2. Fusion có sử dụng tính bổ sung không?

Phân tích trong cùng checkpoint **A1.3**:

| Tình huống | Điều cần đếm |
|---|---|
| Aux Global sai, Aux Local đúng, Fusion đúng | `fusion_rescue_success` |
| Aux Global sai, Aux Local đúng, Fusion sai | `fusion_missed_rescue` |
| Aux Global đúng, Aux Local sai, Fusion sai | `fusion_harmed_global` |
| Aux Global sai, Aux Local sai, Fusion đúng | `fusion_synergy` (cần kiểm tra kỹ) |

Các tỷ lệ:

```text
fusion_rescue_efficiency = fusion_rescue_success /
    (fusion_rescue_success + fusion_missed_rescue)

fusion_harm_rate = fusion_harmed_global /
    count(aux_global_correct AND aux_local_wrong)
```

Chỉ tính tỷ lệ khi mẫu số > 0; luôn in cả tử số và mẫu số. Phân tích thêm theo lớp Normal/Abnormal; không chỉ dựa vào tổng gộp.

### 7.3. Gate có bỏ qua Local không?

Với từng ảnh, trong chế độ **eval**:

```text
delta_local = s * g ⊙ z_l
local_contrib_ratio = ||delta_local||_2 / (||z_g||_2 + 1e-8)
logit_delta = Head(z_g + delta_local) - Head(z_g)
```

- Theo dõi `s` có dấu âm/dương và biến thiên qua epoch. **`s` gần 0 là dấu hiệu** Local ít đóng góp, không tự nó kết luận Local vô dụng.
- Báo cáo `mean(g)`, percentiles 5/50/95, tỷ lệ gate gần 0 hoặc 1, và `local_contrib_ratio` theo seed và lớp.
- `logit_delta` phải được tính với **chính fusion head đang huấn luyện** và `model.eval()`, không thay một classifier khác.
- Phân tích trường hợp `local_contrib_ratio` nhỏ nhưng Local auxiliary dự đoán đúng khi Global auxiliary sai.
- Nếu có thể, kiểm thử ablation **chỉ khi đánh giá**: đặt `z_l=0` ở đường fusion nhưng giữ head, weights, normalization và mọi tham số không đổi; tính thay đổi các metrics. Đây là *diagnostic inference ablation*, không phải model huấn luyện thứ tư.

### 7.4. ROI inspection

Xuất tối thiểu các overlay có phân nhóm:

1. Global sai, Local đúng;
2. Global sai, Local sai;
3. Global đúng, Local sai;
4. Global và Local đúng nhưng Fusion sai (nếu có).

Trong mỗi nhóm, sắp xếp theo mức confidence và chọn ví dụ tiêu biểu **trên validation để quyết định hướng phát triển**. Không đổi ROI selector dựa trên việc soi nhãn external test.

---

## 8. Chỉ số và kiểm định

### 8.1. Metrics bắt buộc

- Accuracy; Macro-F1 (primary checkpoint selection); Balanced Accuracy.
- Sensitivity = Recall(Abnormal); Specificity = Recall(Normal).
- ROC-AUC; Precision, Recall và F1 theo từng lớp; Confusion Matrix.
- PR-AUC (Average Precision cho Abnormal) nếu xác suất dự đoán khả dụng.
- Mean ± **sample SD** (`ddof=1`) qua seed; luôn công bố kết quả từng seed.

### 8.2. Phân tích cặp và khoảng tin cậy

- Đối chiếu McNemar exact cho correctness của A1.1–A1.2, A1.1–A1.3 **trên cùng ảnh và cùng seed**.
- Bootstrap paired theo image ID để ước lượng CI 95% của chênh lệch Macro-F1 / Balanced Accuracy; với ROC-AUC có thể dùng paired bootstrap hoặc paired DeLong.
- Tránh tuyên bố *có ý nghĩa thống kê* chỉ dựa trên mean ± SD của 3 seed.
- Khi báo cáo kết quả các seed trên cùng tập test, không coi dự đoán từ 3 seed như 3 tập bệnh nhân độc lập.

---

## 9. Chính sách Validation, Internal Test và External Test

1. **Train:** chỉ học tham số, selector và tạo crop train theo protocol.
2. **Validation:** chọn checkpoint, r và các cấu hình đã chốt ở giai đoạn thiết kế; đánh giá A1, quyết định A2/A3.
3. **Internal Test:** chỉ báo cáo sau khi cấu hình A1 đã khóa. Do đã có báo cáo trước, mọi phép tái sử dụng internal test để so sánh là phân tích lặp, cần ghi rõ.
4. **External Test:** chỉ chạy đánh giá khi không còn điều chỉnh A1. Vì các chỉ số external đã được xem trong `report.md`, bộ test này **không còn là holdout hoàn toàn chưa được quan sát** cho cả chương trình nghiên cứu. Kết quả Level 1 trên external là *đánh giá khám phá/xác nhận hạn chế*; muốn tuyên bố mạnh về generalization cần thêm cohort/holdout độc lập chưa sử dụng để ra quyết định.
5. Không sửa architecture/hyperparameter/threshold sau khi xem kết quả test rồi báo lại như lần đánh giá độc lập.

---

## 10. Cấu trúc thư mục đầu ra đề nghị

```text
experiments/
└── gf_vit_level1/
    ├── README.md
    ├── resolved_config.yaml
    ├── environment.txt
    ├── manifests/                  # bản ghi định danh, không sửa nhãn
    ├── seed42/
    │   ├── a1_1_global/
    │   │   ├── best.pt
    │   │   ├── predictions_valid.csv
    │   │   └── metrics_valid.json
    │   ├── a1_2_local/
    │   │   ├── best.pt
    │   │   ├── boxes_train.json
    │   │   ├── boxes_valid.json
    │   │   └── predictions_valid.csv
    │   └── a1_3_gate_online/
    │       ├── best.pt
    │       ├── predictions_valid.csv
    │       ├── gate_stats_valid.csv
    │       └── overlays/
    ├── seed46/                     # tương tự
    ├── seed49/                     # tương tự
    ├── analysis/
    │   ├── per_seed_metrics.csv
    │   ├── mean_std.csv
    │   ├── pairwise_global_local.csv
    │   ├── fusion_rescue_analysis.csv
    │   ├── gate_summary.csv
    │   ├── confusion_matrices/
    │   └── roi_case_studies/
    └── A1_FINDINGS.md
```

**Định dạng lưu:** Mỗi checkpoint cần chứa `seed`, `model_config`, `pretrained_id`, `label_map`, `selector_id` (nếu có), `epoch`, `metric_on_validation`, `use_ema`, `input_size`, `crop_ratio` và `git_commit` khi có.

---

## 11. Gợi ý interface mã nguồn (không phải API hiện có)

```python
# Diagnostic mode thêm vào A1.3; không thay đường dự đoán chính.
@torch.no_grad()
def predict_diagnostics(model, x_global, x_high_res, image_ids):
    model.eval()
    out = model.forward_diagnostics(x_global, x_high_res)
    # forward_diagnostics PHẢI chạy auxiliary heads khi eval,
    # KHÔNG bật dropout, KHÔNG dùng nhãn thật để chọn ROI.
    return {
        "image_ids": image_ids,
        "logits_fused": out["logits_fused"],
        "logits_aux_global": out["logits_aux_global"],
        "logits_aux_local": out["logits_aux_local"],
        "boxes": out["boxes"],
        "gate": out["gate"],
        "s": out["s"],
        "local_contrib_ratio": out["local_contrib_ratio"],
    }
```

**Lưu ý:** `forward_diagnostics` là **hàm cần bổ sung**, không phải API đang tồn tại. Nếu cần xuất ảnh/chỉ số cho nhiều mẫu, nên lưu thống kê gọn vào CSV và chỉ lưu toàn bộ 768 giá trị gate/tensor đặc trưng ở dạng NPZ/Parquet khi thực sự cần phân tích.

### 11.1. Kiểm thử kỹ thuật tối thiểu

- [ ] `A1.1` trả logits `[B, C]` và có thể eval không cần tạo local view.
- [ ] `A1.2` không tính gradient cho selector.
- [ ] `A1.3` dùng **một** backbone ViT với shared weights qua hai lượt forward.
- [ ] Trong eval, `forward_diagnostics(...)["logits_fused"]` khớp với `forward(...)` (sai số floating point cho phép) trên cùng input/checkpoint.
- [ ] Đặt `s=0` **trong cùng checkpoint A1.3** thì fusion output bằng `Head(z_g)` của **A1.3**, không mặc nhiên bằng checkpoint A1.1 độc lập.
- [ ] Boxes nằm trong ảnh và crops không rỗng; không jitter khi eval.
- [ ] Join trên `image_id` khớp 100% giữa các mô hình; không trùng, không thiếu.
- [ ] Đổi thứ tự DataLoader không làm thay đổi bảng phân tích trên cùng seed/checkpoint.
- [ ] Các metric tái lập từ prediction CSV khớp metrics JSON.
- [ ] Bật diagnostic mode không làm thay đổi logits, chỉ bổ sung đầu ra để phân tích.

---

## 12. Quy tắc kết luận và chọn hướng sau A1

Không đặt một ngưỡng thành công tùy tiện như "phải tăng 5%". Chọn hướng bằng **mẫu lỗi, khả năng sửa lỗi, độ ổn định qua seed và khoảng tin cậy**, chủ yếu trên validation.

| Bằng chứng từ A1 | Diễn giải ưu tiên | Bước tiếp theo |
|---|---|---|
| Local có ít ảnh cứu Global, và cả hai cùng sai nhiều | ROI/đặc trưng local chưa bổ sung đủ | **A2** — thử ROI selector tốt hơn |
| Local cứu Global thường xuyên nhưng Fusion vẫn sai | Fusion chưa khai thác tín hiệu bổ sung | **A3** — thử Concat/Cross-Attention (có thể kiểm tra gate trước) |
| Local cứu Global, Fusion cũng cứu phần lớn | Gate có thể đã đủ; không cần phức tạp hóa ngay | Giữ A1.3 làm ứng viên, cân nhắc A4 sau |
| Khi tắt Local, dự đoán gần như không đổi, `local_contrib_ratio` thấp | Có khả năng Local bị gate bỏ qua | Kiểm tra optimization/gate, rồi A3 |
| Cả Global và Local tốt trên validation nhưng cùng giảm ở test | A1 xác nhận vấn đề tổng quát hóa chưa được giải quyết; không đủ bằng chứng để quy cho ROI hay fusion | Trình bày giới hạn và tiếp tục kiểm chứng có kiểm soát |

**Điều kiện hoàn thành Level 1:**

- [ ] Có checkpoint + metrics + predictions cho A1.1/A1.2/A1.3 ở đủ 3 seed.
- [ ] Có hai ma trận 2×2 cho từng seed: cặp standalone và cặp auxiliary trong A1.3.
- [ ] Có thống kê `fusion_rescue_efficiency`, `fusion_harm_rate`, gate và tỷ lệ đóng góp Local.
- [ ] Có overlays ví dụ đúng/sai phân nhóm; minh họa không làm thay đổi selector.
- [ ] Có báo cáo mean ± SD, CI cho các so sánh chính và công bố chi phí suy luận.
- [ ] Có `A1_FINDINGS.md` kết luận **cần sửa ROI, cần sửa Fusion, cả hai, hay chưa đủ bằng chứng**.

### 12.1. Mẫu kết luận bắt buộc trong `A1_FINDINGS.md`

```markdown
# Level 1 — Findings

## 1. Cấu hình đã khóa
- Pretrained / ROI / augmentation / head / loss / seeds / best checkpoint criterion

## 2. Kết quả 3 seed
- Bảng Accuracy, Macro-F1, Balanced Accuracy, Sensitivity, Specificity, ROC-AUC

## 3. Complementarity
- Global wrong & Local correct: ... / ... (...%)
- Both wrong: ... / ... (...%)
- Fusion rescued: ... / ... (...%)
- Fusion missed rescue: ... / ... (...%)

## 4. Gate usage
- s: ...
- mean(g): ...
- local contribution ratio: ...
- Fusion output thay đổi thế nào khi tắt Local: ...

## 5. Giới hạn và đối chứng
- A1.2 frozen selector vs A1.3 online selector
- Auxiliary head vs main head
- Tình trạng external test đã được quan sát trước đây

## 6. Quyết định
- Ưu tiên A2 / A3 / giữ A1 / chưa đủ bằng chứng
- Bằng chứng dẫn tới quyết định
```

---

## 13. Những việc tuyệt đối không làm trong Level 1

- Không đưa nhãn thật hoặc nhãn test vào quá trình chọn vùng.
- Không tinh chỉnh ROI, chọn threshold hay chọn best seed bằng external test.
- Không thay đồng thời selector, fusion và số ROI, vì sẽ mất khả năng xác định điểm nghẽn.
- Không so A1.2 Local-only **chỉ tính Local encoder** với A1.3 **tính cả Global + Local** như thể có cùng chi phí suy luận.
- Không kết luận "attention crop sai" chỉ vì Local-only kém; phải có ablation vùng ở Level 2.
- Không kết luận "gate vô dụng" chỉ vì `s` nhỏ; cần xem delta-logits, ablation và các trường hợp cần Local.
- Không coi 3 seed là 3 cohort độc lập để tăng giả tạo độ tin cậy thống kê.

---

## 14. Kết quả mong muốn của Level 1

Một báo cáo kỹ thuật **có thể trả lời bằng chứng cụ thể**:

> Trong Glance–Focus ViT, nhánh Local có sửa được lỗi của Global hay không? Nếu có, Gate Fusion đã tận dụng phần bổ sung đó đến mức nào? Nếu không, phải ưu tiên sửa bước chọn ROI trước khi tăng độ phức tạp của Fusion.

**Sau Level 1**, chỉ lựa chọn thay đổi cho Level 2/3 dựa trên kết quả phân tích, không mặc định kiến trúc phức tạp hơn sẽ tốt hơn.
