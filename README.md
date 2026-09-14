# Hardware-Aware Mixed-Precision NAS for ECG Arrhythmia Classification with FPGA

## 1. Giới thiệu

Dự án xây dựng hệ thống phân loại rối loạn nhịp tim ECG theo 5 lớp:

- `N` — Normal
- `L` — Left bundle branch block beat
- `R` — Right bundle branch block beat
- `V` — Premature ventricular contraction
- `A` — Atrial premature beat

Hệ thống kết hợp:

- Xử lý dữ liệu ECG từ MIT-BIH.
- Neural Architecture Search (NAS) để tìm kiến trúc CNN 1D phù hợp.
- Hardware-Aware objective dựa trên độ chính xác, số tham số và MACs.
- Mixed Precision INT8 / INT4.
- Quantization-Aware Training (QAT).
- Integer-only deployment.
- Bộ tăng tốc FPGA trên PYNQ-Z2.
- Dashboard trên laptop để xem ECG, chọn heartbeat, chạy inference, chạy NAS và đánh giá toàn bộ tập test.

Tên đề tài:

> **Hardware-Aware Mixed-Precision NAS for ECG Arrhythmia Classification with a Systolic-in-Systolic FPGA Accelerator**

---

## 2. Kiến trúc hệ thống

Luồng chính:

```text
MIT-BIH ECG
    ↓
Preprocessing
    ↓
Baseline Model
    ↓
NAS Search
    ↓
Train từng Candidate
    ↓
Chọn Best Candidate
    ↓
Mixed-Precision Search
    ↓
QAT
    ↓
Calibration
    ↓
Integer Deployment
    ↓
Laptop / PYNQ-Z2 FPGA
    ↓
Dashboard
```

Luồng inference FPGA:

```text
Dashboard trên Laptop
        ↓
HTTP
        ↓
PYNQ ARM
        ↓
DMA
        ↓
FPGA Accelerator
        ↓
PYNQ ARM
        ↓
Kết quả dự đoán
        ↓
Dashboard
```

---

## 3. Dữ liệu

Dataset sử dụng MIT-BIH Arrhythmia Database.

Mỗi heartbeat sau preprocessing có dạng:

```text
320 samples × 1 channel
```

Preprocessing chính:

1. Đọc record ECG.
2. Xác định tâm heartbeat.
3. Crop/pad về 320 samples.
4. Chuẩn hóa z-score theo từng segment.
5. Gán nhãn `N/L/R/V/A`.

Tập test mặc định:

```text
datasets/splits/test.csv
```

---

## 4. NAS

NAS sinh nhiều candidate kiến trúc khác nhau.

Mỗi candidate được train một số epoch ngắn để đánh giá:

```text
Candidate #0
→ train
→ validation
→ Macro F1
→ Accuracy
→ Params
→ MACs
```

Sau đó hệ thống chọn candidate theo objective.

Các objective hiện hỗ trợ:

```text
Balanced
Highest Macro F1
Efficient
```

### Balanced

Cân bằng giữa:

- Macro F1.
- Params.
- MACs.

### Highest Macro F1

Chọn candidate có Macro F1 validation cao nhất.

### Efficient

Ưu tiên candidate có MACs thấp trong nhóm có F1 gần candidate tốt nhất.

---

## 5. Checkpoint của NAS Candidate

Từ dashboard V5.19, mỗi candidate của một NAS run mới được lưu riêng:

```text
artifacts/
└── dashboard_nas_jobs/
    └── <job_id>/
        ├── candidates.json
        ├── best_candidate.json
        │
        ├── candidate_000/
        │   ├── config.json
        │   ├── checkpoint.weights.h5
        │   ├── history.json
        │   └── metrics.json
        │
        ├── candidate_001/
        │   ├── config.json
        │   ├── checkpoint.weights.h5
        │   ├── history.json
        │   └── metrics.json
        └── ...
```

Ý nghĩa:

```text
config.json
→ kiến trúc candidate.

checkpoint.weights.h5
→ best weights theo validation loss.

history.json
→ lịch sử train.

metrics.json
→ Accuracy, Macro F1, Params, MACs...
```

Candidate nên được chọn khi:

```text
NAS Run = completed
Candidate = BEST hoặc được chọn thủ công
Checkpoint = SAVED
```

Các NAS run cũ trước V5.19 có thể hiện:

```text
Checkpoint: NOT SAVED
```

và không có weights riêng để inference trực tiếp.

---

## 6. Mixed Precision

Sau khi chọn kiến trúc NAS, hệ thống mới chạy Mixed Precision.

Luồng:

```text
Selected NAS Candidate
        ↓
Sensitivity Analysis
        ↓
Mixed-Precision Search
        ↓
Khóa INT8 / INT4
```

Cấu hình deployment hiện tại có 4 weighted layers INT4:

```text
block2_b3_sepconv
block3_b2_sepconv
block4_b1_sepconv
block4_b2_sepconv
```

Các weighted layer còn lại dùng INT8.

---

## 7. QAT

QAT là:

```text
Quantization-Aware Training
```

QAT không chọn layer nào INT4/INT8.

Nhiệm vụ của QAT là train lại model trong điều kiện mô phỏng lượng tử hóa để model thích nghi với sai số INT4/INT8.

Luồng:

```text
Mixed-Precision Search
        ↓
best_layer_precision_config.json
        ↓
QAT
        ↓
best_mixed_precision_qat.keras
```

---

## 8. Calibration và deployment

Sau QAT:

```text
calibrate_integer_graph.py
```

được dùng để tính scale lượng tử hóa.

Deployment cuối hiện tại:

```text
artifacts/final_w4a4_p99_9/
```

Ý nghĩa tên:

```text
W4
→ Weight 4-bit ở các layer INT4

A4
→ Activation 4-bit ở các layer INT4

p99_9
→ calibration dùng percentile 99.9%
```

Đây là model deployment đã khóa để chạy Integer Reference và PYNQ FPGA Hybrid.

---

## 9. FPGA

Board:

```text
PYNQ-Z2
Device: xc7z020clg400-1
```

Clock hiện tại:

```text
40 MHz
```

FPGA accelerator hỗ trợ các weighted operation như:

- Conv1D.
- Depthwise Conv1D.
- Pointwise Conv1D.
- Projection Conv.
- Dense dạng K=1.

Giao tiếp:

```text
ARM
↓
AXI DMA
↓
FPGA Accelerator
↓
AXI DMA
↓
ARM
```

Tài nguyên implementation hiện tại:

```text
LUT   : 18,248 / 53,200   = 34.30%
FF    :  6,536 / 106,400  =  6.14%
BRAM  :      2 / 140      =  1.43%
DSP   :      7 / 220      =  3.18%
```

---

## 10. Chạy PYNQ

### 10.1. SSH vào PYNQ

Trên Windows PowerShell:

```powershell
ssh xilinx@192.168.2.99
```

Sau đó nhập password của tài khoản `xilinx`.

### 10.2. Chuyển sang root

```bash
sudo -i
```

### 10.3. Kích hoạt môi trường PYNQ

```bash
source /etc/profile.d/pynq_venv.sh
export XILINX_XRT=/usr
```

### 10.4. Vào runtime

```bash
cd /home/xilinx/didong/ecg_fpga/pynq_runtime
```

### 10.5. Chạy server

```bash
python3 server.py \
  --host 0.0.0.0 \
  --port 9000 \
  --deployment /home/xilinx/didong/ecg_fpga/final_w4a4_p99_9 \
  --bitstream /home/xilinx/didong/ecg_fpga/overlay/ecg_accelerator.bit \
  --dma axi_dma_0
```

Giữ terminal này mở trong lúc chạy dashboard.

### 10.6. Kiểm tra server từ laptop

Mở PowerShell khác:

```powershell
Invoke-RestMethod http://192.168.2.99:9000/health
```

Kết quả mong đợi:

```text
ok      : True
runtime : PYNQ_FPGA_HYBRID
```

---

## 11. Chạy dashboard

Project root:

```text
D:\HCMUTE\HK I nam 4\DiDong\ECG_NAS_LOCAL
```

Mở PowerShell:

```powershell
cd "D:\HCMUTE\HK I nam 4\DiDong\ECG_NAS_LOCAL"
```

Kích hoạt virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Chạy FastAPI dashboard:

```powershell
python -m uvicorn dashboard_full.main:app --reload --host 127.0.0.1 --port 8000
```

Mở trình duyệt:

```text
http://127.0.0.1:8000
```

Nếu vừa thay code dashboard:

```text
Ctrl + F5
```

---

## 12. Các chế độ inference

Dashboard hiện hỗ trợ:

```text
Local Integer
PYNQ FPGA Hybrid
```

### Local Integer

Chạy integer deployment trên laptop.

Dùng để:

- kiểm tra numerical result.
- test nhanh.
- chạy toàn bộ test set nhanh hơn FPGA.

### PYNQ FPGA Hybrid

Weighted operation chạy trên FPGA.

Các operation còn lại như pooling, activation, concat, residual, GAP/GMP và orchestration vẫn có phần chạy trên ARM/Python runtime.

---

## 13. Ý nghĩa latency

Ví dụ:

```text
Total Latency          690.844 ms
FPGA Weighted           69.131 ms
Non-FPGA Runtime       621.714 ms
```

### Total Latency

Tổng thời gian xử lý một heartbeat end-to-end.

### FPGA Weighted

Tổng thời gian các weighted operation được thực thi thông qua FPGA accelerator.

### Non-FPGA Runtime

Phần còn lại gồm:

- ARM computation.
- Python orchestration.
- DMA setup.
- buffer allocation/copy.
- synchronization.
- pooling.
- activation.
- concat.
- residual.
- GAP/GMP.

Không nên hiểu `621 ms` là chỉ riêng thời gian CPU ARM tính toán.

---

## 14. Chạy toàn bộ tập test

Dashboard có chức năng:

```text
Run Full Test Set · Local Integer
```

hoặc:

```text
Run Full Test Set · PYNQ FPGA Hybrid
```

Nó đọc toàn bộ:

```text
datasets/splits/test.csv
```

và tính:

- Accuracy.
- Macro F1.
- Confusion Matrix.
- Average Total Latency.
- Average FPGA Weighted Latency.
- Failed Samples.
- Elapsed Time / ETA.

Kết quả được lưu tại:

```text
artifacts/test_runs/
```

Ví dụ:

```text
20260914_xxxxxx_local_full_test.csv
20260914_xxxxxx_local_summary.json
```

hoặc:

```text
20260914_xxxxxx_fpga_full_test.csv
20260914_xxxxxx_fpga_summary.json
```

---

## 15. Lưu ý về test set

Test set chỉ dùng để đánh giá cuối cùng.

Không dùng test set để:

- chọn NAS candidate.
- tune hyperparameter.
- chọn INT4/INT8.
- chọn percentile calibration.

Quy trình đúng:

```text
Train
↓
Validation
↓
NAS Selection
↓
Mixed Precision Selection
↓
QAT
↓
Calibration
↓
LOCK MODEL
↓
Test Set
```

---

## 16. Quy trình chạy đầy đủ đề xuất

### Bước 1 — Chạy dashboard

```powershell
cd "D:\HCMUTE\HK I nam 4\DiDong\ECG_NAS_LOCAL"
.\.venv\Scripts\Activate.ps1
python -m uvicorn dashboard_full.main:app --reload --host 127.0.0.1 --port 8000
```

### Bước 2 — Chạy NAS mới

Trong dashboard:

```text
Kernel candidates
Channel candidates
Trials
Epochs / candidate
Objective
```

Sau đó:

```text
Start NAS
```

Chờ:

```text
Status = completed
```

### Bước 3 — Chọn Candidate

Ưu tiên:

```text
BEST
Checkpoint = SAVED
```

Không nên chọn run còn:

```text
running
```

hoặc candidate:

```text
Checkpoint = NOT SAVED
```

nếu muốn dùng lại weights của candidate.

### Bước 4 — Chạy pipeline tối ưu

```text
Selected Candidate
↓
Full Training
↓
Sensitivity Analysis
↓
Mixed-Precision Search
↓
QAT
↓
Calibration
↓
Integer Deployment
```

### Bước 5 — Chạy PYNQ

```bash
ssh xilinx@192.168.2.99
sudo -i
source /etc/profile.d/pynq_venv.sh
export XILINX_XRT=/usr
cd /home/xilinx/didong/ecg_fpga/pynq_runtime
```

Sau đó:

```bash
python3 server.py \
  --host 0.0.0.0 \
  --port 9000 \
  --deployment /home/xilinx/didong/ecg_fpga/final_w4a4_p99_9 \
  --bitstream /home/xilinx/didong/ecg_fpga/overlay/ecg_accelerator.bit \
  --dma axi_dma_0
```

### Bước 6 — Inference

Dashboard:

```text
Load Selected Heartbeat
↓
Local Integer
hoặc
PYNQ FPGA Hybrid
↓
Run Inference
```

### Bước 7 — Full Test

Sau khi model đã khóa:

```text
Run Full Test Set
```

---

## 17. Các file chính

Một số file/script quan trọng:

```text
scripts/train_baseline.py
→ train baseline.

scripts/run_nas_search.py
→ NAS search.

scripts/train_best_nas.py
→ train lại candidate đã chọn.

scripts/analyze_layer_sensitivity.py
→ sensitivity analysis.

scripts/search_layer_precision.py
→ mixed-precision search.

scripts/run_qat.py
→ Quantization-Aware Training.

scripts/calibrate_integer_graph.py
→ calibration.

scripts/build_deployment_model.py
→ export deployment integer.

scripts/verify_integer_reference.py
→ kiểm tra integer reference.

scripts/final_test_qat.py
→ final evaluation.
```

FPGA:

```text
fpga/rtl/requantizer.v
fpga/rtl/conv1d_engine.v
fpga/rtl/ecg_accelerator_axis.v
```

PYNQ:

```text
pynq_runtime/server.py
pynq_runtime/hybrid_model.py
pynq_runtime/fpga_backend.py
pynq_runtime/deployment_loader.py
pynq_runtime/integer_nonweighted.py
```

Dashboard:

```text
dashboard_full/
```

---

## 18. Trạng thái hiện tại

Hệ thống hiện đã thực hiện được:

- Đọc và hiển thị ECG.
- Chọn heartbeat.
- Local integer inference.
- PYNQ FPGA hybrid inference.
- NAS search.
- Candidate selection.
- Lưu checkpoint riêng cho NAS candidate mới.
- Mixed INT8/INT4 deployment.
- QAT.
- Integer reference.
- FPGA weighted execution.
- Resource utilization display.
- Latency display.
- Full test evaluation.
- Confusion matrix.
- Lưu CSV/JSON kết quả test.

---

## 19. Ghi chú

Model deployment hiện tại:

```text
final_w4a4_p99_9
```

là model đã khóa sau:

```text
NAS
→ Mixed Precision
→ QAT
→ Calibration
→ Integer Export
```

Candidate đang chọn trong phần NAS không tự động thay thế deployment hiện tại.

Muốn một NAS candidate mới chạy trên FPGA, candidate đó phải đi tiếp qua:

```text
Full Training
→ Mixed Precision
→ QAT
→ Calibration
→ Deployment Export
→ copy deployment sang PYNQ
```

---

## 20. Lệnh chạy nhanh

### Laptop

```powershell
cd "D:\HCMUTE\HK I nam 4\DiDong\ECG_NAS_LOCAL"
.\.venv\Scripts\Activate.ps1
python -m uvicorn dashboard_full.main:app --reload --host 127.0.0.1 --port 8000
```

### PYNQ

```bash
ssh xilinx@192.168.2.99
sudo -i
source /etc/profile.d/pynq_venv.sh
export XILINX_XRT=/usr
cd /home/xilinx/didong/ecg_fpga/pynq_runtime
python3 server.py \
  --host 0.0.0.0 \
  --port 9000 \
  --deployment /home/xilinx/didong/ecg_fpga/final_w4a4_p99_9 \
  --bitstream /home/xilinx/didong/ecg_fpga/overlay/ecg_accelerator.bit \
  --dma axi_dma_0
```

### Health check

```powershell
Invoke-RestMethod http://192.168.2.99:9000/health
```
