# ECG NAS FPGA Dashboard V5 — Integrated

Bản này không chỉ đổi giao diện. Nó nối ba phần:

```text
MIT-BIH test set
      ↓
Laptop Dashboard
      ├─ Local integer model
      ├─ NAS architecture search
      └─ Send selected heartbeat
                ↓
             PYNQ-Z2
                ↓
           AXI DMA / FPGA
                ↓
       Hybrid classification
                ↓
        N / L / R / V / A
```

## A. Dashboard đọc tập test

Dashboard tự dò theo thứ tự:

```text
datasets/splits/test.csv
datasets/test.csv
```

Record WFDB tự dò:

```text
datasets/mitbih/*.dat + *.hea
datasets/*.dat + *.hea
```

Trên giao diện có:

```text
Rhythm class:
All / N / L / R / V / A

Find record:
100 / 101 / 230 ...

ECG record:
100.dat

Heartbeat:
Beat 1 · N · sample ...
Beat 2 · V · sample ...
```

Chỉ những segment thuộc `test.csv` mới được dùng cho test inference.

## B. Local integer inference

Dashboard dùng các module hiện tại trong:

```text
server/
├── deployment_loader.py
├── integer_ops.py
├── runtime_to_pool1.py
└── model_block.py
```

và deployment:

```text
artifacts/final_w4a4_p99_9/
```

Kết quả trả:

```text
N / L / R / V / A
logits
probabilities
tensor trace
```

## C. NAS search thật từ dashboard

`Start NAS Search` bây giờ gọi trực tiếp source NAS của project:

```text
data/loader.py
models/nas_model.py
nas/cost.py
training/train.py
```

Runner:

```text
dashboard/nas_runner.py
```

Nó dùng:

```text
train.csv → train candidate
val.csv   → Macro F1 / architecture selection
test.csv  → KHÔNG dùng để tuning
```

Candidate được lưu tại:

```text
artifacts/dashboard_nas_jobs/<job_id>/
├── config.json
├── progress.json
├── candidates.json
├── best_candidate.json
├── stdout.log
└── stderr.log
```

## D. PYNQ-Z2 full heartbeat inference

Copy thư mục:

```text
pynq_runtime/
```

sang PYNQ, ví dụ:

```text
/home/xilinx/didong/ecg_fpga/pynq_runtime/
```

Copy deployment sang:

```text
/home/xilinx/didong/ecg_fpga/final_w4a4_p99_9/
```

Giữ overlay:

```text
/home/xilinx/didong/ecg_fpga/overlay/ecg_accelerator.bit
```

Trên PYNQ:

```bash
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

Test health từ laptop:

```powershell
Invoke-RestMethod http://<PYNQ_IP>:9000/health
```

Kỳ vọng:

```json
{
  "ok": true,
  "runtime": "PYNQ_FPGA_HYBRID"
}
```

Sau đó dashboard:

```text
PYNQ IP = địa chỉ board
Port    = 9000
```

bấm:

```text
Send Selected Heartbeat
```

Luồng PYNQ:

```text
normalized ECG 320×1
      ↓
Stem Conv                  FPGA
      ↓
Block 1 weighted ops       FPGA
GELU/Pooling/Concat/Add    ARM
      ↓
Block 2 weighted ops       FPGA
      ↓
Block 3 weighted ops       FPGA
      ↓
Block 4 weighted ops       FPGA
      ↓
Head Dense                 FPGA
      ↓
class_output INT32 logits  ARM
      ↓
N/L/R/V/A
```

`class_output 24→5` để trên ARM vì deployment hiện xuất `INT32_LOGITS`,
trong khi accelerator hiện tại requantize output INT4/INT8.

## E. Chạy dashboard trên laptop

Copy `dashboard/` vào root project:

```text
D:\HCMUTE\HK I nam 4\DiDong\ECG_NAS_LOCAL\dashboard
```

Cài:

```powershell
cd "D:\HCMUTE\HK I nam 4\DiDong\ECG_NAS_LOCAL"
.\.venv\Scripts\Activate.ps1
pip install -r .\dashboard\requirements.txt
```

Chạy:

```powershell
python -m uvicorn dashboard.main:app --reload --host 127.0.0.1 --port 8000
```

Mở:

```text
http://127.0.0.1:8000
```

## F. Lưu ý

NAS architecture search ở V5 search:

```text
kernel triplet cho mỗi Inception block
channel của Block1..Block4
```

Precision INT4/INT8 trong form được giữ để phục vụ bước mixed-precision,
nhưng `dashboard/nas_runner.py` hiện chủ yếu search kiến trúc FP32 giống
NAS source ban đầu của project. Mixed-precision deployment chính thức vẫn
là artifact `final_w4a4_p99_9`.

Không dùng `test.csv` để tối ưu NAS.
