# PYNQ runtime

Copy this folder to:

/home/xilinx/didong/ecg_fpga/pynq_runtime

Laptop example:

```powershell
scp -r ".\pynq_runtime" xilinx@192.168.2.99:/home/xilinx/didong/ecg_fpga/
```

Run on PYNQ:

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
