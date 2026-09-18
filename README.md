# ECG Arrhythmia Classification with Hardware-Aware NAS on FPGA

This project develops an ECG arrhythmia classification system optimized for FPGA deployment using Hardware-Aware Neural Architecture Search (NAS).

The model architecture is optimized based on Macro F1 score, parameter count, and MACs to achieve a balance between classification performance and hardware efficiency.

Mixed-precision INT4/INT8 quantization with Quantization-Aware Training (QAT) is applied to reduce computational cost and model size while maintaining classification accuracy.

The final system is deployed on the PYNQ-Z2 platform using a hybrid inference architecture:
- FPGA handles weighted operations such as convolution and fully connected layers.
- ARM processor handles non-weighted operations and control tasks.

## Key Features

- ECG arrhythmia classification
- Hardware-Aware NAS
- CNN1D and lightweight model exploration
- Mixed-precision INT4/INT8 quantization
- Quantization-Aware Training (QAT)
- FPGA acceleration on PYNQ-Z2
- Hybrid FPGA–ARM inference
- Hardware-oriented evaluation using Macro F1, parameters, MACs, and latency

## System Overview

The system processes ECG signals through a complete pipeline from preprocessing to hardware deployment. First, ECG data is prepared and used to train candidate CNN1D-based models. Hardware-Aware NAS is then applied to search for suitable architectures based on classification performance and hardware-related metrics such as Macro F1 score, parameter count, and MACs.

After the model is selected, mixed-precision INT4/INT8 quantization and Quantization-Aware Training are applied to reduce computational cost while preserving accuracy. The optimized model is then deployed on the PYNQ-Z2 platform using a hybrid inference architecture, where FPGA logic accelerates weighted operations such as convolution and fully connected layers, while the ARM processor handles non-weighted operations and control tasks.

The final output is the predicted ECG arrhythmia class together with evaluation metrics used to assess both model accuracy and hardware efficiency.
