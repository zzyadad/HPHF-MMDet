````markdown
<div align="center">

# HPHF-MMDet
### Multimodal Object Detection for Training, Validation, and Inference

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" />
  <img src="https://img.shields.io/badge/PyTorch-Supported-red.svg" />
  <img src="https://img.shields.io/badge/Platform-Linux-lightgrey.svg" />
  <img src="https://img.shields.io/badge/Status-Active-success.svg" />
</p>

<p align="center">
  <b>Guohua Lv</b><sup>*1,2</sup>,
  <b>Zhengyang Zhang</b><sup>*1,2</sup>,
  <b>Guangxiao Ma</b><sup>†3</sup>,
  <b>Yongbiao Gao</b><sup>1,2</sup>,
  <b>Jihao Jin</b><sup>1,2</sup>,
  <b>Qian Xu</b><sup>4</sup>
</p>

<p align="center">
  <sup>*</sup> These authors contributed equally to this work. <br>
  <sup>†</sup> Corresponding author: Guangxiao Ma (mgx@sdust.edu.cn)
</p>

</div>

---

## 📌 Overview

**HPHF-MMDet** is a multimodal object detection project built on an Ultralytics/RT-DETR-style framework.  
This repository provides a clean and practical pipeline for:

- training
- validation
- inference
- single-GPU and multi-GPU execution
- flexible dataset and model configuration

The project is organized for efficient experimentation on common multimodal detection benchmarks such as **MFAD**, **M3FD**, and **LLVIP**.

---

## ✨ Highlights

- Clean and easy-to-follow project structure
- Simple environment setup
- Support for both **single-GPU** and **distributed multi-GPU** training
- Flexible configuration for multimodal datasets and models
- Straightforward validation and inference pipeline
- Convenient for research reproduction and further development

---

## 📝 Abstract

> Replace the following paragraph with your final paper abstract.

This project focuses on multimodal object detection by jointly leveraging complementary information from different sensing modalities. It provides a unified framework for model training, evaluation, and deployment, and is designed to support flexible backbone configuration, reproducible experiments, and efficient multi-GPU execution.

---

## 🖼️ Framework

<div align="center">
  <img src="assets/framework.png" alt="Framework of HPHF-MMDet" width="90%">
  <p><em>Overall framework of HPHF-MMDet.</em></p>
</div>

> Please place your framework figure at `assets/framework.png`.  
> If your actual filename is different, replace the image path above.

---

## 📂 Project Structure

```text
/home/s-zhangzy/RT-DETR/mm
├── train.py
├── val.py
├── predict.py
├── pyproject.toml
└── ultralytics
    └── cfg
        ├── datasets
        │   ├── MFAD.yaml
        │   ├── M3FD.yaml
        │   └── LLVIP.yaml
        └── models
            └── HPHF-MMDet
````

---

## 📁 Important Paths

### Project Root

```bash
/home/s-zhangzy/RT-DETR/mm
```

### Dataset Configuration Directory

```bash
/home/s-zhangzy/RT-DETR/mm/ultralytics/cfg/datasets
```

### Model Architecture Directory

```bash
/home/s-zhangzy/RT-DETR/mm/ultralytics/cfg/models/HPHF-MMDet
```

### Common Dataset Configurations

```bash
/home/s-zhangzy/RT-DETR/mm/ultralytics/cfg/datasets/MFAD.yaml
/home/s-zhangzy/RT-DETR/mm/ultralytics/cfg/datasets/M3FD.yaml
/home/s-zhangzy/RT-DETR/mm/ultralytics/cfg/datasets/LLVIP.yaml
```

---

## ⚙️ Environment Setup

### 1. Create a Conda Environment

```bash
conda create -n HPHF-MMDet python=3.10 -y
conda activate HPHF-MMDet
```

### 2. Install PyTorch

Example for **CUDA 11.8**:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install Project Dependencies

This project does not provide a separate `requirements.txt` under `mm`.
Dependencies are managed through `pyproject.toml`.

```bash
cd /home/s-zhangzy/RT-DETR/mm
pip install -e .
```

---

## 🚀 Training

> `train.py` currently uses script-level parameters.
> Please set your actual **model path** and **dataset path** in `train.py` before running.

### Example Setup in `train.py`

```python
model = RTDETR("/absolute/path/to/model.yaml/or/weights.pt")
model.train(data="/absolute/path/to/dataset.yaml", ...)
```

### Single-GPU Training

```bash
cd /home/s-zhangzy/RT-DETR/mm
python train.py
```

### Multi-GPU Training with `torchrun`

#### 2 GPUs

```bash
cd /home/s-zhangzy/RT-DETR/mm
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29000 train.py
```

#### 4 GPUs

```bash
cd /home/s-zhangzy/RT-DETR/mm
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29000 train.py
```

### Legacy Multi-GPU Command

```bash
cd /home/s-zhangzy/RT-DETR/mm
python -m torch.distributed.launch --nproc_per_node=4 --master_port=29000 train.py
```

---

## ✅ Validation

> `val.py` also uses script-level parameters.
> Please set the correct **model path** and **dataset path** in `val.py` first.

```bash
cd /home/s-zhangzy/RT-DETR/mm
python val.py
```

---

## 🔍 Inference

> `predict.py` uses script-level parameters.
> Please set the following items before running:

* `model = RTDETR("/absolute/path/to/weights.pt")`
* `source = "/absolute/path/to/image/or/folder")`

```bash
cd /home/s-zhangzy/RT-DETR/mm
python predict.py
```

---

## 📊 Supported Datasets

This project is currently organized to support the following common multimodal datasets:

* **MFAD**
* **M3FD**
* **LLVIP**

You can switch datasets by modifying the corresponding dataset YAML configuration file.

---



## 📬 Contact

For questions, suggestions, or collaboration:

* **Email:** [10431240210@stu.qlu.edu.cn](mailto:10431240210@stu.qlu.edu.cn)



## 🙏 Acknowledgment

This project is developed based on the excellent open-source ecosystem of **PyTorch** and **Ultralytics**.
We sincerely thank the community for their valuable contributions.

```
```
