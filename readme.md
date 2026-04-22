<div align="center">

# HPHF-MMDet: Heterogeneous Perception and Hyperspherical Fusion for Multimodal Object Detection

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

## 📝 Abstract

Multimodal object detection aims to leverage the complementary characteristics of visible (RGB) and infrared (IR) images to address perception challenges in complex environments.  However, existing methods typically utilize structurally uniform backbones that neglect the intrinsic disparities in imaging mechanisms between RGB and IR modalities. Moreover, existing fusion methods fail to effectively address the feature distribution discrepancies across heterogeneous modalities.  To address these issues, we propose Heterogeneous Perception and Hyperspherical Fusion for Multimodal Object Detection (HPHF-MMDet).
We devise a Modality-Specific Heterogeneous Backbone (MSHB) that diverges from the conventional symmetric feature extraction paradigm.
Specifically, for the RGB modality, we employ a Context-Guided Detail Enhancement Module (CGDEM) to dynamically modulate detailed features, suppressing background interference while highlighting target textures.
For the IR modality, a Multi-Scale Windmill Convolution (MSWC) is carefully designed to exploit asymmetric receptive fields for capturing the morphological features of thermal targets. 
Furthermore, to bridge the feature distribution discrepancies between heterogeneous modalities, we introduce a Cross-Modal Complementary Hyperspherical Fusion (CM-CHF) mechanism, which aggregates complementary information through bidirectional hyperspherical alignment.
We conduct extensive experiments on the LLVIP, M3FD, and MFAD datasets.
Our method achieves mAP50 of 97.9\%, 89.8\%, and 80.8\% 
respectively, significantly outperforming existing state-of-the-art 
approaches.

---

## 🖼️ Framework

<div align="center">
  <img src="over.pdf" alt="Framework" width="90%">
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
