# HPHF-MMDet Multimodal Project Guide (Training / Validation / Inference)

## 1. Directory and Configuration Locations

Project root:
- /home/s-zhangzy/RT-DETR/mm

Dataset configuration directory:
- /home/s-zhangzy/RT-DETR/mm/ultralytics/cfg/datasets

Model architecture directory:
- /home/s-zhangzy/RT-DETR/mm/ultralytics/cfg/models/HPHF-MMDet

Common dataset configs:
- /home/s-zhangzy/RT-DETR/mm/ultralytics/cfg/datasets/MFAD.yaml
- /home/s-zhangzy/RT-DETR/mm/ultralytics/cfg/datasets/M3FD.yaml
- /home/s-zhangzy/RT-DETR/mm/ultralytics/cfg/datasets/LLVIP.yaml

## 2. Environment Setup

### 2.1 Create a Conda Environment

```bash
conda create -n HPHF-MMDet python=3.10 -y
conda activate HPHF-MMDet
```

### 2.2 Install PyTorch (Choose by CUDA Version)

Example for CUDA 11.8:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2.3 Install Project Dependencies

This project does not provide a separate requirements.txt under mm.
Dependencies are defined in pyproject.toml.

```bash
cd /home/s-zhangzy/RT-DETR/mm
pip install -e .
```

## 3. Training Commands (Single-GPU / Multi-GPU)

Note: train.py currently uses script-level parameters. Set your actual model and data paths in train.py first.

Example setup in train.py:
- model = RTDETR("/absolute/path/to/model.yaml/or/weights.pt")
- model.train(data="/absolute/path/to/dataset.yaml", ...)

### 3.1 Single-GPU Training

```bash
cd /home/s-zhangzy/RT-DETR/mm
python train.py
```

### 3.2 Multi-GPU Training (Recommended: torchrun)

2 GPUs:

```bash
cd /home/s-zhangzy/RT-DETR/mm
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29000 train.py
```

4 GPUs:

```bash
cd /home/s-zhangzy/RT-DETR/mm
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29000 train.py
```

### 3.3 Legacy Multi-GPU Command

```bash
cd /home/s-zhangzy/RT-DETR/mm
python -m torch.distributed.launch --nproc_per_node=4 --master_port=29000 train.py
```

## 4. Validation Command

Note: val.py also uses script-level parameters. Set model and data paths in val.py first.

```bash
cd /home/s-zhangzy/RT-DETR/mm
python val.py
```

## 5. Inference Command

Note: predict.py uses script-level parameters. Set the following first:
- model = RTDETR("/absolute/path/to/weights.pt")
- source = "/absolute/path/to/image/or/folder"

```bash
cd /home/s-zhangzy/RT-DETR/mm
python predict.py
```

## 6. Contact

For questions or collaboration:

10431240210@stu.qlu.edu.cn

