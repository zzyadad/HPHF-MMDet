<div align="center">

# HPHF-MMDet
### A Multimodal Object Detection Project for Training, Validation, and Inference

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" /></a>
  <a href="#"><img src="https://img.shields.io/badge/pytorch-supported-red.svg" /></a>
  <a href="#"><img src="https://img.shields.io/badge/platform-linux-lightgrey.svg" /></a>
  <a href="#"><img src="https://img.shields.io/badge/status-active-success.svg" /></a>
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

**HPHF-MMDet** is a multimodal object detection project built on the Ultralytics/RT-DETR-style framework.  
This repository provides a clean and practical pipeline for:

- **Training**
- **Validation**
- **Inference**
- **Single-GPU and Multi-GPU execution**
- **Flexible dataset and model configuration**

The project is organized for convenient experimentation on common multimodal detection benchmarks such as **MFAD**, **M3FD**, and **LLVIP**.

---

## 📝 Abstract

> Replace this paragraph with your paper abstract.

Example format:

This project focuses on multimodal object detection by jointly leveraging complementary information from different sensing modalities.  
It provides a unified framework for model training, evaluation, and deployment, and is designed to support flexible backbone configuration, reproducible experiments, and efficient multi-GPU execution.

---

## 🖼️ Framework

<div align="center">
  <img src="assets/framework.png" alt="Framework of HPHF-MMDet" width="90%">
  <p><em>Overall framework of HPHF-MMDet.</em></p>
</div>

> Put your framework figure into `assets/framework.png`.  
> If your image filename is different, replace the path above accordingly.

---

## ✨ Features

- Clean project structure
- Easy environment setup
- Support for **single-GPU** and **distributed multi-GPU** training
- Compatible with multiple multimodal datasets
- Simple validation and inference workflow
- Convenient for research reproduction and further development

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
