# HPHF-Det: Heterogeneous Perception and Hyperspherical Fusion for Multimodal Detection

<div align="center">
  <p>
    <a href="https://github.com/zzyadad/HPHF-Det" target="_blank">
      <img width="100%" src="https://raw.githubusercontent.com/zzyadad/HPHF-Det/master/ultralytics/over.jpg" alt="HPHF-Det Framework Architecture"></a>
  </p>
</div>

## 🎯 核心创新

- ✨ **Dual-branch Heterogeneous Backbone**: 打破传统的对称特征提取范式，设计了针对模态差异的双分支异构骨干网络。
- 🌈 **Context-Guided Detail Enhancement Module (CGDEM)**: 针对 **RGB** 模态，动态调节细节特征，在抑制背景干扰的同时增强目标纹理。
- 🌀 **Multi-Scale Windmill Convolution (MSWC)**: 针对 **IR** 模态，利用不对称感受野有效捕捉热目标的形态特征，解决弱纹理和类高斯分布问题。
- 🔗 **Cross-Modal Complementary Hyperspherical Fusion (CM-CHF)**: 跨模态互补超球面融合机制，通过双向超球面对齐策略在全局尺度上聚合互补信息。
- 🚀 **SOTA Performance**:
    - **LLVIP**: 97.9% mAP50
    - **M3FD**: 89.8% mAP50
    - **MFAD**: 77.8% mAP50

## 📖 项目简介

> **Multimodal object detection aims to leverage the complementary characteristics of visible (RGB) and infrared (IR) images to address perception challenges in complex environments.**

However, existing methods typically utilize structurally uniform backbones for feature extraction, which neglect the intrinsic disparities in imaging mechanisms between RGB and IR data. Consequently, such unified architectures result in insufficient feature extraction capability for heterogeneous modalities. Moreover, existing fusion methods fail to effectively address the feature distribution discrepancies across heterogeneous modalities.

To address these limitations, we propose a multimodal object detection framework, named **HPHF-Det**.

Specifically:
1.  We devise a **dual-branch heterogeneous backbone network**, diverging from the conventional symmetric feature extraction paradigm.
2.  For the RGB modality, we employ a **Context-Guided Detail Enhancement Module (CGDEM)** to dynamically modulate detail features, suppressing background interference while highlighting target textures.
3.  Parallelly, for the IR modality, we propose a **Multi-Scale Windmill Convolution (MSWC)** that exploits asymmetric receptive fields to capture the morphological features of thermal targets.
4.  To bridge the feature distribution discrepancies between heterogeneous modalities, we introduce a **Cross-Modal Complementary Hyperspherical Fusion (CM-CHF)** mechanism.

## 📅 代码发布计划

> **重要说明**: 本项目的完整代码将在论文被期刊/会议正式录用后公开发布。

**预计发布内容**:
- [ ] 完整的 HPHF-Det 训练与推理代码
- [ ] 预训练权重 (LLVIP, M3FD, MFAD)
- [ ] 异构骨干网络 (Heterogeneous Backbone) 实现细节
- [ ] 配置文件与复现脚本

## 📧 联系方式

如有学术交流或合作意向，请联系：
- 📧 Email: [10431240210@stu.qlu.edu.cn](mailto:10431240210@stu.qlu.edu.cn)

## 📄 许可证

本项目遵循 [MIT License](LICENSE)

---

⭐ **如果您对 HPHF-Det 感兴趣，请点击 Star 关注项目进展！**

**💡 提示**: 请持续关注本仓库，我们会在论文录用后第一时间发布完整代码。
