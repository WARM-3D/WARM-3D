# 🚗 WARM-3D: A Weakly-Supervised Sim2Real Domain Adaptation Framework for Roadside Monocular 3D Object Detection

This repository contains the official implementation of our paper "WARM-3D: A Weakly-Supervised Sim2Real Domain Adaptation Framework for Roadside Monocular 3D Object Detection" (https://ieeexplore.ieee.org/document/10919929).

## 📝 Introduction

Existing roadside perception systems are limited by the absence of publicly available, large-scale, high-quality 3D datasets. Exploring the use of cost-effective, extensive synthetic datasets offers a viable solution to tackle this challenge and enhance the performance of roadside monocular 3D detection. In this study, we introduce the TUMTraf Synthetic Dataset, offering a diverse and substantial collection of high-quality 3D data to augment scarce real-world datasets. Besides, we present WARM-3D, a concise yet effective framework to aid the Sim2Real domain transfer for roadside monocular 3D detection. Our method leverages cheap synthetic datasets and 2D labels from an off-the-shelf 2D detector for weak supervision. We show that WARM-3D significantly enhances performance, achieving a +12.40% increase in mAP3D over the baseline with only pseudo-2D supervision. With 2D GT as weak labels, WARM-3D even reaches performance close to the Oracle baseline. Moreover, WARM-3D improves the ability of 3D detectors to unseen sample recognition across various real-world environments, highlighting its potential for practical applications.

<div align="center">
  <img src="Asset/TUMTraff-Synthetic.png" width="500"/>
  <br>
  <em>Overview of TUMTraffic Synthetic Dataset</em>
</div>

## 📊 Main Results

[Your results will be added here]

## 🛠️ Installation

1. Clone this repository:
```bash
git clone https://github.com/WARM-3D/WARM-3D.git
cd WARM-3D
```

2. Create and activate a conda environment:
```bash
conda create -n warm3d python=3.8
conda activate warm3d
```

3. Install PyTorch and dependencies:
```bash
pip install -r requirements.txt
```

## 📥 Data Preparation

1. Download the TUMTraf A9 Highway Dataset:
   - Visit [TUMTraf Dataset Registration Page](https://a9-dataset.innovation-mobility.com/register)
   - Register and download the dataset
   - TUMTraffic Synthetic (for training) 🎮
   - TUMTraffic Intersection (for evaluation) 🚦
   - For more information about the dataset, visit [TUMTraf Dataset Page](https://innovation-mobility.com/en/project-providentia/a9-dataset/)

<div align="center">
  <img src="Asset/TUMTraff-Synthetic-distribution.png" width="400"/>
  <br>
  <em>Distribution of TUMTraffic Synthetic Dataset</em>
</div>

## 🚀 Training & Ecvaluation

1. Train and evaluation on TUMTraffic dataset:
```bash
bash MonoDETR/train.bash MonoDETR/configs/monodetr.yaml
```

<div align="center">
<table>
<tr>
<td>
  <img src="Asset/2D_pesudo_label.png" width="400"/>
  <br>
  <em>2D Pseudo Label Generation</em>
</td>
</tr>
<tr>
<td>
  <img src="Asset/matching1.png" width="400"/>
  <br>
  <em> Matching Process 1 </em>
</td>
<td>
  <img src="Asset/matching2.png" width="400"/>
  <br>
  <em> Matching Process 2</em>
</td>
<td>
  <img src="Asset/matching3.png" width="400"/>
  <br>
  <em> Matching Process 3</em>
</td>
</tr>
</table>
</div>

## 🔍 Detect

1. Evaluate on test set:
```bash
bash MonoDETR/test.bash MonoDETR/configs/monodetr_detect.yaml
```

## 📚 Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@INPROCEEDINGS{warm3d-tumtraffic-synthetic,
  author={Zhou, Xingcheng and Fu, Deyu and Zimmer, Walter and Liu, Mingyu and Lakshminarasimhan, Venkatnarayanan and Strand, Leah and Knoll, Alois C.},
  booktitle={2024 IEEE 27th International Conference on Intelligent Transportation Systems (ITSC)}, 
  title={WARM-3D: A Weakly-Supervised Sim2Real Domain Adaptation Framework for Roadside Monocular 3D Object Detection}, 
  year={2024},
  volume={},
  number={},
  pages={3489-3496},
  keywords={Three-dimensional displays;Weak supervision;Detectors;Object detection;Intelligent transportation systems;Synthetic data},
  doi={10.1109/ITSC58415.2024.10919929}}
```

## 🙏 Acknowledgments

This work is built upon the excellent [MonoDETR](https://github.com/ZrrSkywalker/MonoDETR) codebase. We thank the authors for their great work and for making their code publicly available.

