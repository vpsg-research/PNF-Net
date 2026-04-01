<div align="center">
<h1> [TCSVT'26] Prototype Memory-based Neighboring Feature Fusion Network for Image Manipulation Localization </h1>
</div>

## 📢 News
* **[2026-4]** Our paper is accepted by **TCSVT 2026**! 🎉
* **[2026-4]** The code are being organized and will be released shortly. Please star this repo for updates!
* **[2025-11]** Our paper proposing the **first scribble-supervised weakly supervised framework for image manipulation localization** is accepted by **AAAI 2026**! 🎉(https://github.com/vpsg-research/SCAF)

## ✨ Contributions
⚠️ We propose a novel PNF-Net for IML, which uses memory priors to guide representation learning for precise localization of tampered regions. It effectively integrates valuable information from neighboring features to form robust fused feature representations. Extensive experiments show that our model achieves SOTA performance.

🚀 We propose an MLM inspired by ``grandmother cells” in the primary visual cortex (V1) of macaque monkeys. The MLM uses semantic clusters to accumulate consistencies and anomalies between tampered regions and the background in images, forming memory priors that guide IML representation learning.

🧩We design a hierarchically collaborative feature refinement mechanism comprising the  NFIM and  VFM. NFIM aggregates adjacent complementary cues to handle scale variations, while VFM establishes cross-level channel dependencies and enforces structural integrity via multi-level supervision. This synergy enables robust representations for localizing tampering artifacts.


##  Introduction
Official repository for the TCSVT2026 paper “*Prototype Memory-based Neighboring Feature Fusion Network for Image Manipulation Localization*”

<div align="center">
    <img width="600" alt="image" src="1.png?raw=true">
</div>

Existing IML methods generally fall into two categories: (a) local structural modeling, which leverages boundary, noise, or frequency cues to capture local artifacts; and (b) global semantic modeling, which enhances semantic discrepancies between tampered regions and the background via network design and feature interactions. (c) We construct a clusterable memory bank that aggregates manipulation cues into reusable memory priors, which proactively guide representation learning.

## 🎮 Getting Started

### 1. Install Environment

To set up the experimental environment, please follow the specific requirements for each baseline model. Taking **PNF-Net** as an example, you can create and install the environment using the provided script:

```bash
conda env create -f PNFNet.yml
```
### 2. Prepare Datasets
| Dataset     | Nums        |  #CM          | #SP          | #IP          |  #Train          |  #Test          | 
| :----:      |    :----:   |         :----:|:----:        |    :----:    |         :----:   |         :----:  |
| CASIAv2   | 5123        | 3295          |1828          |    0         |        5123      |        0        |
| CASIAv1   | 920         | 459           |461           |    0         |        0         |        920      |
| Coverage    | 100         | 100           |0             |    0         |        70        |        30       |
| Columbia    | 180         | 0             |180           |    0         |         130      |        50       |
| NIST16      | 564         | 68            |288           |    208       |        414       |        150      |
|Korus   | 220 | -|-|-|0|220|
|IMD2020|2010|-|-|-|0|2010|

- CASIAv2 [Download](https://github.com/SunnyHaze/IML-Dataset-Corrections)
- CASIAv1 [Download](https://github.com/SunnyHaze/IML-Dataset-Corrections)
- Columbia  [Download](https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/)
- Coverage  [Download](https://github.com/wenbihan/coverage?tab=readme-ov-file)
- NIST16    [Download](https://mfc.nist.gov/users/sign_in)
- Korus [Download](https://pkorus.pl/downloads/dataset-realistic-tampering)
- IMD2020 [Download](https://staff.utia.cas.cz/novozada/db/)

### 3. Train

To train our PNF-Net, first ensure that the pretrained [PVTv2-B2](https://github.com/whai362/PVT) weights are ready, and then simply run the corresponding training script.
```bash
# Make sure your directory paths are set correctly!
python train.py
```
### 4. Test
```bash
python test.py
```

### 5. Evaluation
After confirming that the paths to your prediction and ground-truth files are correct, run the following script to compute pixel-level F1, IoU, and AUC.
```bash
python eval.py
```

## Citation
If you find our code useful, please consider citing us and give us a star!

```
@inproceedings{LiSCAF,
  title={Prototype Memory-based Neighboring Feature Fusion Network for Image Manipulation Localization},
  author={Li, Songlin and and Guo, Zhiqing and Miao, Changtao and Wenzhong, Yang and Liejun, Wang and Yang, Gaobo and Liao, Xin},
  booktitle={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2026}
}
```
