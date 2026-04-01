<div align="center">
<h1> [TCSVT'26] Prototype Memory-based Neighboring Feature Fusion Network for Image Manipulation Localization </h1>
</div>

## 📢 News
* **[2026-4]** Our paper is accepted by **TCSVT 2026**! 🎉
* **[2026-4]** The code are being organized and will be released shortly. Please star this repo for updates!

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
