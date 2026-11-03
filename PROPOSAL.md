# Project Proposal: SlideShift
## Robust Foundation-Model-Driven Whole-Slide Classification and Real-Time Distribution Shift Monitoring

**Principal Investigator / Author:** Computational Pathology Research Lead  
**Target Venue / Audience:** Academic Supervisor / Senior ML Engineering Lead  
**Scope:** 8-Week Research & Engineering Project  

---

### 1. Executive Summary & Thesis
Whole-slide imaging (WSI) in computational pathology faces a persistent translation barrier: while Multiple-Instance Learning (MIL) aggregators trained on frozen pathology foundation models (UNI, CONCH, Virchow, Phikon) achieve high in-distribution AUROC (>0.95), **they are not immune to real-world domain shifts**. Deployments routinely degrade when an institution purchases a new scanner model, alters chemical staining reagents, or receives biopsies from external referral centers.

> **Central Thesis:** Foundation models substantially improve feature expressiveness over ImageNet baselines, but do not provide invariant slide representations across scanner, stain, and hospital shifts. SlideShift quantifies this degradation, identifies which foundation architecture is most resilient, benchmarks mitigations, and introduces a deployable real-time **Slide Shift Monitor** that flags out-of-distribution slides before erroneous clinical predictions occur.

---

### 2. Research Questions
1. **Model Robustness Ranking:** Under controlled synthetic and natural domain shifts, how does the AUROC degradation and Expected Calibration Error (ECE) of pathology foundation models (UNI, CONCH, Virchow, Phikon) compare against an ImageNet baseline?
2. **Heatmap Faithfulness & Stability:** Do attention heatmaps remain stable under scanner and stain perturbations, or do models latch onto non-biological shortcuts (e.g., surgical pen marks, coverslip air bubbles, tissue folds)?
3. **Mitigation Trade-Offs:** What is the cost-benefit trade-off of stain normalization (Macenko/Reinhard), per-slide feature standardization, and domain-adversarial training (DANN)?
4. **Shift Detection:** Can a lightweight monitor combining CIELAB stain color statistics and Mahalanobis feature distance reliably flag slides that suffer performance degradation?

---

### 3. Dataset & Cohort Design
To prevent scope creep while ensuring gold-standard ground truth, SlideShift evaluates on two primary cohorts:
1. **CAMELYON16 & CAMELYON17 (Breast Lymph Node Metastasis):**
   - *Why:* Pixel-level tumor annotations enable quantitative validation of attention heatmaps (AUROC of tile attention vs. pathologist masks). CAMELYON16 features two scanner types across two medical centers; CAMELYON17 provides a built-in 5-center site shift benchmark.
2. **TCGA-NSCLC (Lung Adenocarcinoma vs. Squamous Cell Carcinoma):**
   - *Why:* The universally accepted benchmark for pathology foundation models, allowing direct comparison with published literature.
3. **External Check:** Held-out TCGA Tissue Source Sites (TSS) or CPTAC external cohort.

*Data Hygiene & Contamination Audit:* Pretraining datasets of candidate foundation models are cross-referenced to ensure test cohorts were not seen during self-supervised pretraining (e.g., verifying whether TCGA tiles were included in Phikon's iBOT training set).

---

### 4. Pipeline Architecture
```
[ Gigapixel WSI (.svs, .tif) ]
              │
              ▼
   [ Stage 1: Ingestion & QC ] ───► Saturation/Otsu tissue segmentation + Laplacian blur & artifact flags
              │
              ▼
   [ Stage 2: Physical Tiling ] ───► Resample to calibrated 0.50 µm/px (eliminates fake scanner magnification bugs)
              │
              ▼
   [ Stage 3: Feature Extraction ] ──► Frozen Foundation Models (UNI, CONCH, Virchow, Phikon, ResNet-50)
              │                     (Stream tiles, save half-precision .h5 bags, discard raw slides)
              │
              ├─────────────────────────────────────────┐
              ▼                                         ▼
   [ Stage 4: MIL Aggregation ]               [ Stage 5: Shift Monitor ]
   - ABMIL (Gated Attention)                  - Stain CIELAB moments
   - CLAM (Instance Clustering)               - Embedding Mahalanobis distance
   - TransMIL (Correlated Transformer)        - Calibrated threshold on validation sites
              │                                         │
              ▼                                         ▼
   [ Slide Prediction & Heatmaps ]            [ Real-time OOD Warning Flag ]
```

---

### 5. Robustness Benchmark Matrix
We evaluate on a fixed held-out test cohort of 150–200 slides across 6 perturbation classes at 3 severity levels:
1. **Natural Site & Scanner Shift:** Center A vs. Center B in CAMELYON16/17; scanner cross-testing.
2. **H&E Optical Density Jitter:** Ruifrok stain deconvolution with randomized hematoxylin/eosin concentration jitter ($\pm 10\%$ to $\pm 45\%$).
3. **Stain Saturation & Contrast Shift:** Simulating chemical batch variability.
4. **Optical Defocus Blur:** Gaussian filter radii simulating z-axis focal plane drift.
5. **Lossy Compression:** JPEG compression (quality 65, 40, 18) simulating PACS transmission.
6. **Resolution & MPP Mismatch:** Simulating 40× downsampling discrepancies.

**Evaluated Metrics:**
- $\Delta\text{AUROC}$ and Balanced Accuracy drop
- Prediction Flip Rate (percentage of slide labels changing under shift)
- Expected Calibration Error (ECE) and Brier Score
- Heatmap Spearman rank correlation and Top-10% Jaccard overlap
- Shortcut Baseline AUROC (predicting tumor solely from site metadata)

---

### 6. Expected Deliverables
1. **SlideShift Python Package & CLI:** Production-grade, pip-installable toolchain (`slideshift ingest`, `extract`, `train`, `robustness`, `monitor`).
2. **Interactive Streamlit Demo:** Web-based diagnostic console displaying slide thumbnails, ground-truth overlays, attention heatmaps, perturbation controls, and the real-time Shift Monitor.
3. **Robustness Benchmark Report:** Comprehensive ablation table comparing foundation models $\times$ aggregators $\times$ shifts.
4. **Honest Heatmap Gallery:** Tri-panel exhibits (original, pathologist ground truth, model attention) highlighting best cases, random cases, and worst artifact-driven failures.
5. **Research Paper / Technical Report:** Formatted for submission to a medical imaging or machine learning conference (e.g., MICCAI, MIDL, or CVPR Workshop).

---

### 7. Resource & Compute Planning
- **Storage:** Feature bags cached in half precision (.h5) require ~20–40 MB per slide. A 500-slide cohort fits in <20 GB (raw gigapixel slides streamed and deleted).
- **Compute:** Local development and testing on NVIDIA RTX 3080 Laptop GPU (8GB VRAM) using mixed-precision (FP16/BF16). MIL training runs on cached features in minutes.

---

### 8. Ethical & Clinical Disclaimers
This software is developed strictly for translational machine learning research and benchmarking on public, de-identified datasets. It is **not approved by the FDA or CE-marked as a diagnostic medical device** and must not be used for direct clinical patient diagnosis or treatment decisions.

