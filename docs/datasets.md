# SlideShift: Dataset Access & Ingestion Guide

## 1. Primary Datasets

### CAMELYON16 & CAMELYON17 (Main Task)
- **Clinical Task:** Detection of breast cancer lymph node metastases.
- **Why It Matters:**
  - Includes pixel-level tumor annotations (XML coordinate polygons) curated by expert pathologists. This allows direct, quantitative validation of attention heatmaps (AUROC of tile attention against tumor ground truth).
  - CAMELYON16 features two different scanner types across two medical centers (Radboud UMC and UMC Utrecht).
  - CAMELYON17 features five distinct medical centers, providing a built-in cross-hospital domain shift benchmark.
- **Access Points:**
  - [GigaDB CAMELYON16](http://gigadb.org/dataset/100439)
  - [AWS Open Data Registry / Grand Challenge](https://camelyon17.grand-challenge.org/)
- **Contamination Risk:** UNI, Virchow, and CONCH did not report CAMELYON16/17 in their SSL pretraining datasets.

---

### TCGA-NSCLC (Secondary Benchmark Task)
- **Clinical Task:** Lung adenocarcinoma (LUAD) vs. Lung squamous cell carcinoma (LUSC) slide-level subtyping.
- **Why It Matters:** The gold-standard benchmark in modern computational pathology literature, enabling direct comparison of AUROC against published numbers.
- **Access Point:** [NIH Genomic Data Commons (GDC) Portal](https://portal.gdc.cancer.gov/) via the `gdc-client` tool.
- **Contamination Risk:** **High for Phikon** (which trained on ~40M TCGA tiles with iBOT). UNI and Virchow used proprietary clinical archives (Mass-100K and MSKCC) rather than TCGA for primary pretraining.

---

### CPTAC / Held-out TCGA Sites (External Generalization Check)
- **Clinical Task:** Independent test cohort from medical centers unseen during training.
- **Access Point:** [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/).

---

## 2. Storage & Streaming Best Practices
- **Raw Slide Footprint:** Raw WSIs range from 1 to 3 GB per slide. A 500-slide cohort is ~1 TB.
- **Streaming Pipeline:** SlideShift streams each slide, computes quality control metrics and tissue masks on low-resolution thumbnails, tiles at calibrated physical resolution ($0.50\ \mu\text{m/px}$), runs foundation model inference in half-precision, caches features to `.h5` / `.pt` bags (~20–40 MB per slide), and immediately discards the raw file.
- **Storage Savings:** A full 500-slide study reduces from 1 TB to under 20 GB!

