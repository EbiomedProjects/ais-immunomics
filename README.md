# AIS-Immunomics: Clinical-Grade Multi-Cohort Transcriptomic Intelligence System

[![Python 3.13+](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![R 4.6.0](https://img.shields.io/badge/R-4.6.0-276DC2.svg)](https://www.r-project.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**A systematic computational framework for immune landscape characterization and clinical decision support in acute ischemic stroke.**

> **Key finding:** Neutrophil degranulation is the dominant immune signal in stroke peripheral blood (Δ=+0.805, t=7.15, p<0.001), validated by GO enrichment (p=2.1×10⁻⁴⁰) and WGCNA co-expression network (r=0.883, p=1.1×10⁻²¹).

---

## System Architecture

A 6-layer clinical-grade framework integrating Python ML and R biostatistics:

```
┌─────────────────────────────────────────────────┐
│  Layer 6: Clinical Decision Output               │
│  Layer 5: Multi-Task Predictive Models (ML)      │
│  Layer 4: Immune Landscape Analysis (Python/R)   │
│  Layer 3: Cross-Platform Data Integration        │
│  Layer 2: Multi-Source Data Ingestion            │
│  Layer 1: Infrastructure (Python + R + GPU)      │
└─────────────────────────────────────────────────┘
```

## 5 Algorithmic Innovations

| # | Innovation | Method | Key Metric |
|---|-----------|--------|------------|
| 1 | **Cascade Probe Resolver** | GEO → BioMart → Bioconductor 3-tier fallback | Coverage: 83%→87% |
| 2 | **AIMS** (Adaptive Immune Module Scoring) | Co-expression network connectivity-weighted scoring | AUC up to 0.911 |
| 3 | **SNF Multi-Network Subtyping** | 3-view similarity network fusion + spectral clustering | 4 subtypes, Chi²=25.4, p<0.0001 |
| 4 | **SHAP-Driven Diagnostic Panel** | LASSO → RF → SHAP interaction-value feature optimization | 28-gene panel, CV AUC=0.990 |
| 5 | **Heterogeneous Design Meta-Synthesis** | Fisher's method + Temporal Responsiveness Index | 2,916 genes FDR<0.05 |

## Dataset

| GEO ID | Platform | Samples | Design | Tissue |
|--------|----------|---------|--------|--------|
| [GSE16561](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE16561) | GPL6883 (Illumina) | 39 Stroke + 24 Control | Case-Control | Whole Blood |
| [GSE22255](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE22255) | GPL570 (Affymetrix) | 20 IS + 20 Control | Case-Control | PBMC |
| [GSE37587](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE37587) | GPL6883 (Illumina) | 34 patients × 2 timepoints | Paired Temporal | PBMC |

**Total: 171 samples, 13,278 common genes across 3 platforms.**

## Key Biological Findings

### Immune Module Scores (GSE16561)

| Module | Stroke | Control | Δ | t-stat | AUC |
|--------|--------|---------|---|--------|-----|
| NETosis | +0.306 | -0.499 | **+0.805** | 7.15*** | 0.899 |
| Metabolic Stress | +0.131 | -0.210 | **+0.341** | 7.72*** | 0.911 |
| Inflammasome | +0.176 | -0.286 | **+0.462** | 4.65* | 0.783 |
| Complement | +0.209 | -0.330 | **+0.540** | 3.37* | 0.693 |

### GO/KEGG Enrichment

**Up-regulated pathways (355 genes):**
- Neutrophil Degranulation: p=2.1×10⁻⁴⁰ (Reactome)
- Innate Immune System: p=2.7×10⁻⁴⁰ (Reactome)
- NET Formation: p=3.0×10⁻⁶ (KEGG)
- Inflammatory Response: p=1.0×10⁻⁷ (GO:BP)

**Down-regulated pathways (480 genes):**
- Translation / Ribosome: p=2.3×10⁻¹¹ (GO:BP)
- T cell receptor signaling: p=1.2×10⁻⁵ (KEGG)
- Th17 cell differentiation: p=8.5×10⁻⁵ (KEGG)

### 28-Gene Diagnostic Panel

```
ALG9, ANTXR2, C5AR1, CHD1L, CTSS, CTSZ, DFFB, DOCK8, FGL2, HEATR1,
HMGCR, LAMP2, LIG1, MARCKS, MRPL49, MTPN, NPEPPS, PDK4, PLXDC2,
PPIH, RGS2, ST8SIA4, STK3, TLE4, TNFSF13B, VNN3, ZNF134, ZNF419
```

### Drug Repurposing Candidates

| Drug | p-value | Mechanism |
|------|---------|-----------|
| Valproic Acid | 6.4×10⁻¹² | HDAC inhibitor (stroke clinical trials) |
| Arsenic Trioxide | 5.2×10⁻¹³ | Anti-inflammatory |
| Doxorubicin | 4.7×10⁻¹³ | Immunomodulator |
| Thioguanine | 1.8×10⁻¹⁰ | Purine analog, immunosuppressant |

## Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/ais-immunomics.git
cd ais-immunomics

# Install Python dependencies
pip install pandas numpy scipy scikit-learn matplotlib seaborn umap-learn gseapy shap GEOparse mygene statsmodels

# R packages required (install in R/RStudio):
# install.packages(c("WGCNA", "sva", "limma", "GSVA", "clusterProfiler", "SNFtool"))
```

## Usage

### Full Pipeline (3 stages)

```bash
# Stage 1: Data + Immune Layer (probe mapping, ComBat, AIMS, WGCNA)
python ais_immunomics/pipeline_day1.py

# Stage 2: AI + Clinical Layer (SNF subtyping, ML diagnostic panel, drug repurposing)
python ais_immunomics/pipeline_day2.py

# Stage 3: Enrichment + Benchmark (GO/KEGG, method comparison, final report)
python ais_immunomics/pipeline_day3.py
```

### Individual Modules

```python
# Cascade Probe Resolver (Innovation 1)
from ais_immunomics.data.resolver import CascadeResolver
resolver = CascadeResolver()
gene_expr = resolver.resolve('GPL570', expression_df)

# AIMS Scoring (Innovation 2)
from ais_immunomics.immuno.aims import AIMS
aims = AIMS(use_weights=True)
scores = aims.fit_transform(gene_expression_df)

# Cross-Platform Harmonization
from ais_immunomics.harmony.harmonizer import harmonize_cross_platform
harmonized, metrics = harmonize_cross_platform(gene_exprs_dict)
```

### R Integration

```python
from ais_immunomics.bridge import run_r_script
ok, stdout, stderr, elapsed = run_r_script('path/to/script.R', args=[...])
```

## Requirements

- **Python** 3.10+ (tested on 3.13)
- **R** 4.0+ (tested on 4.6.0) with packages: WGCNA, sva, limma, GSVA, clusterProfiler, SNFtool
- **GPU** optional (for Autoencoder; RTX 4060 tested)
- **OS**: Windows/Linux/macOS

## Output Structure

```
results/
├── GO_enrichment_up.csv / GO_enrichment_down.csv
├── GSE*_gene_expression_v2.csv       (gene-level expression)
├── GSE*_harmonized.csv               (ComBat-corrected)
├── GSE*_AIMS_scores.csv              (immune module scores)
├── GSE*_DE_results.csv               (differential expression)
├── meta_analysis_results.csv         (cross-study synthesis)
├── wgcna_*.csv                       (co-expression modules)
├── snf_*.csv                         (immune subtypes)
├── full_benchmark.csv                (method comparison)
├── diagnostic_model.json             (28-gene panel)
├── drug_repurposing.json             (drug candidates)
├── FINAL_REPORT.md                   (comprehensive report)
└── figures/                          (10 publication-ready figures)
```



## Author

[Haocheng Zhao] - [Huazhong University of Science and Technology]
