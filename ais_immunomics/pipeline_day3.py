"""
DAY 3: Functional Enrichment + Benchmark + Final Report
========================================================
Step 11: GO/KEGG enrichment (gseapy + clusterProfiler R)
Step 12: Full benchmark comparison (AIMS vs ssGSEA/GSVA)
Step 13: Final comprehensive report generation
"""
import pandas as pd, numpy as np
from pathlib import Path
import sys, time, json, os

sys.path.insert(0, str(Path(__file__).parent))
from bridge import run_r_script

RESULT_DIR = Path("D:/Ncz/results")
FIG_DIR = RESULT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def log_step(n, title):
    print(f"\n{'='*70}")
    print(f"[{time.strftime('%H:%M:%S')}] STEP {n}: {title}")
    print(f"{'='*70}")

# ================================================================
# STEP 11: GO/KEGG Enrichment + GSEA
# ================================================================
log_step(11, "GO/KEGG Functional Enrichment & GSEA")

import gseapy as gp

# Load GSE16561 DE results
de = pd.read_csv(RESULT_DIR / "GSE16561_DE_results.csv")
print(f"\n  GSE16561: {len(de)} genes, {(de['p_adj']<0.05).sum()} sig (FDR<0.05)")
print(f"  Up-regulated (log2FC>0.3): {(de['p_adj']<0.05)&(de['log2FC']>0.3)} genes")
print(f"  Down-regulated (log2FC<-0.3): {(de['p_adj']<0.05)&(de['log2FC']<-0.3)} genes")

# --- GO Enrichment ---
print("\n  Running GO enrichment (gseapy)...")
up_genes = de[(de['p_adj'] < 0.05) & (de['log2FC'] > 0.5)]['gene'].tolist()
down_genes = de[(de['p_adj'] < 0.05) & (de['log2FC'] < -0.3)]['gene'].tolist()

go_results = {}
for direction, genes, title in [
    ('up', up_genes, 'Up-regulated in Stroke'),
    ('down', down_genes, 'Down-regulated in Stroke')
]:
    if len(genes) < 5:
        print(f"    {title}: too few genes ({len(genes)}), skipping")
        continue
    print(f"    {title}: {len(genes)} genes")
    try:
        enr = gp.enrichr(
            gene_list=genes,
            gene_sets=['GO_Biological_Process_2023', 'KEGG_2021_Human', 'Reactome_2022'],
            organism='human',
            outdir=None,
            no_plot=True
        )
        if enr and hasattr(enr, 'results'):
            go_res = enr.results
            go_results[direction] = go_res
            # Top terms
            for lib in go_res['Gene_set'].unique()[:1]:
                top = go_res[go_res['Gene_set']==lib].head(10)
                print(f"      [{lib}] Top 5:")
                for _, row in top.head(5).iterrows():
                    print(f"        {row['Term'][:60]}: p={row['Adjusted P-value']:.1e}, overlap={row['Overlap']}")
            go_res.to_csv(RESULT_DIR / f"GO_enrichment_{direction}.csv", index=False)
    except Exception as e:
        print(f"      Error: {e}")

# --- GSEA (pre-ranked) ---
print("\n  Running GSEA pre-ranked analysis...")
de_clean = de.dropna(subset=['log2FC', 'p_value'])
de_clean = de_clean.sort_values('log2FC', ascending=False)
rnk = de_clean[['gene', 'log2FC']].copy()
rnk = rnk.set_index('gene')

# Use gseapy prerank
try:
    gsea_res = gp.prerank(
        rnk=rnk,
        gene_sets='KEGG_2021_Human',
        permutation_num=100,
        outdir=str(RESULT_DIR / "_gsea"),
        seed=42,
        min_size=5,
        max_size=500
    )
    gsea_df = gsea_res.results if hasattr(gsea_res, 'results') else gsea_res.get('results')
    if gsea_df is not None and hasattr(gsea_df, '__len__') and len(gsea_df) > 0:
        if not hasattr(gsea_df, 'to_csv'):
            gsea_df = pd.DataFrame(gsea_df)
        gsea_df.to_csv(RESULT_DIR / "GSEA_prerank_results.csv", index=False)
        sig_gsea = gsea_df[gsea_df['fdr'] < 0.25]
        print(f"  GSEA terms (FDR<0.25): {len(sig_gsea)}")
        for _, row in sig_gsea.head(10).iterrows():
            direction = 'UP' if row['nes'] > 0 else 'DOWN'
            print(f"    {row['Term'][:55]}: NES={row['nes']:+.2f} [{direction}], FDR={row['fdr']:.3f}")
    else:
        print("  GSEA returned no results")
except Exception as e:
    print(f"  GSEA error: {e}")

# --- GO enrichment plot ---
if go_results:
    fig, axes = plt.subplots(2, 1, figsize=(14, 16))
    for idx, (direction, title) in enumerate([('up', 'Up-regulated'), ('down', 'Down-regulated')]):
        if direction not in go_results:
            continue
        ax = axes[idx]
        df = go_results[direction]
        go_bp = df[df['Gene_set'] == 'GO_Biological_Process_2023'].head(15)
        if len(go_bp) > 0:
            terms = [t[:70] for t in go_bp['Term']]
            pvals = [-np.log10(go_bp['Adjusted P-value'].values)]
            # Actually let's use the overlap ratio
            overlap_pct = []
            for ov in go_bp['Overlap']:
                parts = ov.split('/')
                overlap_pct.append(int(parts[0]) / int(parts[1]) * 100)
            colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(terms)))
            ax.barh(range(len(terms)), overlap_pct, color=colors)
            ax.set_yticks(range(len(terms)))
            ax.set_yticklabels(terms, fontsize=8)
            ax.set_xlabel('Gene Ratio (%)')
            ax.set_title(f'GO Biological Process: {title} in Stroke', fontsize=13)
            ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(FIG_DIR / "GO_enrichment.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  GO enrichment plot saved: figures/GO_enrichment.png")

# ================================================================
# STEP 12: Full Benchmark Comparison
# ================================================================
log_step(12, "Comprehensive Benchmark: AIMS vs Baseline Methods")

print("\n  Comparing methods across all modules:")

# Load results
aims_scores = pd.read_csv(RESULT_DIR / "GSE16561_AIMS_scores.csv", index_col=0)
simple = pd.read_csv(RESULT_DIR / "GSE16561_module_scores.csv", index_col=0)
meta_16561 = pd.read_csv(RESULT_DIR / "GSE16561_metadata.csv", index_col=0)
meta_16561['group'] = meta_16561['title'].apply(lambda x: 'Stroke' if 'Stroke' in str(x) else 'Control')
y = (meta_16561['group'] == 'Stroke').astype(int)

from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

# Also run GSVA via R for comparison
print("\n  Running GSVA benchmark (R)...")

# Write expression for GSVA
expr_path = RESULT_DIR / "_gsva_expr.csv"
harmonized = pd.read_csv(RESULT_DIR / "GSE16561_harmonized.csv", index_col=0)
harmonized.to_csv(expr_path)

# Build GSVA gene sets as GMT
from immuno.aims import DEFAULT_MODULES
gmt_path = RESULT_DIR / "_gsva_genesets.gmt"
with open(gmt_path, 'w') as f:
    for name, genes in DEFAULT_MODULES.items():
        available = [g for g in genes if g in harmonized.index]
        if len(available) >= 3:
            f.write(f"{name}\t{name}\t" + "\t".join(available) + "\n")

# GSVA R script
gsva_r = """
args <- commandArgs(trailingOnly=TRUE)
suppressPackageStartupMessages(library(GSVA))
expr <- as.matrix(read.csv(args[1], row.names=1, check.names=FALSE))
gmt <- args[2]
gsva_res <- gsva(expr, gmt, method="ssgsea", kcdf="Gaussian", min.sz=2)
write.csv(t(gsva_res), file=args[3])
"""
gsva_r_path = RESULT_DIR / "_gsva_run.R"
with open(gsva_r_path, 'w') as f:
    f.write(gsva_r)

gsva_out = RESULT_DIR / "_gsva_scores.csv"
ok, stdout, stderr, elapsed = run_r_script(
    gsva_r_path, [str(expr_path), str(gmt_path), str(gsva_out)], timeout=120
)

# Compile benchmark
bench_rows = []
module_names = [c.replace('_AIMS', '') for c in aims_scores.columns if c.endswith('_AIMS')]

for mod in module_names:
    row = {'module': mod}

    # AIMS
    aims_col = f"{mod}_AIMS"
    if aims_col in aims_scores.columns:
        row['AIMS_AUC'] = roc_auc_score(y.loc[aims_scores.index], aims_scores[aims_col])

    # Simple z-score
    if mod in simple.columns:
        row['Simple_AUC'] = roc_auc_score(y.loc[simple.index], simple[mod])

    # GSVA
    if ok and gsva_out.exists():
        gsva_df = pd.read_csv(gsva_out, index_col=0)
        if mod in gsva_df.columns:
            common = gsva_df.index.intersection(y.index)
            if len(common) > 5:
                row['GSVA_AUC'] = roc_auc_score(y.loc[common], gsva_df.loc[common, mod])

    # Correlation between methods
    if aims_col in aims_scores.columns and mod in simple.columns:
        common_idx = aims_scores.index.intersection(simple.index)
        r, p = pearsonr(aims_scores.loc[common_idx, aims_col], simple.loc[common_idx, mod])
        row['AIMS_vs_Simple_r'] = r

    bench_rows.append(row)

bench_df = pd.DataFrame(bench_rows)
print("\n  Benchmark Results:")
print(bench_df.to_string(index=False))
bench_df.to_csv(RESULT_DIR / "full_benchmark.csv", index=False)

# Benchmark plot
fig, ax = plt.subplots(figsize=(10, 8))
x = np.arange(len(bench_df))
width = 0.25
for i, (method, col, color) in enumerate([
    ('AIMS', 'AIMS_AUC', '#E74C3C'),
    ('Simple', 'Simple_AUC', '#3498DB'),
    ('GSVA', 'GSVA_AUC', '#2ECC71')
]):
    vals = bench_df[col].values if col in bench_df.columns else [0]*len(bench_df)
    ax.bar(x + i*width, vals, width, label=method, color=color, alpha=0.8)

ax.set_xticks(x + width)
ax.set_xticklabels(bench_df['module'], rotation=45, ha='right')
ax.set_ylabel('AUC (Stroke vs Control)')
ax.set_title('Immune Module Scoring: Method Comparison', fontsize=14)
ax.legend()
ax.axhline(0.5, color='grey', ls='--', alpha=0.5, label='Random')
plt.tight_layout()
fig.savefig(FIG_DIR / "benchmark_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Benchmark plot: figures/benchmark_comparison.png")

# ================================================================
# STEP 13: Final Comprehensive Report
# ================================================================
log_step(13, "Final Comprehensive Report")

report = f"""# AIS-Immunomics: 临床级多队列转录组智能诊疗系统
## 最终分析报告

**日期:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**数据:** 3个GEO数据集 (GSE16561, GSE22255, GSE37587), 171样本, 13,278共同基因

---

### 1. 多源探针级联解析器 (创新1)

| 数据集 | 平台 | 覆盖率 |
|--------|------|--------|
| GSE16561 | GPL6883 | 100.0% (24,424/24,424) |
| GSE22255 | GPL570 | 86.8% (47,440/54,675) |
| GSE37587 | GPL6883 | 100.0% (24,526/24,526) |

三级级联策略: GEO(83.7%) → BioMart(+2.2%) → Bioconductor+MyGene(+0.9%)

### 2. 跨平台批次校正 (ComBat)

| 指标 | 校正前 | 校正后 |
|------|--------|--------|
| 批次PCA方差 | 0.995 | 0.000 |
| 组内相关性 | 0.654 | 0.917 |
| 组间相关性 | 0.226 | 0.916 |

### 3. 免疫模块评分 (AIMS)

| 模块 | AIMS AUC | 简单z-score AUC | Δ |
|------|---------|-----------------|-----|
"""
for _, row in bench_df.iterrows():
    aims = row.get('AIMS_AUC', float('nan'))
    simple = row.get('Simple_AUC', float('nan'))
    delta = aims - simple if not (pd.isna(aims) or pd.isna(simple)) else float('nan')
    report += f"| {row['module']} | {aims:.3f} | {simple:.3f} | {delta:+.3f} |\n"

report += f"""
### 4. WGCNA共表达网络

- 软阈值: power=7, 8个共表达模块
- ME3 ~ NETosis: r=+0.883, p=1.1×10⁻²¹
- ME1 ~ Kynurenine: r=+0.971, p=1.1×10⁻³⁹
- ME4 ~ Stroke (负相关): r=-0.652, p=7.0×10⁻⁹

### 5. SNF免疫分子亚型 (创新3)

- 3网络融合 (表达+AIMS+WGCNA), 4个亚型
- 与临床分组高度相关: Chi²=25.4, p<0.0001
- 亚型1: 纯卒中 (14/14), 亚型4: 富集对照 (12/16)

### 6. 28基因诊断Panel (创新4)

- LASSO筛选 28基因 from 13,278
- 5-fold CV AUC: 0.990 ± 0.020
- 跨组织验证 (GSE22255 PBMC): AUC=0.528
- GSE37587基线预测: 79%正确分类为卒中

**28基因Panel:**
ALG9, ANTXR2, C5AR1, CHD1L, CTSS, CTSZ, DFFB, DOCK8, FGL2, HEATR1,
HMGCR, LAMP2, LIG1, MARCKS, MRPL49, MTPN, NPEPPS, PDK4, PLXDC2,
PPIH, RGS2, ST8SIA4, STK3, TLE4, TNFSF13B, VNN3, ZNF134, ZNF419

### 7. 药物重定位 (创新5)

| 药物 | p值 | 来源 |
|------|-----|------|
| Valproic Acid | 6.4×10⁻¹² | GEO Perturbations |
| Arsenic Trioxide | 5.2×10⁻¹³ | GEO Perturbations |
| Doxorubicin | 4.7×10⁻¹³ | DrugMatrix |
| Thioguanine | 1.8×10⁻¹⁰ | DrugMatrix |
| Oxaliplatin | 4.1×10⁻¹⁰ | DrugMatrix |

### 8. 功能富集分析

GO/KEGG富集和GSEA预排名分析结果详见:
- `GO_enrichment_up.csv` / `GO_enrichment_down.csv`
- `GSEA_prerank_results.csv`

### 9. 所有产出文件

**数据文件 (20+):** 基因表达 (raw + gene-level + harmonized) ×3, DE结果 ×3, AIMS评分 ×3, WGCNA ×6, SNF ×4
**图表 (17):** UMAP ×3, Volcano ×3, Module Boxplot ×3, ComBat PCA, AIMS Comparison, WGCNA Heatmap, SNF Subtypes, SHAP Summary, ROC, Meta Scatter, GO Enrichment, Benchmark, Day2 Summary
**报告 (5 JSON):** coverage, combat_metrics, day1_summary, diagnostic_model, drug_repurposing

---
*系统运行环境: Python 3.13 + R 4.6.0, NVIDIA RTX 4060*
"""

report_path = RESULT_DIR / "FINAL_REPORT.md"
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)
print(f"\n  Final report saved: {report_path}")
print(f"  Report size: {report_path.stat().st_size} bytes")

print(f"\n{'='*70}")
print(f"DAY 3 + FULL PIPELINE COMPLETE")
print(f"{'='*70}")
print(f"\nTotal output files: {len(list(RESULT_DIR.glob('**/*')))}")
print(f"Figures: {len(list(FIG_DIR.glob('*.png')))}")
print(f"Final report: {report_path}")
