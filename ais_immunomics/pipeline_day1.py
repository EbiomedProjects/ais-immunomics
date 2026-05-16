"""
DAY 1 Pipeline: Data Layer + Immune Layer
===========================================
1. Cascade Probe Resolver (Innovation 1)
2. ComBat Cross-Platform Harmonization (Innovation 2 substrate)
3. AIMS Adaptive Immune Module Scoring (Innovation 2)
4. WGCNA Co-expression Network
5. Result compilation and visualization

Each step logs to console, validates outputs, and saves intermediate results.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys, os, time, json

# Add project root
sys.path.insert(0, str(Path(__file__).parent))
from bridge import check_r, run_r_script, r_to_csv, csv_from_r
from data.resolver import build_cascade_mapping, collapse_to_genes, get_cascade_coverage, print_coverage_report
from harmony.harmonizer import harmonize_cross_platform
from immuno.aims import AIMS, benchmark_aims, compare_with_standard

# === Configuration ===
RESULT_DIR = Path("D:/Ncz/results")
FIG_DIR = RESULT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

R_WGCNA = Path("D:/Ncz/ais_immunomics/network/wgcna.R")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def log_step(step_num, total, title):
    """Print formatted step header."""
    ts = time.strftime("%H:%M:%S", time.localtime())
    print(f"\n{'='*70}")
    print(f"[{ts}] STEP {step_num}/{total}: {title}")
    print(f"{'='*70}")

def check_output(filepath, description):
    """Verify output file exists and is non-trivial."""
    path = Path(filepath)
    if path.exists():
        size_kb = path.stat().st_size / 1024
        print(f"  [OK] {description}: {path.name} ({size_kb:.1f} KB)")
        return True
    else:
        print(f"  [FAIL] {description}: MISSING!")
        return False

# ================================================================
# STEP 0: Environment Verification
# ================================================================
log_step(0, 5, "Environment Verification")
r_ok, r_ver = check_r()
print(f"  R engine: {'OK' if r_ok else 'FAILED'} - {r_ver}")

import platform
print(f"  Python: {platform.python_version()}")
import sklearn, umap
print(f"  sklearn: {sklearn.__version__}, umap: {umap.__version__}")

# ================================================================
# STEP 1: Cascade Probe Resolver (Innovation 1)
# ================================================================
log_step(1, 5, "Cascade Probe Resolver - Multi-source annotation")

# Load expression data
print("\n  Loading expression matrices...")
expr_raw = {}
for gse_id in ['GSE16561', 'GSE22255', 'GSE37587']:
    df = pd.read_csv(RESULT_DIR / f"{gse_id}_expression.csv", index_col=0)
    df.index = df.index.str.strip('"')
    expr_raw[gse_id] = df
    print(f"  {gse_id}: {df.shape[0]} probes × {df.shape[1]} samples")

# Apply cascade resolver to each platform
coverage_report = {}
gene_exprs = {}
for gse_id, platform_id in [('GSE16561', 'GPL6883'), ('GSE22255', 'GPL570'), ('GSE37587', 'GPL6883')]:
    print(f"\n  --- {gse_id} ({platform_id}) ---")
    mapping, coverage = get_cascade_coverage(platform_id, expr_raw[gse_id].index)
    print_coverage_report(coverage)
    coverage_report[gse_id] = coverage

    gene_expr = collapse_to_genes(expr_raw[gse_id], mapping)
    gene_expr.to_csv(RESULT_DIR / f"{gse_id}_gene_expression_v2.csv")
    gene_exprs[gse_id] = gene_expr
    print(f"  Gene-level: {gene_expr.shape[0]} genes × {gene_expr.shape[1]} samples")

# Save coverage report
with open(RESULT_DIR / "coverage_report.json", 'w') as f:
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'tier_contributions'}
               for k, v in coverage_report.items()}, f, indent=2)

# Find common genes
gene_sets = [set(df.index) for df in gene_exprs.values()]
common_genes = gene_sets[0]
for gs in gene_sets[1:]:
    common_genes = common_genes & gs
print(f"\n  Common genes across all datasets: {len(common_genes)}")

# Restrict to common genes
for gse_id in gene_exprs:
    gene_exprs[gse_id] = gene_exprs[gse_id].loc[list(common_genes)]

# === Critical: Normalize data scales ===
print(f"\n  Data scale normalization:")
for gse_id in ['GSE16561', 'GSE22255', 'GSE37587']:
    vals = gene_exprs[gse_id].values.flatten()
    vals = vals[~np.isnan(vals)]
    med = np.median(vals)
    print(f"  {gse_id}: median={med:.2f}, 1%={np.percentile(vals,1):.2f}, 99%={np.percentile(vals,99):.2f}")

# GSE37587 is in linear scale -> log2 transform
print(f"\n  [FIX] Log2-transforming GSE37587 (was linear scale)...")
gse37587_expr = gene_exprs['GSE37587']
gse37587_expr = np.log2(gse37587_expr.clip(lower=1))
gene_exprs['GSE37587'] = gse37587_expr
vals = gse37587_expr.values.flatten()
print(f"  GSE37587 after log2: median={np.median(vals):.2f}, range=[{vals.min():.1f}, {vals.max():.1f}]")

# GSE16561 has NaN values -> fill with gene median
nan_count = gene_exprs['GSE16561'].isna().sum().sum()
if nan_count > 0:
    print(f"  [FIX] Filling {nan_count} NaN values in GSE16561 with gene median...")
    gse16561_filled = gene_exprs['GSE16561'].T.fillna(gene_exprs['GSE16561'].median(axis=1)).T
    gene_exprs['GSE16561'] = gse16561_filled
    remaining = gene_exprs['GSE16561'].isna().sum().sum()
    if remaining > 0:
        print(f"    {remaining} NaN remain after median fill -> filling with 0")
        gene_exprs['GSE16561'] = gene_exprs['GSE16561'].fillna(0)

# ================================================================
# STEP 2: ComBat Cross-Platform Harmonization (Innovation 2 substrate)
# ================================================================
log_step(2, 5, "ComBat Cross-Platform Harmonization")

# Run ComBat via R
harmonized, combat_metrics = harmonize_cross_platform(gene_exprs)

if harmonized is None:
    print("  WARNING: ComBat failed, using original expression")
    harmonized = gene_exprs

# Save harmonized data
for gse_id, df in harmonized.items():
    df.to_csv(RESULT_DIR / f"{gse_id}_harmonized.csv")
    check_output(RESULT_DIR / f"{gse_id}_harmonized.csv", f"{gse_id} harmonized")

# Save metrics
with open(RESULT_DIR / "combat_metrics.json", 'w') as f:
    json.dump(combat_metrics, f, indent=2, default=str)

# Generate pre/post PCA comparison plot
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Pre-ComBat
combined_raw = pd.concat([gene_exprs[gse] for gse in gene_exprs], axis=1)
batch_labels_raw = []
for gse in gene_exprs:
    batch_labels_raw.extend([gse] * gene_exprs[gse].shape[1])

X_raw = StandardScaler().fit_transform(np.nan_to_num(combined_raw.T.values, 0))
pca_raw = PCA(n_components=2).fit_transform(X_raw)

# Post-ComBat
combined_harm = pd.concat([harmonized[gse] for gse in harmonized], axis=1)
batch_labels_harm = []
for gse in harmonized:
    batch_labels_harm.extend([gse] * harmonized[gse].shape[1])

X_harm = StandardScaler().fit_transform(np.nan_to_num(combined_harm.T.values, 0))
pca_harm = PCA(n_components=2).fit_transform(X_harm)

for ax, pcs, title, labels in [
    (axes[0], pca_raw, 'Pre-ComBat', batch_labels_raw),
    (axes[1], pca_harm, 'Post-ComBat', batch_labels_harm)
]:
    for lbl in set(labels):
        mask = np.array(labels) == lbl
        ax.scatter(pcs[mask, 0], pcs[mask, 1], label=lbl, alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9)
    ax.set_xlabel(f'PC1'); ax.set_ylabel(f'PC2')

plt.tight_layout()
fig.savefig(FIG_DIR / "combat_pca_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  PCA comparison saved: figures/combat_pca_comparison.png")

# ================================================================
# STEP 3: AIMS Adaptive Immune Module Scoring (Innovation 2)
# ================================================================
log_step(3, 5, "AIMS Adaptive Immune Module Scoring")

# Use GSE16561 harmonized data for AIMS fitting (largest dataset, clearest signal)
print("\n  Fitting AIMS on GSE16561...")
aims = AIMS(use_weights=True, weight_method='intramodular_connectivity')
aims_scores_16561 = aims.fit_transform(harmonized['GSE16561'])
aims_scores_16561.to_csv(RESULT_DIR / "GSE16561_AIMS_scores.csv")

# Compute traditional z-score for benchmark
print("\n  Computing benchmark (simple z-score)...")
from immuno.aims import DEFAULT_MODULES
simple_scores = {}
for gse_id in ['GSE16561', 'GSE22255', 'GSE37587']:
    scores = pd.DataFrame(index=harmonized[gse_id].columns)
    for name, genes in DEFAULT_MODULES.items():
        available = [g for g in genes if g in harmonized[gse_id].index]
        if len(available) >= 3:
            sub = harmonized[gse_id].loc[available]
            z = sub.subtract(sub.mean(axis=1), axis=0).divide(sub.std(axis=1).replace(0, 1), axis=0)
            scores[name] = z.mean()
    simple_scores[gse_id] = scores

# Load group labels
meta_16561 = pd.read_csv(RESULT_DIR / "GSE16561_metadata.csv", index_col=0)
meta_16561['group'] = meta_16561['title'].apply(lambda x: 'Stroke' if 'Stroke' in str(x) else 'Control')

# AIMS benchmark
print("\n  --- AIMS Benchmark (GSE16561) ---")
bench = benchmark_aims(aims_scores_16561, meta_16561['group'], simple_scores['GSE16561'])
print(bench[['module', 'AUC', 'cohens_d', 'p_value']].to_string(index=False))

# Compare AIMS vs simple
comparison = compare_with_standard(aims_scores_16561, simple_scores['GSE16561'], meta_16561['group'])
print("\n  --- AIMS vs Simple Z-Score ---")
print(comparison.to_string(index=False))

bench.to_csv(RESULT_DIR / "AIMS_benchmark.csv", index=False)
comparison.to_csv(RESULT_DIR / "AIMS_vs_simple.csv", index=False)

# Compute AIMS for all datasets using weights from GSE16561
for gse_id in ['GSE22255', 'GSE37587']:
    aims_scores = aims.transform(harmonized[gse_id])
    aims_scores.to_csv(RESULT_DIR / f"{gse_id}_AIMS_scores.csv")

# AIMS comparison plot
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, module_name in enumerate(list(DEFAULT_MODULES.keys())[:6]):
    ax = axes[idx]
    data = []
    for gse_id in ['GSE16561', 'GSE22255', 'GSE37587']:
        scores = pd.read_csv(RESULT_DIR / f"{gse_id}_AIMS_scores.csv", index_col=0)
        col = f"{module_name}_AIMS"
        if col in scores.columns:
            if gse_id == 'GSE16561':
                meta = meta_16561
                for grp in ['Control', 'Stroke']:
                    vals = scores[col][meta[meta['group'] == grp].index.intersection(scores.index)].dropna()
                    data.append({'Dataset': f'{gse_id}_{grp}', 'Score': vals.median()})
            else:
                vals = scores[col].dropna()
                data.append({'Dataset': gse_id, 'Score': vals.median()})

    if data:
        pd.DataFrame(data).set_index('Dataset')['Score'].plot(kind='bar', ax=ax, color='steelblue')
    ax.set_title(f'{module_name}', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylabel('AIMS Score')

plt.suptitle('AIMS Adaptive Immune Module Scores Across Datasets', fontsize=15, y=1.01)
plt.tight_layout()
fig.savefig(FIG_DIR / "AIMS_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  AIMS comparison plot saved: figures/AIMS_comparison.png")

# ================================================================
# STEP 4: WGCNA Co-expression Network
# ================================================================
log_step(4, 5, "WGCNA Co-expression Network Analysis")

# Prepare data for WGCNA: use top variable genes to reduce computation
print("\n  Preparing WGCNA input...")
expr_for_wgcna = harmonized['GSE16561'].copy()
gene_vars = expr_for_wgcna.var(axis=1)
top_genes = gene_vars.nlargest(5000).index.tolist()
expr_for_wgcna = expr_for_wgcna.loc[top_genes]
print(f"  Top {len(top_genes)} variable genes for WGCNA")

# Save input
wgcna_input = RESULT_DIR / "_wgcna_expr.csv"
wgcna_trait = RESULT_DIR / "_wgcna_trait.csv"
r_to_csv(expr_for_wgcna, wgcna_input)

# Build trait data
aims_all = pd.read_csv(RESULT_DIR / "GSE16561_AIMS_scores.csv", index_col=0)
# Binary stroke trait
meta_16561['stroke_binary'] = (meta_16561['group'] == 'Stroke').astype(int)
trait_df = meta_16561[['stroke_binary']].copy()
# Add AIMS module scores as traits
for col in aims_all.columns:
    trait_df[col.replace('_AIMS', '')] = aims_all[col]
trait_df.to_csv(wgcna_trait)

print("\n  Running WGCNA (R)...")
wgcna_prefix = str(RESULT_DIR / "_wgcna")
ok, stdout, stderr, elapsed = run_r_script(
    R_WGCNA, [str(wgcna_input), str(wgcna_trait), wgcna_prefix], timeout=600
)
print(stdout)
if not ok:
    print(f"  WGCNA WARNING: {stderr[:500]}")

# Load WGCNA results
wgcna_outputs = {}
for suffix in ['_sft.csv', '_module_trait_cor.csv', '_module_trait_pvalue.csv',
               '_gene_modules.csv', '_connectivity.csv', '_hub_genes.csv']:
    fpath = RESULT_DIR / f"_wgcna{suffix}"
    if fpath.exists():
        wgcna_outputs[suffix] = pd.read_csv(fpath)
        print(f"  Loaded: _wgcna{suffix} ({wgcna_outputs[suffix].shape})")

# WGCNA Module-Trait heatmap
if '_module_trait_cor.csv' in wgcna_outputs:
    cor_df = pd.read_csv(RESULT_DIR / "_wgcna_module_trait_cor.csv", index_col=0)
    pval_df = pd.read_csv(RESULT_DIR / "_wgcna_module_trait_pvalue.csv", index_col=0)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(cor_df, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                ax=ax, cbar_kws={'label': 'Pearson r'})
    ax.set_title('WGCNA: Module-Trait Associations (GSE16561)', fontsize=14)
    ax.set_ylabel('Module (ME color)')
    plt.tight_layout()
    fig.savefig(FIG_DIR / "WGCNA_module_trait_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Module-trait heatmap saved: figures/WGCNA_module_trait_heatmap.png")

# Hub genes report
if '_hub_genes.csv' in wgcna_outputs:
    hub = wgcna_outputs['_hub_genes.csv']
    stroke_modules = []
    if '_module_trait_cor.csv' in wgcna_outputs:
        cor_df = pd.read_csv(RESULT_DIR / "_wgcna_module_trait_cor.csv", index_col=0)
        if 'stroke_binary' in cor_df.columns:
            # Find modules with |r| > 0.3 with stroke
            stroke_cors = cor_df['stroke_binary']
            stroke_modules = list(stroke_cors[abs(stroke_cors) > 0.3].index)
            stroke_modules = [s.replace('ME', '') for s in stroke_modules]
            print(f"\n  Stroke-associated modules (|r|>0.3): {stroke_modules}")

    hub_stroke = hub[hub['moduleLabel'].isin(stroke_modules)] if stroke_modules else pd.DataFrame()
    if not hub_stroke.empty:
        print(f"  Hub genes in stroke modules:\n{hub_stroke[['moduleLabel', 'gene', 'kWithin']].to_string(index=False)}")

# ================================================================
# STEP 5: Result Compilation & Summary
# ================================================================
log_step(5, 5, "Result Compilation & Summary Report")

# Generate comprehensive summary
summary = {
    'date': time.strftime("%Y-%m-%d %H:%M:%S"),
    'probe_coverage': coverage_report,
    'combat_metrics': combat_metrics,
    'aims_modules': list(DEFAULT_MODULES.keys()),
    'wgcna_params': {
        'genes_input': len(top_genes),
        'modules_found': len(wgcna_outputs.get('_gene_modules.csv', pd.DataFrame()).get('module', pd.Series()).unique()) if '_gene_modules.csv' in wgcna_outputs else 'N/A'
    },
    'common_genes': len(common_genes),
    'datasets': {
        'GSE16561': {'samples': gene_exprs['GSE16561'].shape[1], 'genes': gene_exprs['GSE16561'].shape[0]},
        'GSE22255': {'samples': gene_exprs['GSE22255'].shape[1], 'genes': gene_exprs['GSE22255'].shape[0]},
        'GSE37587': {'samples': gene_exprs['GSE37587'].shape[1], 'genes': gene_exprs['GSE37587'].shape[0]},
    }
}

with open(RESULT_DIR / "day1_summary.json", 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print(f"\n{'='*70}")
print(f"DAY 1 PIPELINE COMPLETE")
print(f"{'='*70}")
print(f"\nOutput files:")
for f in sorted(RESULT_DIR.glob("*AIMS*")) + sorted(RESULT_DIR.glob("*harmonized*")) + sorted(RESULT_DIR.glob("*_wgcna*")) + sorted(RESULT_DIR.glob("day1*")) + sorted(RESULT_DIR.glob("combat*")) + sorted(RESULT_DIR.glob("coverage*")):
    if f.is_file():
        print(f"  {f.name}")
for f in sorted(FIG_DIR.glob("*AIMS*")) + sorted(FIG_DIR.glob("*combat*")) + sorted(FIG_DIR.glob("*WGCNA*")):
    print(f"  figures/{f.name}")

print(f"\nFile count: {len(list(RESULT_DIR.glob('*'))) + len(list(FIG_DIR.glob('*')))}")
