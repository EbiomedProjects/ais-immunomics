"""
Cross-Platform Harmonizer: batch correction + platform integration.
Wraps R/ComBat via subprocess for robust cross-platform normalization.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys, os, json

# Add parent for bridge import
sys.path.insert(0, str(Path(__file__).parent.parent))
from bridge import run_r_script, r_to_csv, csv_from_r

RESULT_DIR = Path("D:/Ncz/results")
COMBAT_R = Path("D:/Ncz/ais_immunomics/harmony/combat.R")


def harmonize_cross_platform(gene_exprs_dict, platform_map=None):
    """
    Apply ComBat to harmonize gene expression across platforms.

    Parameters:
    - gene_exprs_dict: {gse_id: DataFrame} gene-level expression matrices
    - platform_map: {gse_id: platform_id} optional platform mapping

    Returns:
    - harmonized_df: combined DataFrame with batch-corrected expression
    - quality_metrics: dict of pre/post batch effect metrics
    """
    if platform_map is None:
        platform_map = {'GSE16561': 'GPL6883', 'GSE22255': 'GPL570', 'GSE37587': 'GPL6883'}

    print("\n[Harmonizer] Cross-Platform Batch Correction")
    print(f"  Datasets: {list(gene_exprs_dict.keys())}")

    # Find common genes
    gene_sets = [set(df.index) for df in gene_exprs_dict.values()]
    common_genes = gene_sets[0]
    for gs in gene_sets[1:]:
        common_genes = common_genes & gs
    print(f"  Common genes: {len(common_genes)}")

    # Assemble combined matrix
    combined_parts = []
    batch_labels = []
    for gse_id, expr_df in gene_exprs_dict.items():
        sub = expr_df.loc[list(common_genes)]
        combined_parts.append(sub)
        platform = platform_map.get(gse_id, 'unknown')
        batch_labels.extend([f"{gse_id}_{platform}"] * sub.shape[1])

    combined = pd.concat(combined_parts, axis=1, sort=True)
    print(f"  Combined matrix: {combined.shape[0]} genes × {combined.shape[1]} samples")

    # Pre-batch-effect metrics
    metrics = {'pre': _batch_metrics(combined, batch_labels)}

    # Save input for R
    input_csv = RESULT_DIR / "_combat_input.csv"
    batch_csv = RESULT_DIR / "_combat_batches.csv"
    output_csv = RESULT_DIR / "_combat_output.csv"

    r_to_csv(combined, input_csv)
    batch_df = pd.DataFrame({'batch': batch_labels}, index=combined.columns)
    r_to_csv(batch_df, batch_csv)

    # Run ComBat
    print(f"\n  Running ComBat (R/sva)...")
    ok, stdout, stderr, elapsed = run_r_script(
        COMBAT_R, [str(input_csv), str(batch_csv), str(output_csv)], timeout=300
    )
    print(stdout)
    if not ok:
        print(f"  ComBat ERROR: {stderr}")
        return None, metrics

    # Load corrected data
    corrected = csv_from_r(output_csv)
    print(f"  ComBat completed in {elapsed:.1f}s")

    # Post-batch-effect metrics
    metrics['post'] = _batch_metrics(corrected, batch_labels)
    metrics['runtime_s'] = elapsed
    metrics['genes'] = len(common_genes)
    metrics['samples'] = combined.shape[1]

    # Print comparison
    _print_batch_effect_report(metrics)

    # Clean temp files
    for f in [input_csv, batch_csv, output_csv]:
        if f.exists():
            f.unlink()

    # Split corrected matrix back to per-dataset
    harmonized_dict = {}
    offset = 0
    for gse_id in gene_exprs_dict:
        n = gene_exprs_dict[gse_id].shape[1]
        harmonized_dict[gse_id] = corrected.iloc[:, offset:offset+n]
        offset += n

    return harmonized_dict, metrics


def _batch_metrics(expr_df, batch_labels):
    """Compute batch effect metrics."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from scipy import stats

    # PCA-based batch variance explained
    X = StandardScaler().fit_transform(expr_df.T.values)
    X = np.nan_to_num(X, 0)
    pca = PCA(n_components=min(10, X.shape[0]-1))
    pcs = pca.fit_transform(X)

    # For each PC, test association with batch
    batches = pd.Series(batch_labels, index=expr_df.columns)
    batch_var_explained = 0
    for i in range(min(5, pcs.shape[1])):
        pc = pcs[:, i]
        # ANOVA of PC ~ batch
        batch_groups = [pc[batches == b] for b in batches.unique()]
        if len(batch_groups) >= 2:
            f_stat, p_val = stats.f_oneway(*batch_groups)
            if p_val < 0.05:
                batch_var_explained += pca.explained_variance_ratio_[i]

    # Average pairwise correlation within vs between batches
    unique_batches = list(batches.unique())
    within_corrs = []
    between_corrs = []
    for i, b1 in enumerate(unique_batches):
        s1 = expr_df.columns[batches == b1]
        for j, b2 in enumerate(unique_batches):
            s2 = expr_df.columns[batches == b2]
            if len(s1) < 2 or len(s2) < 2:
                continue
            corr = np.nanmean(np.corrcoef(expr_df[s1].T, expr_df[s2].T)[:len(s1), len(s1):])
            if not np.isnan(corr):
                if i == j:
                    within_corrs.append(corr)
                else:
                    between_corrs.append(corr)

    return {
        'batch_pca_variance': float(batch_var_explained),
        'within_batch_corr': float(np.nanmean(within_corrs)) if within_corrs else 0.0,
        'between_batch_corr': float(np.nanmean(between_corrs)) if between_corrs else 0.0,
        'corr_ratio': float(np.nanmean(within_corrs) / np.nanmean(between_corrs)) if within_corrs and between_corrs and np.nanmean(between_corrs) > 0 else 1.0,
    }


def _print_batch_effect_report(metrics):
    """Pretty-print batch effect comparison."""
    pre = metrics['pre']
    post = metrics['post']
    print(f"\n  {'Metric':<30s} {'Pre-ComBat':>12s} {'Post-ComBat':>12s} {'Improvement':>12s}")
    print(f"  {'-'*66}")
    for key in ['batch_pca_variance', 'within_batch_corr', 'between_batch_corr']:
        pre_v = pre.get(key, 0)
        post_v = post.get(key, 0)
        if key == 'batch_pca_variance':
            if pre_v > 0 and not np.isnan(pre_v) and not np.isnan(post_v):
                improvement = f"{(pre_v-post_v)/pre_v*100:+.1f}%"
            else:
                improvement = 'N/A'
        else:
            if not np.isnan(pre_v) and not np.isnan(post_v):
                improvement = f"{post_v-pre_v:+.4f}"
            else:
                improvement = 'N/A'
        label = key.replace('_', ' ').title()
        print(f"  {label:<30s} {pre_v:>12.4f} {post_v:>12.4f} {improvement:>12s}")
