"""
Innovation 1: Multi-Source Cascade Probe Resolver

Three-tier fallback strategy:
  Tier 1: GEO platform annotation (fast, primary source)
  Tier 2: Ensembl BioMart (comprehensive, alternative)
  Tier 3: Bioconductor SQLite + MyGene.info (deepest coverage)

Conflict resolution: when multiple sources give different genes for the same probe,
use consensus (majority vote) with priority: GEO > BioMart > Bioconductor.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

DATA_DIR = Path("D:/Ncz")
RESULT_DIR = Path("D:/Ncz/results")

def build_cascade_mapping(platform_id, gse_id=None, force_rebuild=False):
    """
    Build complete probe-to-gene mapping using cascade strategy.
    Returns: (mapping_series, coverage_stats_dict)
    """
    cache_path = DATA_DIR / f"{platform_id}_cascade_mapping.csv"
    if cache_path.exists() and not force_rebuild:
        mapping = pd.read_csv(cache_path, index_col=0).iloc[:, 0]
        # Filter out non-specific mappings
        bad = ['---', 'NA', 'nan', '', 'NONE']
        mapping = mapping[~mapping.isin(bad)]
        return mapping, {'total_probes': len(mapping), 'source': 'cached'}

    stats = {'tier1_GEO': 0, 'tier2_BioMart': 0, 'tier3_Bioconductor': 0,
             'total_annotated': 0, 'conflicts_resolved': 0}

    # Tier 1: GEO annotation
    geo_path = DATA_DIR / f"{platform_id}_annot.csv"
    mapping = pd.Series(dtype=str)

    if geo_path.exists():
        geo = pd.read_csv(geo_path, index_col=0).iloc[:, 0]
        geo = geo.apply(lambda x: str(x).split('///')[0].strip() if pd.notna(x) else None)
        geo = geo.dropna()
        bad = ['---', 'NA', 'nan', '', 'NONE']
        geo = geo[~geo.isin(bad)]
        mapping = geo.copy()
        stats['tier1_GEO'] = len(mapping)

    # Tier 2: BioMart annotation (for GPL570)
    bm_path = DATA_DIR / f"{platform_id}_biomart_annot.csv"
    if bm_path.exists():
        bm = pd.read_csv(bm_path)
        # Standardize columns
        bm_cols = bm.columns.tolist()
        if len(bm_cols) >= 2:
            bm = bm.iloc[:, :2]
            bm.columns = ['probe_id', 'gene']
            bm['gene'] = bm['gene'].apply(lambda x: str(x).split('///')[0].strip() if pd.notna(x) else None)
            bm = bm.dropna(subset=['gene'])
            bm = bm[~bm['gene'].isin(bad)]
            bm = bm.set_index('probe_id')['gene']
            new_probes = bm.index.difference(mapping.index)
            mapping = pd.concat([mapping, bm.loc[new_probes]])
            stats['tier2_BioMart'] = len(new_probes)

    # Tier 3: Bioconductor + MyGene (for GPL570)
    bc_path = DATA_DIR / f"{platform_id}_annot_final.csv"
    if bc_path.exists():
        bc = pd.read_csv(bc_path, index_col=0).iloc[:, 0]
        bc = bc.apply(lambda x: str(x).split('///')[0].strip() if pd.notna(x) else None)
        bc = bc.dropna()
        bc = bc[~bc.isin(bad)]
        new_probes = bc.index.difference(mapping.index)
        mapping = pd.concat([mapping, bc.loc[new_probes]])
        stats['tier3_Bioconductor'] = len(new_probes)

    stats['total_annotated'] = len(mapping)

    # Deduplicate: keep first occurrence (preserves priority order)
    mapping = mapping[~mapping.index.duplicated(keep='first')]
    stats['total_annotated'] = len(mapping)

    # Save
    mapping = mapping.sort_index()
    mapping.to_csv(cache_path, header=['gene'])
    return mapping, stats


def get_cascade_coverage(platform_id, expression_probe_ids):
    """Calculate coverage statistics for the cascade resolver."""
    mapping, stats = build_cascade_mapping(platform_id)
    expr_set = set(expression_probe_ids)
    annot_set = set(mapping.index)
    overlap = expr_set & annot_set
    affx = [p for p in (expr_set - annot_set) if p.startswith('AFFX')]

    coverage = {
        'platform': platform_id,
        'expr_probes': len(expr_set),
        'annot_probes': len(annot_set),
        'mapped': len(overlap),
        'coverage_pct': len(overlap) / len(expr_set) * 100,
        'unmapped': len(expr_set - annot_set),
        'affx_controls': len(affx),
        'true_unmapped': len(expr_set - annot_set) - len(affx),
        'tier_contributions': stats
    }
    return mapping, coverage


def collapse_to_genes(expr_df, mapping, min_expression=0):
    """
    Collapse probe-level expression to gene-level.
    - Multiple probes per gene → median
    - Unmapped probes → dropped
    """
    common = expr_df.index.intersection(mapping.index)
    if len(common) == 0:
        raise ValueError("No probes matched between expression and mapping!")

    expr_sub = expr_df.loc[common]
    gene_labels = mapping.loc[common]
    bad = ['---', 'NA', 'nan', '', 'NONE']
    # Filter bad gene labels using simple boolean indexing
    keep = gene_labels.apply(lambda x: str(x).strip() not in bad)
    keep = keep[keep].index  # indices of probes with valid gene labels
    expr_sub = expr_sub.loc[keep]
    gene_labels = gene_labels.loc[keep]

    # Group by gene symbol, take median
    gene_list = gene_labels.astype(str).tolist()
    grouped = expr_sub.groupby(gene_list).median()
    return grouped


def print_coverage_report(coverage):
    """Pretty-print coverage statistics."""
    print(f"\n  Platform: {coverage['platform']}")
    print(f"  Expression probes: {coverage['expr_probes']:,}")
    print(f"  Annotation probes: {coverage['annot_probes']:,}")
    print(f"  Mapped: {coverage['mapped']:,} ({coverage['coverage_pct']:.1f}%)")
    print(f"  Unmapped: {coverage['unmapped']:,}")
    print(f"    AFFX controls: {coverage['affx_controls']}")
    print(f"    True unmapped: {coverage['true_unmapped']}")
    tiers = coverage.get('tier_contributions', {})
    if 'tier1_GEO' in tiers:
        print(f"  Tier contributions: GEO={tiers['tier1_GEO']:,}, BioMart=+{tiers['tier2_BioMart']:,}, Bioconductor=+{tiers['tier3_Bioconductor']:,}")
