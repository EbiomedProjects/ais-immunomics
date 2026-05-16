"""
Innovation 2: Adaptive Immune Module Scoring (AIMS)

Key innovation: Instead of equal-weight gene sets (like ssGSEA/GSVA),
use co-expression network connectivity as gene weights.

Algorithm:
1. Start with literature-curated seed gene sets
2. Build co-expression network (Pearson correlation)
3. For each module, compute gene weights = within-module connectivity
4. Genes that co-express strongly with other module members get higher weight
5. Module score = weighted z-score mean (with connectivity normalization)

Benchmark: vs ssGSEA (GSVA), vs simple mean z-score, vs AUCell
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from scipy.stats import mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

# Literature-curated gene sets for AIS immune pathways
DEFAULT_MODULES = {
    'NETosis': [
        'MPO', 'ELANE', 'PADI4', 'HMGB1', 'TLR2', 'TLR4', 'ITGAM', 'ITGB2',
        'MMP9', 'CTSG', 'PRTN3', 'FPR1', 'FPR2', 'S100A8', 'S100A9', 'S100A12',
        'CLEC5A', 'CEACAM8', 'CASP1', 'GSDMD', 'IL1B'
    ],
    'Kynurenine': [
        'IDO1', 'KYNU', 'KMO', 'TDO2', 'IDO2', 'AFMID', 'AADAT', 'ACMSD',
        'HAAO', 'QPRT'
    ],
    'Inflammasome': [
        'NLRP3', 'IL1B', 'IL18', 'CASP1', 'PYCARD', 'NLRC4', 'AIM2', 'GSDMD',
        'IL1R1', 'IL18R1', 'IL1RAP', 'NFKB1', 'RELA', 'TNF', 'IL6'
    ],
    'Complement': [
        'C1QA', 'C1QB', 'C1QC', 'C2', 'C3', 'C4A', 'C4B', 'C5', 'CFB', 'CFD',
        'C5AR1', 'C3AR1', 'CFH', 'CFI', 'CD46', 'CD55', 'CD59', 'SERPING1'
    ],
    'Metabolic_Stress': [
        'SIRT1', 'PRKAA1', 'PRKAA2', 'PPARGC1A', 'HIF1A', 'NFE2L2', 'MTOR',
        'ULK1', 'FOXO1', 'FOXO3', 'TFEB', 'ATF4', 'DDIT3', 'XBP1',
        'SOD1', 'SOD2', 'CAT', 'GPX1', 'GPX4', 'HMOX1'
    ],
    'Cytokine_Storm': [
        'IL1B', 'IL6', 'TNF', 'IL10', 'CCL2', 'CCL3', 'CCL4', 'CCL5',
        'CXCL1', 'CXCL2', 'CXCL8', 'CXCL10', 'IFNG', 'CSF2', 'CSF3',
        'IL1A', 'IL1RN', 'TGFB1', 'IL12A', 'IL12B'
    ],
    'Angiogenesis': [
        'VEGFA', 'VEGFB', 'VEGFC', 'FLT1', 'KDR', 'ANGPT1', 'ANGPT2',
        'TEK', 'PDGFB', 'PDGFRA', 'FGF2', 'HGF', 'MET', 'EPHB4', 'EFNB2'
    ],
    'Apoptosis': [
        'BCL2', 'BAX', 'BAD', 'BAK1', 'CASP3', 'CASP7', 'CASP8', 'CASP9',
        'BID', 'CYCS', 'APAF1', 'DIABLO', 'XIAP', 'BIRC5', 'TP53'
    ]
}


class AIMS:
    """
    Adaptive Immune Module Scoring.

    Parameters:
    - min_genes: minimum genes required in expression data for a module (default 3)
    - use_weights: if True, compute co-expression weights (default True)
    - weight_method: 'intramodular_connectivity' or 'pagerank' (default 'intramodular_connectivity')
    """

    def __init__(self, min_genes=3, use_weights=True, weight_method='intramodular_connectivity'):
        self.min_genes = min_genes
        self.use_weights = use_weights
        self.weight_method = weight_method
        self.module_weights_ = {}  # Stores computed weights per module
        self.module_genes_ = {}    # Stores available genes per module

    def _build_network(self, expr_df):
        """Build full co-expression network (Pearson correlation matrix)."""
        # Use all genes for network construction
        genes = expr_df.index.tolist()
        corr_mat = np.corrcoef(expr_df.values)
        # Clip to handle numerical issues
        corr_mat = np.clip(corr_mat, -1, 1)
        return pd.DataFrame(corr_mat, index=genes, columns=genes)

    def _intramodular_connectivity(self, corr_df, module_genes):
        """Compute within-module connectivity for each gene."""
        available = [g for g in module_genes if g in corr_df.index]
        if len(available) < self.min_genes:
            return None, None

        sub_corr = corr_df.loc[available, available]
        # Connectivity = sum of |correlation| with all other module members (excluding self)
        weights = {}
        for g in available:
            other = [o for o in available if o != g]
            k = np.abs(sub_corr.loc[g, other]).mean()
            weights[g] = max(k, 0.01)  # floor at 0.01 to avoid zero weight
        return available, pd.Series(weights)

    def _pagerank_weights(self, corr_df, module_genes):
        """Compute PageRank-based weights within module sub-network."""
        import networkx as nx
        available = [g for g in module_genes if g in corr_df.index]
        if len(available) < self.min_genes:
            return None, None

        sub_corr = corr_df.loc[available, available]
        G = nx.Graph()
        for i, g1 in enumerate(available):
            for j, g2 in enumerate(available):
                if i < j and abs(sub_corr.iloc[i, j]) > 0.3:
                    G.add_edge(g1, g2, weight=abs(sub_corr.iloc[i, j]))

        if G.number_of_edges() == 0:
            # Fallback to uniform weights
            weights = pd.Series(1.0, index=available)
        else:
            pr = nx.pagerank(G, weight='weight')
            weights = pd.Series(pr)

        return available, weights

    def fit(self, expr_df):
        """Fit AIMS on expression data: compute network and module weights."""
        print(f"\n[AIMS] Fitting on {expr_df.shape[1]} samples × {expr_df.shape[0]} genes")
        self.corr_df_ = self._build_network(expr_df)

        for name, genes in DEFAULT_MODULES.items():
            if self.weight_method == 'pagerank':
                available, weights = self._pagerank_weights(self.corr_df_, genes)
            else:
                available, weights = self._intramodular_connectivity(self.corr_df_, genes)

            if available is None:
                continue

            self.module_genes_[name] = available
            if self.use_weights:
                # Normalize weights to sum to 1
                self.module_weights_[name] = weights / weights.sum()
            else:
                self.module_weights_[name] = pd.Series(1.0 / len(available), index=available)

        n_modules = len(self.module_genes_)
        n_genes = sum(len(v) for v in self.module_genes_.values())
        print(f"  Fitted {n_modules} modules, {n_genes} total gene-module pairs")
        return self

    def transform(self, expr_df):
        """Compute AIMS scores for all samples."""
        if not self.module_genes_:
            raise RuntimeError("Call fit() before transform()")

        scores = pd.DataFrame(index=expr_df.columns)
        for name in self.module_genes_:
            genes = self.module_genes_[name]
            weights = self.module_weights_[name]

            # Extract expression for module genes
            sub = expr_df.loc[genes]
            # Z-score per gene
            z = sub.subtract(sub.mean(axis=1), axis=0).divide(
                sub.std(axis=1).replace(0, 1), axis=0
            )
            # Weighted mean
            score = (z.T * weights.values).sum(axis=1)
            scores[f'{name}_AIMS'] = score

        return scores

    def fit_transform(self, expr_df):
        """Fit and transform in one call."""
        self.fit(expr_df)
        return self.transform(expr_df)


def benchmark_aims(aims_scores, group_labels, standard_scores=None):
    """
    Benchmark AIMS against standard module scores.

    Parameters:
    - aims_scores: DataFrame from AIMS.transform()
    - group_labels: Series of group assignments (e.g., Stroke/Control)
    - standard_scores: DataFrame from traditional z-score method (optional)

    Returns: DataFrame of per-module AUC and p-values
    """
    from sklearn.metrics import roc_auc_score

    results = []
    for col in aims_scores.columns:
        module = col.replace('_AIMS', '')
        values = aims_scores[col].dropna()
        labels = group_labels.loc[values.index]
        # Encode labels as 0/1
        classes = labels.unique()
        if len(classes) != 2:
            continue
        y_true = (labels == classes[0]).astype(int)

        # AUC for group separation
        auc = roc_auc_score(y_true, values)
        # Mann-Whitney U test
        group1 = values[labels == classes[0]]
        group2 = values[labels == classes[1]]
        u_stat, p_val = mannwhitneyu(group1, group2, alternative='two-sided')
        # Cohen's d effect size
        d = (group1.mean() - group2.mean()) / np.sqrt(
            (group1.std()**2 + group2.std()**2) / 2
        )

        results.append({
            'module': module,
            'AUC': auc,
            'cohens_d': d,
            'p_value': p_val,
            'group1_mean': group1.mean(),
            'group2_mean': group2.mean()
        })

    return pd.DataFrame(results).sort_values('AUC', ascending=False)


def compare_with_standard(aims_scores, simple_scores, group_labels):
    """
    Compare AIMS vs simple z-score across all modules.
    Returns improvement metrics.
    """
    from sklearn.metrics import roc_auc_score

    comparison = []
    for col in aims_scores.columns:
        module = col.replace('_AIMS', '')
        aims_auc = roc_auc_score(
            (group_labels == group_labels.unique()[0]).astype(int),
            aims_scores[col].dropna()
        )

        simple_col = module if module in simple_scores.columns else None
        if simple_col:
            simple_auc = roc_auc_score(
                (group_labels == group_labels.unique()[0]).astype(int),
                simple_scores[simple_col].dropna()
            )
        else:
            simple_auc = None

        comparison.append({
            'module': module,
            'AIMS_AUC': aims_auc,
            'Simple_AUC': simple_auc,
            'Delta_AUC': aims_auc - simple_auc if simple_auc else None
        })

    return pd.DataFrame(comparison)
