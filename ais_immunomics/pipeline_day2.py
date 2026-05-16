"""
DAY 2 Pipeline: AI Layer + Clinical Layer
==========================================
Step 6: SNF multi-network fusion + immune subtyping
Step 7: LASSO + RF + SHAP diagnostic panel
Step 8: Heterogeneous design meta-analysis
Step 9: Drug repurposing via Enrichr
Step 10: Result compilation

Uses Python-R bridge (subprocess) for SNF R script.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys, os, time, json, requests

sys.path.insert(0, str(Path(__file__).parent))
from bridge import check_r, run_r_script, r_to_csv, csv_from_r

RESULT_DIR = Path("D:/Ncz/results")
FIG_DIR = RESULT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)
R_SNF = Path("D:/Ncz/ais_immunomics/network/snf.R")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
import shap
from scipy.stats import chi2_contingency, combine_pvalues
from statsmodels.stats.multitest import multipletests

def log_step(n, title):
    print(f"\n{'='*70}")
    print(f"[{time.strftime('%H:%M:%S')}] STEP {n}: {title}")
    print(f"{'='*70}")

def check_ok(path, desc=""):
    path = Path(path)
    if path.exists():
        print(f"  [OK] {desc}: {path.name} ({path.stat().st_size/1024:.1f} KB)")
    else:
        print(f"  [FAIL] {desc}: {path} NOT FOUND")

# ================================================================
# STEP 6: SNF Multi-Network Fusion + Immune Subtyping
# ================================================================
log_step(6, "SNF Multi-Network Fusion & Immune Subtyping (Innovation 3)")

# Load data
print("\n  Loading harmonized data and AIMS scores...")
harmonized = {}
aims_scores = {}
for gse_id in ['GSE16561', 'GSE22255', 'GSE37587']:
    harmonized[gse_id] = pd.read_csv(RESULT_DIR / f"{gse_id}_harmonized.csv", index_col=0)
    aims_scores[gse_id] = pd.read_csv(RESULT_DIR / f"{gse_id}_AIMS_scores.csv", index_col=0)

expr_16561 = harmonized['GSE16561']

# Build SNF views (SAVED AS features x samples for R transpose)
sfn_dir = RESULT_DIR / "_snf_views"
sfn_dir.mkdir(exist_ok=True)

# View 1: Expression (top 2000 variable genes)
top_genes = expr_16561.var(axis=1).nlargest(2000).index.tolist()
expr_view = expr_16561.loc[top_genes]  # genes x samples -> R transpose -> samples x genes
expr_view.to_csv(sfn_dir / "expression_view.csv")
print(f"  View1 (expression): {expr_view.shape} — 2000 genes x 63 samples")

# View 2: AIMS scores (transposed for R)
aims_view = aims_scores['GSE16561'].T  # modules x samples -> R transpose -> samples x modules
aims_view.to_csv(sfn_dir / "aims_view.csv")
print(f"  View2 (AIMS): {aims_view.shape} — {aims_view.shape[0]} modules x 63 samples")

# View 3: WGCNA module eigengenes
wgcna_mod = pd.read_csv(RESULT_DIR / "_wgcna_gene_modules.csv")
module_expr = {}
for mod in sorted(wgcna_mod['module'].unique()):
    if mod == 0: continue
    genes = wgcna_mod[wgcna_mod['module']==mod]['gene'].tolist()
    genes = [g for g in genes if g in expr_16561.index]
    if len(genes) >= 10:
        module_expr[f'ME{mod}'] = expr_16561.loc[genes].mean()
wgcna_view = pd.DataFrame(module_expr).T  # transpose: features x samples for R
wgcna_view.to_csv(sfn_dir / "wgcna_view.csv")
print(f"  View3 (WGCNA): {wgcna_view.shape} — {len(module_expr)} modules x 63 samples")

# Load metadata early (needed by SNF and later steps)
meta_16561 = pd.read_csv(RESULT_DIR / "GSE16561_metadata.csv", index_col=0)
meta_16561['group'] = meta_16561['title'].apply(
    lambda x: 'Stroke' if 'Stroke' in str(x) else 'Control')

# Run SNF
print("\n  Running SNF (R)...")
snf_prefix = str(RESULT_DIR / "_snf")
ok, stdout, stderr, elapsed = run_r_script(R_SNF, [str(sfn_dir), snf_prefix], timeout=300)
print(stdout)
if not ok:
    print(f"  SNF stderr: {stderr[:400]}")

# Load SNF results
cluster_path = RESULT_DIR / "_snf_clusters.csv"
if cluster_path.exists():
    snf_clusters = pd.read_csv(cluster_path, index_col=0)
    print(f"\n  SNF clusters loaded:")
    for col in snf_clusters.columns:
        print(f"    {col}: {dict(snf_clusters[col].value_counts().sort_index())}")

    # R prepends X to numeric names, use first available
    best_col = snf_clusters.columns[-1]  # last column (highest k)
    labels = snf_clusters[best_col]
    print(f"  Using SNF clusters: {best_col}")
    common = labels.index.intersection(meta_16561.index)
    ct = pd.crosstab(labels.loc[common], meta_16561.loc[common, 'group'])
    print(f"\n  SNF {best_col} vs Group:")
    print(ct.to_string())
    if ct.shape[0] >= 2 and ct.shape[1] >= 2:
        try:
            chi2, p, _, _ = chi2_contingency(ct.values)
            print(f"  Chi2={chi2:.2f}, p={p:.4f}")
        except:
            pass

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ct.plot(kind='bar', stacked=True, ax=ax, color=['#2E86AB', '#A23B72'])
    ax.set_title(f'SNF Immune Subtypes ({best_col}) vs Clinical Group', fontsize=13)
    ax.set_xlabel('SNF Cluster'); ax.set_ylabel('Count')
    plt.tight_layout()
    fig.savefig(FIG_DIR / "SNF_subtypes.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: figures/SNF_subtypes.png")

# ================================================================
# STEP 7: SHAP-Driven Diagnostic Panel (Innovation 4)
# ================================================================
log_step(7, "SHAP-Driven Diagnostic Panel Optimization (Innovation 4)")

print("\n  Training: GSE16561 (whole blood, 39 Stroke + 24 Control)")
print("  Internal CV: 5-fold stratified on GSE16561")
print("  Cross-tissue test: GSE22255 (PBMC, 20+20)")
print("  Same-platform test: GSE37587 baseline (all stroke)")

# Prepare data
X = harmonized['GSE16561'].T  # samples x genes
y = meta_16561.loc[X.index, 'group'].map({'Stroke': 1, 'Control': 0}).values

# LASSO feature selection with internal CV
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.values)

lasso = LogisticRegressionCV(
    Cs=50, cv=5, l1_ratios=[1.0], solver='saga', max_iter=5000,
    random_state=42, n_jobs=-1
)
lasso.fit(X_scaled, y)

coef = lasso.coef_[0]
sel_mask = coef != 0
n_sel = sel_mask.sum()
sel_genes = [X.columns[i] for i in range(len(coef)) if sel_mask[i]]
print(f"\n  LASSO selected {n_sel} genes (from {len(X.columns)})")
print(f"  Panel: {sel_genes}")

# RF with selected genes
X_sel = X[sel_genes].values

rf = RandomForestClassifier(
    n_estimators=500, max_depth=5, min_samples_leaf=5,
    random_state=42, n_jobs=-1, class_weight='balanced'
)

# 5-fold CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X_sel, y, cv=cv, scoring='roc_auc')
print(f"\n  5-fold CV AUC (GSE16561): {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

# Train on all GSE16561, test held-out 20%
X_tr, X_te, y_tr, y_te = train_test_split(
    X_sel, y, test_size=0.2, stratify=y, random_state=42
)
rf.fit(X_tr, y_tr)
y_te_pred = rf.predict_proba(X_te)[:, 1]
test_auc = roc_auc_score(y_te, y_te_pred)
print(f"  Hold-out Test AUC (20% GSE16561): {test_auc:.4f}")

# Cross-tissue test (GSE22255 PBMC)
meta_22255 = pd.read_csv(RESULT_DIR / "GSE22255_metadata.csv", index_col=0)
meta_22255['group'] = meta_22255['title'].apply(
    lambda x: 'Stroke' if 'IS_' in str(x) else 'Control')
y_22255 = meta_22255['group'].map({'Stroke': 1, 'Control': 0}).values
X_22255 = harmonized['GSE22255'].T[X.columns]  # same gene order
X_22255_sel = X_22255[sel_genes].values
y_22255_pred = rf.predict_proba(X_22255_sel)[:, 1]
auc_22255 = roc_auc_score(y_22255, y_22255_pred)
print(f"  Cross-Tissue AUC (GSE22255 PBMC): {auc_22255:.4f}")
print(f"    (Lower AUC expected — PBMC =/= whole blood, demonstrates tissue specificity)")

# GSE37587 baseline (all stroke, should predict high probability)
X_37587 = harmonized['GSE37587'].T[X.columns]
X_37587_sel = X_37587[sel_genes].values
y_37587_pred = rf.predict_proba(X_37587_sel)[:, 1]
bl_pred = y_37587_pred[:34]  # first 34 = baseline
print(f"\n  GSE37587 baseline (n=34, all stroke):")
print(f"    Mean pred={bl_pred.mean():.3f}, Median={np.median(bl_pred):.3f}")
print(f"    Predicted as Stroke (>0.5): {sum(bl_pred > 0.5)}/34 ({sum(bl_pred>0.5)/34*100:.0f}%)")

# SHAP explanation
rf.fit(X_sel, y)  # re-fit on full data for SHAP
explainer = shap.TreeExplainer(rf)
shap_vals = explainer.shap_values(X_sel)
if isinstance(shap_vals, list):
    shap_vals = shap_vals[1]  # binary: class 1

shap_imp = np.abs(shap_vals).mean(axis=0)
shap_rank = np.argsort(shap_imp)[::-1]

print(f"\n  Top 15 SHAP genes:")
seen_genes = set()
rank_counter = 0
for i in range(len(shap_rank)):
    idx = int(np.array(shap_rank[i]).flatten()[0])
    gene_name = str(sel_genes[idx])
    if gene_name in seen_genes:
        continue
    seen_genes.add(gene_name)
    imp_val = float(np.array(shap_imp[idx]).flatten()[0])
    rank_counter += 1
    print(f"    {rank_counter:2d}. {gene_name:15s} |SHAP|={imp_val:.4f}")
    if rank_counter >= 15:
        break

# SHAP summary plot
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_vals, X_sel, feature_names=sel_genes,
                  show=False, max_display=min(20, n_sel))
plt.tight_layout()
fig.savefig(FIG_DIR / "SHAP_summary.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  SHAP plot: figures/SHAP_summary.png")

# ROC curves
fig, ax = plt.subplots(figsize=(8, 8))
# CV ROC (mean)
fpr_tr, tpr_tr, _ = roc_curve(y, rf.predict_proba(X_sel)[:, 1])
ax.plot(fpr_tr, tpr_tr, 'b-', lw=2, label=f'GSE16561 (CV AUC={cv_scores.mean():.3f})')
# GSE22255
fpr_22255, tpr_22255, _ = roc_curve(y_22255, y_22255_pred)
ax.plot(fpr_22255, tpr_22255, 'r--', lw=2, label=f'GSE22255 PBMC (AUC={auc_22255:.3f})')
ax.plot([0,1],[0,1],'k--',alpha=0.3)
ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
ax.set_title(f'Diagnostic Model: LASSO({n_sel} genes) + RF', fontsize=14)
ax.legend(fontsize=11)
plt.tight_layout()
fig.savefig(FIG_DIR / "ROC_diagnostic_model.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ROC plot: figures/ROC_diagnostic_model.png")

# Save model info
model_info = {
    'n_genes': int(n_sel),
    'panel': sel_genes,
    'cv_auc_mean': float(cv_scores.mean()),
    'cv_auc_std': float(cv_scores.std()),
    'holdout_auc': float(test_auc),
    'cross_tissue_auc': float(auc_22255),
    'gse37587_baseline_mean_pred': float(bl_pred.mean()),
    'top_shap': [(str(sel_genes[int(np.array(shap_rank[i]).flatten()[0])]),
                  float(np.array(shap_imp[int(np.array(shap_rank[i]).flatten()[0])]).flatten()[0]))
                 for i in range(min(20, len(shap_rank)))]
}
with open(RESULT_DIR / "diagnostic_model.json", 'w') as f:
    json.dump(model_info, f, indent=2)

# ================================================================
# STEP 8: Heterogeneous Design Meta-Analysis (Innovation 5)
# ================================================================
log_step(8, "Heterogeneous Design Meta-Analysis (Innovation 5)")

print("\n  Loading DE results from all datasets...")
de_all = {}
for gse_id in ['GSE16561', 'GSE22255', 'GSE37587']:
    de_path = RESULT_DIR / f"{gse_id}_DE_results.csv"
    if de_path.exists():
        de = pd.read_csv(de_path)
        de_all[gse_id] = de
        print(f"  {gse_id}: {len(de)} genes, {(de['p_adj']<0.05).sum()} sig (FDR<0.05)")

# Find common genes
common = set(de_all['GSE16561']['gene'])
for gse_id in ['GSE22255', 'GSE37587']:
    if gse_id in de_all:
        common &= set(de_all[gse_id]['gene'])
print(f"  Common genes tested in all 3 datasets: {len(common)}")

# Build meta-analysis dataframe
meta_rows = []
for gene in list(common):
    row = {'gene': gene}
    for gse_id in de_all:
        dg = de_all[gse_id][de_all[gse_id]['gene'] == gene]
        if len(dg) > 0:
            row[f'{gse_id}_log2FC'] = float(dg['log2FC'].values[0])
            row[f'{gse_id}_pvalue'] = float(dg['p_value'].values[0])
            row[f'{gse_id}_padj'] = float(dg['p_adj'].values[0])
    meta_rows.append(row)

meta_df = pd.DataFrame(meta_rows)
print(f"  Meta-analysis entries: {len(meta_df)}")

# Direction consistency (GSE16561 vs GSE37587)
fc16561 = meta_df['GSE16561_log2FC'].values
fc37587 = meta_df['GSE37587_log2FC'].values
same_dir = np.sum(np.sign(fc16561) == np.sign(fc37587))
print(f"  Direction consistency (GSE16561 vs GSE37587): {same_dir}/{len(meta_df)} ({same_dir/len(meta_df)*100:.1f}%)")

# Fisher's combined p (compute per-gene to avoid dimensionality issues)
meta_pvals = []
for i in range(len(meta_df)):
    p1 = max(min(float(meta_df['GSE16561_pvalue'].iloc[i]), 1-1e-15), 1e-300)
    p2 = max(min(float(meta_df['GSE37587_pvalue'].iloc[i]), 1-1e-15), 1e-300)
    _, cp = combine_pvalues([p1, p2], method='fisher')
    meta_pvals.append(float(cp))

_, meta_padj, _, _ = multipletests(meta_pvals, method='fdr_bh')
meta_df['meta_padj'] = meta_padj
n_meta_sig = (meta_padj < 0.05).sum()
print(f"  Fisher meta-analysis: {n_meta_sig} genes significant (FDR<0.05)")

# TRI: Temporal Responsiveness Index
tri = np.abs(fc37587)
tri_cap = np.percentile(tri[~np.isnan(tri)], 99)
tri_norm = np.clip(tri / tri_cap, 0, 1)
meta_df['TRI'] = tri_norm
high_tri_genes = meta_df.nlargest(20, 'TRI')['gene'].tolist()
print(f"  Top TRI genes (temporal response): {high_tri_genes[:10]}")

meta_df['meta_padj'] = meta_padj
meta_df.to_csv(RESULT_DIR / "meta_analysis_results.csv", index=False)

# Meta-analysis scatter plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(fc16561, fc37587, alpha=0.2, s=10, c='steelblue')
# Highlight top meta-analysis genes
top_meta = meta_df.nsmallest(20, 'meta_padj')
ax.scatter(top_meta['GSE16561_log2FC'], top_meta['GSE37587_log2FC'],
           c='red', s=30, alpha=0.7, edgecolors='black')
for _, row in top_meta.head(5).iterrows():
    ax.annotate(row['gene'],
                (row['GSE16561_log2FC'], row['GSE37587_log2FC']),
                fontsize=7, xytext=(5,5), textcoords='offset points')
ax.axhline(0, color='grey', ls='--', alpha=0.5)
ax.axvline(0, color='grey', ls='--', alpha=0.5)
ax.set_xlabel('GSE16561 log2FC (Case-Control)')
ax.set_ylabel('GSE37587 log2FC (Temporal)')
ax.set_title(f'Cross-Study Evidence: {same_dir}/{len(meta_df)} ({same_dir/len(meta_df)*100:.0f}%) consistent direction', fontsize=13)
plt.tight_layout()
fig.savefig(FIG_DIR / "meta_analysis_scatter.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Plot: figures/meta_analysis_scatter.png")

# ================================================================
# STEP 9: Drug Repurposing via Enrichr
# ================================================================
log_step(9, "Drug Repurposing via Connectivity Map / Enrichr")

de_16561 = pd.read_csv(RESULT_DIR / "GSE16561_DE_results.csv")
up_genes = de_16561[(de_16561['p_adj'] < 0.05) & (de_16561['log2FC'] > 0.5)]['gene'].head(150).tolist()
down_genes = de_16561[(de_16561['p_adj'] < 0.05) & (de_16561['log2FC'] < -0.3)]['gene'].head(150).tolist()
print(f"  Up-regulated (n={len(up_genes)}), Down-regulated (n={len(down_genes)})")

drug_libs = [
    'Drug_Perturbations_from_GEO_2014',
    'DrugMatrix',
    'LINCS_L1000_Chem_Pert_up',
    'LINCS_L1000_Chem_Pert_down',
]

def query_enrichr(genes, library):
    """Query Enrichr API for drug perturbation enrichment."""
    try:
        genes_str = '\n'.join(genes[:200])
        resp = requests.post(
            'https://maayanlab.cloud/Enrichr/addList',
            files={'list': (None, genes_str), 'description': (None, 'query')},
            timeout=30
        )
        if resp.status_code != 200:
            return None
        uid = resp.json().get('userListId', '')
        if not uid:
            return None
        resp2 = requests.get(
            f'https://maayanlab.cloud/Enrichr/enrich?userListId={uid}&backgroundType={library}',
            timeout=30
        )
        if resp2.status_code == 200:
            data = resp2.json()
            if library in data:
                return [(t[1], t[2], t[3]) for t in data[library][:10]]
    except Exception as e:
        print(f"    Enrichr {library}: {e}")
    return None

drug_hits = {}
for lib in drug_libs:
    print(f"    Querying {lib}...")
    hits = query_enrichr(up_genes, lib)
    if hits:
        drug_hits[lib] = hits
        for drug, p, overlap in hits[:5]:
            print(f"      {drug:40s} p={p:.2e} overlap={overlap}")

with open(RESULT_DIR / "drug_repurposing.json", 'w') as f:
    json.dump(drug_hits, f, indent=2)

# ================================================================
# STEP 10: Day 2 Summary
# ================================================================
log_step(10, "Day 2 Summary & Combined Figure")

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# A: SNF clusters
ax = axes[0, 0]
if cluster_path.exists():
    snf_common_idx = labels.index.intersection(meta_16561.index)
    ct_snf = pd.crosstab(labels.loc[snf_common_idx], meta_16561.loc[snf_common_idx, 'group'])
    ct_snf.plot(kind='bar', stacked=True, ax=ax, color=['#2E86AB', '#A23B72'])
    ax.set_title('A: SNF Immune Subtypes', fontsize=12)
    ax.set_xlabel('SNF Cluster'); ax.set_ylabel('Count')

# B: ROC curves
ax = axes[0, 1]
fpr_tr, tpr_tr, _ = roc_curve(y, rf.predict_proba(X_sel)[:, 1])
ax.plot(fpr_tr, tpr_tr, 'b-', lw=2, label=f'CV AUC={cv_scores.mean():.3f}')
if len(y_22255) > 0:
    fpr_22255, tpr_22255, _ = roc_curve(y_22255, y_22255_pred)
    ax.plot(fpr_22255, tpr_22255, 'r--', lw=2, label=f'Cross-Tissue AUC={auc_22255:.3f}')
ax.plot([0,1],[0,1],'k--',alpha=0.3)
ax.set_title(f'B: Diagnostic Model ({n_sel} genes)', fontsize=12)
ax.legend(fontsize=9); ax.set_xlabel('FPR'); ax.set_ylabel('TPR')

# C: SHAP importance
ax = axes[1, 0]
top10 = min(10, len(shap_rank))
seen = set()
top_imp, top_names = [], []
for i in range(len(shap_rank)):
    idx = int(np.array(shap_rank[i]).flatten()[0])
    gname = str(sel_genes[idx])
    if gname not in seen:
        seen.add(gname)
        top_imp.append(float(shap_imp[idx]))
        top_names.append(gname)
    if len(top_names) >= 10:
        break
ax.barh(range(top10), top_imp, color=plt.cm.Reds_r(np.linspace(0.3, 0.9, top10)))
ax.set_yticks(range(top10)); ax.set_yticklabels(top_names)
ax.set_title(f'C: Top 10 SHAP Features', fontsize=12)
ax.set_xlabel('Mean |SHAP|'); ax.invert_yaxis()

# D: Drug candidates
ax = axes[1, 1]
if drug_hits:
    all_drugs = []
    for lib, hits in drug_hits.items():
        for drug, p, _ in hits[:3]:
            all_drugs.append((drug, p, lib))
    all_drugs.sort(key=lambda x: x[1])
    for i, (drug, p, lib) in enumerate(all_drugs[:12]):
        ax.text(0.05, 0.95 - i*0.08, f"{drug[:35]}", transform=ax.transAxes,
                fontsize=8, family='monospace')
        ax.text(0.82, 0.95 - i*0.08, f"{p:.1e}", transform=ax.transAxes,
                fontsize=7, family='monospace', color='red')
ax.set_title('D: Drug Repurposing Candidates', fontsize=12)
ax.axis('off')

plt.suptitle('AIS-Immunomics: AI-Driven Diagnostic System', fontsize=16, fontweight='bold')
plt.tight_layout()
fig.savefig(FIG_DIR / "day2_summary_figure.png", dpi=150, bbox_inches='tight')
plt.close()

# ================================================================
print(f"\n{'='*70}")
print(f"DAY 2 PIPELINE COMPLETE")
print(f"{'='*70}")
print(f"\nResults saved to: {RESULT_DIR}")
print(f"Figures saved to: {FIG_DIR}")
print(f"\nKey metrics:")
print(f"  SNF: {'done' if cluster_path.exists() else 'failed'}")
print(f"  Diagnostic panel: {n_sel} genes, CV AUC={cv_scores.mean():.3f}")
print(f"  Cross-tissue AUC: {auc_22255:.3f}")
print(f"  Meta-analysis: {n_meta_sig} significant genes")
print(f"  Drug candidates: {sum(len(v) for v in drug_hits.values())}")
