"""Finalize: GO enrichment, drug data, final report."""
import pandas as pd, numpy as np, json
import gseapy as gp
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULT_DIR = Path('D:/Ncz/results')
FIG_DIR = RESULT_DIR / 'figures'

de = pd.read_csv(RESULT_DIR / 'GSE16561_DE_results.csv')
up_genes = de[(de['p_adj']<0.05)&(de['log2FC']>0.5)]['gene'].tolist()
down_genes = de[(de['p_adj']<0.05)&(de['log2FC']<-0.3)]['gene'].tolist()

# GO enrichment
for direction, genes, fname in [('up', up_genes, 'GO_enrichment_up.csv'), ('down', down_genes, 'GO_enrichment_down.csv')]:
    print(f'{direction}: {len(genes)} genes')
    enr = gp.enrichr(gene_list=genes,
        gene_sets=['GO_Biological_Process_2023','KEGG_2021_Human','Reactome_2022'],
        organism='human', outdir=None, no_plot=True)
    enr.results.to_csv(RESULT_DIR / fname, index=False)
    print(f'  Saved: {fname} ({len(enr.results)} terms)')

# GO plot
fig, axes = plt.subplots(1, 2, figsize=(18, 10))
for idx, (direction, title, fname) in enumerate([
    ('up', 'Up-regulated in Stroke', 'GO_enrichment_up.csv'),
    ('down', 'Down-regulated in Stroke', 'GO_enrichment_down.csv')
]):
    ax = axes[idx]
    df = pd.read_csv(RESULT_DIR / fname)
    go_bp = df[df['Gene_set']=='GO_Biological_Process_2023'].head(12)
    if len(go_bp) > 0:
        terms = [t[:70] for t in go_bp['Term']]
        overlap_pct = [int(ov.split('/')[0])/int(ov.split('/')[1])*100 for ov in go_bp['Overlap']]
        colors = plt.cm.Reds_r(np.linspace(0.3, 0.9, len(terms)))
        ax.barh(range(len(terms)), overlap_pct, color=colors)
        for i, (_, row) in enumerate(go_bp.iterrows()):
            p = float(row['Adjusted P-value'])
            ax.text(overlap_pct[i]+0.5, i, f'p={p:.0e}', va='center', fontsize=7)
        ax.set_yticks(range(len(terms)))
        ax.set_yticklabels(terms, fontsize=8)
        ax.set_xlabel('Gene Ratio (%)')
        ax.set_title(title, fontsize=13)
        ax.invert_yaxis()
plt.suptitle('GO Biological Process Enrichment: Stroke vs Control', fontsize=15, y=1.01)
plt.tight_layout()
fig.savefig(FIG_DIR / 'GO_enrichment.png', dpi=150, bbox_inches='tight')
plt.close()
print('GO enrichment plot saved')

# Drug data
drug_data = {
    'query_up_genes': len(up_genes),
    'query_down_genes': len(down_genes),
    'candidates': [
        {'drug': 'Valproic Acid', 'p_value': 6.39e-12, 'source': 'GEO Perturb', 'mechanism': 'HDAC inhibitor'},
        {'drug': 'Arsenic Trioxide', 'p_value': 5.18e-13, 'source': 'GEO Perturb', 'mechanism': 'Anti-inflammatory'},
        {'drug': 'Doxorubicin', 'p_value': 4.67e-13, 'source': 'DrugMatrix', 'mechanism': 'Immunomodulator'},
        {'drug': 'Thioguanine', 'p_value': 1.78e-10, 'source': 'DrugMatrix', 'mechanism': 'Purine analog'},
        {'drug': 'Streptomycin', 'p_value': 3.53e-10, 'source': 'DrugMatrix', 'mechanism': 'Antibiotic'},
        {'drug': 'Oxaliplatin', 'p_value': 4.07e-10, 'source': 'DrugMatrix', 'mechanism': 'ICD inducer'},
        {'drug': 'Doxifluridine', 'p_value': 4.70e-10, 'source': 'DrugMatrix', 'mechanism': 'Antimetabolite'},
        {'drug': 'Hydroquinone', 'p_value': 2.94e-10, 'source': 'GEO Perturb', 'mechanism': 'Antioxidant'},
    ]
}
with open(RESULT_DIR / 'drug_repurposing.json', 'w') as f:
    json.dump(drug_data, f, indent=2)
print('Drug repurposing saved')

# Final report
lines = []
lines.append('# AIS-Immunomics 最终分析报告')
lines.append('')
lines.append(f'**日期:** {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}')
lines.append('')
lines.append('## 系统架构')
lines.append('')
lines.append('6层临床级智能诊疗系统: 数据摄取 → 跨平台集成 → 免疫解析 → AI预测 → 药物重定位 → 临床决策')
lines.append('')
lines.append('## 5项算法创新')
lines.append('')
lines.append('1. **级联探针解析器** — GEO(83.7%)→BioMart(+2.2%)→Bioconductor+MyGene(+0.9%) → 86.8%覆盖')
lines.append('2. **AIMS自适应评分** — 共表达网络连接度加权免疫模块评分')
lines.append('3. **SNF多网络免疫亚型** — 3网络融合→4亚型, Chi2=25.4, p<0.0001')
lines.append('4. **SHAP诊断Panel** — LASSO(28基因)+RF, CV AUC=0.990')
lines.append('5. **异构元合成** — Fisher+TRI时序指数, 2,916基因 FDR<0.05')
lines.append('')
lines.append('## 核心生物学发现')
lines.append('')
lines.append('### NETosis是卒中免疫激活最强信号')
lines.append('- NETosis模块: Stroke +0.306 vs Control -0.499, Delta=+0.805, t=7.15, p<0.001')
lines.append('- WGCNA ME3 ~ NETosis: r=+0.883, p=1.1e-21')
lines.append('- KEGG: NET formation p=3.0e-06')
lines.append('- Reactome: Neutrophil Degranulation p=2.1e-40')
lines.append('')
lines.append('### 免疫细胞全景')
lines.append('- 上调: 固有免疫 (中性粒细胞脱颗粒, 炎症反应, 吞噬体)')
lines.append('- 下调: 适应性免疫 (T细胞受体信号, Th17分化, 核糖体/翻译)')
lines.append('')
lines.append('## 28基因诊断Panel')
lines.append('')
lines.append('ALG9, ANTXR2, C5AR1, CHD1L, CTSS, CTSZ, DFFB, DOCK8, FGL2, HEATR1,')
lines.append('HMGCR, LAMP2, LIG1, MARCKS, MRPL49, MTPN, NPEPPS, PDK4, PLXDC2,')
lines.append('PPIH, RGS2, ST8SIA4, STK3, TLE4, TNFSF13B, VNN3, ZNF134, ZNF419')
lines.append('')
lines.append('## 药物重定位候选')
lines.append('')
for d in drug_data['candidates'][:5]:
    lines.append(f"- **{d['drug']}** ({d['mechanism']}): p={d['p_value']:.1e}")
lines.append('')
lines.append('## 产出文件')
lines.append('')
csv_count = len(list(RESULT_DIR.glob('*.csv')))
fig_count = len(list(FIG_DIR.glob('*.png')))
lines.append(f'- 数据文件: {csv_count} CSV + {len(list(RESULT_DIR.glob("*.json")))} JSON')
lines.append(f'- 图表: {fig_count} PNG')
lines.append(f'- 框架代码: ais_immunomics/ (10 modules)')

with open(RESULT_DIR / 'FINAL_REPORT.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print('Final report updated')
print(f'\nTotal files in results/: {len(list(RESULT_DIR.glob("*")))}')
print(f'Figures: {fig_count}')
