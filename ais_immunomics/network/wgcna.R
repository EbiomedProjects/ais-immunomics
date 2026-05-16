#!/usr/bin/env Rscript
# WGCNA: Weighted Gene Co-expression Network Analysis
# Usage: Rscript wgcna.R <input_csv> <trait_csv> <output_prefix>
#   input_csv: gene expression matrix (genes x samples)
#   trait_csv: sample trait data (sample_id, group, module1, module2, ...)
#   output_prefix: path prefix for output files

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
    stop("Usage: Rscript wgcna.R <input_csv> <trait_csv> <output_prefix>")
}

input_csv <- args[1]
trait_csv <- args[2]
output_prefix <- args[3]

suppressPackageStartupMessages(library(WGCNA))

# Enable multi-threading
allowWGCNAThreads(nThreads = 4)

cat(sprintf("[WGCNA] Loading expression data: %s\n", input_csv))
expr <- as.matrix(read.csv(input_csv, row.names = 1, check.names = FALSE))
cat(sprintf("[WGCNA] Input: %d genes x %d samples\n", nrow(expr), ncol(expr)))

# Transpose: WGCNA expects samples x genes
datExpr <- t(expr)

# Check for excessive missing values
gsg <- goodSamplesGenes(datExpr, verbose = 3)
if (!gsg$allOK) {
    cat(sprintf("[WGCNA] Removing %d outlier samples and %d outlier genes\n",
        sum(!gsg$goodSamples), sum(!gsg$goodGenes)))
    datExpr <- datExpr[gsg$goodSamples, gsg$goodGenes]
}

# ---- Step 1: Soft-threshold selection ----
cat("\n[WGCNA] Step 1: Selecting soft-threshold power...\n")
powers <- c(1:20, seq(22, 30, by = 2))
sft <- pickSoftThreshold(datExpr, powerVector = powers, verbose = 2)

# Save SFT results
sft_df <- data.frame(
    Power = sft$fitIndices[, 1],
    SFT.R.sq = sft$fitIndices[, 2],
    mean.k = sft$fitIndices[, 3],
    median.k = sft$fitIndices[, 4]
)
write.csv(sft_df, file = paste0(output_prefix, "_sft.csv"), row.names = FALSE)

# Select power where R^2 > 0.8 or best available
best_power <- sft$powerEstimate
if (is.na(best_power)) {
    # Pick the first power with R^2 > 0.8
    good_powers <- which(sft$fitIndices[, 2] > 0.8)
    if (length(good_powers) > 0) {
        best_power <- sft$fitIndices[good_powers[1], 1]
    } else {
        best_power <- 6  # default fallback
    }
}
cat(sprintf("[WGCNA] Selected soft power: %d\n", best_power))

# ---- Step 2: Network construction ----
cat("\n[WGCNA] Step 2: Building co-expression network...\n")
net <- blockwiseModules(
    datExpr,
    power = best_power,
    TOMType = "unsigned",
    minModuleSize = 30,
    reassignThreshold = 1e-6,
    mergeCutHeight = 0.25,
    numericLabels = TRUE,
    pamRespectsDendro = FALSE,
    saveTOMs = FALSE,
    verbose = 2,
    maxBlockSize = ncol(datExpr) + 1  # single block
)

cat(sprintf("[WGCNA] Found %d modules\n", length(unique(net$colors))))

# ---- Step 3: Module-trait association ----
cat("\n[WGCNA] Step 3: Module-trait association...\n")
traits <- read.csv(trait_csv, row.names = 1, check.names = FALSE)

# Align traits with expression samples
common_samples <- intersect(rownames(datExpr), rownames(traits))
datTraits <- traits[common_samples, , drop = FALSE]

# Convert factor columns to numeric (binary)
for (col in colnames(datTraits)) {
    if (is.character(datTraits[[col]]) || is.factor(datTraits[[col]])) {
        datTraits[[col]] <- as.numeric(as.factor(datTraits[[col]])) - 1
    }
}

# Module eigengenes
MEs <- moduleEigengenes(datExpr, colors = net$colors)$eigengenes
MEs <- orderMEs(MEs)

# Correlate MEs with traits
moduleTraitCor <- cor(MEs, datTraits, use = "pairwise.complete.obs")
moduleTraitPvalue <- corPvalueStudent(moduleTraitCor, nrow(datExpr))

# Save module-trait associations
write.csv(moduleTraitCor, file = paste0(output_prefix, "_module_trait_cor.csv"))
write.csv(moduleTraitPvalue, file = paste0(output_prefix, "_module_trait_pvalue.csv"))

# ---- Step 4: Export module assignments ----
cat("\n[WGCNA] Step 4: Exporting module assignments...\n")
module_df <- data.frame(
    gene = colnames(datExpr),
    module = net$colors,
    moduleLabel = labels2colors(net$colors),
    stringsAsFactors = FALSE
)
write.csv(module_df, file = paste0(output_prefix, "_gene_modules.csv"), row.names = FALSE)

# ---- Step 5: Intramodular connectivity ----
cat("[WGCNA] Step 5: Computing intramodular connectivity...\n")
# Compute adjacency and TOM for connectivity
adj <- adjacency(datExpr, power = best_power)
TOM <- TOMsimilarity(adj)
# Intramodular connectivity for each gene
kIM <- intramodularConnectivity(TOM, net$colors, scaleByMax = FALSE)
write.csv(kIM, file = paste0(output_prefix, "_connectivity.csv"))

# ---- Step 6: Hub genes per module ----
cat("[WGCNA] Step 6: Identifying hub genes...\n")
hub_genes <- data.frame()
for (mod in unique(net$colors)) {
    if (mod == 0) next  # skip grey (unassigned)
    mod_genes <- module_df$gene[module_df$module == mod]
    mod_kIM <- kIM[mod_genes, , drop = FALSE]
    # Top 10 by intramodular connectivity
    top_idx <- order(mod_kIM$kWithin, decreasing = TRUE)[1:min(10, length(mod_genes))]
    hub_genes <- rbind(hub_genes, data.frame(
        module = mod,
        moduleLabel = labels2colors(mod),
        gene = mod_genes[top_idx],
        kWithin = mod_kIM$kWithin[top_idx],
        stringsAsFactors = FALSE
    ))
}
write.csv(hub_genes, file = paste0(output_prefix, "_hub_genes.csv"), row.names = FALSE)

# ---- Summary ----
cat("\n[WGCNA] === Summary ===\n")
cat(sprintf("  Input: %d genes, %d samples\n", ncol(datExpr), nrow(datExpr)))
cat(sprintf("  Soft power: %d\n", best_power))
cat(sprintf("  Modules found: %d\n", length(unique(net$colors)) - 1))  # exclude grey
cat(sprintf("  Grey (unassigned) genes: %d\n", sum(net$colors == 0)))

# Top module-trait associations
cat("\n  Top module-trait associations:\n")
for (trait_col in colnames(moduleTraitCor)) {
    cors <- moduleTraitCor[, trait_col]
    pvals <- moduleTraitPvalue[, trait_col]
    # Sort by absolute correlation
    idx <- order(abs(cors), decreasing = TRUE)
    for (i in idx[1:min(3, length(idx))]) {
        if (cors[i] != 0) {
            cat(sprintf("    %s ~ %s: r=%.3f, p=%.3e\n",
                names(cors)[i], trait_col, cors[i], pvals[i]))
        }
    }
}

cat(sprintf("\n[WGCNA] Output files saved with prefix: %s\n", output_prefix))
cat("Done.\n")
