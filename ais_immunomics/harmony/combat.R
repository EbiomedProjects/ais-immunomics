#!/usr/bin/env Rscript
# ComBat batch correction for cross-platform harmonization
# Usage: Rscript combat.R <input_csv> <batch_csv> <output_csv>
#   input_csv: gene expression matrix (genes x samples)
#   batch_csv: batch labels per sample (sample_id, batch)
#   output_csv: corrected expression matrix

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
    stop("Usage: Rscript combat.R <input_csv> <batch_csv> <output_csv>")
}

input_csv <- args[1]
batch_csv <- args[2]
output_csv <- args[3]

suppressPackageStartupMessages(library(sva))

# Load data
cat(sprintf("[ComBat] Loading expression matrix: %s\n", input_csv))
expr <- as.matrix(read.csv(input_csv, row.names = 1, check.names = FALSE))

cat(sprintf("[ComBat] Input: %d genes x %d samples\n", nrow(expr), ncol(expr)))

# Load batch labels
batches <- read.csv(batch_csv, row.names = 1, stringsAsFactors = FALSE)
batch_vec <- as.factor(batches[colnames(expr), 1])
cat(sprintf("[ComBat] Batch levels: %s\n", paste(levels(batch_vec), collapse = ", ")))
cat(sprintf("[ComBat] Samples per batch: %s\n", paste(table(batch_vec), collapse = ", ")))

# Remove genes with zero variance (ComBat requirement)
gene_var <- apply(expr, 1, var, na.rm = TRUE)
zero_var <- sum(gene_var == 0 | is.na(gene_var))
if (zero_var > 0) {
    cat(sprintf("[ComBat] Removing %d genes with zero variance\n", zero_var))
    expr <- expr[gene_var > 0 & !is.na(gene_var), ]
}

# Run ComBat
cat("[ComBat] Running ComBat adjustment...\n")
# Use parametric adjustment, no covariates
expr_corrected <- ComBat(dat = expr, batch = batch_vec, mod = NULL, par.prior = TRUE)

cat(sprintf("[ComBat] Output: %d genes x %d samples\n", nrow(expr_corrected), ncol(expr_corrected)))

# Save corrected matrix
write.csv(expr_corrected, file = output_csv, row.names = TRUE)
cat(sprintf("[ComBat] Corrected matrix saved to: %s\n", output_csv))

# Quick summary statistics
cat("\n[ComBat] Summary:\n")
cat(sprintf("  Mean expression range: [%.4f, %.4f]\n", min(expr_corrected, na.rm=TRUE), max(expr_corrected, na.rm=TRUE)))
cat(sprintf("  Overall mean: %.4f\n", mean(expr_corrected, na.rm=TRUE)))
cat("Done.\n")
