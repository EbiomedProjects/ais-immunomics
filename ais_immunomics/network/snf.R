#!/usr/bin/env Rscript
# Similarity Network Fusion (SNF) for multi-view immune subtyping
# Usage: Rscript snf.R <input_dir> <output_prefix>
#   input_dir: directory containing CSV files for each view
#   output_prefix: path prefix for output files

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
    stop("Usage: Rscript snf.R <input_dir> <output_prefix>")
}

input_dir <- args[1]
output_prefix <- args[2]

suppressPackageStartupMessages(library(SNFtool))

cat(sprintf("[SNF] Loading multi-view data from: %s\n", input_dir))

# Load all CSV views from input directory
view_files <- list.files(input_dir, pattern = "_view\\.csv$", full.names = TRUE)
cat(sprintf("[SNF] Found %d view files\n", length(view_files)))

if (length(view_files) < 2) {
    stop("SNF requires at least 2 data views")
}

# Load each view
views <- list()
view_names <- c()
for (f in view_files) {
    vname <- gsub("_view\\.csv$", "", basename(f))
    view_names <- c(view_names, vname)
    dat <- as.matrix(read.csv(f, row.names = 1, check.names = FALSE))
    # Transpose: SNF expects samples x features
    views[[vname]] <- t(dat)
    cat(sprintf("  %s: %d samples x %d features\n", vname, nrow(views[[vname]]), ncol(views[[vname]])))
}

# Verify all views have same samples
n_samples <- nrow(views[[1]])
sample_names <- rownames(views[[1]])
for (i in 2:length(views)) {
    if (nrow(views[[i]]) != n_samples) {
        stop(sprintf("View %s has %d samples, expected %d", view_names[i], nrow(views[[i]]), n_samples))
    }
}

cat(sprintf("\n[SNF] All views aligned: %d samples\n", n_samples))

# ---- Step 1: Build similarity networks for each view ----
cat("\n[SNF] Step 1: Building per-view similarity networks...\n")
dist_matrices <- list()
affinity_matrices <- list()

for (i in 1:length(views)) {
    vname <- view_names[i]
    dat <- views[[vname]]

    # Standardize
    dat_scaled <- scale(dat)

    # Euclidean distance
    dist_mat <- as.matrix(dist(dat_scaled))
    dist_matrices[[vname]] <- dist_mat

    # Affinity matrix
    K <- 20  # number of neighbors
    alpha <- 0.5
    aff_mat <- affinityMatrix(dist_mat, K, alpha)
    affinity_matrices[[vname]] <- aff_mat

    cat(sprintf("  %s: distance range [%.3f, %.3f]\n", vname, min(dist_mat), max(dist_mat)))
}

# ---- Step 2: Fuse networks ----
cat("\n[SNF] Step 2: Fusing similarity networks...\n")
# Convert to list of matrices
aff_list <- lapply(affinity_matrices, function(x) x)
W <- SNF(aff_list, K = 20, t = 20)
cat(sprintf("  Fused network: %d x %d\n", nrow(W), ncol(W)))

# ---- Step 3: Spectral clustering on fused network ----
cat("\n[SNF] Step 3: Spectral clustering...\n")

# Estimate optimal number of clusters
# Use rotation cost or eigen-gap
eigen_vals <- eigen(W)$values
n_max <- min(10, n_samples - 1)

# Simple eigen-gap method
gaps <- abs(diff(eigen_vals[1:n_max]))
best_k <- which.max(gaps) + 1
cat(sprintf("  Estimated optimal k from eigen-gap: %d\n", best_k))

# Cluster with k=2,3,4 for comparison
cluster_results <- list()
for (k in 2:4) {
    labels <- spectralClustering(W, k)
    names(labels) <- sample_names
    cluster_results[[as.character(k)]] <- labels
    cat(sprintf("  k=%d: cluster sizes = %s\n", k,
        paste(table(labels), collapse = ", ")))
}

# ---- Step 4: Cluster evaluation ----
cat("\n[SNF] Step 4: Evaluating cluster stability...\n")

# Silhouette score
calc_silhouette <- function(W, labels) {
    n <- nrow(W)
    k <- length(unique(labels))
    if (k < 2) return(NA)

    sils <- numeric(n)
    for (i in 1:n) {
        a_i <- mean(W[i, labels == labels[i]]) - W[i, i]
        b_i <- Inf
        for (cl in unique(labels)) {
            if (cl != labels[i]) {
                b_cand <- mean(W[i, labels == cl])
                if (b_cand < b_i) b_i <- b_cand
            }
        }
        sils[i] <- (b_i - a_i) / max(a_i, b_i)
    }
    mean(sils, na.rm = TRUE)
}

silhouette_scores <- sapply(cluster_results, function(l) calc_silhouette(W, l))
cat("  Silhouette scores:\n")
for (k in names(silhouette_scores)) {
    cat(sprintf("    k=%s: %.4f\n", k, silhouette_scores[k]))
}

# ---- Save results ----
cat("\n[SNF] Saving results...\n")

# Save fused network
write.csv(W, file = paste0(output_prefix, "_fused_network.csv"))

# Save cluster labels
cluster_df <- as.data.frame(cluster_results)
colnames(cluster_df) <- paste0("k", colnames(cluster_df))
write.csv(cluster_df, file = paste0(output_prefix, "_clusters.csv"))

# Save silhouette scores
write.csv(data.frame(k = names(silhouette_scores), silhouette = silhouette_scores),
    file = paste0(output_prefix, "_silhouette.csv"), row.names = FALSE)

# Save per-view affinity matrices
for (i in 1:length(affinity_matrices)) {
    write.csv(affinity_matrices[[i]],
        file = paste0(output_prefix, "_affinity_", view_names[i], ".csv"))
}

# ---- Summary ----
cat(sprintf("\n[SNF] === Summary ===\n"))
cat(sprintf("  Views: %s\n", paste(view_names, collapse = ", ")))
cat(sprintf("  Samples: %d\n", n_samples))
cat(sprintf("  Optimal k: %d\n", best_k))
cat(sprintf("  Best silhouette: %.4f (k=%s)\n",
    max(silhouette_scores), names(which.max(silhouette_scores))))
cat("Done.\n")
