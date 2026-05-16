"""
Python-R Bridge via subprocess + CSV exchange.
More robust than rpy2 on Windows, works with any R installation.
"""
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import os, sys, json, tempfile, time

RSCRIPT = "D:/R-4.6.0/bin/Rscript.exe"

def check_r():
    """Verify R is working."""
    result = subprocess.run(
        [RSCRIPT, "--version"],
        capture_output=True, text=True, timeout=10
    )
    ok = result.returncode == 0
    version = result.stdout.strip() if ok else result.stderr.strip()
    return ok, version

def run_r_script(script_path, args=None, timeout=300):
    """
    Execute an R script with optional arguments.
    Returns (success, stdout, stderr, elapsed_seconds).
    """
    cmd = [RSCRIPT, str(script_path)]
    if args:
        cmd.extend([str(a) for a in args])

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    elapsed = time.time() - t0
    return result.returncode == 0, result.stdout, result.stderr, elapsed

def r_to_csv(expr_df, filepath):
    """Save Python DataFrame to CSV for R consumption."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    expr_df.to_csv(filepath, float_format='%.6f')

def csv_from_r(filepath):
    """Load CSV produced by R back into Python."""
    return pd.read_csv(filepath, index_col=0)

def annot_to_csv(annot_series, filepath):
    """Save annotation Series (probe_id -> gene) to CSV for R."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    annot_series.to_csv(filepath, header=['gene'])

# Verify on import
r_ok, r_ver = check_r()
if r_ok:
    print(f"[Bridge] R connected: {r_ver}")
else:
    print(f"[Bridge] R ERROR: {r_ver}")
    sys.exit(1)
