#!/usr/bin/env python3
"""
Per-cluster gene-level percent identity using VSEARCH (recommended) or Biopython fallback.

Input
-----
- Table (TSV/CSV/XLSX) with at least: genome_id, gene_id, <cluster_col>
- Directory with FASTA files named: <genome_id>_genes.ffn
  FASTA headers look like: >NODE_233_length_..._ID_465_1:2-76(-)
  The script matches gene_id to header by stripping anything after the first ':' in the header.

What it does
------------
For each cluster (default column: `leiden_cluster`):
  1) Collect the nucleotide sequence for each (genome_id, gene_id).
  2) Compute all-vs-all global identities within the cluster.
     - Default backend: VSEARCH `--allpairs_global` (fast, scalable).
     - Fallback backend: Biopython pairwise2 (slow; use only for small clusters).
  3) Write a summary CSV of identity stats per cluster, and optional identity matrices.

Notes
-----
- This measures **gene** sequence identity, not genome ANI. FastANI is unsuitable for single genes.
- Large clusters are O(n^2) pairs; control matrix output size with `--matrix-limit`.

Usage
-----
python cluster_gene_ani.py \
  --table path/to/table.tsv \
  --ffn-dir /path/to/ffn \
  --out-dir out_geneANI \
  --cluster-col leiden_cluster \
  --threads 16 \
  --backend vsearch

Dependencies
------------
- Required: pandas, numpy, biopython (for FASTA parsing)
- Recommended: vsearch (conda install -c bioconda vsearch)
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from Bio import SeqIO

# --------------------------
# Helpers
# --------------------------

def read_table_any(path: Path) -> pd.DataFrame:
    """Read TSV/CSV/XLSX into a dataframe with all columns as strings."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Table not found: {p}")
    ext = p.suffix.lower()
    if ext in {".xlsx", ".xls"}:
        df = pd.read_excel(p, dtype=str)
    else:
        # sep=None lets pandas sniff CSV vs TSV
        df = pd.read_csv(p, sep=None, engine="python", dtype=str)
    return df


def norm_gene_id(header_or_id: str) -> str:
    """Normalize to the portion before the first ':' (matches example headers)."""
    return str(header_or_id).split(":", 1)[0]


def load_sequences_for_needed(ffn_dir: Path, needed: Dict[str, set]) -> Dict[str, Dict[str, str]]:
    """Return {genome_id: {gene_key: SEQ}} only for needed gene IDs per genome."""
    seqs_by_genome: Dict[str, Dict[str, str]] = {}
    missing = []
    for genome, wanted in needed.items():
        ffn_path = ffn_dir / f"{genome}_genes.ffn"
        if not ffn_path.exists():
            missing.append(genome)
            seqs_by_genome[genome] = {}
            continue
        idx: Dict[str, str] = {}
        try:
            for rec in SeqIO.parse(str(ffn_path), "fasta"):
                key = norm_gene_id(rec.id)
                if key in wanted:
                    # Bio.SeqIO already gives a contiguous sequence string
                    idx[key] = str(rec.seq).upper()
        except Exception as e:
            logging.error(f"Failed parsing {ffn_path}: {e}")
        seqs_by_genome[genome] = idx
    if missing:
        logging.warning(f"Missing FFNs for {len(missing)} genomes (e.g., {missing[:3]} ...); those rows will be skipped.")
    return seqs_by_genome


# --------------------------
# Identity backends
# --------------------------

def vsearch_allpairs_identities(names: List[str], seqs: List[str], threads: int = 1) -> List[Tuple[int, int, float]]:
    """Compute pairwise identities using vsearch --allpairs_global.
    Returns list of (i, j, identity_in_0_to_1).
    """
    if len(seqs) < 2:
        return []

    if shutil.which("vsearch") is None:
        raise RuntimeError("vsearch not found in PATH. Install it or use --backend biopython.")

    with tempfile.TemporaryDirectory() as td:
        fasta_path = Path(td) / "cluster.fasta"
        blast6_path = Path(td) / "pairs.tsv"
        with open(fasta_path, "w", encoding="utf-8") as fh:
            for i, s in enumerate(seqs):
                if not s:
                    continue
                fh.write(f">S{i}\n{s}\n")
        cmd = [
            "vsearch", "--allpairs_global", str(fasta_path),
            "--acceptall",
            "--blast6out", str(blast6_path),
            "--threads", str(max(1, int(threads))),
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"vsearch failed: {e.stderr.decode(errors='ignore')}")

        pairs: List[Tuple[int, int, float]] = []
        if not blast6_path.exists():
            return pairs
        with open(blast6_path, "r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.rstrip().split("\t")
                if len(parts) < 3:
                    continue
                qid, sid, pident = parts[0], parts[1], parts[2]
                if not (qid.startswith("S") and sid.startswith("S")):
                    continue
                i = int(qid[1:]); j = int(sid[1:])
                if i == j:
                    continue
                a, b = (i, j) if i < j else (j, i)
                ident = float(pident) / 100.0
                pairs.append((a, b, ident))
        # Deduplicate directions
        uniq: Dict[Tuple[int, int], float] = {}
        for a, b, iden in pairs:
            uniq[(a, b)] = max(iden, uniq.get((a, b), 0.0))
        return [(i, j, iden) for (i, j), iden in sorted(uniq.items())]


def biopython_global_identities(names: List[str], seqs: List[str], max_pairs: int | None = None, rng_seed: int = 1337) -> List[Tuple[int, int, float]]:
    """Slow but dependency-free fallback using Biopython pairwise2 global alignment."""
    if len(seqs) < 2:
        return []
    try:
        from Bio import pairwise2  # import here so Biopython isn't required for vsearch-only users
    except Exception as e:
        raise RuntimeError("Biopython pairwise2 not available. Install biopython or use --backend vsearch.")

    # Build list of pairs
    all_pairs = [(i, j) for i, j in combinations(range(len(seqs)), 2)]
    if max_pairs is not None and max_pairs < len(all_pairs):
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(len(all_pairs), size=max_pairs, replace=False)
        pairs = [all_pairs[k] for k in idx]
    else:
        pairs = all_pairs

    def identity(i: int, j: int) -> float:
        # Score: match=1, mismatch=0, gap_open=-1, gap_extend=-1
        aln = pairwise2.align.globalms(seqs[i], seqs[j], 1, 0, -1, -1, one_alignment_only=True)
        if not aln:
            return np.nan
        a_aln, b_aln, score, start, end = aln[0]
        L = len(a_aln)
        if L == 0:
            return np.nan
        return float(score) / float(L)

    out: List[Tuple[int, int, float]] = []
    for i, j in pairs:
        out.append((i, j, identity(i, j)))
    return out


def summarize(values: List[float]) -> dict:
    arr = np.array([v for v in values if not (v is None or np.isnan(v))], dtype=float)
    if arr.size == 0:
        return {
            "pair_count": 0,
            "mean_identity": np.nan,
            "median_identity": np.nan,
            "min_identity": np.nan,
            "max_identity": np.nan,
            "std_identity": np.nan,
        }
    return {
        "pair_count": int(arr.size),
        "mean_identity": float(np.mean(arr)),
        "median_identity": float(np.median(arr)),
        "min_identity": float(np.min(arr)),
        "max_identity": float(np.max(arr)),
        "std_identity": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
    }


# --------------------------
# Main
# --------------------------

def main():
    ap = argparse.ArgumentParser(description="Per-cluster gene-level identity (VSEARCH or Biopython)")
    ap.add_argument("--table", required=True, help="TSV/CSV/XLSX with genome_id, gene_id, and cluster col")
    ap.add_argument("--ffn-dir", required=True, help="Directory containing <genome_id>_genes.ffn files")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--cluster-col", default="leiden_cluster", help="Cluster column (default: leiden_cluster)")
    ap.add_argument("--genome-col", default="genome_id", help="Genome ID column (default: genome_id)")
    ap.add_argument("--gene-col", default="gene_id", help="Gene ID column (default: gene_id)")
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 1) // 2), help="Threads for vsearch")
    ap.add_argument("--matrix-limit", type=int, default=500, help="Write matrix only if cluster size ≤ this (default: 500)")
    ap.add_argument("--min-seqs", type=int, default=2, help="Skip clusters with < N sequences (default: 2)")
    ap.add_argument("--backend", choices=["vsearch", "biopython"], default="vsearch", help="Computation backend")
    ap.add_argument("--max-pairs", type=int, default=None, help="(biopython only) Subsample to at most this many pairs")
    ap.add_argument("--deduplicate", action="store_true", help="Drop exact duplicate sequences within a cluster")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"], help="Logging level")
    args = ap.parse_args()

    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=getattr(logging, args.log_level))

    table_path = Path(args.table)
    ffn_dir = Path(args.ffn_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "matrices").mkdir(exist_ok=True)

    # Load table
    logging.info(f"Reading table: {table_path}")
    df = read_table_any(table_path)

    for col in (args.cluster_col, args.genome_col, args.gene_col):
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}'. Columns present: {list(df.columns)}")

    # Normalize gene ids used for FASTA lookup
    df["_gene_key"] = df[args.gene_col].map(norm_gene_id)

    # Build per-genome set of required gene IDs to limit FASTA parsing
    needed_by_genome: Dict[str, set] = defaultdict(set)
    for g, k in zip(df[args.genome_col], df["_gene_key"]):
        needed_by_genome[str(g)].add(str(k))

    logging.info("Indexing FFNs for required genomes...")
    seqs_by_genome = load_sequences_for_needed(ffn_dir, needed_by_genome)

    summary_rows = []

    # Iterate clusters in input order
    for cluster, sub in df.groupby(args.cluster_col, sort=False):
        names: List[str] = []
        seqs: List[str] = []
        missing = 0
        for _, row in sub.iterrows():
            genome = str(row[args.genome_col])
            gkey = str(row["_gene_key"])  # normalized
            seq = seqs_by_genome.get(genome, {}).get(gkey)
            if seq:
                names.append(f"{genome}|{gkey}")
                seqs.append(seq)
            else:
                missing += 1
        n = len(seqs)
        if n < args.min_seqs:
            logging.info(f"Cluster {cluster}: {n} sequences (<{args.min_seqs}), skipping.")
            continue

        if args.deduplicate:
            uniq: Dict[str, str] = {}
            dedup_names: List[str] = []
            dedup_seqs: List[str] = []
            for nm, sq in zip(names, seqs):
                if sq not in uniq:
                    uniq[sq] = nm
                    dedup_names.append(nm)
                    dedup_seqs.append(sq)
            if len(dedup_seqs) != len(seqs):
                logging.info(f"Cluster {cluster}: dedup {len(seqs)} -> {len(dedup_seqs)}")
                names, seqs = dedup_names, dedup_seqs
                n = len(seqs)

        logging.info(f"Cluster {cluster}: {n} sequences (missing {missing}) → computing identities with {args.backend}...")

        if args.backend == "vsearch":
            pairs = vsearch_allpairs_identities(names, seqs, threads=args.threads)
        else:
            pairs = biopython_global_identities(names, seqs, max_pairs=args.max_pairs)

        idents = [iden for _, _, iden in pairs if not (iden is None or np.isnan(iden))]
        stats = summarize(idents)
        stats.update({
            "cluster": cluster,
            "n_sequences": n,
            "n_missing": int(missing),
            "n_pairs_computed": len(pairs),
            "backend": args.backend,
        })
        summary_rows.append(stats)

        # Optional matrix
        if n <= args.matrix_limit:
            mat = np.full((n, n), np.nan, dtype=float)
            np.fill_diagonal(mat, 1.0)
            for i, j, iden in pairs:
                mat[i, j] = iden
                mat[j, i] = iden
            mat_df = pd.DataFrame(mat, index=names, columns=names)
            mat_path = out_dir / "matrices" / f"cluster_{cluster}_identity.csv"
            mat_df.to_csv(mat_path)
        else:
            logging.info(f"Cluster {cluster}: size {n} > matrix-limit {args.matrix_limit}; skipping matrix write")

    # Write summary
    summary_df = pd.DataFrame(summary_rows)
    # Order columns nicely if present
    col_order = [
        "cluster", "n_sequences", "n_missing", "n_pairs_computed", "backend",
        "pair_count", "mean_identity", "median_identity", "min_identity", "max_identity", "std_identity",
    ]
    for c in col_order:
        if c not in summary_df.columns:
            summary_df[c] = np.nan
    summary_df = summary_df[col_order]
    out_csv = Path(out_dir) / "cluster_ani_summary.csv"
    summary_df.to_csv(out_csv, index=False)
    logging.info(f"Wrote summary: {out_csv}")


if __name__ == "__main__":
    main()

