#!/usr/bin/env python3
"""
Per-cluster gene/protein percent identity.

Supports:
- DNA (default): fast all-vs-all identities with VSEARCH (--allpairs_global) or Biopython fallback.
- Protein (AA): identities via Biopython global alignment with BLOSUM62 (VSEARCH **does not** support proteins).

Input
-----
- Table (TSV/CSV/XLSX) with at least: genome_id, gene_id, <cluster_col>
- Directory of FASTA files per genome. Use --file-pattern to specify filenames.
  * Default DNA: "{genome}_genes.ffn"
  * Default AA : "{genome}_genes.faa"
  FASTA headers often look like: >NODE_233_..._ID_465_1:2-76(-)
  We match gene_id to the header *before* the first ':'

Outputs
-------
- out_dir/cluster_ani_summary.csv: per-cluster statistics (mean/median/min/max identity, counts).
- out_dir/matrices/cluster_<cluster>_identity.csv: identity matrix for clusters ≤ --matrix-limit sequences.

Notes
-----
- This measures per-gene/protein identity, not whole-genome ANI.
- Complexity is O(n^2) per cluster; use --matrix-limit to avoid huge matrices.

Usage
-----
python cluster_gene_ani.py \
  --table path/to/table.tsv \
  --ffn-dir /path/to/ff_dir \
  --out-dir out_geneANI \
  --cluster-col leiden_cluster \
  --seq-type dna \
  --threads 16 \
  --backend vsearch

Dependencies
------------
- pandas, numpy, biopython
- For fast DNA mode: vsearch (conda install -c bioconda vsearch)
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
        df = pd.read_csv(p, sep=None, engine="python", dtype=str)
    return df


def norm_gene_id(header_or_id: str) -> str:
    """Normalize to the portion before the first ':' (matches example headers)."""
    return str(header_or_id).split(":", 1)[0]


def load_sequences_for_needed(ff_dir: Path, needed: Dict[str, set], file_pattern: str) -> Dict[str, Dict[str, str]]:
    """Return {genome_id: {gene_key: SEQ}} only for needed gene IDs per genome."""
    seqs_by_genome: Dict[str, Dict[str, str]] = {}
    missing = []
    for genome, wanted in needed.items():
        fpath = ff_dir / file_pattern.format(genome=genome)
        if not fpath.exists():
            missing.append(genome)
            seqs_by_genome[genome] = {}
            continue
        idx: Dict[str, str] = {}
        try:
            for rec in SeqIO.parse(str(fpath), "fasta"):
                key = norm_gene_id(rec.id)
                if key in wanted:
                    idx[key] = str(rec.seq).upper()
        except Exception as e:
            logging.error(f"Failed parsing {fpath}: {e}")
        seqs_by_genome[genome] = idx
    if missing:
        show = ", ".join(missing[:3]) + (" ..." if len(missing) > 3 else "")
        logging.warning(f"Missing FASTAs for {len(missing)} genomes (e.g., {show}); those rows will be skipped.")
    return seqs_by_genome


# --------------------------
# Identity backends
# --------------------------

def vsearch_allpairs_identities(names: List[str], seqs: List[str], threads: int = 1) -> List[Tuple[int, int, float]]:
    """Compute pairwise identities using vsearch --allpairs_global (DNA only).
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


def biopython_global_identities_dna(seqs: List[str], max_pairs: int | None = None, rng_seed: int = 1337) -> List[Tuple[int, int, float]]:
    """DNA identities via global alignment with simple match/mismatch scoring."""
    if len(seqs) < 2:
        return []
    from Bio import pairwise2

    all_pairs = [(i, j) for i, j in combinations(range(len(seqs)), 2)]
    if max_pairs is not None and max_pairs < len(all_pairs):
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(len(all_pairs), size=max_pairs, replace=False)
        pairs = [all_pairs[k] for k in idx]
    else:
        pairs = all_pairs

    out: List[Tuple[int, int, float]] = []
    for i, j in pairs:
        aln = pairwise2.align.globalms(seqs[i], seqs[j], 1, 0, -1, -1, one_alignment_only=True)
        if not aln:
            out.append((i, j, np.nan)); continue
        a_aln, b_aln, score, _, _ = aln[0]
        L = len(a_aln)
        ident = float(score) / float(L) if L else np.nan
        out.append((i, j, ident))
    return out


def biopython_global_identities_aa(seqs: List[str], gap_open: float = -10.0, gap_extend: float = -1.0,
                                   max_pairs: int | None = None, rng_seed: int = 1337) -> List[Tuple[int, int, float]]:
    """Protein identities via global alignment with BLOSUM62.

    Identity is computed as (# identical residue pairs) / (# aligned positions where both are not gaps).
    """
    if len(seqs) < 2:
        return []
    from Bio import pairwise2
    from Bio.Align import substitution_matrices
    blosum62 = substitution_matrices.load("BLOSUM62")

    all_pairs = [(i, j) for i, j in combinations(range(len(seqs)), 2)]
    if max_pairs is not None and max_pairs < len(all_pairs):
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(len(all_pairs), size=max_pairs, replace=False)
        pairs = [all_pairs[k] for k in idx]
    else:
        pairs = all_pairs

    def pct_ident(a: str, b: str) -> float:
        aln = pairwise2.align.globalds(a, b, blosum62, gap_open, gap_extend, one_alignment_only=True)
        if not aln:
            return np.nan
        a_aln, b_aln, _, _, _ = aln[0]
        denom = 0
        matches = 0
        for x, y in zip(a_aln, b_aln):
            if x != '-' and y != '-':
                denom += 1
                if x == y:
                    matches += 1
        return (matches / denom) if denom else np.nan

    out: List[Tuple[int, int, float]] = []
    for i, j in pairs:
        out.append((i, j, pct_ident(seqs[i], seqs[j])))
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
    ap = argparse.ArgumentParser(description="Per-cluster gene/protein identity (DNA: VSEARCH/Biopython; AA: Biopython)")
    ap.add_argument("--table", required=True, help="TSV/CSV/XLSX with genome_id, gene_id, and cluster col")
    ap.add_argument("--ffn-dir", required=True, help="Directory containing per-genome FASTA files")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--cluster-col", default="leiden_cluster", help="Cluster column (default: leiden_cluster)")
    ap.add_argument("--genome-col", default="genome_id", help="Genome ID column (default: genome_id)")
    ap.add_argument("--gene-col", default="gene_id", help="Gene ID column (default: gene_id)")
    ap.add_argument("--seq-type", choices=["dna", "aa"], default="dna", help="Sequence type: dna or aa (protein)")
    ap.add_argument("--file-pattern", default=None, help="Filename pattern with {genome}. If omitted, uses {genome}_genes.ffn for dna or {genome}_genes.faa for aa")
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 1) // 2), help="Threads for vsearch (DNA)")
    ap.add_argument("--matrix-limit", type=int, default=500, help="Write matrix only if cluster size ≤ this (default: 500)")
    ap.add_argument("--min-seqs", type=int, default=2, help="Skip clusters with < N sequences (default: 2)")
    ap.add_argument("--backend", choices=["vsearch", "biopython"], default="vsearch", help="Computation backend (DNA only: vsearch is fastest)")
    ap.add_argument("--max-pairs", type=int, default=None, help="(biopython only) Subsample to at most this many pairs")
    ap.add_argument("--deduplicate", action="store_true", help="Drop exact duplicate sequences within a cluster")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"], help="Logging level")
    args = ap.parse_args()

    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=getattr(logging, args.log_level))

    table_path = Path(args.table)
    ff_dir = Path(args.ffn_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "matrices").mkdir(exist_ok=True)

    # Decide file pattern
    if args.file_pattern is None:
        file_pattern = "{genome}_genes.ffn" if args.seq_type == "dna" else "{genome}_genes.faa"
    else:
        # Allow users to pass e.g. "{genome}.faa" or "{genome}_prot.faa"
        if "{genome}" not in args.file_pattern:
            raise ValueError("--file-pattern must contain '{genome}' placeholder")
        file_pattern = args.file_pattern

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

    logging.info("Indexing FASTAs for required genomes...")
    seqs_by_genome = load_sequences_for_needed(ff_dir, needed_by_genome, file_pattern)

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

        logging.info(f"Cluster {cluster}: {n} sequences (missing {missing}) → computing identities for seq-type={args.seq_type} backend={args.backend} ...")

        # Compute pairwise identities
        if args.seq_type == "dna":
            if args.backend == "vsearch":
                pairs = vsearch_allpairs_identities(names, seqs, threads=args.threads)
            else:
                pairs = biopython_global_identities_dna(seqs, max_pairs=args.max_pairs)
        else:  # protein
            if args.backend == "vsearch":
                logging.warning("VSEARCH does not support protein sequences; switching to biopython.")
            pairs = biopython_global_identities_aa(seqs, max_pairs=args.max_pairs)

        idents = [iden for _, _, iden in pairs if not (iden is None or np.isnan(iden))]
        stats = summarize(idents)
        stats.update({
            "cluster": cluster,
            "n_sequences": n,
            "n_missing": int(missing),
            "n_pairs_computed": len(pairs),
            "backend": ("vsearch" if (args.seq_type == "dna" and args.backend == "vsearch") else "biopython"),
            "seq_type": args.seq_type,
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
    col_order = [
        "cluster", "seq_type", "backend", "n_sequences", "n_missing", "n_pairs_computed",
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

