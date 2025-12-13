import os
import glob
import sys
import argparse
import pandas as pd
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.spatial.distance import directed_hausdorff
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import igraph as ig
import leidenalg
import h5py
import faulthandler; faulthandler.enable()
import csv
import matplotlib.pyplot as plt

# Check for ESM availability
try:
    import gc, torch
    import esm
    HAS_ESM = True
except ImportError:
    HAS_ESM = False

# Check for BioPython availability
try:
    from Bio import SeqIO
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    logging.warning("FAISS not available. Install with: pip install faiss-cpu or faiss-gpu")

try:
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Global variables for model and configuration
MODEL = None
BATCH_CONVERTER = None
DEVICE = None
EMBEDDING_DIM = 1152

def get_device(device_arg):
    """
    Determine torch device based on argument or availability,
    including Apple Silicon MPS support.
    """
    if not HAS_ESM:
        return None

    # Auto‐select: prefer CUDA, then MPS, then CPU
    if device_arg == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        # MPS is Apple’s Metal-backed GPU support
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')

    # Explicit device selection with validation
    if device_arg == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError("CUDA device requested but not available")
        return torch.device('cuda')
    if device_arg == 'mps':
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            raise ValueError("MPS device requested but not available")
        return torch.device('mps')
    if device_arg == 'cpu':
        return torch.device('cpu')

    # Fallback for any other string
    return torch.device(device_arg)


def load_esm_model(model_name: str = 'esmc_600m', device_arg: str = 'auto'):
    """Load ESM-C model for protein embedding generation."""
    global MODEL, BATCH_CONVERTER, DEVICE, EMBEDDING_DIM

    if not HAS_ESM:
        raise ImportError("ESM not available")

    DEVICE = get_device(device_arg)

    try:
        # Try to load ESM-C 600M - adjust model loading based on actual ESM-C availability
        # Note: ESM-C might require different loading approach - this is a placeholder
        if model_name == 'esmc_600m':
            # For now, use ESM-2 as ESM-C might not be directly available yet
            MODEL, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            EMBEDDING_DIM = 1280  # ESM-2 dimension
            logging.info("Using ESM-2 650M (ESM-C 600M not yet available)")
        elif model_name == 'esmc_300m':
            MODEL, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
            EMBEDDING_DIM = 640
            logging.info("Using ESM-2 150M as proxy for ESM-C 300M")
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        MODEL.eval()
        if DEVICE == "cuda" and torch.cuda.device_count() > 1:
            MODEL = torch.nn.DataParallel(MODEL)  # 🔁 wrap for multi-GPU
        if DEVICE:
            MODEL = MODEL.to(DEVICE)

        BATCH_CONVERTER = alphabet.get_batch_converter()
        logging.info(f"Loaded {model_name} on {DEVICE}")

    except Exception as e:
        logging.error(f"Failed to load ESM model: {e}")
        raise


def merge_sample(sample: str, genomad_dir: str,  output_dir,mge_gene_metadata, host: str):
    """
    Merge gene-level, plasmid summary, virus summary, and gene-level annotations
    for a single sample, using contig and coordinates.
    """
    logging.info(f"Processing sample: {sample}")

    base_pattern = os.path.join(genomad_dir, "*_annotate", "*_genes.tsv")
    plasmid_summary_pattern = os.path.join(genomad_dir, "*_summary", "*_plasmid_summary.tsv")
    virus_summary_pattern = os.path.join(genomad_dir, "*_summary", "*_virus_summary.tsv")
    plasmid_genes_pattern = os.path.join(genomad_dir, "*_summary", "*_plasmid_genes.tsv")
    virus_genes_pattern = os.path.join(genomad_dir, "*_summary", "*_virus_genes.tsv")

    # Locate files
    gene_files = glob.glob(base_pattern)
    ps_files = glob.glob(plasmid_summary_pattern)
    vs_files = glob.glob(virus_summary_pattern)
    pg_files = glob.glob(plasmid_genes_pattern)
    vg_files = glob.glob(virus_genes_pattern)

    if not gene_files:
        logging.warning(f"No gene file found for {sample}")
        return

    # Read base genes
    gene_file = gene_files[0]
    logging.debug(f"Reading base gene file: {gene_file}")
    base = pd.read_csv(gene_file, sep='\t')

    # Derive contig by stripping the last '_<gene_index>' suffix
    base['contig'] = base['gene'].str.rsplit('_', n=1).str[0]
    # Preserve seq_name for output
    base['seq_name'] = base['contig']

    # Merge plasmid summary on contig
    if ps_files:
        ps = pd.read_csv(ps_files[0], sep='\t')
        # Remove any '|...' then strip final '_<index>'
        ps['contig'] = ps['seq_name']#.apply(lambda x: x.split('|', 1)[0] if '|' in x else x.rsplit('_', 1)[0])
        #ps = ps.drop(columns=['seq_name'])
        ps = ps.rename({c: f"{c}_plasmid_summary" for c in ps.columns if c != 'contig'}, axis=1)
        base = base.merge(ps, on='contig', how='left')
        logging.debug(f"Joined plasmid summary columns: {ps.columns.tolist()[1:]} to base")
    else:
        logging.warning(f"No plasmid summary file for sample {sample}")

    # Merge virus summary on contig
    if vs_files:
        vs = pd.read_csv(vs_files[0], sep='\t')
       # vs['contig'] = vs['seq_name'].apply(lambda x: x.split('|', 1)[0] if '|' in x else x.rsplit('_', 1)[0])
        vs['contig'] = vs['seq_name'].str.split('|').str[0]
        #vs = vs.drop(columns=['seq_name'])
        vs = vs.rename({c: f"{c}_virus_summary" for c in vs.columns if c != 'contig'}, axis=1)
        base = base.merge(vs, on='contig', how='left')
        logging.debug(f"Joined virus summary columns: {vs.columns.tolist()[1:]} to base")
    else:
        logging.warning(f"No virus summary file for sample {sample}")

    # Merge plasmid gene annotations by contig, start, end
    if pg_files:
        pg = pd.read_csv(pg_files[0], sep='\t')
        # Normalize contig: split '|', then strip last suffix
        pg['contig'] = pg['gene'].apply(lambda x: x.split('|', 1)[0] if '|' in x else x.rsplit('_', 1)[0])
        # Suffix all annotation columns except merge keys
        suf_pg = {c: f"{c}_plasmid_gene" for c in pg.columns if c not in ['contig', 'start', 'end', 'seq_name']}
        pg = pg.rename(suf_pg, axis=1)
        base = base.merge(pg, on=['contig', 'start', 'end'], how='left')
        logging.debug(f"Joined plasmid gene annotation columns: {list(suf_pg.values())}")
    else:
        logging.warning(f"No plasmid gene annotation file for sample {sample}")

    # Merge virus gene annotations by contig, start, end
    if vg_files:
        vg = pd.read_csv(vg_files[0], sep='\t')
        vg['contig'] = vg['gene'].apply(lambda x: x.split('|', 1)[0] if '|' in x else x.rsplit('_', 1)[0])
        suf_vg = {c: f"{c}_virus_gene" for c in vg.columns if c not in ['contig', 'start', 'end', 'seq_name']}
        vg = vg.rename(suf_vg, axis=1)
        base = base.merge(vg, on=['contig', 'start', 'end'], how='left')
        logging.debug(f"Joined virus gene annotation columns: {list(suf_vg.values())}")
    else:
        logging.warning(f"No virus gene annotation file for sample {sample}")

    # Drop helper contig column
    result = base#.drop(columns=['contig'])

    # Write to Excel
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{sample}_merged.xlsx")
    result.to_excel(out_file, index=False)
    logging.info(f"Saved merged output to {out_file}")
    # MGE-related columns
    mge_columns = [
        'marker','annotation_amr', 'annotation_amr_plasmid_gene', 'annotation_amr_virus_gene',
        'annotation_conjscan', 'annotation_conjscan_plasmid_gene', 'annotation_conjscan_virus_gene',
        'plasmid_hallmark', 'virus_hallmark',
        'marker_plasmid_gene', 'marker_virus_gene',
        'plasmid_hallmark_plasmid_gene', 'virus_hallmark_virus_gene'
    ]

    # Filter for genes with MGE annotations
    #mge_genes = base[base[mge_columns].notnull().any(axis=1)].copy()

    mask = ((base[mge_columns] != 0) &
            (base[mge_columns] != '') &
            (base[mge_columns].notna())).any(axis=1)

    mge_genes = base[mask].copy()
    print(f"Filtered {len(base)} -> {len(mge_genes)} MGE genes")

    mge_genes_filtered = mge_genes[['gene', 'contig']].drop_duplicates(subset=['gene', 'contig'])
    mge_genes_filtered['genome_id'] = sample
    mge_genes_filtered['host'] =host  # Update accordingly if host information is available

    # Append to unified gene_metadata.csv
    metadata_tsv = os.path.join(output_dir,mge_gene_metadata)
    if not os.path.exists(metadata_tsv):
        # Write new TSV with header
        mge_genes_filtered.to_csv(metadata_tsv, sep='\t', mode='w', header=True, index=False)
    else:
        # Append without header
        mge_genes_filtered.to_csv(metadata_tsv, sep='\t', mode='a', header=False, index=False)

    logging.info(f"Appended MGE genes to {metadata_tsv}")


def extract_mge_sequences_only(genomad_dir, genome_id, mge_metadata_file, output_dir):
    """
    Extract sequences only for genes that are in the MGE metadata.
    This ensures perfect alignment between metadata and embeddings.
    """
    if not HAS_BIOPYTHON:
        raise ImportError("BioPython not available. Install with: pip install biopython")

    # Load MGE metadata to get the list of MGE genes for this genome
    metadata_file = os.path.join(output_dir + "_merge", mge_metadata_file)
    if not os.path.exists(metadata_file):
        logging.error(f"MGE metadata file not found: {metadata_file}")
        return None, None

    mge_metadata = pd.read_csv(metadata_file, sep='\t')
    genome_mge_genes = mge_metadata[mge_metadata['genome_id'] == genome_id]['gene'].tolist()

    if not genome_mge_genes:
        logging.warning(f"No MGE genes found for {genome_id}")
        return [], []

    logging.info(f"Found {len(genome_mge_genes)} MGE genes for {genome_id}")

    # Load protein sequences
    proteins_file = os.path.join(genomad_dir, "*_annotate", "*_proteins.faa")
    prot_files = glob.glob(proteins_file)

    if not prot_files:
        logging.warning(f"No protein fasta found for {genome_id}")
        return None, None

    proteins_file = prot_files[0]
    prot_dict = {rec.id: str(rec.seq).replace('*', '').strip()
                 for rec in SeqIO.parse(proteins_file, 'fasta')}

    # Extract sequences only for MGE genes
    mge_sequences = []
    valid_mge_genes = []
    missing = 0

    for gene_id in genome_mge_genes:
        seq = prot_dict.get(gene_id, '')
        if seq and all(aa in 'ACDEFGHIKLMNPQRSTVWYXZ' for aa in seq.upper()):
            mge_sequences.append(seq.upper())
            valid_mge_genes.append(gene_id)
        else:
            logging.warning(f"Missing or invalid sequence for MGE gene {gene_id}")
            missing += 1

    logging.info(f"Extracted {len(mge_sequences)}/{len(genome_mge_genes)} valid MGE sequences")
    # Verify alignment
    assert len(mge_sequences) == len(valid_mge_genes), "Sequences and gene IDs must be aligned"

    return mge_sequences, valid_mge_genes


def generate_and_save_embeddings(
        model, batch_converter, final_layer,emb_dim, device,
        mge_sequences, valid_mge_genes, genome_id,output_h5_file, batch_size
):
    """
    Generate embeddings only for MGE genes.
    Returns embeddings that perfectly align with valid_mge_genes.
    """
    if model is None:
        raise ValueError("ESM model is not loaded")

    if not mge_sequences:
        logging.warning(f"No valid MGE sequences for {genome_id}")
        return np.zeros((0, 0)), []

        # Verify alignment
    if len(mge_sequences) != len(valid_mge_genes):
        raise ValueError(f"Sequence count ({len(mge_sequences)}) != gene count ({len(valid_mge_genes)})")

        # Prepare batch entries using actual gene IDs as labels
   # entries = [(gene_id, seq) for gene_id, seq in zip(valid_mge_genes, mge_sequences)]

    num_genes = len(valid_mge_genes)
   # embeddings = np.zeros((len(mge_sequences), emb_dim))
    with h5py.File(output_h5_file, 'w') as h5f:
        embeddings_ds = h5f.create_dataset('embeddings', shape=(num_genes, emb_dim), dtype='float32')
        gene_ids_ds = h5f.create_dataset('gene_ids', data=np.array(valid_mge_genes, dtype='S'))

        for start in range(0, num_genes, batch_size):
            end = min(start + batch_size, num_genes)
            batch_seqs = mge_sequences[start:end]
            batch_entries = [(valid_mge_genes[i], seq) for i, seq in enumerate(batch_seqs, start)]

            labels, strs, toks = batch_converter(batch_entries)
            toks = toks.to(device)

            with torch.no_grad():
                out = model(toks, repr_layers=[final_layer], return_contacts=False)
                reps = out['representations'][final_layer]
                batch_emb = reps[:, 1:-1].mean(1).cpu().numpy()

            embeddings_ds[start:end, :] = batch_emb
            print(f"Saved embeddings for genes {start} to {end}")
            torch.cuda.empty_cache()

def h5_to_memmap(h5_folder, memmap_file, gene_metadata_file, host_labels_df):
    h5_files = glob.glob(os.path.join(h5_folder, "*_embeddings.h5"))


    # First pass: get total embeddings and embedding dimension
    total_embeddings = 0
    emb_dim = None
    gene_ids_all = []
    hosts_all = []
    genome_ids_all = []

    for h5_file in h5_files:
        genome_id = os.path.basename(h5_file).replace("_embeddings.h5", "")
        host_label = host_labels_df.get(genome_id, 'unknown')

        with h5py.File(h5_file, 'r') as h5f:
            num_emb = h5f['embeddings'].shape[0]
            emb_dim = h5f['embeddings'].shape[1]
            total_embeddings += num_emb

            gene_ids = h5f['gene_ids'][:].astype(str)
            gene_ids_all.extend(gene_ids)
            hosts_all.extend([host_label] * num_emb)
            genome_ids_all.extend([genome_id] * num_emb)

    # Create memmap for embeddings
    embeddings_memmap = np.memmap(memmap_file, dtype='float32', mode='w+', shape=(total_embeddings, emb_dim))

    # Second pass: fill memmap
    current_index = 0
    for h5_file in h5_files:
        with h5py.File(h5_file, 'r') as h5f:
            emb = h5f['embeddings'][:]
            num_emb = emb.shape[0]
            embeddings_memmap[current_index:current_index+num_emb, :] = emb
            current_index += num_emb

    embeddings_memmap.flush()

    # Save combined metadata file with genome IDs, gene IDs, and host labels
    gene_metadata = pd.DataFrame({
        'genome_id': genome_ids_all,
        'gene_id': gene_ids_all,
        'host': hosts_all
    })

    gene_metadata.to_csv(gene_metadata_file, sep='\t', index=False)

    print(f"✅ Consolidated {total_embeddings} embeddings into {memmap_file}")
    print(f"✅ Unified gene metadata (with hosts) saved to {gene_metadata_file}")

def embedding_clusters_from_memmap(
    memmap_file: str,
    metadata_file: str,
    output_dir: str,
    embedding_dim: int,
    k_neighbors: int = 10,
    leiden_resolution: float = 0.5
):
   # output_dir = Path(output_dir)
   # output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    metadata_df = pd.read_csv(metadata_file, sep='\t')
    n = len(metadata_df)

    # new: load into RAM once
    tmp = np.memmap(memmap_file, mode='r', dtype='float32', shape=(n, embedding_dim))
    embeddings = np.array(tmp, copy=True)   # makes a contiguous in‑RAM array
    del tmp

    if embeddings.shape[0] != n:
        raise ValueError(f"Embeddings shape {embeddings.shape} does not match metadata ({n} rows)")

    logging.info(f"Loaded {n} embeddings from memmap")

    # Hausdorff distance matrix
    hosts_unique = metadata_df['host'].unique()
    hausdorff_matrix = pd.DataFrame(index=hosts_unique, columns=hosts_unique, dtype=float)

    for host_a in hosts_unique:
        emb_a = embeddings[metadata_df['host'] == host_a]
        for host_b in hosts_unique:
            emb_b = embeddings[metadata_df['host'] == host_b]
            if len(emb_a) > 0 and len(emb_b) > 0:
                d_ab = directed_hausdorff(emb_a, emb_b)[0]
                d_ba = directed_hausdorff(emb_b, emb_a)[0]
                hausdorff_matrix.loc[host_a, host_b] = max(d_ab, d_ba)

    hausdorff_tsv = os.path.join(output_dir,"hausdorff_distances.csv")
    hausdorff_matrix.to_csv(hausdorff_tsv, sep='\t', index=False)
    logging.info("Saved Hausdorff distance matrix")
    # drop large objects
    del hausdorff_matrix
    gc.collect()

    # free PyTorch GPU memory
    torch.cuda.empty_cache()
   # Hausdorff distance per genome


    genome_ids_unique = metadata_df['genome_id'].unique()
    hausdorff_genome_matrix = pd.DataFrame(index=genome_ids_unique, columns=genome_ids_unique, dtype=float)

    for genome_a in genome_ids_unique:
        emb_a = embeddings[metadata_df['genome_id'] == genome_a]
        for genome_b in genome_ids_unique:
            emb_b = embeddings[metadata_df['genome_id'] == genome_b]
            if len(emb_a) > 0 and len(emb_b) > 0:
                d_ab = directed_hausdorff(emb_a, emb_b)[0]
                d_ba = directed_hausdorff(emb_b, emb_a)[0]
                hausdorff_genome_matrix.loc[genome_a, genome_b] = max(d_ab, d_ba)

    hausdorff_pergenome_tsv = os.path.join(output_dir,"hausdorff_distances_per_genome.csv")
    hausdorff_genome_matrix.to_csv(hausdorff_pergenome_tsv, sep='\t', index=False)
    logging.info("Saved Hausdorff distance matrix per genome")
    # drop large objects
    del hausdorff_genome_matrix
    gc.collect()

    # free PyTorch GPU memory
    torch.cuda.empty_cache()

    # FAISS + Leiden clustering
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    distances, neighbors = index.search(embeddings, k_neighbors + 1)

    edges = set()
    for i, neighbor_ids in enumerate(neighbors):
        for j in neighbor_ids[1:]:  # skip self
            edge = tuple(sorted((i, j)))
            edges.add(edge)

    graph = ig.Graph(edges=list(edges), directed=False)
    graph.simplify()

    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=leiden_resolution
    )
    metadata_df['leiden_cluster'] = partition.membership
    
    # drop large objects

    del graph
    gc.collect()

    # free PyTorch GPU memory
    torch.cuda.empty_cache()

     # DBSCAN clustering
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    dbscan = DBSCAN(eps=0.5, min_samples=5, metric='cosine').fit(embeddings_scaled)
    metadata_df['dbscan_cluster'] = dbscan.labels_

    finalcluster_tsv = os.path.join(output_dir,"final_clusters.tsv")
    metadata_df.to_csv(finalcluster_tsv, sep='\t', index=False)

    # Host distributions with cluster_id as first column
    dbscan_host_distribution = metadata_df.groupby(['dbscan_cluster', 'host']).size().unstack(fill_value=0)
    dbscan_host_distribution = dbscan_host_distribution.reset_index()  # Move cluster_id from index to column
    dbscan_cluster_host_distribution_tsv = os.path.join(output_dir,"dbscan_cluster_host_distribution.tsv")
    metadata_df.to_csv(dbscan_cluster_host_distribution_tsv, sep='\t', index=False)

    #dbscan_host_distribution.to_csv(output_dir / "dbscan_cluster_host_distribution.csv", index=False)

    leiden_host_distribution = metadata_df.groupby(['leiden_cluster', 'host']).size().unstack(fill_value=0)
    leiden_host_distribution = leiden_host_distribution.reset_index()  # Move cluster_id from index to column
    leiden_cluster_host_distribution_tsv = os.path.join(output_dir,"leiden_cluster_host_distribution.tsv")
    metadata_df.to_csv(leiden_cluster_host_distribution_tsv, sep='\t', index=False)
    #leiden_host_distribution.to_csv(output_dir / "leiden_cluster_host_distribution.csv", index=False)    

    # Analyze clusters (DBSCAN)
    cluster_stats = analyze_cluster_composition(metadata_df.rename(columns={'dbscan_cluster': 'cluster'}), output_dir, method='dbscan')
    cluster_stats = analyze_cluster_composition(metadata_df.rename(columns={'leiden_cluster': 'cluster'}), output_dir, method='leiden')

    # Visualize both cluster types
    visualize_clusters(embeddings_scaled, metadata_df['dbscan_cluster'].values, metadata_df, output_dir, method='dbscan')
    visualize_clusters(embeddings_scaled, metadata_df['leiden_cluster'].values, metadata_df, output_dir, method='leiden')

def analyze_cluster_composition(metadata, output_dir, method):
    """Analyze the composition of clusters by host and genome."""
    cluster_stats = []

    for cluster_id in metadata['cluster'].unique():
        if cluster_id == -1:  # Skip noise
            continue

        cluster_data = metadata[metadata['cluster'] == cluster_id]

        # Calculate host distribution
        host_counts = cluster_data['host'].value_counts()
        most_common_host = host_counts.index[0] if len(host_counts) > 0 else None
        host_purity = host_counts.iloc[0] / len(cluster_data) if len(host_counts) > 0 else 0

        # Calculate genome distribution
        genome_counts = cluster_data['genome_id'].value_counts()

        stats = {
            'cluster_id': cluster_id,
            'size': len(cluster_data),
            'unique_hosts': cluster_data['host'].nunique(),
            'unique_genomes': cluster_data['genome_id'].nunique(),
            'most_common_host': most_common_host,
            'host_purity': host_purity,
            'host_diversity': 1 - (host_counts ** 2).sum() / (host_counts.sum() ** 2),  # Simpson's diversity
            'genome_diversity': 1 - (genome_counts ** 2).sum() / (genome_counts.sum() ** 2),
            'cross_host': cluster_data['host'].nunique() > 1,
            'cross_genome': cluster_data['genome_id'].nunique() > 1
        }

        cluster_stats.append(stats)

    stats_df = pd.DataFrame(cluster_stats)

    if len(stats_df) == 0:
        logging.warning("No clusters found for analysis")
        return stats_df

    # Save detailed statistics
    out_file = os.path.join(output_dir, f'{method}_cluster_statistics.tsv')
    stats_df.to_csv(out_file, sep='\t', index=False)

    # Generate summary statistics
    summary = {
        'total_clusters': len(stats_df),
        'avg_cluster_size': stats_df['size'].mean(),
        'median_cluster_size': stats_df['size'].median(),
        'largest_cluster': stats_df['size'].max(),
        'cross_host_clusters': stats_df['cross_host'].sum(),
        'cross_genome_clusters': stats_df['cross_genome'].sum(),
        'avg_host_purity': stats_df['host_purity'].mean(),
        'avg_host_diversity': stats_df['host_diversity'].mean()
    }

    out_file = os.path.join(output_dir, f'{method}_cluster_summary.tsv')
    summary_df = pd.DataFrame([summary])  # Convert single dict to DataFrame
    summary_df.to_csv(out_file, sep='\t', index=False)

   # with open(out_file, "w", newline="") as tsvfile:
    # Define the fieldnames (column headers) based on dictionary keys
   #     fieldnames = summary[0].keys()
   #     writer = csv.DictWriter(tsvfile, fieldnames=fieldnames, delimiter='\t')

        # Write the header row
   #     writer.writeheader()

        # Write each dictionary as a row
    #    writer.writerows(summary)
    #    print(f"Data saved to {output_file}")
    
    logging.info(f"Cluster analysis: {summary['total_clusters']} clusters, "
                 f"avg size: {summary['avg_cluster_size']:.1f}, "
                 f"{summary['cross_host_clusters']} cross-host clusters")

    return stats_df

def visualize_clusters(embeddings, cluster_labels,
                       metadata, output_dir, method):
    """Create visualizations of clustering results."""
    if not HAS_SKLEARN:
        logging.warning("Scikit-learn not available - skipping visualizations")
        return

    # Reduce to 2D for visualization
    if embeddings.shape[1] > 2:
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
    else:
        embeddings_2d = embeddings

    # Plot clusters
    unique_clusters = np.unique(cluster_labels)
    plt.figure(figsize=(12, 8))
    if len(unique_clusters) > 20:
        # Use a different colormap or cycle through colors
        colors = plt.cm.tab20(np.linspace(0, 1, 20))
        colors = np.tile(colors, (len(unique_clusters) // 20 + 1, 1))[:len(unique_clusters)]
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))

    # Plot clusters
    plt.figure(figsize=(12, 8))
    for cluster_id, color in zip(unique_clusters, colors):
        mask = cluster_labels == cluster_id
        if cluster_id == -1:
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                        c='black', alpha=0.3, s=20, label='Noise')
        else:
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                        c=[color], alpha=0.7, s=30, label=f'Cluster {cluster_id}')

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(f'Protein Embedding Clusters - {method.upper()}')
    if len(unique_clusters) <= 20:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        # Skip legend if too many clusters
        pass
    
    plt.tight_layout()
    out_file = os.path.join(output_dir, f'{method}_clusters_visualization.png')
    
    # Issue 6: Add error handling for file saving
    try:
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        print(f"Cluster visualization saved to {out_file}")
    except Exception as e:
        logging.error(f"Failed to save cluster visualization: {e}")
    
    plt.close()  # This is good - prevents memory issues

    # Plot by host
    plt.figure(figsize=(12, 8))
    unique_hosts = metadata['host'].unique()
    #host_colors = plt.cm.Set1(np.linspace(0, 1, len(unique_hosts)))
    if len(unique_hosts) > 9:  # Set1 has 9 colors
        host_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_hosts)))
    else:
        host_colors = plt.cm.Set1(np.linspace(0, 1, len(unique_hosts)))


    for host, color in zip(unique_hosts, host_colors):
        mask = metadata['host'] == host
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                    c=[color], alpha=0.7, s=30, label=host)

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Protein Embeddings by Host')
#    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    if len(unique_hosts) <= 15:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    out_file = os.path.join(output_dir, f'{method}_hosts_visualization.png')
    #plt.savefig(out_file, dpi=300, bbox_inches='tight')
    try:
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        print(f"Host visualization saved to {out_file}")
    except Exception as e:
        logging.error(f"Failed to save host visualization: {e}")
    
    plt.close()

def load_host_labels(tsv_file):
    """Load genome host labels from TSV file with case-insensitive validation"""
    try:
        df = pd.read_csv(tsv_file, sep='\t')
        if 'sample_id' not in df.columns or 'host_label' not in df.columns:
            raise ValueError("TSV file must have 'sample_id' and 'host_label' columns")

        # Normalize host labels to lowercase and strip whitespace
        df['host_label'] = df['host_label'].str.strip().str.lower()
        host_labels = dict(zip(df['sample_id'], df['host_label']))

        # Case-insensitive validation
        valid_hosts = {'human', 'chicken', 'turkey', 'pork'}
        invalid_hosts = set(host_labels.values()) - valid_hosts
        if invalid_hosts:
            print(f"Warning: Found invalid host labels: {invalid_hosts}")
            print(f"Valid hosts are: {', '.join(sorted(valid_hosts))}")

        print(f"Loaded {len(host_labels)} genome host labels")
        return host_labels

    except Exception as e:
        print(f"Error loading host labels: {e}")
        return {}

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='merge geNomad ouputs')
    parser.add_argument('phase', choices=['merge', 'generate_embeddings'],
                        help='Pipeline phase to run')

    # Common arguments
    parser.add_argument('--host-labels', '-l', required=True,
                        help='Path to TSV file with genome IDs and host labels')
    parser.add_argument('--genomad-dir', '-g', required=True,
                        help='Directory containing geNomad output folders')

    # MGE detection arguments
    parser.add_argument('--output-prefix', '-o', default='debug_070225',
                        help='Output prefix  (creates .json and .tsv) (default: debug_070625)')
    parser.add_argument('--mge-gene-metadata', '-m', default='mge_functional_modules',
                        help='Output prefix for filtered MGE only results (creates .tsv) (default: mge_functional_modules)')


    # Embedding generation specific arguments
    parser.add_argument('--embedding-output-prefix', '-e', default='mge_gene_embeddings',
                        help='Output file for gene embeddings (default: mge_gene_embeddings)')
    parser.add_argument('--batch-size', '-b', type=int, default=16,
                        help='Batch size for embedding generation')
    parser.add_argument('--esm-model', default='esmc_600m',
                        choices=['esmc_600m', 'esmc_300m'],
                        help='ESM-C model to use')
    parser.add_argument('--device', default='cuda', choices=['auto', 'cpu', 'cuda', 'mps'],
                        help='Device for computation')

    # Debug mode
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with default parameters')

    return parser.parse_args()
def main():
    """Main pipeline execution"""
    # Debug mode: provide default arguments when no args given
    if len(sys.argv) == 1:
        print("No arguments provided. Running in debug mode...")
        print("To run normally, use: python script.py detect_mges --host-labels <file> --genomad-dir <dir>")

        # Set debug arguments - MODIFY THESE FOR YOUR SETUP
        sys.argv = [
            sys.argv[0],  # script name
            'merge',  # 'generate_embeddings', # or 'merge' or 'prepare_training'
            '--host-labels',
            '/scratch/liu_price_lab/mlaziz/incubator_2025/deepMGEhost_project/Data/sampletest/samplelist_1.txt',
            '--genomad-dir',
            '/scratch/liu_price_lab/mlaziz/incubator_2025/deepMGEhost_project/Data/sampletest',
            '--output-prefix', '/scratch/liu_price_lab/mlaziz/incubator_2025/deepMGEhost_project/output/debug_070625',
            '--mge-gene-metadata', 'mge_functional_modules.tsv',
            '--debug'
        ]
        print(f"Debug args: {' '.join(sys.argv[1:])}")

    args = parse_arguments()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s'
    )

    if torch.cuda.is_available():
        logging.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        logging.info(f"Memory allocated: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
        # Limit memory use to 50% of the GPU
       # torch.cuda.set_per_process_memory_fraction(0.5, device=0)

    if args.phase == "merge":
        # Phase 1: MGE Detection
        print("=== Phase 1: geNomad merge ===")


        # Load host labels to get genome list
        host_labels = load_host_labels(args.host_labels)
        if not host_labels:
            logging.error("Error: Could not load host labels")
            return


        for genome_id in host_labels.keys():
            genomad_genome_dir = os.path.join(args.genomad_dir, genome_id)

            if os.path.exists(genomad_genome_dir):
                try:
                    outprefix=args.output_prefix+"_merge"
                    merge_sample(genome_id, genomad_genome_dir, outprefix,args.mge_gene_metadata, host_labels[genome_id])
                except Exception as e:
                    logging.error(f"Error processing {genome_id}: {e}")
                    return
    elif args.phase == 'generate_embeddings':
        logging.info("=== Phase 2: MGE-Only Embedding Generation ===")
        logging.info(f"ESM model: {args.esm_model}, device: {args.device}, batch size: {args.batch_size}")

        # If embeddings file already exists, load and skip regeneration
        outputdir = args.output_prefix+"_embedding"
        embeddings_filename = os.path.join(outputdir, args.embedding_output_prefix)
         # Map model to embedding dimension
        final_layer = 33
        dim_map = {6: 320, 8: 320, 12: 480, 30: 640, 33: 1280}
        emb_dim = dim_map.get(final_layer, 320)

        if os.path.exists(embeddings_filename+".memmap"):
            memmap_file = embeddings_filename + ".memmap"
            gene_metadata_file = embeddings_filename + "_metadata.tsv"
            # Load unified gene metadata
            #metadata_df = pd.read_csv(gene_metadata_file, sep='\t')

            # Load embeddings via memmap
            #embeddings = np.memmap(memmap_file, dtype='float32', mode='r',
            #                       shape=(len(metadata_df), emb_dim))
            embedding_clusters_from_memmap(memmap_file, gene_metadata_file,outputdir,emb_dim)
            return
        else:
            # Load ESM model
            load_esm_model(args.esm_model, args.device)

            # Load host labels
            host_labels = load_host_labels(args.host_labels)
            if not host_labels:
                logging.error("Error: Could not load host labels")
                return

            os.makedirs(outputdir, exist_ok=True)

            for genome_id in host_labels.keys():
                genomad_genome_dir = os.path.join(args.genomad_dir, genome_id)

                if os.path.exists(genomad_genome_dir):
                    logging.info(f"Processing MGE sequences for {genome_id}")

                    try:
                        # Extract only MGE sequences
                        mge_sequences, valid_mge_genes = extract_mge_sequences_only(
                            genomad_genome_dir, genome_id, args.mge_gene_metadata, args.output_prefix
                        )

                        if mge_sequences is None:
                            continue

                        if not mge_sequences:
                            logging.warning(f"No valid MGE sequences found for {genome_id}")
                            continue

                        # Generate embeddings for MGE genes only
                        output_h5_file = os.path.join(outputdir, f"{genome_id}_embeddings.h5")

                        generate_and_save_embeddings(
                            model=MODEL,
                            batch_converter=BATCH_CONVERTER,
                            final_layer=33,
                            emb_dim=emb_dim,
                            device=DEVICE,
                            mge_sequences=mge_sequences,
                            valid_mge_genes=valid_mge_genes,
                            genome_id=genome_id,
                            output_h5_file=output_h5_file,
                            batch_size=args.batch_size
                        )

                    except Exception as e:
                        logging.error(f"Error processing {genome_id}: {e}")
                        continue
            memmap_file = embeddings_filename+'.memmap'
            gene_metadata_file = embeddings_filename+"_embeddings_metadata.tsv"
            # Check for existence of any *_embeddings.h5 files
            h5_files = glob.glob(os.path.join(outputdir, "*_embeddings.h5"))

            if not h5_files:
                print(f"❌ No *_embeddings.h5 files found in {outputdir}. Aborting!.")
                exit()
            print(f"✅ Found {len(h5_files)} H5 files. Proceeding with consolidation.")
            h5_to_memmap(
                outputdir, memmap_file, gene_metadata_file, host_labels
            )

            embedding_clusters_from_memmap(memmap_file, gene_metadata_file, outputdir,emb_dim)

if __name__ == '__main__':
    main()

