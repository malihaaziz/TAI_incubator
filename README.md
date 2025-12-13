# Evaluating Fidelity of Protein LLM Embeddings for Host-Association Analysis

**Author**: Maliha Aziz, Dr. Lance Price
**Institution**: The George Washington University
**Affiliation**: GW TAI Trustworthy AI Antibiotic Resistance Action Center

## Overview

This project evaluates the fidelity of protein language model (LLM) embeddings, specifically ESM-2, for analyzing host associations in *E. coli* and mobile genetic elements (MGEs). The pipeline processes genomic data to identify and cluster MGE-associated proteins across different bacterial hosts (human, chicken, turkey, pork) to understand horizontal gene transfer patterns and identify host-associated genes.

**Key Innovation**: Leverages alignment-free protein embeddings from ESM-2 (trained on ~65M UniRef sequences) to accelerate pangenome-like grouping and identify host-associated gene clusters, reducing reliance on computationally expensive sequence alignments while maintaining interpretability.

## Background & Motivation

Mobile genetic elements (MGEs) such as plasmids and viruses facilitate horizontal gene transfer between bacteria, enabling the spread of traits like antibiotic resistance. Understanding which MGE proteins are shared across different hosts is critical for:

- Tracking antibiotic resistance gene transmission
- Identifying cross-host MGE transfer events
- Understanding structure-function relationships in MGE proteins
- Analyzing evolutionary relationships among microbial communities

Protein embeddings from large language models (ESM-2, ESM-C) capture functional and structural features of proteins, potentially revealing biologically meaningful clusters that correspond to:
- Gene gain/loss events
- Structure-aware protein families
- Cross-host transmission patterns

## Research Questions

1. **Can protein LLM embeddings group genes into coherent families that go beyond raw sequence identity?**
   - Similar to what pangenomes aim to achieve, but alignment-free
   - Using ESM-2's evolutionary and structural signals encoded from 65M UniRef sequences

2. **Can these embedding-based clusters help identify host-associated genes or gene sets?**
   - Do clusters reveal host-specific vs. cross-host patterns in MGEs?
   - Can we build predictive models for host association using cluster information?

3. **How well do different clustering algorithms perform on protein embeddings?**
   - Leiden community detection on k-NN graphs
   - DBSCAN density-based clustering
   - Planned: HDBSCAN, spherical k-means

## Pipeline Overview

### Phase 1: Data Integration (`merge`)
Merges geNomad annotation outputs including:
- Gene-level annotations
- Plasmid summary data
- Virus summary data
- AMR (antimicrobial resistance) annotations
- Conjugation system annotations

Filters for MGE-associated genes based on hallmark markers.

### Phase 2: Embedding Generation & Clustering (`generate_embeddings`)
1. **Extract MGE protein sequences** from geNomad outputs
2. **Generate embeddings** using ESM-2 650M parameter model (1280-dimensional vectors)
3. **Cluster proteins** using:
   - **Leiden algorithm**: Graph-based community detection via k-NN graph (FAISS)
   - **DBSCAN**: Density-based clustering
4. **Calculate distances**:
   - Hausdorff distances between host groups
   - Pairwise genome distances
5. **Analyze cluster composition**:
   - Host purity and diversity (Simpson's index)
   - Genome diversity within clusters
   - Cross-host vs. host-specific clusters
   - Presence/absence matrices (clusters × genomes/hosts)

## Key Features

- **Protein language model embeddings**: Leverages ESM-2 (Evolutionary Scale Modeling) for semantic protein representations
- **Multi-host analysis**: Tracks MGE distribution across human, chicken, turkey, and pork bacterial samples
- **Dual clustering approach**: Compares Leiden (graph-based) and DBSCAN (density-based) methods
- **Comprehensive statistics**:
  - Cluster composition metrics
  - Host association patterns
  - Presence/absence matrices for downstream analysis
- **GPU acceleration**: Optimized for CUDA with batch processing
- **Memory-efficient**: Uses HDF5 and memory-mapped arrays for large-scale data

## Installation

### Requirements
- Python 3.8+
- CUDA-capable GPU (recommended)
- Dependencies:
  ```bash
  pip install torch esm-py biopython pandas numpy scipy scikit-learn
  pip install igraph leidenalg h5py matplotlib
  pip install faiss-gpu  # or faiss-cpu for CPU-only
  ```

### Clone Repository
```bash
git clone <repository-url>
cd TAI_incubator
```

## Usage

### Phase 1: Merge geNomad Outputs
```bash
python mge_llm_cluster_v1.py merge \
  --host-labels data/samplelist.txt \
  --genomad-dir data/genomad_outputs/ \
  --output-prefix results/analysis \
  --mge-gene-metadata mge_functional_modules.tsv
```

**Input**:
- `--host-labels`: TSV with columns `sample_id` and `host_label` (human/chicken/turkey/pork)
- `--genomad-dir`: Directory containing geNomad output folders (one per genome)

**Output**:
- `results/analysis_merge/`: Merged Excel files per genome
- `results/analysis_merge/mge_functional_modules.tsv`: Unified MGE gene metadata

### Phase 2: Generate Embeddings & Cluster
```bash
python mge_llm_cluster_v1.py generate_embeddings \
  --host-labels data/samplelist.txt \
  --genomad-dir data/genomad_outputs/ \
  --output-prefix results/analysis \
  --mge-gene-metadata mge_functional_modules.tsv \
  --esm-model esmc_600m \
  --device cuda \
  --batch-size 16
```

**Output**:
- `results/analysis_embedding/*.h5`: Per-genome embeddings
- `results/analysis_embedding/mge_gene_embeddings.memmap`: Consolidated embeddings
- `results/analysis_embedding/mge_gene_embeddings_metadata.tsv`: Gene metadata with cluster assignments
- `results/analysis_embedding/final_clusters.tsv`: Cluster assignments (Leiden + DBSCAN)
- Hausdorff distance matrices (host-level and genome-level)
- Cluster statistics and visualizations
- **Presence/absence matrices**: Binary matrices showing cluster distribution

### SLURM Execution (HPC)
```bash
sbatch run_deepmgehost_v1.slurm
```

## Output Files

| File | Description |
|------|-------------|
| `final_clusters.tsv` | Gene-level cluster assignments (both algorithms) |
| `leiden_cluster_statistics.tsv` | Detailed statistics per Leiden cluster |
| `dbscan_cluster_statistics.tsv` | Detailed statistics per DBSCAN cluster |
| `leiden_cluster_summary.tsv` | Summary metrics for Leiden clustering |
| `dbscan_cluster_summary.tsv` | Summary metrics for DBSCAN clustering |
| `leiden_cluster_genome_presence_matrix.tsv` | Binary matrix: clusters (rows) × genomes (cols) |
| `leiden_cluster_host_presence_matrix.tsv` | Binary matrix: clusters (rows) × hosts (cols) |
| `hausdorff_distances.csv` | Pairwise host distances |
| `hausdorff_distances_per_genome.csv` | Pairwise genome distances |
| `*_visualization.png` | PCA projections of clusters and host distributions |

## Key Findings

### Current Results

1. **Embeddings Produce Cohesive Clusters** ✅
   - ESM-2 embeddings successfully group functionally related proteins
   - Validated using all-vs-all pairwise sequence identities
   - Mean/median amino acid identity metrics confirm cluster coherence
   - Alignment-free grouping works as a practical pangenome alternative

2. **Challenge: Clusters Are Too Coarse** ⚠️
   - First-pass clusters reflect broad structural/evolutionary signals
   - Fine functional distinctions can be blurred
   - Requires refinement before host-association modeling

3. **Host Association Patterns**:
   - Identifies both host-specific and cross-host MGE protein clusters
   - Quantifies host purity using Simpson's diversity index
   - Presence/absence matrices reveal potential horizontal gene transfer events

4. **Algorithm Performance**:
   - Leiden clustering reveals hierarchical community structure
   - DBSCAN identifies dense core clusters with noise detection
   - Need for resolution tuning and algorithm comparison

### Dataset

**Primary Analysis**: *E. coli* ST131-H22 genomes with host labels from:
> Liu CM, Stegger M, Aziz M et al. *Escherichia coli* ST131-H22 as a Foodborne Uropathogen. *mBio*. 2018. PMID: 30154256

## Project Structure

```
TAI_incubator/
├── mge_llm_cluster_v1.py          # Main pipeline (enhanced version)
├── code_versions/                  # Version control for scripts
│   └── mge_llm_cluster.py         # Original version (backup)
├── ANI_clusters/                   # ANI analysis scripts
│   ├── calc_cluster_ANI.py
│   └── calc_cluster_nucl_aa_avgid.py
├── run_deepmgehost*.slurm         # SLURM job scripts
├── data/                           # Input data (not tracked)
├── GW TAI basic poster -mlaziz-final/  # Research presentation
└── README.md                       # This file
```

## Future Work

### Immediate Next Steps: Clustering Refinement

1. **Embedding Preprocessing**:
   - Normalize and lightly denoise embeddings
   - Apply PCA to reduce dimensionality while preserving signal
   - Optimize for cosine geometry

2. **Algorithm Tuning & Comparison**:
   - Tune k-NN graph parameters and Leiden resolution
   - Re-cluster large, heterogeneous groups
   - Compare HDBSCAN and spherical k-means performance
   - Add light priors (protein length, domain cues) without overwhelming embeddings

3. **Validation**:
   - Refine all-vs-all identity validation
   - Assess cluster biological coherence with functional annotations
   - Measure improvement in functional distinction

### Genome-Level Host Association Modeling

4. **Early Fusion Architecture**:
   - Concatenate ESM-2 gene embeddings with cluster-ID embeddings
   - Pool embeddings at genome level
   - Train elastic-net and tree-based models

5. **Validation Strategy**:
   - Leave-lineage-out cross-validation
   - Produce ranked, interpretable list of candidate host-associated clusters
   - Enable follow-up biological validation

### Long-Term Goals

- Expand to additional protein language models (ESM-C, ProtTrans)
- Integrate phylogenetic analysis with embedding-based clustering
- Develop predictive models for cross-host transmission risk
- Create practical front-end tools for host-association analysis
- Validate with experimental functional annotation data

## Methods

### Embedding Generation
- **Model**: ESM-2 650M (33 layers, 1280-dimensional embeddings)
- **Pooling**: Mean pooling over sequence length (excluding start/end tokens)
- **Batch processing**: Configurable batch size with GPU memory management

### Clustering Algorithms

**Current Implementation**:
- **Leiden**: Resolution parameter = 0.5, RBConfigurationVertexPartition on k-NN graph
- **DBSCAN**: eps=0.5, min_samples=5, cosine distance on standardized embeddings
- **k-NN graph**: FAISS IndexFlatIP with k=10 neighbors for efficient nearest-neighbor search

**Validation**:
- All-vs-all pairwise amino acid identity within clusters
- Mean/median identity metrics for cluster coherence
- Functional annotation overlay for biological validation

**Planned Enhancements**:
- HDBSCAN for hierarchical density-based clustering
- Spherical k-means for normalized embedding space
- Resolution parameter tuning for Leiden
- Light domain-based priors to sharpen cluster boundaries

### Distance Metrics
- **Hausdorff distance**: Maximum of directed Hausdorff distances (host and genome comparisons)
- **Cosine similarity**: For k-NN graph construction

### Statistical Analysis
- **Simpson's diversity index**: Measures host/genome heterogeneity within clusters
- **Host purity**: Proportion of dominant host in each cluster
- **Presence/absence matrices**: Binary representation for comparative genomics

## Citation

If you use this pipeline in your research, please cite:

```
Aziz, M., & Price, L. (2024). Evaluating Fidelity of Protein LLM Embeddings
for Host-Association Analysis. The George Washington University.
```

## Main Takeaways

1. **ESM-2 embeddings accelerate pangenome-like grouping**: Alignment-free clustering achieves cohesive protein families comparable to traditional alignment-based methods

2. **Refinement needed for host-association modeling**: Current clusters capture broad evolutionary/structural signals but require tuning for finer functional distinctions

3. **Practical path forward**: Modest clustering refinements + genome-level modeling with early fusion can create an interpretable, efficient front-end for host-association analysis

4. **Reduced computational burden**: Eliminates need for expensive all-vs-all sequence alignments while maintaining biological interpretability

## Contact

For questions or collaborations:
- **Maliha Aziz** - The George Washington University
- **Dr. Lance Price** - Antibiotic Resistance Action Center

## Acknowledgments

This work was conducted at The George Washington University's Trustworthy AI (TAI) program in collaboration with the Antibiotic Resistance Action Center.

## License

[Add appropriate license]
