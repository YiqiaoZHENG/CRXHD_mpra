# Data analysis for MPRA experiments in CRX homeodomain variants
This directory contains all scripts and metadata information used to analyze the MPRA data obtained from P8 retinal ex plants. The scripts should be run in the numbered order. All the intermediate and processed data generated are stored under subfolders in the main repository. The raw data can be downloaded from [GEO](link_to_mpra_GEO).

## Brief description of scripts and metadata files
- `demux_and_BCcount.sbatch` demultiplexes the raw fastq files, generate barcode count tables, and barcode abundance histogram.
  - `annotations/fastq_meta.tsv` and `annotations/030523_p1Index_lookup.tsv` record the internal index sequences used to parse the raw fastq files.
- `replicateCorrelation_QC.ipynb` performs basic sample-wise QC metrics and filtering on raw barcode counts.
- `readDepthNormalization_DESeq2.ipynb` takes a raw count table of rna samples and perform median ratio normalization through R package DEseq2.
- `calculate_CREactivity.ipynb` calculates the enhancer/silencer activity of each CRE.
- `calculate_motifActivity.ipynb` calculates the HD motif activity of each CRE.


## Reference
Prior studies that used similar library designs as in my study here.
- A massively parallel reporter assay reveals context-dependent activity of homeodomain binding sites in vivo [Hughes et al., 2018](https://genome.cshlp.org/content/28/10/1520.full).
- Information content differentiates enhancers from silencers in mouse photoreceptors [Friedman et al., 2021](https://elifesciences.org/articles/67403).