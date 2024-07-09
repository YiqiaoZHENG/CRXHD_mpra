# Aberrant homeodomain-DNA cooperative dimerization underlies distinct developmental defects in two dominant _CRX_ retinopathy models
This repository contains all codes used for MPRA library design and data analysis in [Zheng et al.](https://pubmed.ncbi.nlm.nih.gov/38559186/)

## Prerequisites and Data availability
Necessary software packages with versions used to process data are described in text file software_versions.txt
- MPRA oligo library design was performed on [WSL:Ubuntu-20.04](https://docs.microsoft.com/en-us/windows/wsl/).
- MPRA data analysis was performed on the WashU High Throughput Computing Facility ([HTCF](https://htcf.wustl.edu/docs/)) using [SLURM](https://slurm.schedmd.com/documentation.html).
- Further processing of intermediate data and visualization of processed data was performed on the WashU High Throughput Computing Facility.
- Raw data and additional processed data for this manuscript can be downloaded from GEO under SuperSeries [GEO](link_to_mpra_GEO).

## Repository organization
- `hdmuts_library_design` and `hdmuts_retina_analysis` contain scripts, metadata text files, and any intermediate data generated for MPRA library design and data analysis, respectively Detailed descriptions of the usage of these scripts can be found in the README.md under each sub-directory.