#!/bin/bash

#SBATCH --job-name=yq_DemuxBCcount
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/sclab/030523_MPRA/slurm_outputs/DemuxBCcount_%A_%a.out
#SBATCH --array=4

echo "Node ID allocates ${SLURM_JOB_NODELIST}"
echo "Number of cpus requested per task ${SLURM_CPUS_PER_TASK}"

basedir="/scratch/sclab/030523_MPRA"
scriptdir="${basedir}/scripts/utils"
demux_count_script="${scriptdir}/demultiplexAndBarcodeCount.sh"
fastqToSample="${basedir}/annotations/fastq_meta.tsv"
p1_to_sample="${basedir}/annotations/030523_p1Index_lookup.tsv"
spacerSeq="CACCTGTTCCTGTAGGCATGC"
bcSize=10

# run the demultiplex and bc count script in conda mpra env
sh ${demux_count_script} ${fastqToSample} ${spacerSeq} ${bcSize} ${scriptdir} ${SLURM_ARRAY_TASK_ID}

echo "ha! tihs is the end of the script!"