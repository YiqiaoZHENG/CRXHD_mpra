#!/bin/bash

# Scan a FASTA file with a list of MEME motifs using FIMO 
if [ "$#" -ne 6 ]
then
    echo "Usage: meme_fimo_scanning.sh path/to/this/script path/to/mpra/output/directory path/to/meme/motif/file fimo_threshold path/to/fimo_meta.txt sampleID"
    echo "  - fimo_meta.txt is formatted as [sample name],[input fasta full path],[markov background output prefix],[output directory]."
    echo "  - Example: mpraAllCRE,allCRE/mpraAllCRE.fa,allCRE/mpraAllCRE_background,allCRE_fimo"
    exit 1
fi

# parse bash command line arguments
basedir=$1
mpraout_dir=$2 #"/mnt/v/yqzheng/qiaoer/PhD Thesis/Experiment/MPRA/hdmuts_library"
motif_file=$3 #"${mpraout_dir}/all_chip_pwm.meme"
fimo_th=$4
#motif_file="${mpraout_dir}/photoreceptorAndEnrichedMotifs.meme"
sample_file=$5 #($(<${basedir}/mpra_fimo_meta.txt))
lineNum=$6

# make a new directory based on fimo threshold
fimo_dir="${mpraout_dir}/fimo_${fimo_th}/"
mkdir -p "${fimo_dir}"

# parse the sample metadata file into array
IFS=$'\n'
sample_list=($(<${sample_file}))
IFS=',' read -a array <<< "${sample_list[${lineNum}]}"
sample_name=${fimo_dir}${array[0]}
query_fa=${fimo_dir}${array[1]}
markov_bg=${fimo_dir}${array[2]}.txt
new_dir=${fimo_dir}${array[3]}

echo "working directory: ${basedir}"
echo "query fasta: ${query_fa}"

# generate first order Markov model from a FASTA file of sequences
fasta-get-markov -m 1 -dna ${query_fa} ${markov_bg}

# rename the background file
#mv ${markov_bg} "${markov_bg}.txt"
#markov_bg="${markov_bg}.txt"

echo "Scanning with threshold ${fimo_th}"
echo "FIMO output will be written to ${new_dir}"
# run fimo and use 0-order Markov Background model generated from MEME default threshold 1.0E-3
fimo --bfile ${markov_bg} --oc ${new_dir} --verbosity 1 --thresh ${fimo_th} ${motif_file} ${query_fa}

echo "ha! this is the end of the script!"