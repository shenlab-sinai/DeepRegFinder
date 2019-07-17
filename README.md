# training_pipeline
To Run:
Fill out input file: training_data.yaml
In command line type:
>python main.py training_data.yaml

Pipeline

Making windowed + formatted genome

Making all-together-not-normed-DONE
featureCounts
awk to get average of columns for reps + formatting into bed
•	Input
o	Path to windowed.filtered.saf genome
o	Path to histone folder, has BAM + bai files
•	Pipeline: histone_pipeline
1.	For each histone mark, each in own directory download multiple replicates of targeted ChIP-seq assay for specific histone mark from encode
a.	Download as bam files
b.	MAKE ABLE TO HANDLE MULTIPLE REPLICATES
2.	Use Bedtools to partition your genome into 100 bp windows
3.	Filter the genome and throw away nonstandard chromosomes 
4.	
5.	Program:featureCounts v1.6.2; Command:"featureCounts" "ENCFF596SHE-h3k27me3-rep1.bam" "-a" "../genome_100bpbins.filtered.saf" "-O" "-F" "SAF" "-o" "r1bam-bincounts.txt" 
•	Final file: alltogether_notnormed.txt

Make TSS_data-DONE
•	Pipeline
o	tss_pipeline
o	format_tss.sh
1.	Download human_tss.tsv 
2.	human_coding_clustered_TSS.tsv
	Remove_tss_redundancy.ipynb-to cluster
•	Final file: true_tss.bed

Need dnase_data, just downloaded from Ensembl BioMart

Make Dnase data (DHS)-DONE 
ALL PYTHON
1.	Download DHS data from Ensembl Biomart
•	Final file: h1-dnase-seq-peaks.bed


Make p300 data
•	Final File: 'h1-p300-narrowpeaks.bed'

Make enhancer data-DONE
ALL PYTHON
	-get_enhancers.py to get data
1.	Sort the enhancers
2.	ADD MORE
•	Final file: 
o	potential-enhancers.bed
o	potential-enhancer-centers.bed

Make TFBS data-DONE
ALL PYTHON
•	Question: Should the p300 data used for TFBS, TPMs, and background data be intersected with DHS and distal to TSS to make potential enhancers?
o	Or just use p300 plain

•	Input
o	genome name
o	DHS
o	p300
o	tss
•	QUESTION: 
o	Should the features be merged?
♣	Using cat merges them currently
1.	Slop TSS by 1kb on each side
o	QUESTION: Should TSS be slopped for everything?
♣	LARGER QUESTION
2.	Subtract true_slopped_tss from TFBS
3.	Adding all the TFBS files together
4.	Sort the files
•	Final file: final_tfbs.bed

Make TPMs data-DONE
ALL PYTHON
•	Input: 
o	genome name
o	DHS
o	p300
o	tss
o	TFBS
o	Additional parameter for window size 

•	In same file as TFBS creation
1.	Slop TSS by 1kb on each side
2.	Subtract true_slopped_tss from TPMS
a.	TPMS-DHS + p300 + TFBS
3.	Adding all the TPMS files together
4.	Sort the files
•	Final file:  final_tpms.bed

Make background data-Needs more work
ALL PYTHON
•	Input: 
o	Genome name
o	DHS
o	p300
o	all tss from tss creation pipeline
♣	Not just ones intersected with DHS
o	TFBS
•	Pipeline: background_pipeline
1.	Window the genome by 2kb
2.	Subtract inputs from genome windowed using bedtools
•	Final file: final_bg.bed

Make sequence data-DONE
ALL PYTHON
•	Input:
o	Genome fasta file
o	Genome windowed and filtered by 100 bp and filtered (from histone + genome windowing creation)
•	Pipeline: sequence_pipeline
1.	Getfasta turns the fasta file into 100 bp features and filters out chromosomes using the provided bed file
•	Final file: filtered_binned.fa

Making GRO-seq data
featureCounts
awk to get average of columns for reps + formatting into bed
•	Input
o	Path to enhancer bed file
o	Path to GRO-seq folder, has BAM + bai files
•	Pipeline: histone_pipeline
6.	For each histone mark, each in own directory download multiple replicates of targeted ChIP-seq assay for specific histone mark from encode
a.	Download as bam files
b.	MAKE ABLE TO HANDLE MULTIPLE REPLICATES
7.	Use Bedtools to partition your genome into 100 bp windows
8.	Filter the genome and throw away nonstandard chromosomes 
9.	
10.	Program:featureCounts v1.6.2; Command:"featureCounts" "ENCFF596SHE-h3k27me3-rep1.bam" "-a" "../genome_100bpbins.filtered.saf" "-O" "-F" "SAF" "-o" "r1bam-bincounts.txt" 
•	Final file: alltogether_notnormed.txt
•	In /ap_data


