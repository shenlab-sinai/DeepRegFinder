import numpy as np
import subprocess
import glob
import os
import sys
from pybedtools import BedTool
import pysam
#from Bio import SeqIO
import pandas as pd
from re import sub
from sklearn.cluster import KMeans
import shlex
from torch.utils.data import Dataset, DataLoader


#Function to convert input bed file to saf file
def bed_to_saf(bed_file, saf_file):
    with open(bed_file, 'r') as bed:
        with open(saf_file, 'w+') as saf:
            saf.write("GeneID\tChr\tStart\tEnd\tStrand\n" )
            strand = "+"
            for i, line in enumerate(bed):
                words = line.split(sep = '\t')
                chrom = words[0]
                start = str(int(words[1]))
                end = str(int(words[2]))
                gene_id = '.'.join((chrom, start, end))
                saf.write(gene_id + "\t" + chrom + "\t" + start + "\t" + end + "\t" + strand + "\n")

#Function to window genome by set width and save it by save name
def window_sort_genome(genome, window_width, genome_save_name, filtered_save_name):
    genome_windowed = BedTool().window_maker(genome=genome, w=window_width).saveas()
    genome_windowed.saveas(genome_save_name)
    cmnd = 'grep -w -f valids.txt ' + genome_save_name + '>' + filtered_save_name
    os.system(cmnd)
    
def process_genome(genome, window_width, bg_window_width, valids):
    #Creation of text file of valid chromosomes to find in larger genome file
    valids = np.array(valids)
    with open("valids.txt", "w") as f:
        np.savetxt(f, valids, fmt='%s')
    
    window_sort_genome(genome, window_width, 'windowed.bed', 'windowed.filtered.bed')
    window_sort_genome(genome, bg_window_width, '2kbwindowed.bed', '2kbwindowed.filtered.bed')
    bed_to_saf('windowed.filtered.bed', 'windowed.filtered.saf')

    

    

    