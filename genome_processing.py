import numpy as np
import os
import sys
from pybedtools import BedTool

"""
Converts the input bed file to a saf file
Saves the saf file as name passed in as saf_file
"""
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

"""
Windows passsed in genome by window_width, saves it as genome_save_name
Filters this windowed genome so it only contains chromosomes in valids file
"""
#Function to window genome by set width and save it by save name
def window_sort_genome(genome, window_width, genome_save_name, filtered_save_name, valids_file):
    genome_windowed = BedTool().window_maker(genome=genome, w=window_width).saveas()
    genome_windowed.saveas(genome_save_name)
    cmnd = 'grep -w -f '+ valids_file + ' '+ genome_save_name + '>' + filtered_save_name
    os.system(cmnd)

"""
Creates windowed.filtered.bed which is the genome windowed by window_width and filtered to only
contain set valid chromosomes
Also converts this windowed.filtered.bed into saf format
Creates bgwindowed.filtered.bed which is genome windowed by bg_window_width and filtered to only
contain set valid chromosomes
"""
def process_genome(genome, window_width, number_of_windows, valids, output_folder):
    gene_out_folder = './' + output_folder + '/genome_data/'
    if not os.path.exists(gene_out_folder):
        os.mkdir(gene_out_folder)
        
    bg_window_width = window_width * number_of_windows

    #Creation of text file of valid chromosomes to find in larger genome file
    valids = np.array(valids)
    valids_file = gene_out_folder + "valids.txt"
    with open(valids_file, "w") as f:
        np.savetxt(f, valids, fmt='%s')
    
    window_sort_genome(genome, window_width, gene_out_folder + 'windowed.bed', gene_out_folder + 'windowed.filtered.bed', valids_file)
    window_sort_genome(genome, bg_window_width, gene_out_folder + 'bgwindowed.bed', gene_out_folder + 'bgwindowed.filtered.bed', valids_file)
    bed_to_saf(gene_out_folder + 'windowed.filtered.bed', gene_out_folder + 'windowed.filtered.saf')

    

    

    