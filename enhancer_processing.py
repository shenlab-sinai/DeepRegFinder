from pybedtools import BedTool
import os
from genome_processing import bed_to_saf

"""
Selects sites that are p300 co-activator binding sites 
Next uses a BED file of human TSS to isolate those p300 binding sites that are distal to the TSS. 

Also creates a saf file from this bed file of enhancers
"""
def process_enhancers(p300, tss, gene_bodies, H3K4me3, enhancer_distal_num, genome, output_folder):
    enhancer_out_folder = './' + output_folder + '/enhancer_data/'
    if not os.path.exists(enhancer_out_folder):
        os.mkdir(enhancer_out_folder)
    
    #Converting to Bed format
    p300 = BedTool(p300)
    tss = BedTool(tss)
    gene_bodies = BedTool(gene_bodies)
    H3K4me3_peaks = BedTool(H3K4me3)
    
    H3K4me3_slopped = H3K4me3_peaks.slop(b=enhancer_distal_num, genome=genome)
    
    
    intergenic = p300.subtract(gene_bodies, A = True)
    tss_distal_sites = intergenic.subtract(tss, A=True)
    tss_histone_distal_sites = tss_distal_sites.subtract(H3K4me3_slopped, A=True)

    sorted_sites = tss_histone_distal_sites.sort()
    sorted_sites.saveas(enhancer_out_folder + 'strict_enhancers.bed')
    valids_file = './' + output_folder + '/genome_data/valids.txt'
    
    
    cmnd = 'grep -w -f '+ valids_file + ' '+ enhancer_out_folder + 'strict_enhancers.bed' + '>' + enhancer_out_folder + 'strict_enhancers_filtered.bed'
    os.system(cmnd)
    
    bed_to_saf(enhancer_out_folder + 'strict_enhancers_filtered.bed', enhancer_out_folder + 'strict_enhancers_filtered.saf')
    
    
    
    
    
    
    