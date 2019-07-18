from pybedtools import BedTool
import os
from genome_processing import bed_to_saf

"""
Selects sites that are both p300 co-activator binding sites and DHS by finding the sites that overlap 
in the 2 BED files representing p300 peaks/binding sites and DNase-seq peaks/DHS. 
(i.e. the sites on the genome that are both DHS and p300 binding sites). Next uses a BED file of human
TSS to isolate those DHS p300 binding sites that are distal to the TSS. 

Also creates a saf file from this bed file of enhancers
"""
def process_enhancers(p300, DHS, tss, output_folder):
    enhancer_out_folder = './' + output_folder + '/enhancer_data/'
    if not os.path.exists(enhancer_out_folder):
        os.mkdir(enhancer_out_folder)
    
    #Converting to Bed format
    p300 = BedTool(p300)
    dhs = BedTool(DHS)
    tss = BedTool(tss)
    
    overlapping_sites = p300.intersect(dhs)
    distal_sites = overlapping_sites.subtract(tss, A=True)

    sorted_sites = distal_sites.sort()
    sorted_sites.saveas(enhancer_out_folder + 'strict_enhancers.bed')
    bed_to_saf(enhancer_out_folder + 'strict_enhancers.bed', enhancer_out_folder + 'strict_enhancers.saf')