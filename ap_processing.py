from pybedtools import BedTool
from genome_processing import bed_to_saf
import os
import glob
import subprocess

"""
Runs feature counts on all bam files in the active_poised folder
Returns names of the files created after running featureCounts
"""
def process_ap(enhancers_saf, sense_bam_file, antisense_bam_file, output_folder):
    ap_out_folder = './' + output_folder + '/active_poised_data/'
    if not os.path.exists(ap_out_folder):
        os.mkdir(ap_out_folder)
    
    files = [sense_bam_file, antisense_bam_file]
    file_type = ['sense', 'antisense']
 
    for counter, rep in enumerate(files):
        out_name = ap_out_folder + file_type[counter] + "_bam-bincounts.txt"
        subprocess.call(["featureCounts", rep, "-a", enhancers_saf, "-O", "-F", "SAF", "-o", out_name]) 
