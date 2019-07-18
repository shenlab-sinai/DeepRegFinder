from pybedtools import BedTool
from genome_processing import bed_to_saf
import os
import glob
import subprocess

"""
Runs feature counts on all bam files in the active_poised folder
Returns names of the files created after running featureCounts
"""
def process_ap(enhancers_saf, ap_path, output_folder):
    ap_out_folder = './' + output_folder + '/active_poised_data/'
    if not os.path.exists(ap_out_folder):
        os.mkdir(ap_out_folder)
    
    out_named = []
    counter = 1
    for rep in glob.glob(ap_path + "/*.bam"):
        out_name = ap_out_folder + "srr_r" + str(counter) + "bam-bincounts.txt"
        out_named.append(out_name)
        subprocess.call(["featureCounts", rep, "-a", enhancers_saf, "-O", "-F", "SAF", "-o", out_name]) 
        counter += 1
    return out_named