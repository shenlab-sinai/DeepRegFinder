from pybedtools import BedTool
from genome_processing import bed_to_saf
import os
import glob
import subprocess

"""
Runs feature counts on all bam files in the active_poised folder
Returns names of the files created after running featureCounts
"""
def process_ap(saf_file, sense_bam_file, antisense_bam_file, output_folder, groseq_logtrans):
    #extract name + change it 
    ap_out_folder = './' + output_folder + '/active_poised_data/'
    if not os.path.exists(ap_out_folder):
        os.mkdir(ap_out_folder)
    
    #histone = histone_folder.split('/')[split_num]
    file_name = saf_file.split('/')[-1].split('.')[0]
    files = [sense_bam_file, antisense_bam_file]
    file_type = [file_name + '_sense', file_name + '_antisense']
    
    out_files = []
    for counter, rep in enumerate(files):
        out_name = ap_out_folder + file_type[counter] + "_bam-bincounts.txt"
        subprocess.call(["featureCounts", rep, "-a", saf_file, "-O", "-F", "SAF", "--fracOverlap", "0.5", "-f", "-o",  out_name]) 
        out_files.append(out_name)
    
    if groseq_logtrans:
        for file in out_files:
            out_name = file.split('.txt')[0] + '_logtrans.txt'
            subprocess.call(['./log_transform.sh', file, out_name])
  
            
        
