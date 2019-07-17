from pybedtools import BedTool
from genome_processing import bed_to_saf
import os
import glob
import subprocess

#Path to enhancer bed file
# filt_genome_path = 'strict_enhancers.bed'

# #Path to the folder with rep data
# ap_path = '/home/kims/work/enhancer-prediction-data/ap_data/'

def process_ap(enhancers, ap_path):
    saf_name = enhancers.split('.')[1] + '.saf'
    bed_to_saf(enhancers, saf_name)
   
    counter = 1
    os.chdir(ap_path)

    for rep in glob.glob(ap_path + "/*.bam"):
        print(os.getcwd())
        rep_name =  os.path.basename(rep)
        out_name = "srr_r" + str(counter) + "bam-bincounts.txt"
        saf_path = "../" + saf_name

        subprocess.call(["featureCounts", rep_name, "-a", saf_path, "-O", "-F", "SAF", "-o", out_name]) 
        counter += 1