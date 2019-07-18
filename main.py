import csv
import numpy as np
import random
import sys
from genome_processing import process_genome
from tss_processing import process_tss
from enhancer_processing import process_enhancers
from tfbs_tpms_processing import process_tfbs, process_tpms
from background_processing import process_background
from ap_processing import process_ap
from histone_processing import process_histones
from compress_index_files import compress_index_files, remove_header_histone_file
from train_histone import make_histone_tensors, clustering
from train_sequence import make_sequence_tensors
from generate_unified_indices import make_training_testing_tensors
import yaml
import sys
import os

"""
First argument: input .yaml file 
Second argument: name for output folder where files stored
"""
def main():
    """
    Taking in input data from yaml file
    """
    params = sys.argv[1]
    with open(params) as f:
        # use safe_load instead load
        dataMap = yaml.safe_load(f)
        
    #Data for Genome Processing (Making 100 kb windowed bed + saf + bg windowed)
    genome = dataMap['genome']
    window_width = dataMap['window_resolution']
    bg_window_width = dataMap['number_of_windows']
    valids = dataMap['valid_chromosomes']

    #Peak data for DHS, tss, p300
    DHS_file, tss_file, p300_file = dataMap['DHS_file'], dataMap['tss_file'], dataMap['p300_file']
 
    #Data for enhancer file creation 
    distal_num = dataMap['distal_bp_distance']
    
    #Data for TFBS and TPMs file creation
    TFBS, TPMs = dataMap['TFBS'], dataMap['TPMs']
    
    #Data for active poised clustering
    sense_bam_file, antisense_bam_file= dataMap['sense_bam_file'], dataMap['antisense_bam_file']
    
    #Data for histone mark file creation
    histone_path = dataMap['histone_folder']
    
    
    
    
    """
    Calling functions to process data
    """
    # Creating a tmp directory based on input
    output_folder = sys.argv[2]
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
        
    #process_genome(genome, window_width, bg_window_width, valids, output_folder)
    print('Finished processing genome')
    bg_genome = './'+  output_folder + '/genome_data/bgwindowed.filtered.bed'
    genome_saf_format = './'+  output_folder + '/genome_data/windowed.filtered.saf'
                                 
    #process_tss(tss_file, DHS_file, genome, distal_num, output_folder)
    print('Finished processing tss')
    tss_no_dhs = './'+  output_folder + '/tss_data/tss_no_dhs_intersect.bed'
    slopped_tss_no_dhs = './'+  output_folder + '/tss_data/tss_no_dhs_intersect_slopped.bed'
    slopped_tss = './'+  output_folder + '/tss_data/true_slopped_tss.bed'
    true_tss = './'+  output_folder + '/tss_data/true_tss.bed'
    
    #process_enhancers(p300_file, DHS_file, slopped_tss_no_dhs, output_folder)
    print('Finished processing enhancers')
    enhancers = './'+  output_folder + '/enhancer_data/strict_enhancers.bed'
    enhancers_saf = './'+  output_folder + '/enhancer_data/strict_enhancers.saf'
    
    #process_tfbs(slopped_tss, TFBS, output_folder)
    print('Finished processing TFBS')
    final_tfbs_file = './'+  output_folder + '/tfbs_data/final_tfbs.bed'
    
    #process_tpms(slopped_tss, TPMs, final_tfbs_file, output_folder)
    print('Finished processing TPMs')

    #process_background(bg_genome, slopped_tss_no_dhs, DHS_file, enhancers, final_tfbs_file, output_folder)
    print('Finished processing background')
    final_background = './'+  output_folder + '/background_data/final_bg.bed'
    
    #process_ap(enhancers_saf, sense_bam_file, antisense_bam_file, output_folder)
    sense_file = './'+  output_folder + '/active_poised_data/sense_bam-bincounts.txt'
    antisense_file = './'+  output_folder + '/active_poised_data/antisense_bam-bincounts.txt'
    print('Finished processing active/poised')
    
    #process_histones(genome_saf_format, histone_path, output_folder)
    print('Finished processing histones')
    all_histone_data = './'+  output_folder + '/histone_data/alltogether_notnormed.txt '
    
    #Compressing + indexing files for tensor creation
#     files_to_compress = [true_tss, final_background, enhancers]
#     file_folders = ['./'+output_folder+'/tss_data/', './'+output_folder+'/background_data/', './'+output_folder+'/enhancer_data/', './'+output_folder+'/histone_data/']
#     files_to_compress.append(remove_header_histone_file(all_histone_data))
#     compress_index_files(files_to_compress)
    print('Finished compressing files')
    


    groseq_positive, groseq_negative = clustering(sense_file, sense_bam_file, antisense_file, antisense_bam_file)
    #make_histone_tensors(groseq_positive, groseq_negative, 3)
    print('Finished making histone tensors')
    #make_sequence_tensors(groseq_positive, groseq_negative)
    print('Finished making sequence tensors')
    #make_training_testing_tensors('sevenmark_train_dataset.pt', 'sequence_train_dataset.pt', 50)
    print('Finished making tensors for training')
    

    
    
    
    
    return
if __name__ == '__main__':
    main()

