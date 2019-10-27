import csv
import numpy as np
import random
import sys
from preprocessing_functions import *
# from genome_processing import process_genome
# from tss_processing import process_tss
# from enhancer_processing import process_enhancers
# from tfbs_tpms_processing import process_tfbs, process_tpms
# from background_processing import process_background
# from ap_processing import process_ap
# from histone_processing import process_histones
# from compress_index_files import compress_index_files, remove_header_histone_file
# from train_histone import make_histone_tensors, clustering, tss_clustering
# from train_sequence import make_sequence_tensors
# from generate_unified_indices import make_training_testing_dataloaders
import yaml
import sys
import os
import time

"""
First argument: input .yaml file 
Second argument: name for output folder where files stored

Outputs used for training pipeline can be found in tensor_data folder
"""
def main():
    """
    Taking in input data from yaml file
    """
    start = time.time()
    params = sys.argv[1]
    with open(params) as f:
        # use safe_load instead load
        dataMap = yaml.safe_load(f)
        
    #Data for Genome Processing (Making 100 kb windowed bed + saf + bg windowed)
    genome, genome_fasta_file = dataMap['genome'], dataMap['genome_fasta_file']
    window_width = dataMap['window_width']
    number_of_windows = dataMap['number_of_windows']
    valids = dataMap['valid_chromosomes']

    #Peak data for DHS, tss, p300
    DHS_file, tss_file, p300_file = dataMap['DHS_file'], dataMap['tss_file'], dataMap['p300_file']
    
    #Data for enhancer file creation 
    enhancer_distal_num = dataMap['enhancer_distal_bp_distance']
    gene_bodies = dataMap['gene_bodies']
    H3K4me3_file = dataMap['H3K4me3_file']
    
    #Data for TFBS file creation
    distal_num = dataMap['distal_bp_distance']
    TFBS = dataMap['TFBS']
    
    #Data for active poised clustering
    sense_bam_file, antisense_bam_file= dataMap['sense_bam_file'], dataMap['antisense_bam_file']
    groseq_logtrans = dataMap['groseq_log_transformation']
    
    #Data for histone mark file creation
    histone_path, nb_mark = dataMap['histone_folder'], dataMap['nb_mark']
    hist_logtrans = dataMap['histone_log_transformation']
    
    #Window width of sequence
    sequence_window_width = dataMap['sequence_window_width']
                                                              
    
    
    """
    Calling functions to process data
    """
    # Creating a tmp directory based on input
    output_folder = sys.argv[2]
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
        
    process_genome(genome, window_width, number_of_windows, valids, output_folder)
    print('Finished processing genome')
    bg_genome = output_folder + '/genome_data/bgwindowed.filtered.bed'
    genome_saf_format = output_folder + '/genome_data/windowed.filtered.saf'
    
    process_tss(tss_file, genome, enhancer_distal_num, distal_num, output_folder)
    print('Finished processing tss')
    unslopped_tss = output_folder + '/tss_data/true_tss_filtered.bed'
    enhancer_slopped_tss =  output_folder + '/tss_data/true_enhancer_slopped_tss.bed'
    slopped_tss = output_folder + '/tss_data/true_slopped_tss.bed'
    slopped_tss_saf = output_folder + '/tss_data/true_slopped_tss.saf'
    
    
    process_enhancers(p300_file, enhancer_slopped_tss, gene_bodies, H3K4me3_file, enhancer_distal_num, genome, output_folder)
    print('Finished processing enhancers')
    enhancers = output_folder + '/enhancer_data/strict_enhancers_filtered.bed'
    enhancers_saf = output_folder + '/enhancer_data/strict_enhancers_filtered.saf'
    
    process_tfbs(slopped_tss, TFBS, output_folder)
    print('Finished processing TFBS')
    final_tfbs_file = output_folder + '/tfbs_data/final_tfbs.bed'
    
    process_background(bg_genome, slopped_tss, DHS_file, enhancers, final_tfbs_file, output_folder)
    print('Finished processing background')
    final_background = output_folder + '/background_data/final_bg.bed'
    
    process_ap(enhancers_saf, sense_bam_file, antisense_bam_file, output_folder, groseq_logtrans)
    process_ap(slopped_tss_saf, sense_bam_file, antisense_bam_file, output_folder, groseq_logtrans)
    if groseq_logtrans:
        enh_sense_file = output_folder + '/active_poised_data/strict_enhancers_filtered_sense_bam-bincounts_logtrans.txt'
        enh_antisense_file = output_folder + '/active_poised_data/strict_enhancers_filtered_antisense_bam-bincounts_logtrans.txt'
        
        tss_sense_file = output_folder + '/active_poised_data/true_slopped_tss_sense_bam-bincounts_logtrans.txt'
        tss_antisense_file = output_folder + '/active_poised_data/true_slopped_tss_antisense_bam-bincounts_logtrans.txt'
    else:   
        enh_sense_file = output_folder + '/active_poised_data/strict_enhancers_filtered_sense_bam-bincounts.txt'
        enh_antisense_file = output_folder + '/active_poised_data/strict_enhancers_filtered_antisense_bam-bincounts.txt'
        
        tss_sense_file = output_folder + '/active_poised_data/true_slopped_tss_sense_bam-bincounts.txt'
        tss_antisense_file = output_folder + '/active_poised_data/true_slopped_tss_antisense_bam-bincounts.txt'
    print('Finished processing active/poised')
    
    process_histones(genome_saf_format, histone_path, output_folder, hist_logtrans)
    all_histone_data = output_folder + '/histone_data/alltogether_notnormed.txt'
    if hist_logtrans:
        all_histone_data = output_folder + '/histone_data/alltogether_notnormed_logtrans.txt'
    else:
        all_histone_data = output_folder + '/histone_data/alltogether_notnormed.txt'
    print('Finished processing histones')    
    

    #Compressing + indexing files for tensor creation
    files_to_compress = [unslopped_tss, final_background, enhancers]
    files_to_compress.append(remove_header_histone_file(all_histone_data))
    compress_index_files(files_to_compress)
    print('Finished compressing files')
    
    tss_compressed = unslopped_tss + '.gz'
    background_compressed = final_background + '.gz'
    enhancer_compressed = enhancers + '.gz' 
    histone_compressed = files_to_compress[-1] + '.gz'
    #histone_compressed = output_folder +'/histone_data/alltogether_notnormed_noheader.txt.gz'
  
    groseq_positive, groseq_negative, _ = clustering(enh_sense_file, sense_bam_file, enh_antisense_file, antisense_bam_file)
  
    groseq_tss = tss_clustering(tss_sense_file, sense_bam_file, tss_antisense_file, antisense_bam_file)
    
    tensor_out_folder = output_folder + '/tensor_data/'
    if not os.path.exists(tensor_out_folder):
        os.mkdir(tensor_out_folder)
    
    make_histone_tensors(groseq_positive, groseq_negative, groseq_tss, nb_mark, enhancer_compressed, tss_compressed, background_compressed, histone_compressed, window_width, number_of_windows, tensor_out_folder, output_folder)
    print('Finished making histone tensors')
    histone_dataset = tensor_out_folder + 'histone_train_dataset.pt'
    sequence_dataset = tensor_out_folder + 'sequence_train_dataset.pt'
    used_bg_sorted = output_folder + '/background_data/used_bg_sorted.bed.gz'

    make_sequence_tensors(groseq_positive, groseq_negative, enhancer_compressed, tss_compressed, used_bg_sorted, histone_compressed, genome_fasta_file, sequence_window_width, tensor_out_folder)
    print('Finished making sequence tensors')
    
    make_training_testing_dataloaders(histone_dataset, sequence_dataset, 50, tensor_out_folder)
    print('Finished making tensors for training')
    
    time_len = time.time() - start
    print("Amount of time to process: " + str(time_len))
   
    
    
    return
if __name__ == '__main__':
    main()

