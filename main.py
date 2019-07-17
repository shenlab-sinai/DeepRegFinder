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
from compress_index_files import compress_index_files, compress_index_histone_file
from train_histone import make_histone_tensors, clustering
from train_sequence import make_sequence_tensors
from generate_unified_indices import make_training_testing_tensors
import yaml
import sys

def main():
    """
    Taking in input data from yaml file
    """
    params = sys.argv[1]
    print(params)
    with open(params) as f:
        # use safe_load instead load
        dataMap = yaml.safe_load(f)
        
 
    #Data for Genome Processing (Making 100 kb windowed bed + saf + 2kb windowed)
    genome = dataMap['genome']
    window_width = dataMap['window_width']
    bg_window_width = dataMap['background_window_width']
    valids = dataMap['valid_chromosomes']

    #DNase/DHS file
    DHS_file = dataMap['DHS_file']
    
    #Data for TSS file creation
    tss_file = dataMap['tss_file']
    
    #p300 file
    p300_file = dataMap['p300_file']
    
    #Data for enhancer file creation 
    distal_num = dataMap['distal_bp_distance']
    tss_for_enhancer = dataMap['tss_for_enhancer'] #BED file of human TSS to isolate those DHS p300 binding sites that are distal to the TSS. 
    
    #Data for tfbs file creation
    TFBS = dataMap['TFBS']

    
    #Data for tpms file creation
    TPMs = []
    TPMs.append(DHS_file)
    TPMs.append(p300_file)
    
    #Data for AP file creation
    #Path to the folder with rep data
    ap_path = dataMap['active_poised_folder']
    
    #Data for histone file creation
    histone_path = dataMap['histone_folder']
    
    #Data to make pytorch files
    sense_bam_file = dataMap['sense_bam_file']
    antisense_bam_file = dataMap['antisense_bam_file']
    
    
    """
    Calling functions to process data
    """
    #process_genome(genome, window_width, bg_window_width, valids)
    print('Finished processing genome')
                                 
    #process_tss(tss_file, DHS_file, genome)
    print('Finished processing tss')
    
    #TSS data!!!!!
    #process_enhancers(p300=p300_file, DHS=DHS_file, distal_num=distal_num, set_genome = genome,tss = tss_for_enhancer)
    print('Finished processing enhancers')
    
    #Data for tfbs file creation
    #TSS Data!!!!
    tss_for_tfbs_tpms = 'true_slopped_tss.bed'
    
    #process_tfbs(tss_for_tfbs_tpms, TFBS)
    print('Finished processing TFBS')
    final_tfbs_file = 'final_tfbs.bed'
    
    
    process_tpms(tss_for_tfbs_tpms, TPMs, final_tfbs_file)
    print('Finished processing TPMs')
    
    #Data for background file creation
    #Genome is from processing pipeline and is windowed by 100 bp, filtered, and sorted
    bg_genome = '2kbwindowed.filtered.bed'
    #TSS input is all TSS, not just TSS not intersected with DHS
    bg_tss = 'hcc_sorted_chr.bed'
    #P300 DATA????
    bg_p300 = 'strict_enhancers.bed'
    process_background(bg_genome, bg_tss, DHS_file, bg_p300, final_tfbs_file)
    print('Finished processing background')
    
    ap_enhancers = 'strict_enhancers.bed'
    process_ap(ap_enhancers, ap_path)
    print('Finished processing active/poised')
    
    genome_saf = 'windowed.filtered.saf'
    process_histones(genome_saf, histone_path)
    print('Finished processing histones')
    
    #Compressing + indexing files for pytorch creation
    files = ['true_tss.bed', 'final_bg.bed', 'strict_enhancers.bed']
    files.append(compress_index_histone_file('alltogether_notnormed.txt'))
    compress_index_files(files)
    
    
    sense_file = '/home/kims/work/enhancer-prediction-data/ap_data/srr_r1bam-bincounts.txt'
    antisense_file = '/home/kims/work/enhancer-prediction-data/ap_data/srr_r2bam-bincounts.txt'
    groseq_positive, groseq_negative = clustering(sense_file, sense_bam_file, antisense_file, antisense_bam_file)
    #make_histone_tensors(groseq_positive, groseq_negative, 3)
    #make_sequence_tensors(groseq_positive, groseq_negative)
    
    #make_training_testing_tensors('sevenmark_train_dataset.pt', 'sequence_train_dataset.pt', 50)
    
    

    
    
    
    
    return
if __name__ == '__main__':
    main()

