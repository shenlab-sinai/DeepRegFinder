#!/usr/bin/env python3
from DeepRegFinder.preprocessing_functions import *
from pybedtools import BedTool
import yaml
import sys
import os
import time
import pandas as pd

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
    # Creating a output directory for all intermediate and final results.
    output_folder = sys.argv[2]
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)        

    #Data for Genome Processing (Making 100 kb windowed bed + saf + bg windowed)
    genome = dataMap['genome']
    try:
        # genome_size_file expected format:
        # chr1<TAB>1234567
        # chr2<TAB>1234567
        genome_size_file = dataMap['genome_size_file']
        df = pd.read_csv(genome_size_file, header=None, delim_whitespace=True)
        valids = df.iloc[:, 0].tolist()
    except KeyError:
        valids = dataMap['valid_chromosomes']
        genome_size_file = None
    window_width = dataMap['window_width']
    number_of_windows = dataMap['number_of_windows']

    # Promoter files.
    tss_file = dataMap['tss_file']
    distal_num = dataMap['distal_bp_distance']
    H3K4me3_file = dataMap['H3K4me3_file']

    # Enhancer files.
    enhancer_files = dataMap['enhancer_files']
    enhancer_distal_num = dataMap['enhancer_distal_bp_distance']

    # DNA accessibility.
    DHS_file = dataMap['DHS_file']
        
    #Data for TFBS file creation
    try:
        TFBS = dataMap['TFBS']
    except KeyError:
        TFBS = None
    
    #Data for active poised clustering
    try:
        sense_bam_file = dataMap['sense_bam_file']
        antisense_bam_file = dataMap['antisense_bam_file']
        groseq_bam_file = None
    except KeyError:
        groseq_bam_file = dataMap['groseq_bam_file']
        sense_bam_file = None
        antisense_bam_file = None
    groseq_logtrans = dataMap['groseq_log_transformation']
    
    #Data for histone mark file creation
    histone_path = dataMap['histone_folder']
    hist_logtrans = dataMap['histone_log_transformation']
    bkg_samples = dataMap['bkg_samples']
    nz_cutoff = dataMap['nz_cutoff']
    val_p, test_p = dataMap['val_p'], dataMap['test_p']

    #Data for performance.
    cpu_threads = dataMap['cpu_threads']
    
    """
    Calling functions to process data
    """

    # Create genomic bins.
    process_genome(genome, valids, window_width, number_of_windows, 
                   output_folder, genome_size_file=genome_size_file)
    print('Finished processing genome')
    bg_genome = os.path.join(output_folder, 'genome_data', 'bgwindowed.filtered.bed')
    genome_saf_format = os.path.join(output_folder, 'genome_data', 
                                     'windowed.filtered.saf')
    
    # TSSs are from existing annotations. Clustered TSSs are collapsed.
    process_tss(tss_file, DHS_file, genome, valids, enhancer_distal_num, 
                distal_num, output_folder)
    print('Finished processing tss')
    unslopped_tss = os.path.join(output_folder, 'tss_data', 'true_tss_filtered.bed')
    enhancer_slopped_tss =  os.path.join(output_folder, 'tss_data', 'enhancer_slopped_tss.bed')
    slopped_tss = os.path.join(output_folder, 'tss_data', 'true_slopped_tss.bed')
    slopped_tss_saf = os.path.join(output_folder, 'tss_data', 'true_slopped_tss.saf')
    
    # Enhancers are defined as p300/CBP/etc. peaks (narrow) that are away 
    # from potential TSSs.
    process_enhancers(enhancer_files, DHS_file, enhancer_slopped_tss, 
                      H3K4me3_file, distal_num, genome, valids, output_folder)
    print('Finished processing enhancers')
    enhancers = os.path.join(output_folder, 'enhancer_data', 'strict_enhancers_filtered.bed')
    enhancers_saf = os.path.join(output_folder, 'enhancer_data', 'strict_slopped_enh.saf')
    merged_enh_file = os.path.join(output_folder, 'enhancer_data', 'merged_enh.bed')
    
    # TFBS.
    if TFBS is not None:
        process_tfbs(slopped_tss, TFBS, valids, output_folder)
        print('Finished processing TFBS')
        final_tfbs_file = os.path.join(output_folder, 'tfbs_data', 'final_tfbs.bed')
    else:
        final_tfbs_file = None

    # Background regions are genomic bins minus enhancers, TSS and DHS.
    process_background(bg_genome, valids, enhancer_slopped_tss, DHS_file, 
                       merged_enh_file, final_tfbs_file, enhancer_distal_num, 
                       genome, output_folder)
    print('Finished processing background')
    final_background = os.path.join(output_folder, 'background_data', 'final_bg.bed')

    # True positive markers are used to calculate validation rate after 
    # whole genome prediction.
    process_tpms(slopped_tss, merged_enh_file, DHS_file, final_tfbs_file, 
                 valids, output_folder)
    print('Finished processing True Positive Markers')

    # Get histone mark counts for the above defined regions.
    process_histones(genome_saf_format, histone_path, output_folder, 'histone_data', 
                     hist_logtrans, cpu_threads=cpu_threads)
    if hist_logtrans:
        all_histone_data = os.path.join(
            output_folder, 'histone_data', 'alltogether_notnormed_logtrans.txt')
    else:
        all_histone_data = os.path.join(
            output_folder, 'histone_data', 'alltogether_notnormed.txt')
    print('Finished processing histones')

    # Compressing + indexing files for tensor creation
    bed = BedTool(all_histone_data)
    bed.tabix(force=True, is_sorted=True)
    print('Finished compressing and indexing files')
    histone_compressed = all_histone_data + '.gz'

    # Get GRO-seq counts for enhancers and TSSs.
    process_groseq(enhancers_saf, sense_bam_file, antisense_bam_file, 
                   groseq_bam_file, output_folder, groseq_logtrans, 
                   cpu_threads=cpu_threads)
    process_groseq(slopped_tss_saf, sense_bam_file, antisense_bam_file, 
                   groseq_bam_file, output_folder, groseq_logtrans, 
                   cpu_threads=cpu_threads)
    if groseq_logtrans:
        file_tail = '_logtrans.txt'
    else:
        file_tail = '.txt'
    if sense_bam_file is not None:
        enh_sense_file = os.path.join(
            output_folder, 'groseq_data', 
            'strict_slopped_enh_sense_bam-bincounts' + file_tail)
        enh_antisense_file = os.path.join(
            output_folder, 'groseq_data', 
            'strict_slopped_enh_antisense_bam-bincounts' + file_tail)
        tss_sense_file = os.path.join(
            output_folder, 'groseq_data', 
            'true_slopped_tss_sense_bam-bincounts' + file_tail)
        tss_antisense_file = os.path.join(
            output_folder, 'groseq_data', 
            'true_slopped_tss_antisense_bam-bincounts' + file_tail)
        enh_groseq_file = None
        tss_groseq_file = None
    else:
        enh_groseq_file = os.path.join(
            output_folder, 'groseq_data', 
            'strict_slopped_enh_bam-bincounts' + file_tail)
        tss_groseq_file = os.path.join(
            output_folder, 'groseq_data', 
            'true_slopped_tss_bam-bincounts' + file_tail)
        enh_sense_file, enh_antisense_file = None, None
        tss_sense_file, tss_antisense_file = None, None
    print('Finished processing groseq')
  
    # Define active and poised enhancers and TSSs.
    positive_enh, negative_enh = positive_negative_clustering(
        enh_sense_file, enh_antisense_file, enh_groseq_file)
    positive_tss, negative_tss = positive_negative_clustering(
        tss_sense_file, tss_antisense_file, tss_groseq_file)
    
    make_tensor_dataset(positive_enh, negative_enh, positive_tss, negative_tss, 
                        enhancers, unslopped_tss, final_background, 
                        histone_compressed, window_width, number_of_windows, 
                        output_folder, bkg_samples=bkg_samples, 
                        nz_cutoff=nz_cutoff, val_p=val_p, test_p=test_p)
    print('Finished making train-val-test datasets')

    # Generate input for prediction step
    try:
        histone_pred_path = dataMap['histone_folder_prediction']
        process_histones(genome_saf_format, histone_pred_path, output_folder, 'histone_data_prediction', 
                     hist_logtrans, cpu_threads=cpu_threads)
        if hist_logtrans:
            all_histone_data = os.path.join(
                output_folder, 'histone_data_prediction', 'alltogether_notnormed_logtrans.txt')
        else:
            all_histone_data = os.path.join(
                output_folder, 'histone_data_prediction', 'alltogether_notnormed.txt')
        print('Finished processing histones for prediction')
        
        bed = BedTool(all_histone_data)
        bed.tabix(force=True, is_sorted=True)
        print('Finished compressing and indexing files')

    except KeyError:
        histone_pred_path = None

    print('Finished processing prediction data')
    
    # Print time.
    elapsed = time.time() - start
    print("Time elapsed for preprocessing: {}s".format(int(elapsed)))

    return

if __name__ == '__main__':
    main()

