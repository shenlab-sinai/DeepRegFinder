from pybedtools import BedTool
import os
import subprocess


"""
Creating TFBS file by getting rid of TFBS that are not distal to tss
Merge TFBS and save as final_tfbs.bed
"""
def process_tfbs(slopped_tss, TFBs, output_folder):
    tfbs_out_folder = './' + output_folder + '/tfbs_data/'
    if not os.path.exists(tfbs_out_folder):
        os.mkdir(tfbs_out_folder)
        
    tss = BedTool(slopped_tss)
    tfbs_sub_tss = []
    #Getting rid of TFBS that aren't distal to tss
    #tss file is slopped so range of each interval is 2kb, making sure anything that is 
    #overlapping is at least 1kb away
    for tfbs in TFBs:
        tfbs = BedTool(tfbs)
        tfbs_sub_tss.append(tfbs.subtract(tss, A=True))
    
    final_tfbs = tfbs_sub_tss[0]

    #Merging the features using cat
    for i in range(1, len(TFBs)):
        final_tfbs = final_tfbs.cat(TFBs[i])
    final_tfbs = final_tfbs.sort()
    final_tfbs.saveas(tfbs_out_folder + 'final_tfbs.bed')

"""
Creating TPMs file by getting rid of TPMs that are not distal to tss
Merge TPMs and save as final_tpms.bed
"""
def process_tpms(slopped_tss, TPMs, final_tfbs, output_folder):
    tpms_out_folder = './' + output_folder + '/tpms_data/'
    if not os.path.exists(tpms_out_folder):
        os.mkdir(tpms_out_folder)
        
    tpms_sub_tss = []
    #Getting rid of TFBS that aren't distal to tss
    #tss file is slopped so range of each interval is 2kb, making sure anything that is 
    #overlapping is at least 1kb away

    for tpms in TPMs:
        tpms = BedTool(tpms)
        tpms_sub_tss.append(tpms.subtract(slopped_tss, A=True))

    tpms_sub_tss.append(final_tfbs)
    final_tpms = tpms_sub_tss[0]

    #Merging the features using cat
    for i in range(1, len(TPMs)):
        final_tpms = final_tpms.cat(TPMs[i])
        
    #Sorting the features
    file = tpms_out_folder + 'final_tpms.bed'
    file_compress_name = file + '.gz'
    final_tpms = final_tpms.sort()
    final_tpms.saveas(file)
    subprocess.call(['./index_file.sh', file, file_compress_name])