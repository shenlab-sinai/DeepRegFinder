from pybedtools import BedTool

# TFBs = []
# set_genome = 'hg38'

# TFBs.append('/home/kims/work/enhancer-prediction-data/tfbs/nanog/optimal-idr-peaks-nanog.bed')
# TFBs.append('/home/kims/work/enhancer-prediction-data/tfbs/oct4/IDR_final_optimal.narrowPeak')

# DHS = 'h1-dnase-seq-peaks.bed'

#CHANGE P300 DATA
# p300 = 'potential-enhancers.bed'
# slopped_tss = 'true_tss.bed'
# TPMs = []
# TPMs.append(DHS)
# TPMs.append(p300)

def process_tfbs(slopped_tss, TFBs):
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
    final_tfbs.saveas('final_tfbs.bed')

def process_tpms(slopped_tss, TPMs, final_tfbs):
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
    final_tpms = final_tpms.sort()
    final_tpms.saveas('final_tpms.bed')