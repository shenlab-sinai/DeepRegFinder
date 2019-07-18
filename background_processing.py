from pybedtools import BedTool
import os

"""
Removes tss, DHS, enhancer, and TFBS from genome to get possible background sites
Saves this as final_bg.bed
"""
def process_background(genome_windowed, tss, DHS, enhancers, tfbs, output_folder):
    bg_out_folder = './' + output_folder + '/background_data/'
    if not os.path.exists(bg_out_folder):
        os.mkdir(bg_out_folder)
        
    genome_windowed = BedTool(genome_windowed)
    tss = BedTool(tss)
    DHS = BedTool(DHS)
    enhancers = BedTool(enhancers)
    tfbs = BedTool(tfbs)
    
    #Subtracting off TSS
    bins_minus_T = genome_windowed.subtract(tss, A=True)

    #Subtracting DHS
    bins_minus_TD = bins_minus_T.subtract(DHS, A=True)

    #Subtracting p300 sites
    bins_minus_TDE = bins_minus_TD.subtract(enhancers, A=True)

    #Subtracting TFBS
    final_bg = bins_minus_TDE.subtract(tfbs, A=True)

    final_bg.saveas(bg_out_folder + 'final_bg.bed')