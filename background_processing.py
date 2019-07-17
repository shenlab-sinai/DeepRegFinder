from pybedtools import BedTool

def process_background(genome_windowed, tss, DHS, p300, tfbs):
    genome_windowed = BedTool(genome_windowed)
    tss = BedTool(tss)
    DHS = BedTool(DHS)
    p300 = BedTool(p300)
    tfbs = BedTool(tfbs)
    
    #Subtracting off TSS
    bins_minus_T = genome_windowed.subtract(tss, A=True)

    #Subtracting DHS
    bins_minus_TD = bins_minus_T.subtract(DHS, A=True)

    #Subtracting p300 sites
    bins_minus_TDE = bins_minus_TD.subtract(p300, A=True)

    #Subtracting TFBS
    final_bg = bins_minus_TDE.subtract(tfbs, A=True)

    final_bg.saveas('final_bg.bed')

 

