from pybedtools import BedTool

"""Selects sites that are both p300 co-activator binding sites and DHS by finding the sites that overlap 
	in the 2 BED files representing p300 peaks/binding sites and DNase-seq peaks/DHS. 
	(i.e. the sites on the genome that are both DHS and p300 binding sites). Next uses a BED file of human
	TSS to isolate those DHS p300 binding sites that are distal to the TSS. 
	"""
def process_enhancers(p300, DHS, distal_num, set_genome, tss):
    #Converting to Bed format
    p300 = BedTool(p300)
    dhs = BedTool(DHS)
    tss = BedTool(tss)

    slopped_tss = tss.slop(b=distal_num, genome=set_genome)
    
    overlapping_sites = p300.intersect(dhs)
    distal_sites = overlapping_sites.subtract(slopped_tss, A=True)

    sorted_sites = distal_sites.sort()
    sorted_sites.saveas('strict_enhancers.bed')