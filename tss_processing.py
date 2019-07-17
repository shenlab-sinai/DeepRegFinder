import pandas as pd
import numpy as np
import subprocess
from pybedtools import BedTool

def cluster_gene_tss(g, bps_cutoff=500, verbose=0):
    '''Cluster a gene's TSSs and return the cluster medians
    '''
    geneid = g['Gene stable ID'].iloc[0]
    chrom = g['Chromosome/scaffold name'].iloc[0]
    tss = g['Transcription start site (TSS)']
    med_list = []
    clust_list = []
    n_tss = n_clust = 0
    for t in tss.sort_values():
        if len(clust_list) == 0:
            clust_list.append(t)
        elif (t - np.median(clust_list)) < bps_cutoff:
            clust_list.append(t)
        else:
            if len(clust_list) > 1:
                n_tss += len(clust_list)
                n_clust += 1
            med_list.append(int(np.median(clust_list)))
            clust_list = [t]
    if len(clust_list) > 1:
        n_tss += len(clust_list)
        n_clust += 1
    med_list.append(int(np.median(clust_list)))
    if verbose > 0:
        if n_tss > 0:
            print("%s: %d TSSs were clustered into %d cluster medians." % 
                  (geneid, n_tss, n_clust))
        else:
            print("%s: no clustering was done." % (geneid))
    df = pd.DataFrame({'Gene stable ID': geneid, 
                       'Chromosome/scaffold name': chrom, 
                       'Transcription start site (TSS)': med_list})
    return df[['Gene stable ID', 
               'Chromosome/scaffold name', 
               'Transcription start site (TSS)']]

def process_tss(tss_file, dhs_file, set_genome):
    human_tss_df = pd.read_csv(tss_file, delimiter="\t")
    coding_tss_df = human_tss_df.loc[human_tss_df['Gene type'] == 'protein_coding']
    coding_tss_grouped = coding_tss_df.groupby('Gene stable ID')
    clustered_coding_tss = coding_tss_grouped.apply(cluster_gene_tss, bps_cutoff=500, verbose=0)
  
    clustered_coding_tss.to_csv('human_coding_clustered_TSS.tsv', sep="\t", index=None)
    
    subprocess.call(['./format_tss.sh', 'human_coding_clustered_TSS.tsv'])
    
    tss_temp = BedTool('hcc_sorted_chr.bed')
    dhs = BedTool(dhs_file)
    true_tss = tss_temp.intersect(dhs)
    true_tss.saveas('true_tss.bed')
    slopped_tss = true_tss.slop(b=1000, genome=set_genome)
    slopped_tss.saveas('true_slopped_tss.bed')
    