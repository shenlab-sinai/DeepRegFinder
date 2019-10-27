from pybedtools import BedTool
import os
import glob
import subprocess
import sys
import pandas as pd
import numpy as np
import torch
import numpy as np
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pysam
import time
import math
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import random
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp
from scipy.stats.mstats import zscore
import itertools as itls
import pycm
from re import sub
from torchsummary import summary
import torch.nn as nn
from scipy import interp 

"""
Compresses all files using bgzip, then indexes them with tabix
"""
def compress_index_files(files):
    for file in files:
        file_compress_name = file + '.gz'
        subprocess.call(['./index_file.sh', file, file_compress_name])

"""
Function to remove header from histone file
Returns the name
"""
def remove_header_histone_file(file):
    file_no_head_name = file.split('.txt')[0] + '_noheader.txt' 
    cmnd = "sed '1d' " + file + " > " + file_no_head_name
    os.system(cmnd)
    return file_no_head_name


"""
Converts the input bed file to a saf file
Saves the saf file as name passed in as saf_file
"""
def bed_to_saf(bed_file, saf_file):
    with open(bed_file, 'r') as bed:
        with open(saf_file, 'w+') as saf:
            saf.write("GeneID\tChr\tStart\tEnd\tStrand\n" )
            strand = "+"
            for i, line in enumerate(bed):
                words = line.split(sep = '\t')
                chrom = words[0]
                start = str(int(words[1]))
                end = str(int(words[2]))
                gene_id = '.'.join((chrom, start, end))
                saf.write(gene_id + "\t" + chrom + "\t" + start + "\t" + end + "\t" + strand + "\n")
"""
Runs feature counts on all bam files in the active_poised folder
Returns names of the files created after running featureCounts
"""
def process_ap(saf_file, sense_bam_file, antisense_bam_file, output_folder, groseq_logtrans):
    #extract name + change it 
    ap_out_folder = './' + output_folder + '/active_poised_data/'
    if not os.path.exists(ap_out_folder):
        os.mkdir(ap_out_folder)
    
    #histone = histone_folder.split('/')[split_num]
    file_name = saf_file.split('/')[-1].split('.')[0]
    files = [sense_bam_file, antisense_bam_file]
    file_type = [file_name + '_sense', file_name + '_antisense']
    
    out_files = []
    for counter, rep in enumerate(files):
        out_name = ap_out_folder + file_type[counter] + "_bam-bincounts.txt"
        subprocess.call(["featureCounts", rep, "-a", saf_file, "-O", "-F", "SAF", "--fracOverlap", "0.5", "-f", "-o",  out_name]) 
        out_files.append(out_name)
    
    if groseq_logtrans:
        for file in out_files:
            out_name = file.split('.txt')[0] + '_logtrans.txt'
            subprocess.call(['./log_transform.sh', file, out_name])

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

"""
Selects sites that are p300 co-activator binding sites 
Next uses a BED file of human TSS to isolate those p300 binding sites that are distal to the TSS. 

Also creates a saf file from this bed file of enhancers
"""
def process_enhancers(p300, tss, gene_bodies, H3K4me3, enhancer_distal_num, genome, output_folder):
    enhancer_out_folder = './' + output_folder + '/enhancer_data/'
    if not os.path.exists(enhancer_out_folder):
        os.mkdir(enhancer_out_folder)
    
    #Converting to Bed format
    p300 = BedTool(p300)
    tss = BedTool(tss)
    gene_bodies = BedTool(gene_bodies)
    H3K4me3_peaks = BedTool(H3K4me3)
    
    H3K4me3_slopped = H3K4me3_peaks.slop(b=enhancer_distal_num, genome=genome)
    
    
    intergenic = p300.subtract(gene_bodies, A = True)
    tss_distal_sites = intergenic.subtract(tss, A=True)
    tss_histone_distal_sites = tss_distal_sites.subtract(H3K4me3_slopped, A=True)

    sorted_sites = tss_histone_distal_sites.sort()
    sorted_sites.saveas(enhancer_out_folder + 'strict_enhancers.bed')
    valids_file = './' + output_folder + '/genome_data/valids.txt'
    
    
    cmnd = 'grep -w -f '+ valids_file + ' '+ enhancer_out_folder + 'strict_enhancers.bed' + '>' + enhancer_out_folder + 'strict_enhancers_filtered.bed'
    os.system(cmnd)
    
    bed_to_saf(enhancer_out_folder + 'strict_enhancers_filtered.bed', enhancer_out_folder + 'strict_enhancers_filtered.saf')
   
"""
Windows passsed in genome by window_width, saves it as genome_save_name
Filters this windowed genome so it only contains chromosomes in valids file
"""
#Function to window genome by set width and save it by save name
def window_sort_genome(genome, window_width, genome_save_name, filtered_save_name, valids_file):
    genome_windowed = BedTool().window_maker(genome=genome, w=window_width).saveas()
    genome_windowed.saveas(genome_save_name)
    cmnd = 'grep -w -f '+ valids_file + ' '+ genome_save_name + '>' + filtered_save_name
    os.system(cmnd)
    
"""
Creates windowed.filtered.bed which is the genome windowed by window_width and filtered to only
contain set valid chromosomes
Also converts this windowed.filtered.bed into saf format
Creates bgwindowed.filtered.bed which is genome windowed by bg_window_width and filtered to only
contain set valid chromosomes
"""
def process_genome(genome, window_width, number_of_windows, valids, output_folder):
    gene_out_folder = './' + output_folder + '/genome_data/'
    if not os.path.exists(gene_out_folder):
        os.mkdir(gene_out_folder)
        
    bg_window_width = window_width * number_of_windows

    #Creation of text file of valid chromosomes to find in larger genome file
    valids = np.array(valids)
    valids_file = gene_out_folder + "valids.txt"
    with open(valids_file, "w") as f:
        np.savetxt(f, valids, fmt='%s')
    
    window_sort_genome(genome, window_width, gene_out_folder + 'windowed.bed', gene_out_folder + 'windowed.filtered.bed', valids_file)
    window_sort_genome(genome, bg_window_width, gene_out_folder + 'bgwindowed.bed', gene_out_folder + 'bgwindowed.filtered.bed', valids_file)
    bed_to_saf(gene_out_folder + 'windowed.filtered.bed', gene_out_folder + 'windowed.filtered.saf')

"""
Runs featureCounts on all reps of a chiq-seq assay for each histone mark specified
Finds the average of each repetition for each histone mark 
Stores these averages as columns in alltogether_notnormed.txt
"""
def process_histones(genome_saf, histone_path, output_folder, hist_logtrans):
    histone_out_folder = './' + output_folder + '/histone_data/'
    if not os.path.exists(histone_out_folder):
        os.mkdir(histone_out_folder)
        
    
    out_files = []
    #Looping through every histone folder and running featureCounts on each rep
    for histone_folder in glob.glob(histone_path + '/*/'):
        split_num = -1
        if histone_folder[-1] == '/':
            split_num = -2
        histone = histone_folder.split('/')[split_num]
        out_path = histone_out_folder + histone
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        
        for rep in glob.glob(histone_folder + "/*.bam"):
            rep_name = os.path.basename(rep)
            rep_num = rep_name.split('rep')[1][0]
            out_name = out_path + "/r" + rep_num + "-bincounts.txt"
            out_files.append(out_name)
            subprocess.call(["featureCounts", rep, "-a", genome_saf, "-O", "-F", "SAF","--fracOverlap",  "0.5", "-o", out_name]) 
            
    
    #Log transforming data if necessary
    if hist_logtrans:
        for file in out_files:
            out_name = file.split('.txt')[0] + '_logtrans.txt'
            subprocess.call(['./log_transform.sh', file, out_name])
    
    
    histone_cols = {}
    dict_of_counts = {}
    same_data = {}
    same_data['Chr'] = []
    same_data['Start'] = []
    same_data['End'] = []
    
    #Looping through files made from featureCounts and storing
    #results in a dataframe
    counter = 0
    for histone_folder in glob.glob(histone_out_folder + '/*/'):
        split_num = -1
        if histone_folder[-1] == '/':
            split_num = -2
        histone = histone_folder.split('/')[split_num]
        out_path = histone_out_folder + histone
        histone_cols[histone] = []
        
        ending = "/*bincounts.txt"
        if hist_logtrans:
            ending = "/*bincounts_logtrans.txt"
          
        
        for rep in glob.glob(histone_folder + ending):
            rep_name = os.path.basename(rep)
            rep_num = rep_name.split('r')[1][0]
            col_name = histone + '-rep' + rep_num
            histone_cols[histone].append(col_name)

            dict_of_counts[col_name] = []
            #print(rep_name)
            with open(rep, "r") as infile:
                next(infile)
                next(infile)
                for lines in infile:
                    lines = lines.strip().split("\t")
                    dict_of_counts[col_name].append(float(lines[6]))
                    
                    if counter == 0:
                        same_data['Chr'].append(lines[1])
                        same_data['Start'].append(int(lines[2]))
                        same_data['End'].append(int(lines[3]))
            counter += 1
      
    dataframe = pd.DataFrame(dict_of_counts)
    dataframe = pd.DataFrame(same_data)
    dataframe = dataframe.assign(**dict_of_counts)
    
    #Taking the average of the counts for each histone mark
    avg_cols = {}
    for key in histone_cols:
        avg_col_name = key + '_avg'
        avg_cols[avg_col_name] = dataframe[histone_cols[key]].mean(axis=1)

    dataframe = dataframe.assign(**avg_cols)
    
    kept_cols = ['Chr', 'Start', 'End']
    avgs_cols_list = list(avg_cols.keys())
    new_cols = kept_cols + avgs_cols_list
    new_dataframe = dataframe[new_cols]
    if hist_logtrans:
        new_dataframe.to_csv(histone_out_folder + "alltogether_notnormed_logtrans.txt", sep = '\t', index = False)
    else:
        new_dataframe.to_csv(histone_out_folder + "alltogether_notnormed.txt", sep = '\t', index = False)

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


'''
Cluster a gene's TSSs and return the cluster medians
'''
def cluster_gene_tss(g, bps_cutoff=500, verbose=0):
   
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
'''
Cluster a gene's TSSs 
Sorts and formats these clustered TSSs into BED format and saves as hcc_sorted_chr.bed
Finds the intersection of the TSS with DHS and saves it as true_tss.bed
Slops the TSS file by distal_num, and saves as true_slopped_tss.bed
'''
def process_tss(tss_file, set_genome, enhancer_distal_num, distal_num, output_folder):
    tss_out_folder = './' + output_folder + '/tss_data/'
    if not os.path.exists(tss_out_folder):
        os.mkdir(tss_out_folder)
        
    human_tss_df = pd.read_csv(tss_file, delimiter="\t")
    
    #Get rid of this line later
    coding_tss_df = human_tss_df.loc[human_tss_df['Gene type'] == 'protein_coding']
    coding_tss_grouped = human_tss_df.groupby('Gene stable ID')
    clustered_coding_tss = coding_tss_grouped.apply(cluster_gene_tss, bps_cutoff=500, verbose=0)
  
    clustered_coding_tss.to_csv(tss_out_folder + 'clustered_TSS.tsv', sep="\t", index=None)
    
    valids_file = './' + output_folder + '/genome_data/valids.txt'
    subprocess.call(['./format_tss.sh', tss_out_folder + 'clustered_TSS.tsv', tss_out_folder, valids_file])
    
    #bed_to_saf(tss_out_folder + 'true_tss_filtered.bed', tss_out_folder + 'true_tss_filtered.saf')
    
    true_tss = BedTool(tss_out_folder + 'true_tss_filtered.bed')
    slopped_tss = true_tss.slop(b=enhancer_distal_num, genome=set_genome)
    slopped_tss.saveas(tss_out_folder + 'true_enhancer_slopped_tss.bed')
    
    slopped_tss_no_dhs = true_tss.slop(b=distal_num, genome=set_genome)
    slopped_tss_no_dhs.saveas(tss_out_folder + 'true_slopped_tss.bed')
    bed_to_saf(tss_out_folder + 'true_slopped_tss.bed', tss_out_folder + 'true_slopped_tss.saf')
        
"""
Read GRO-seq reads aligned to an enhancer list, normalize the reads to RPKM values and log transform them, use K-means to cluster the log RPKM values into active and poised classes. This procedure is done for sense and antisense reads separately. An enhancer is classified as active or poised only when both sense and antisense derived class labels agree.
"""
def rpkm_normalize_rnaseq_reads(total_reads, feature_reads, feature_length):
    rpkm = ((10**9) * feature_reads) / (feature_length * total_reads).astype(float)
    return rpkm 

def active_poised_clustering(enhancer_rnaseq_file, sense, seed=12345):
    ap_df = pd.read_table(enhancer_rnaseq_file, skiprows = 1)

    
    ap_df['Chr'] = ap_df.Chr.astype('category')
    #total_reads = ap_df[sense].sum()
    reads = ap_df.iloc[:, -1]
    float_reads = pd.to_numeric(reads, downcast='float')
    total_reads = float_reads.sum()
   
    #ap_df['rpkm'] = rpkm_normalize_rnaseq_reads(total_reads, ap_df[sense], ap_df['Length'])
    ap_df['rpkm'] = rpkm_normalize_rnaseq_reads(total_reads, float_reads, ap_df['Length'])
    ap_df['logrpkm'] = np.log(ap_df['rpkm'] + 1)
    assert(not ap_df.isnull().values.any())
    clusters = KMeans(n_clusters=2, random_state=seed).fit(
        ap_df['logrpkm'].values.reshape(-1, 1))
    return clusters.labels_, clusters.cluster_centers_

def clustering(sense_file, sense_bam_file, antisense_file, antisense_bam_file):
    sense_labels, sense_centers = active_poised_clustering(sense_file, sense_bam_file)
    antisense_labels, antisense_centers = active_poised_clustering(antisense_file, antisense_bam_file)
    groseq_positive = np.logical_and(sense_labels, np.logical_not(antisense_labels))
    groseq_negative = np.logical_and(np.logical_not(sense_labels), antisense_labels)
    groseq_tss = np.logical_or(sense_labels, antisense_labels)
    
    return groseq_positive, groseq_negative, groseq_tss

def tss_clustering(sense_file, sense_bam_file, antisense_file, antisense_bam_file):
    sense_labels, sense_centers = active_poised_clustering(sense_file, sense_bam_file)
    antisense_labels, antisense_centers = active_poised_clustering(antisense_file, antisense_bam_file)
    groseq_tss = np.logical_or(sense_labels, antisense_labels)
    
    return groseq_tss



def build_histone_training_tensors(featurefile, rpkmsfile, isEnhancer, groseq_positive, groseq_negative, window_width, number_of_windows, nb_mark=7):
    """Build a pytorch tensor at designated regions for training
    Args:
        featurefile (str): file for regions.
        rpkmsfile (str): file for genome-wide 100bp bin RPKMs.
        isEnhancer (bool): is enhancer region?
        nb_mark ([int]): number of histone marks. Default is 7.
    """
    features = list(pysam.TabixFile(featurefile).fetch(parser=pysam.asTuple()))
    rpkms = pysam.TabixFile(rpkmsfile)
    elist = []  # list for all regions.
    for i, feat in enumerate(features):
        if isEnhancer:  # ignore ambiguous enhancers.
            if (not groseq_positive[i]) and (not groseq_negative[i]):
                continue
        # Get up and down 2Kb of region centers.
        chrom, start, end = feat[0], int(feat[1]), int(feat[2])
        center = int((start + end)/2)
        
        center_start = int(math.ceil(center/float(window_width)))*window_width
        halfway = (window_width * number_of_windows)/2
        
        if (center_start - halfway < 0):
            continue
        
        rows = list(rpkms.fetch(chrom, center_start - halfway, center_start + halfway, 
                                parser=pysam.asTuple()))
        
        if (len(rows)!=number_of_windows):
            continue
        
        #assert(len(rows) == number_of_windows)
        

        forward_tmp = []
        for row in rows:
            e = [ float(item.strip()) for item in row[3:3+nb_mark]]
            forward_tmp.append(e)
        forward_data = np.array(forward_tmp).T
        elist.append(forward_data)
    out = np.stack(elist)
    return torch.from_numpy(out)

def build_background_data(bgfile, rpkmsfile, background_save_folder, trimmed_bg_file = None):
    """
    Helper to create background dataset
    """
    if trimmed_bg_file:
        #we already have the ~32000 sites we want to use
        return build_histone_training_tensors(trimmed_bg_file, rpkmsfile, isEnhancer=False)
    bg = list(pysam.TabixFile(bgfile).fetch(parser = pysam.asTuple()))
    rpkms = pysam.TabixFile(rpkmsfile)
    elist = []
    print("Gathering all non-zero potential background sites!")
    count = 0
    for i, pb in enumerate(bg):
        chrom, start, end = pb[0], int(pb[1]), int(pb[2])
        rows = np.array(list(rpkms.fetch(chrom, start, end, parser=pysam.asTuple())))
        
        if rows.shape[0] != 20: 
            #We will have fewer than 20 bins if we are at a chromosome end; 
            #we can safely exclude this from our training set
            continue
            
        #discard chr, start, end so we just have histone marks (columns 3:10), and convert to int dtype
        #take transpose so we have channels x rows (histone marks x bins; 7 x 20)
        data = rows[:, 3:].astype(float).T
        
        #if the sample is not all 0, include it in our list of potential samples
        if not np.all(np.isclose(data, 0)):  
            elist.append({'data': data, 'location': (chrom, str(start), str(end))})
            count += 1
            
        if i % 100000 == 0:
            print(str(i) + " regions checked so far")
            print("Gathered " + str(count) + " potential regions so far")
    print("Done gathering potential sites. Selecting 32000 randomly")
    if len(elist) > 32000:
        elist = random.sample(elist, 32000)
    
    out = np.stack([x['data'] for x in elist])
    print("Writing BED file of background sites used in used_bg.bed")
    with open(background_save_folder + "used_bg.bed", "w+") as bed:
        for x in elist:
            c, s, e = x['location'][0], x['location'][1], x['location'][2]
            bed.write(c + "\t" + s + "\t" + e + "\n")
    print("Wrote succesfully")
    return torch.from_numpy(out)

    
def get_statistics(data):
    mean, std = 0., 0.
    total_observed = 0.
    for sample in data:
        x = sample[0]
        x_mean = x.mean(1)
        x_std = x.std(1)
        current_observations = x.shape[1]
        old_mean = mean
        old_std = std
        old_constant = (total_observed/(total_observed + current_observations))
        new_constant = (current_observations/(total_observed + current_observations))
        std_update_term = (total_observed*current_observations) \
            / (np.power((total_observed + current_observations), 2)) \
            * np.power(np.subtract(old_mean, x_mean), 2)
        mean = (old_constant*old_mean) + (new_constant*x_mean)
        std = np.sqrt((old_constant*np.power(old_std, 2)) + (new_constant*np.power(x_std, 2)) + std_update_term)
        total_observed += current_observations
    return mean, std



def gen_pos_lab(gp, gn):
    if gp:
        return 1
    elif gn:
        return 0
    else:
        return -1

"""
poised enhancer: 0
active enhancer: 1
poised tss: 2
active tss: 3
background: 4

"""
def make_histone_tensors(groseq_positive, groseq_negative, groseq_tss, nb_mark, enhancer, tss, background, histone, window_width, number_of_windows, tensor_output_folder, output_folder):
        
    enhancer_tensor = build_histone_training_tensors(enhancer, histone, True, groseq_positive, groseq_negative, window_width, number_of_windows, nb_mark)
    print("Made enhancer histone tensor")
    
    
    tss_tensor = build_histone_training_tensors(tss, histone, False, groseq_positive, groseq_negative, window_width, number_of_windows, nb_mark)
    print("Made tss histone tensor")
    
    background_save_folder = output_folder + '/background_data/'
    
    #This takes so long
    bg_tensor = build_background_data(background, histone, background_save_folder)
    

    used_bg = BedTool(background_save_folder + 'used_bg.bed')
    used_bg_sorted = used_bg.sort()
    used_bg_sorted.saveas(background_save_folder + 'used_bg_sorted.bed')
    subprocess.call(['./index_file.sh', background_save_folder + 'used_bg_sorted.bed', background_save_folder + 'used_bg_sorted.bed.gz'])
    
    bg_tensor = build_histone_training_tensors(background_save_folder + 'used_bg_sorted.bed.gz', histone, False, groseq_positive, groseq_negative, window_width, number_of_windows, nb_mark)
    print("Made background histone tensor")
    
    pos_labels = [gen_pos_lab(gp, gn) for gp, gn in zip(groseq_positive, groseq_negative)]
    pos_labels = np.array(pos_labels)
    pos_labels = pos_labels[pos_labels != -1]
    num_active_enhancer = np.sum(pos_labels)
    num_poised_enhancer = len(pos_labels) - np.sum(pos_labels)
    pos_labels = torch.from_numpy(pos_labels).long()
    
    
    tss_labels = np.ones(len(tss_tensor)) * 2
    tss_labels[groseq_tss] = 3
    num_active_tss = np.sum(tss_labels == 3)
    num_poised_tss = len(tss_labels) - num_active_tss
    tss_labels = torch.from_numpy(tss_labels).long()
    
    bg_labels = torch.full((len(bg_tensor),), 4).long()
    
    histone_class_weights = [num_poised_enhancer, num_active_enhancer, num_poised_tss, num_active_tss, len(bg_labels)]
    histone_class_weights = histone_class_weights / np.sum(histone_class_weights)
    
    seq_class_weights = [num_poised_enhancer + num_active_enhancer, num_poised_tss + num_active_tss, len(bg_labels)]
    seq_class_weights = seq_class_weights / np.sum(seq_class_weights)
    np.savetxt(tensor_output_folder + 'histone_class_weights.txt', histone_class_weights)
    np.savetxt(tensor_output_folder + 'seq_class_weights.txt', seq_class_weights)
    
    posDataset = torch.utils.data.TensorDataset(enhancer_tensor.float(), pos_labels)
    tssDataset = torch.utils.data.TensorDataset(tss_tensor.float(), tss_labels)
    bgDataset = torch.utils.data.TensorDataset(bg_tensor.float(), bg_labels)
    
    data = torch.utils.data.ConcatDataset((posDataset, tssDataset, bgDataset))
    mean, std = get_statistics(data)
    
    final_training_dataset = NormDataset(data, mean=mean, std=std, norm=True, nb_mark=nb_mark)
    torch.save(final_training_dataset, tensor_output_folder + "histone_train_dataset.pt")

def build_sequence_training_tensors(featurefile, seqfile, groseq_positive, groseq_negative, sequence_window_width, isEnhancer=False):
    encoder = {
        'A' : [1, 0, 0, 0],
        'C' : [0, 1, 0, 0], 
        'G' : [0, 0, 1, 0],
        'T' : [0, 0, 0, 1],
        'N' : [0, 0, 0, 0]
    }
    features = list(pysam.TabixFile(featurefile).fetch(parser=pysam.asTuple()))
    seqs = pysam.FastaFile(seqfile)
    finished_sequences = []
    for i, feat in enumerate(features):
        if isEnhancer:
            if (not groseq_positive[i]) and (not groseq_negative[i]):
                continue
        chrom, start, end = feat[0], int(feat[1]), int(feat[2])
        center = int((start+end) /2)
        center_start = int(math.ceil(center / 100.0)) * 100
        half = sequence_window_width / 2
        s = seqs.fetch(chrom, center_start-half, center_start+half)
        s = sub('[a-z]', 'N', s)
        encoded_seq = np.array([encoder[base] for base in s])
        finished_sequences.append(encoded_seq)
    stacked_data = np.stack(finished_sequences)
    return torch.from_numpy(stacked_data)

def build_pos_labels(file, groseq_positive, groseq_negative):
    l = []
    f = list(pysam.TabixFile(file).fetch(parser=pysam.asTuple()))
    for i, feat in enumerate(f):
        if groseq_positive[i] or groseq_negative[i]:
            l.append(0)
    return torch.from_numpy(np.array(l)).long()

    
def make_sequence_tensors(groseq_positive, groseq_negative, enhancer, tss, background, histone, genome_fasta_file, sequence_window_width, output_folder):
    
    enhancer_sequence_tensor = build_sequence_training_tensors(enhancer, genome_fasta_file, groseq_positive, groseq_negative, sequence_window_width, isEnhancer=True)
    print('Made enhancer sequence dataset')
    
    tss_sequence_tensor = build_sequence_training_tensors(tss, genome_fasta_file, groseq_positive, groseq_negative, sequence_window_width)
    print('Made tss sequence dataset')
    
    bg_sequence_tensor = build_sequence_training_tensors(background, genome_fasta_file, groseq_positive, groseq_negative, sequence_window_width)
    print('Made background sequence dataset')
    
    pos_labels = build_pos_labels(enhancer, groseq_positive, groseq_negative)
    
    tss_labels = torch.full((len(tss_sequence_tensor),), 1).long()
    bg_labels = torch.full((len(bg_sequence_tensor),), 2).long()
    
    posDataset = torch.utils.data.TensorDataset(enhancer_sequence_tensor.float(), pos_labels)
    tssDataset = torch.utils.data.TensorDataset(tss_sequence_tensor.float(), tss_labels)
    bgDataset = torch.utils.data.TensorDataset(bg_sequence_tensor.float(), bg_labels)
    
    sequence_training_dataset = torch.utils.data.ConcatDataset((posDataset, tssDataset, bgDataset))
    
    torch.save(sequence_training_dataset, output_folder + "sequence_train_dataset.pt" )


class NormDataset(Dataset):
    def __init__(self, data, mean, std, norm=True, nb_mark=7):
        self.data = data
        self.mean = mean.view(nb_mark, -1)
        self.std = std.view(nb_mark, -1)
        self.norm = norm
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample, label = self.data[idx]
        if self.norm:
            sample = torch.div(sample.sub(self.mean), self.std)
        return sample, label







def check_ordering(hist_dset, seq_dset):
    '''Make sure the labels of histone mark and sequence 
    datasets represent the same classes
    '''
    assert len(hist_dset) == len(seq_dset), "Unordered: Datasets have different lengths!"
    for hist_sample, seq_sample in zip(hist_dset, seq_dset):
        hist_l, seq_l = hist_sample[1].item(), seq_sample[1].item()
        if (hist_l == 0) or (hist_l == 1):
            assert seq_l == 0, "Unordered: Enhancers don't match"
        elif (hist_l == 2) or (hist_l == 3):
            assert seq_l == 1, "Unordered, TSS don't match"
        elif (hist_l == 4):
            assert seq_l == 2, "Unordered, BG don't match"
    return True 

def make_training_testing_dataloaders(his_dataset, seq_dataset, batch_sz, tensor_out_folder):
    histone_dataset = torch.load(his_dataset)
    sequence_dataset = torch.load(seq_dataset)
    check_ordering(histone_dataset, sequence_dataset)
    
    rng = np.random.RandomState(12345)
    eval_size = .2
    hist_train_X, hist_eval_X = [], []
    hist_train_y, hist_eval_y = [], []
    seq_train_X, seq_eval_X = [], []
    seq_train_y, seq_eval_y = [], []
    for hist_sample, seq_sample in zip(histone_dataset, sequence_dataset):
        hist_d, seq_d = hist_sample[0], seq_sample[0]
        hist_l, seq_l = hist_sample[1], seq_sample[1]
        if rng.rand() > eval_size:
            #if hist_l != 3:
            hist_train_X.append(hist_d)
            hist_train_y.append(hist_l)
            seq_train_X.append(seq_d)
            seq_train_y.append(seq_l)
        else:
            #if hist_l != 3:
            hist_eval_X.append(hist_d)
            hist_eval_y.append(hist_l)
            seq_eval_X.append(seq_d)
            seq_eval_y.append(seq_l)
            
    hist_train_X, hist_train_y = torch.stack(hist_train_X), torch.stack(hist_train_y)
    hist_eval_X, hist_eval_y = torch.stack(hist_eval_X), torch.stack(hist_eval_y)
    seq_train_X, seq_train_y = torch.stack(seq_train_X), torch.stack(seq_train_y)
    seq_eval_X, seq_eval_y = torch.stack(seq_eval_X), torch.stack(seq_eval_y)
    
    #Swap last two dimensions for sequence tensors
    seq_train_X = seq_train_X.permute(0, 2, 1)
    seq_eval_X = seq_eval_X.permute(0, 2, 1)
    
    h_train_loader = DataLoader(
    torch.utils.data.TensorDataset(hist_train_X, hist_train_y), 
    batch_size=batch_sz, shuffle=True, num_workers=4) 
    
    h_eval_loader = DataLoader(
    torch.utils.data.TensorDataset(hist_eval_X, hist_eval_y), 
    batch_size=batch_sz, shuffle=False, num_workers=4)
    
    h_train_loader_3cls = DataLoader(
    torch.utils.data.TensorDataset(hist_train_X, seq_train_y), 
    batch_size=batch_sz, shuffle=True, num_workers=4)
    
    h_eval_loader_3cls = DataLoader(
    torch.utils.data.TensorDataset(hist_eval_X, seq_eval_y), 
    batch_size=batch_sz, shuffle=False, num_workers=4)
    
    s_train_loader = DataLoader(
    torch.utils.data.TensorDataset(seq_train_X, seq_train_y), 
    batch_size=batch_sz, shuffle=True, num_workers=4)
    
    s_eval_loader = DataLoader(
    torch.utils.data.TensorDataset(seq_eval_X, seq_eval_y), 
    batch_size=batch_sz, shuffle=False, num_workers=4)
    
    torch.save(h_train_loader, tensor_out_folder + "histone_train_dataloader.pt")
    torch.save(h_eval_loader, tensor_out_folder + "histone_eval_dataloader.pt")
    torch.save(h_train_loader_3cls, tensor_out_folder + "histone_train_dataloader_3cls.pt")
    torch.save(h_eval_loader_3cls, tensor_out_folder + "histone_eval_dataloader_3cls.pt")
    torch.save(s_train_loader, tensor_out_folder + "sequence_train_dataloader.pt")
    torch.save(s_eval_loader, tensor_out_folder + "sequence_eval_dataloader.pt")
    
    