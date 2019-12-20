from pybedtools import BedTool, Interval
from pybedtools.featurefuncs import midpoint
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import torch
from torch.utils.data import Dataset, TensorDataset
import os
import glob
import subprocess
import sys
import pandas as pd
import numpy as np
import pysam
import time
import math


__all__ = ['process_groseq', 'process_background', 'process_enhancers', 
           'process_genome', 'process_histones', 'process_tfbs', 'process_tss', 
           'process_tpms', 'positive_negative_clustering', 'make_tensor_dataset', 
           'ChannNormDataset']


"""
Converts the input bed file to a saf file
Saves the saf file as name passed in as saf_file
"""
def _bed_to_saf(bed_file, saf_file):
    with open(bed_file, 'r') as bed:
        with open(saf_file, 'w+') as saf:
            saf.write("GeneID\tChr\tStart\tEnd\tStrand\n" )
            for i, line in enumerate(bed):
                words = line.split(sep='\t')
                chrom = words[0]
                start = str(int(words[1]) + 1)  # SAF is 1-based.
                end = str(int(words[2]))
                gene_id = '.'.join((chrom, start, end))
                saf.write("\t".join((gene_id, chrom, start, end, ".")) + "\n")

"""
Reads a txt file produced by featureCounts and log-transform the count column.
"""
def _logtrans_featcnt_file(file):
    out_name = os.path.splitext(file)[0] + '_logtrans.txt'
    df = pd.read_csv(file, comment="#", delim_whitespace=True)
    # The last column contains the read counts.
    df.iloc[:, -1] = df.iloc[:, -1].apply(lambda x: np.log2(x + 1))
    df.to_csv(out_name, sep="\t", index=False)
    # subprocess.call(['./log_transform.sh', file, out_name])

"""
Nomralize the read counts from featureCounts.
"""
def _norm_featcnt_file(file):
    df = pd.read_csv(file, comment="#", delim_whitespace=True)
    # Normalize to 100M reads.
    df.iloc[:, -1] *= 100e6/df.iloc[:, -1].sum()
    df.to_csv(file, sep="\t", index=False)


"""
Runs feature counts on all bam files in the active_poised folder
Returns names of the files created after running featureCounts
"""
def process_groseq(saf_file, sense_bam_file, antisense_bam_file, groseq_bam_file, 
                   output_folder, groseq_logtrans=False, cpu_threads=1):
    # import pdb; pdb.set_trace()
    groseq_out_folder = os.path.join(output_folder, 'groseq_data')
    if not os.path.exists(groseq_out_folder):
        os.mkdir(groseq_out_folder)
    
    # Output file names.
    file_base = os.path.basename(saf_file)
    file_base = os.path.splitext(file_base)[0]
    bam_files = [sense_bam_file, antisense_bam_file, groseq_bam_file]
    strands = ['_sense', '_antisense', '']
    
    out_files = []
    for bam, strand in zip(bam_files, strands):
        if bam is None:
            continue
        out_name = os.path.join(
            groseq_out_folder, file_base + strand + "_bam-bincounts.txt")
        print('Running featureCounts: {} -> {}'.format(bam, saf_file), end='...')
        sys.stdout.flush()
        subprocess.call(
            ["featureCounts", bam, "-a", saf_file, "-O", "-F", "SAF", 
             "--fracOverlap", "0.5", "-f", "-o", out_name, 
             "-T", str(cpu_threads)], 
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print('Done')
        sys.stdout.flush()
        out_files.append(out_name)
    
    if groseq_logtrans:
        for file in out_files:
            _logtrans_featcnt_file(file)


"""
Removes tss, DHS, enhancer, and TFBS from genome to get possible background sites
Saves this as final_bg.bed
"""
def process_background(genome_windowed_bed, valids, enh_tss_bed, DHS_bed, 
                       p300_bed, tfbs_bed, enhancer_distal_num, genome, 
                       output_folder):

    bg_out_folder = os.path.join(output_folder, 'background_data')
    if not os.path.exists(bg_out_folder):
        os.mkdir(bg_out_folder)

    # Read and filter genome windows and other lists.
    genome_windowed = BedTool(genome_windowed_bed).filter(
        lambda p: p.chrom in valids)
    enh_tss = BedTool(enh_tss_bed).filter(lambda p: p.chrom in valids)
    DHS = BedTool(DHS_bed).filter(lambda p: p.chrom in valids)
    p300 = BedTool(p300_bed).filter(lambda p: p.chrom in valids)
    tfbs = BedTool(tfbs_bed).filter(lambda p: p.chrom in valids)

    # Expand all the regulatory elements by a large #bps to make 
    # the background windows "purer". 
    # tss = tss.slop(b=enhancer_distal_num, genome=genome)
    DHS = DHS.slop(b=enhancer_distal_num, genome=genome)
    p300 = p300.slop(b=enhancer_distal_num, genome=genome)
    tfbs = tfbs.slop(b=enhancer_distal_num, genome=genome)

    # Subtracting TSS, DHS, p300 and TFBS.
    bins_minus_T = genome_windowed.subtract(enh_tss, A=True)
    bins_minus_TD = bins_minus_T.subtract(DHS, A=True)
    bins_minus_TDE = bins_minus_TD.subtract(p300, A=True)
    final_bg = bins_minus_TDE.subtract(tfbs, A=True)
    final_bg.saveas(os.path.join(bg_out_folder, 'final_bg.bed'))
    # import pdb; pdb.set_trace()

"""
Define enhancers based on p300 peaks that are intergenic 
(excluding genebodies) and away from TSS and H3K4me3 peaks 
by a certain number of bps (default is 3Kb).
Also creates a SAF file from this BED file of enhancers.
"""
def process_enhancers(p300_file, dhs_file, slopped_tss_file, H3K4me3_file, 
                      distal_num, genome, valids, output_folder):

    enhancer_out_folder = os.path.join(output_folder, 'enhancer_data')
    if not os.path.exists(enhancer_out_folder):
        os.mkdir(enhancer_out_folder)

    # Read p300 and overlap with DHS.
    p300 = BedTool(p300_file).filter(lambda p: p.chrom in valids)
    dhs = BedTool(dhs_file).filter(lambda p: p.chrom in valids)
    slopped_dhs = dhs.slop(b=distal_num, genome=genome)
    p300 = p300.intersect(b=slopped_dhs, u=True)
    # Read all the regions that need to subtract.
    slopped_tss = BedTool(slopped_tss_file).filter(lambda p: p.chrom in valids)
    # gene_bodies = BedTool(gene_bodies_file).filter(lambda p: p.chrom in valids)
    H3K4me3_peaks = BedTool(H3K4me3_file).filter(lambda p: p.chrom in valids)
    H3K4me3_slopped = H3K4me3_peaks.slop(b=distal_num, genome=genome)
    # Excluding genebodies, TSS and H3K4me3.
    # intergenic = p300.subtract(gene_bodies, A=True)
    tss_distal_sites = p300.subtract(slopped_tss, A=True)
    tss_histone_distal_sites = tss_distal_sites.subtract(H3K4me3_slopped, A=True)
    # Sorting and writing to BED and SAF files.
    sorted_sites = tss_histone_distal_sites.sort()
    sorted_sites.saveas(
        os.path.join(enhancer_out_folder, 'strict_enhancers_filtered.bed'))
    # Get enhancer modpoints and slop them.
    slopped_enh = sorted_sites.each(midpoint)
    slopped_enh = slopped_enh.slop(b=distal_num, genome=genome)
    slopped_enh.saveas(os.path.join(enhancer_out_folder, 'strict_slopped_enh.bed'))
    _bed_to_saf(
        os.path.join(enhancer_out_folder, 'strict_slopped_enh.bed'), 
        os.path.join(enhancer_out_folder, 'strict_slopped_enh.saf'))

    
"""
Creates windowed.filtered.bed which is the genome windowed by window_width and filtered to only
contain set valid chromosomes
Also converts this windowed.filtered.bed into saf format
Creates bgwindowed.filtered.bed which is genome windowed by bg_window_width and filtered to only
contain set valid chromosomes
"""
def process_genome(genome, valids, window_width, number_of_windows, 
                   output_folder, genome_size_file=None):

    gene_out_folder = os.path.join(output_folder, 'genome_data')
    if not os.path.exists(gene_out_folder):
        os.mkdir(gene_out_folder)

    # Function to window genome by set width and save it.
    def window_genome(window_width_, filtered_save_name):
        if genome_size_file is not None:
            genome_windowed = BedTool().window_maker(g=genome_size_file, 
                                                     w=window_width_)
            genome_windowed.saveas(filtered_save_name)
        else:
            genome_windowed = BedTool().window_maker(genome=genome, 
                                                     w=window_width_)
            genome_windowed = genome_windowed.filter(lambda p: p.chrom in valids)
            genome_windowed.saveas(filtered_save_name)

    # Create windowed BED and SAF files.    
    bg_window_width = window_width*number_of_windows
    window_genome(
        window_width, 
        os.path.join(gene_out_folder, 'windowed.filtered.bed')
    )
    window_genome(
        bg_window_width, 
        os.path.join(gene_out_folder, 'bgwindowed.filtered.bed')
    )
    _bed_to_saf(os.path.join(gene_out_folder, 'windowed.filtered.bed'), 
                os.path.join(gene_out_folder, 'windowed.filtered.saf'))


"""
Runs featureCounts on all reps of chiq-seq samples for each histone mark
Calculates the average of the reps for each histone mark 
Stores these averages as columns in the alltogether_notnormed.txt
"""
def process_histones(genome_saf, histone_path, output_folder, 
                     hist_logtrans=False, cpu_threads=1):
    # import pdb; pdb.set_trace()
    histone_out_folder = os.path.join(output_folder, 'histone_data')
    if not os.path.exists(histone_out_folder):
        os.mkdir(histone_out_folder)
    
    # Looping through every histone folder and running featureCounts on each rep
    out_files = []
    for histone in os.listdir(histone_path):
        histone_folder = os.path.join(histone_path, histone)
        if not os.path.isdir(histone_folder):
            continue
        out_path = os.path.join(histone_out_folder, histone)
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        # Looping through each rep.
        for rep in glob.glob(os.path.join(histone_folder, "*.bam")):
            rep_name = os.path.splitext(os.path.basename(rep))[0]
            rep_num = rep_name.split('rep')[1]
            out_name = os.path.join(out_path, "r" + rep_num + "-bincounts.txt")
            out_files.append(out_name)
            print('Running featureCounts: {} -> {}'.format(rep, genome_saf), end='...')
            sys.stdout.flush()
            subprocess.call(
                ["featureCounts", rep, "-a", genome_saf, "-O", "-F", "SAF",
                 "--fracOverlap", "0.5", "-o", out_name, 
                 "-T", str(cpu_threads)], 
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print('Done')
            sys.stdout.flush()
    
    # Normalize read counts for each rep before averaging them.
    for file in out_files:
        _norm_featcnt_file(file)

    # Log transforming data if necessary
    if hist_logtrans:
        for file in out_files:
            _logtrans_featcnt_file(file)
    
    # Looping through files produced by featureCounts and merging
    # them into a dataframe.
    hist_cnt_dict = {}
    bin_cols = None
    for histone in os.listdir(histone_out_folder):
        histone_folder = os.path.join(histone_out_folder, histone)
        if not os.path.isdir(histone_folder):
            continue
        if hist_logtrans:
            pattern = "r*-bincounts_logtrans.txt"
        else:
            pattern = "r*-bincounts.txt"
        print('Calculating average bincounts for histone: {}'.format(histone), 
              end='...', flush=True)
        rep_cnt_list = []
        for rep in glob.glob(os.path.join(histone_folder, pattern)):
            df = pd.read_csv(rep, comment="#", delim_whitespace=True)
            rep_cnt_list.append(df.iloc[:, -1])
            if bin_cols is None:
                bin_cols = df[['Chr', 'Start', 'End']]
                # Comment the header line so that it can be 
                # processed as BED by tabix.
                bin_cols = bin_cols.rename(columns={'Chr': '#Chr'})
        hist_df = pd.concat(rep_cnt_list, axis=1)
        avg_cnt = hist_df.mean(axis=1)
        hist_cnt_dict[histone + '_avg'] = avg_cnt
        print('Done', flush=True)
    print('Merging bincounts for histones into one dataframe', 
          end='...', flush=True)
    hist_cnt_merged = pd.DataFrame(hist_cnt_dict)
    hist_cnt_df = pd.concat([bin_cols, hist_cnt_merged], axis=1)
    print('Done', flush=True)
    if hist_logtrans:
        df_out_path = os.path.join(histone_out_folder, 
                                   "alltogether_notnormed_logtrans.txt")
    else:
        df_out_path = os.path.join(histone_out_folder, 
                                   "alltogether_notnormed.txt")
    hist_cnt_df.to_csv(df_out_path, sep="\t", index=False)


"""
Creating TFBS file by getting rid of TFBS that are not distal to tss
Merge TFBS and save as final_tfbs.bed
"""
def process_tfbs(slopped_tss, TFBS, valids, output_folder):

    tfbs_out_folder = os.path.join(output_folder, 'tfbs_data')
    if not os.path.exists(tfbs_out_folder):
        os.mkdir(tfbs_out_folder)

    # Slopped TSS list.
    tss = BedTool(slopped_tss)    
    # Getting rid of TFBS that aren't distal to tss
    # tss file is slopped so range of each interval is 2kb, making sure 
    # anything that isn't overlapping is at least 1kb away.
    for i, bed in enumerate(TFBS):
        tfbs = BedTool(bed).filter(lambda p: p.chrom in valids)
        tfbs = tfbs.subtract(tss, A=True)
        if i == 0:
            final_tfbs = tfbs
        else:
            final_tfbs = final_tfbs.cat(tfbs, postmerge=True)
    final_tfbs.sort().saveas(os.path.join(tfbs_out_folder, 'final_tfbs.bed'))


"""
Creating TPMs file by getting rid of TPMs that are not distal to tss
Merge TPMs and save as final_tpms.bed
"""
def process_tpms(slopped_tss_file, p300_file, dhs_file, final_tfbs_file, 
                 valids, output_folder):

    tpms_out_folder = os.path.join(output_folder, 'tpms_data')
    if not os.path.exists(tpms_out_folder):
        os.mkdir(tpms_out_folder)
        
    tss = BedTool(slopped_tss_file)
    p300 = BedTool(p300_file).filter(lambda p: p.chrom in valids)
    dhs = BedTool(dhs_file).filter(lambda p: p.chrom in valids)
    p300 = p300.subtract(tss, A=True)
    dhs = dhs.subtract(tss, A=True)
    if final_tfbs_file is not None:
        tfbs = BedTool(final_tfbs_file)  # already filtered and subtracted.
    else:
        tfbs = None
    
    tpms = p300
    tpms = tpms.cat(dhs, postmerge=True)
    tpms = tpms.cat(tfbs, postmerge=True) if tfbs is not None else tpms
    tpms.sort().saveas(os.path.join(tpms_out_folder, 'final_tpms.bed'))


'''
Cluster a gene's TSSs and return the cluster medians
'''
def _cluster_gene_tss(g, bps_cutoff=500, verbose=0):
   
    geneid = g['geneid'].iloc[0]
    chrom = g['chrom'].iloc[0]
    tss = g['tss']
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
            print("%s: %d TSSs were clustered into %d "
                  "cluster medians." % (geneid, n_tss, n_clust))
        else:
            print("%s: no clustering was done." % (geneid))
    df = pd.DataFrame({'geneid': geneid, 
                       'chrom': chrom, 
                       'tss': med_list})
    return df


'''
Cluster a gene's TSSs 
Sorts and formats these clustered TSSs into BED format and saves as clustered_TSS.tsv
Slops the TSSs by enhancer_distal_num and distal_num and saves as BEDs.
'''
def process_tss(tss_file, dhs_file, set_genome, valids, enhancer_distal_num, 
                distal_num, output_folder):

    tss_out_folder = os.path.join(output_folder, 'tss_data')
    if not os.path.exists(tss_out_folder):
        os.mkdir(tss_out_folder)

    # Merge neighboring TSSs and take the midpoints.
    tss = BedTool(tss_file).filter(lambda p: p.chrom in valids)
    tss_merged = tss.sort().merge(d=500).each(midpoint)
    enhancer_slopped_tss = tss_merged.slop(b=enhancer_distal_num, 
                                           genome=set_genome)
    tss_merged = enhancer_slopped_tss.saveas(
        os.path.join(tss_out_folder, 'enhancer_slopped_tss.bed'))

    # Overlap with DHS to find the true TSSs.
    tss_merged = tss_merged.each(midpoint)  # unslop.
    dhs = BedTool(dhs_file).filter(lambda p: p.chrom in valids)
    slopped_dhs = dhs.slop(b=distal_num, genome=set_genome)
    true_tss = tss_merged.intersect(b=slopped_dhs, u=True)
    true_tss = true_tss.saveas(
        os.path.join(tss_out_folder, 'true_tss_filtered.bed'))
    slopped_tss = true_tss.slop(b=distal_num, genome=set_genome)
    slopped_tss.saveas(os.path.join(tss_out_folder, 'true_slopped_tss.bed'))
    _bed_to_saf(os.path.join(tss_out_folder, 'true_slopped_tss.bed'), 
                os.path.join(tss_out_folder, 'true_slopped_tss.saf'))
    
        
"""
Read GRO-seq counts (maybe log-transformed) for an enhancer/tss list, 
use K-means to cluster the values into active and poised classes. This 
procedure is done for sense and antisense counts separately. An enhancer 
is classified as active or poised only when both sense and antisense derived 
class labels agree.
"""
def positive_negative_clustering(sense_file, antisense_file, groseq_file=None):
    '''
    Two-way clustering using GRO-seq counts.
    '''
    def binary_clustering(groseq_cnt_file, seed=12345):
        df = pd.read_csv(groseq_cnt_file, delim_whitespace=True)
        cnts = df.iloc[:, [-1]].astype('float')
        clustering = KMeans(n_clusters=2, random_state=seed).fit(cnts)
        # Make sure label 0 is always the cluster with lower count.
        # Flip the labels if C0's center > C1's center.
        c0c, c1c = clustering.cluster_centers_[:, 0]
        if c0c > c1c:
            return 1 - clustering.labels_
        return clustering.labels_

    if sense_file is not None:
        sense_labels = binary_clustering(sense_file)
        antisense_labels = binary_clustering(antisense_file)
        positive_labs = np.logical_and(sense_labels, 
                                       antisense_labels)
        negative_labs = np.logical_and(np.logical_not(sense_labels), 
                                       np.logical_not(antisense_labels))
    else:
        positive_labs = binary_clustering(groseq_file)
        negative_labs = np.logical_not(positive_labs)
    return positive_labs, negative_labs


def build_histone_tensors(region_bed, hist_cnt_file, positives, negatives, 
                          window_width, number_of_windows, is_bkg=False, 
                          samples=None, nz_cutoff=5, out_bed=None, 
                          base_label=0, cpu_threads=1):
    """Build a pytorch tensor at designated regions for training
    Args:
        featurefile (str): file for regions.
        rpkmsfile (str): file for genome-wide 100bp bin RPKMs.
        isEnhancer (bool): is enhancer region?
        nb_mark ([int]): number of histone marks. Default is 7.
    """
    assert(is_bkg and positives is None and negatives is None or not is_bkg)
    regions = BedTool(region_bed)
    if is_bkg:
        # merge regions to reduce #queries to improve performance.
        # multiprocessing failed because tabix fetch can't be pickled.
        regions = regions.merge()  # bkg regions are already sorted.
    # pandas was much slower than tabix. 
    hist_cnts = pysam.TabixFile(hist_cnt_file)
    dlist, rlist, ylist = [], [], []
    region_size = window_width*number_of_windows
    for i, r in enumerate(regions):
        if positives is not None and (not positives[i]) and (not negatives[i]):
            continue
        # Once the region passed, it is either positive or negative.
        label = (base_label if positives is None 
                 else base_label + int(positives[i]))
        if not is_bkg:
            r_midpoint = round(((r.start + r.end)//2)/window_width)*window_width
            r_start = r_midpoint - region_size//2 + 1
            r_end = r_midpoint + region_size//2
            if r_start < 1:
                continue
        else:
            r_start = r.start + 1
            r_end = r.end//region_size*region_size  # remove incomplete region.
            if r_end - r_start + 1 < region_size:
                continue
        rows = np.array(
            list(hist_cnts.fetch(r.chrom, r_start, r_end, 
                                 parser=pysam.asTuple()))
        )
        if i > 0 and i % 100000 == 0:
            print('{} regions checked so far'.format(i))
            sys.stdout.flush()
        # discard chr, start, end so we just have histone marks (columns 3:10)
        # take transpose so we have channels x bins (histone marks x bins; 7 x 20)
        d = rows[:, 3:].astype(float).T
        if not is_bkg:
            # if an enhancer or TSS is at the chromosome end, it may have fewer
            # than number_of_windows bins. Remove them from training.
            if d.shape[1] % number_of_windows != 0 or \
                np.count_nonzero(d) < nz_cutoff:
                continue
            dlist.append(d)
            rlist.append((r.chrom, r_start - 1, r_end))
            ylist.append(label)
        else:  # split data and merged region into number_of_windows bin regions.
            # chann x bin (many) -> chann x batch x bin (20)
            ds = d.reshape((d.shape[0], -1, number_of_windows))
            # chann x batch x bin (20) -> batch x chann x bin (20)
            ds = np.transpose(ds, (1, 0, 2))
            rs = [ (r.chrom, s - 1, s - 1 + region_size) 
                   for s in range(r_start, r_end, region_size)]
            for d_, r_ in zip(ds, rs):
                if np.count_nonzero(d_) >= nz_cutoff:
                    dlist.append(d_)
                    rlist.append(r_)
                    ylist.append(label)

    # Assemble the data and do sampling.
    hist_t = np.stack(dlist)
    label_t = np.array(ylist)
    if samples is not None and samples < len(hist_t):
        ix = np.sort(np.random.choice(len(hist_t), samples, replace=False))
        hist_t = hist_t[ix]
        label_t = label_t[ix]
        rlist = [ rlist[x] for x in ix]
    # Output the used regions as BED.
    if out_bed is not None:
        used_regs = [ Interval(r[0], r[1], r[2]) for r in rlist]
        used_bed = BedTool(used_regs)
        used_bed.saveas(out_bed)

    return hist_t, label_t

"""
poised enhancer: 0
active enhancer: 1
poised tss: 2
active tss: 3
background: 4
"""
def make_tensor_dataset(positive_enh, negative_enh, positive_tss, negative_tss, 
                        enhancer, tss, background, hist_cnt_file, window_width, 
                        number_of_windows, output_folder, bkg_samples=100000, 
                        nz_cutoff=5, val_p=0.2, test_p=0.2):
    tensor_out_folder = os.path.join(output_folder, 'tensor_data')
    if not os.path.exists(tensor_out_folder):
        os.mkdir(tensor_out_folder)
    bg_out_folder = os.path.join(output_folder, 'background_data')
    if not os.path.exists(bg_out_folder):
        raise Exception('Subfolder background_data does not exist in', output_folder)
    
    # Build tensors for enhancer, TSS and background.
    enhancer_X, enhancer_y = build_histone_tensors(
        enhancer, hist_cnt_file, positive_enh, negative_enh, 
        window_width, number_of_windows, is_bkg=False, 
        nz_cutoff=nz_cutoff, base_label=0)
    print("Made enhancer histone tensor")
    tss_X, tss_y = build_histone_tensors(
        tss, hist_cnt_file, positive_tss, negative_tss, 
        window_width, number_of_windows, is_bkg=False, 
        nz_cutoff=nz_cutoff, base_label=2)
    print("Made tss histone tensor")
    bkg_bed = os.path.join(bg_out_folder, 'used_bg_sorted.bed.gz')
    bg_X, bg_y = build_histone_tensors(
        background, hist_cnt_file, None, None, 
        window_width, number_of_windows, is_bkg=True, 
        samples=bkg_samples, nz_cutoff=nz_cutoff, 
        out_bed=bkg_bed, base_label=4)
    print("Made background histone tensor")
    
    # Concat enhancer, tss and bkg and then split into train-val-test.
    train_X = np.concatenate([enhancer_X, tss_X, bg_X])
    train_y = np.concatenate([enhancer_y, tss_y, bg_y])
    val_size = int(len(train_X)*val_p)
    test_size = int(len(train_X)*test_p)
    train_X, test_X, train_y, test_y = train_test_split(
        train_X, train_y, test_size=test_size, stratify=train_y)
    train_X, val_X, train_y, val_y = train_test_split(
        train_X, train_y, test_size=val_size, stratify=train_y)
    train_X_t, train_y_t = torch.from_numpy(train_X).float(), torch.from_numpy(train_y).long()
    val_X_t, val_y_t = torch.from_numpy(val_X).float(), torch.from_numpy(val_y).long()
    test_X_t, test_y_t = torch.from_numpy(test_X).float(), torch.from_numpy(test_y).long()

    # Create pytorch datasets.
    train_dataset = TensorDataset(train_X_t, train_y_t)
    val_dataset = TensorDataset(val_X_t, val_y_t)
    test_dataset = TensorDataset(test_X_t, test_y_t)
    # Normalization by mean and std.
    # import pdb; pdb.set_trace()
    chann_mean, chann_std = ChannNormDataset.chann_norm_stats(train_dataset)
    train_dataset = ChannNormDataset(train_dataset, chann_mean, chann_std)
    val_dataset = ChannNormDataset(val_dataset, chann_mean, chann_std)
    test_dataset = ChannNormDataset(test_dataset, chann_mean, chann_std)

    torch.save(
        {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}, 
        os.path.join(tensor_out_folder, 'all_datasets.pth.tar')
    )

    # Output channel-wise mean and std for future reference.
    header = pd.read_csv(hist_cnt_file, nrows=0, delim_whitespace=True)
    marks = header.columns[3:].values.tolist()
    stats = torch.stack((chann_mean, chann_std), dim=1)
    df = pd.DataFrame(stats.cpu().numpy(), columns=['mean', 'std'], index=marks)
    df.to_csv(os.path.join(tensor_out_folder, 'chann_stats.csv'))


'''
A normalized dataset that return bin values with mean subtraction 
and std division for each histone mark separately.
'''
class ChannNormDataset(Dataset):
    def __init__(self, dataset, mean, std, norm=True):
        self.dataset = dataset
        # reshape mean and std to column vectors.
        self.mean = mean.view(-1, 1)
        self.std = std.view(-1, 1)
        self.norm = norm
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample, label = self.dataset[idx]
        if self.norm:
            sample = torch.div(sample.sub(self.mean), self.std)
        return sample, label

    '''
    Return channel-wise mean and std across all regions and bins for a dataset.
    '''
    @staticmethod
    def chann_norm_stats(dataset):
        mean, std = 0., 0.
        total_observed = 0
        for x, _ in dataset:
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


  
    