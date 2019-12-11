from itertools import accumulate
from pybedtools import BedTool, Interval
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

__all__ = ['WholeGenomeDataset', 'FastWholeGenomeDataset', 
           'process_genome_preds', 'post_merge_blocks']


class WholeGenomeDataset(Dataset):
    '''Dataset for whole genome bin count data
    It takes care of converting a linear index into the index within a 
    chromosome and the chromosome name. This avoids reading data across
    the boundary between two chromosomes.
    '''

    def __init__(self, wgbc_tsv, mean, std, norm=True, half_size=10):
        # self.wgbc_df = pd.read_table(wgbc_tsv, low_memory=True)
        self.wgbc_df = pd.read_table(wgbc_tsv)
        # import pdb; pdb.set_trace()
        chroms = self.wgbc_df['#Chr'].unique().tolist()
        self.wgbc_df = self.wgbc_df.set_index('#Chr')
        # build a list for chromosome names and their accumulate lens.
        chr_lens = [ self.wgbc_df.loc[x].shape[0] - half_size*2 + 1 
                     for x in chroms]
        # remove zero or negative length chromosomes.
        chroms_nz = [ chrom for i, chrom in enumerate(chroms) if chr_lens[i] > 0]
        chr_lens_nz = [ clen for i, clen in enumerate(chr_lens) if clen > 0]
        chr_alens = list(accumulate(chr_lens_nz))
        chroms_nz.append('END')
        chr_alens.append(np.inf)
        self.chr_name_alen = list(zip(chroms_nz, chr_alens))

        # convert mean and std into column vectors.
        self.mean, self.std = mean.reshape((-1, 1)), std.reshape((-1, 1))
        self.norm = norm
        self.half_size = half_size
        
    def _chr_name_idx(self, idx):
        '''Convert a linear index into chromosome name and 
           within chromosome index
        '''
        prev_alen = 0
        for name, alen in self.chr_name_alen:
            if idx < alen:
                # import pdb; pdb.set_trace()
                return name, idx - prev_alen
            prev_alen = alen

    def __len__(self):
        return self.chr_name_alen[-2][1]
  
    def __getitem__(self, idx):
        chrom, idx = self._chr_name_idx(idx)
        idx = idx + self.half_size  # shift to the center.
        dat = self.wgbc_df.loc[chrom].\
            iloc[idx - self.half_size : idx + self.half_size, :]
        bincnt = dat.iloc[:, 2:].values.T  # channels x bins.
        if self.norm:
            bincnt = (bincnt - self.mean)/self.std
        # return start position according to the center of the data block.
        start = int(dat.iloc[self.half_size, 0]) - 1  # to 0-based position.
        sample = (bincnt, chrom, start)
        return sample

class FastWholeGenomeDataset(Dataset):
    '''Dataset for whole genome bin count data (40x faster)
    pandas is slow. Storing all data as numpy array offers great speedup.
    It takes care of converting a linear index into the index within a 
    chromosome and the chromosome name. This avoids reading data across
    the boundary between two chromosomes.
    '''

    def __init__(self, wgbc_tsv, mean, std, norm=True, half_size=10):
        df = pd.read_table(wgbc_tsv)
        chroms = df['#Chr'].unique().tolist()
        df = df.set_index('#Chr')

        # build a list for chromosome names and their accumulate lens.
        chr_lens = [ df.loc[x].shape[0] - half_size*2 + 1 
                     for x in chroms]
        # remove zero or negative length chromosomes.
        chroms_nz = [ chrom for i, chrom in enumerate(chroms) if chr_lens[i] > 0]
        chr_lens_nz = [ clen for i, clen in enumerate(chr_lens) if clen > 0]
        chr_alens = list(accumulate(chr_lens_nz))
        chroms_nz.append('END')  # add an end point.
        chr_alens.append(np.inf)
        self.chr_name_alen = list(zip(chroms_nz, chr_alens))

        # extract data into numpy arrays for fast retrieval.
        self.chrom_dat_dict = {x: df.loc[x].values for x in chroms_nz[:-1]}
        del df

        # convert mean and std into column vectors.
        self.mean, self.std = mean.reshape((-1, 1)), std.reshape((-1, 1))
        self.norm = norm
        self.half_size = half_size
        
    def _chr_name_idx(self, idx):
        '''Convert a linear index into chromosome name and 
           within chromosome index
        '''
        prev_alen = 0
        for name, alen in self.chr_name_alen:
            if idx < alen:
                return name, idx - prev_alen
            prev_alen = alen

    def __len__(self):
        return self.chr_name_alen[-2][1]
  
    def __getitem__(self, idx):
        chrom, idx = self._chr_name_idx(idx)
        idx = idx + self.half_size  # shift to the center.
        dat = self.chrom_dat_dict[chrom][
            idx - self.half_size : idx + self.half_size]
        bincnt = dat[:, 2:].T  # channels x bins.
        if self.norm:
            bincnt = (bincnt - self.mean)/self.std
        # return start position according to the center of the data block.
        start = int(dat[self.half_size, 0]) - 1  # to 0-based position.
        sample = (bincnt, chrom, start)
        return sample


def process_genome_preds(preds, chroms, starts, maxprobs, ignore_labels=[4], 
                         maxprob_cutoff=.5, nb_block=None, contrast_probs=None):
    '''Process whole-genome predictions by grouping them
    into blocks and filter out low confidence predictions
    Args:
        preds, chroms, starts: they shall be the results returned by the 
            prediction_loop. preds are predicted labels. starts are 0-based
            positions of the centers of the predicted sites.
        ignore_labels ([list]): the list of labels that 
            shall be ignored. Default=[4] corresponds
            to background.
        nb_block ([int]): #blocks to produce. Default is None.
            This is useful for debug.
        contrast_probs ([2D array]): class probabilities of 
            a comparing dataset.
    Returns:
        A block list containing merged genomic regions that
        have the same labels.
    '''
    # Initialize block list and the working block.
    block_list = []
    block_label, block_chrom, block_start, \
        block_end, block_i1, block_i2, block_maxprob = \
        None, None, None, None, None, None, None

    def _add_block_on_condition(block_label, block_chrom, 
                                block_start, block_end, 
                                block_i1, block_i2,
                                block_maxprob):
        '''An utility function to add a block to list
        '''
        if block_label is not None \
            and not block_label in ignore_labels \
            and block_maxprob > maxprob_cutoff:
            block_info = [block_chrom, block_start, block_end, block_label]
            if contrast_probs is not None:
                block_contrastprob = np.max(
                    contrast_probs[block_i1:block_i2, block_label]
                )
                diff_prob = block_maxprob - block_contrastprob
                block_info.append(diff_prob)
            else:
                block_info.append(block_maxprob)
            block_list.append(tuple(block_info))
            return True
        return False
    
    counter = 0
    for i, (pred, chrom, start, maxprob) in \
        enumerate(zip(preds, chroms, starts, maxprobs)):
        # start a new block?
        if block_chrom != chrom or block_label != pred:
            added = _add_block_on_condition(
                block_label, block_chrom, block_start, 
                block_end, block_i1, block_i2, block_maxprob
            )
            if added: 
                counter += 1
            if nb_block is not None and counter >= nb_block:
                break
            block_label = pred
            block_chrom = chrom
            block_start = block_end = start
            block_i1 = i
            block_i2 = i + 1
            block_maxprob = maxprob
        else: # add the current record to the block.
            block_end = start
            block_i2 = i + 1
            if maxprob > block_maxprob:
                block_maxprob = maxprob
    # add the last one.
    if nb_block is None:
        _add_block_on_condition(
            block_label, block_chrom, block_start, 
            block_end, block_i1, block_i2, block_maxprob
        )
    return block_list


def post_merge_blocks(block_list, window_width=100, number_of_windows=20, 
                      known_tss_file=None):
    '''Merge blocks with the same label and certain distance 
       cutoff into one block
    Returns: merged blocks as a BedTool object.
    '''
    half_width = window_width*number_of_windows//2
    # Collect block lists for different classes separately 
    # and create BedTool objects from them.
    block_dict = {}
    class_lookup = {0: 'Poised_Enh', 1: 'Active_Enh', 2: 'Poised_TSS', 
                    3: 'Active_TSS', 4: 'Background'}
    rgb_lookup = {0: '153,255,51', 1: '255,51,51', 2: '51,51,255', 
                  3: '255,51,255', 4: '160,160,160'}
    half_width = window_width*number_of_windows//2
    # counter = 0
    for block in block_list:
        try:
            chrom, start, end, pred = block
            score = 0
        except ValueError:
            chrom, start, end, pred, score = block
        start -= half_width
        end += half_width
        name = class_lookup[pred]
        strand = '.'
        tstart, tend = start, end
        rgb = rgb_lookup[pred]
        six = (chrom, start, end, name, str(round(score, 3)), strand)
        others = (str(tstart), str(tend), rgb)
        interval = Interval(*six, otherfields=others)
        if name in block_dict:
            block_dict[name].append(interval)
        else:
            block_dict[name] = [interval]
    # merge blocks of the same type if they overlap.
    bed_dict = {
        name: 
        BedTool(ilist).merge(
            c='4,5,6,7,8,9', 
            o='distinct,max,distinct,min,max,distinct'
        )
        for name, ilist in block_dict.items()
    }
    # remove known TSSs from predicted enhancers.
    if known_tss_file is not None:
        tss = BedTool(known_tss_file)
        for name in bed_dict:
            if name.endswith('Enh'):
                bed_dict[name] = bed_dict[name].subtract(tss, A=True)

    return bed_dict






