import subprocess
import glob
import os
import pandas as pd

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
        