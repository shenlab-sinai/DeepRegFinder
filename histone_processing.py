import subprocess
import glob
import os
import pandas as pd



def process_histones(genome_saf, histone_path):
    saf_file = '../../' + genome_saf
    
    for histone_folder in glob.glob(histone_path + '/*/'):
        os.chdir(histone_folder)
     
        for rep in glob.glob(histone_folder + "/*.bam"):
            rep_name = os.path.basename(rep)
            rep_num = rep_name.split('rep')[1][0]
            out_name = "r" + rep_num + "-bincounts.txt"

            subprocess.call(["featureCounts", rep_name, "-a", saf_file, "-O", "-F", "SAF", "-o", out_name]) 
            
    histone_cols = {}
    dict_of_counts = {}
    
#     dict_of_counts['Chr'] = []
#     dict_of_counts['Start'] = []
#     dict_of_counts['End'] = []
    same_data = {}
    same_data['Chr'] = []
    same_data['Start'] = []
    same_data['End'] = []
    

    counter = 0
    for histone_folder in glob.glob(histone_path + '/*/'):
        os.chdir(histone_folder)
        histone = os.getcwd().split('/')[-1]
        histone_cols[histone] = []
        for rep in glob.glob(histone_folder + "/*bincounts.txt"):
            rep_name = os.path.basename(rep)
            rep_num = rep_name.split('r')[1][0]
            col_name = histone + '-rep' + rep_num
            histone_cols[histone].append(col_name)

            dict_of_counts[col_name] = []

            with open(rep_name, "r") as infile:
                next(infile)
                next(infile)
                for lines in infile:
                    lines = lines.strip().split("\t")
               
                    dict_of_counts[col_name].append(int(lines[6]))

                    if counter == 0:
                        same_data['Chr'].append(lines[1])
                        same_data['Start'].append(int(lines[2]))
                        same_data['End'].append(int(lines[3]))
            counter += 1
      
    #dataframe = pd.DataFrame(dict_of_counts)
    dataframe = pd.DataFrame(same_data)
    dataframe = dataframe.assign(**dict_of_counts)

    avg_cols = {}
    for key in histone_cols:
        avg_col_name = key + '_avg'
        avg_cols[avg_col_name] = dataframe[histone_cols[key]].mean(axis=1)

    dataframe = dataframe.assign(**avg_cols)
    
    kept_cols = ['Chr', 'Start', 'End']
    avgs_cols_list = list(avg_cols.keys())
    new_cols = kept_cols + avgs_cols_list
    new_dataframe = dataframe[new_cols]
    new_dataframe.to_csv("../../alltogether_notnormed.txt", sep = '\t', index = False)
