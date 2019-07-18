import subprocess
import os
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
        

