
import subprocess
def compress_index_files(files):
    for file in files:
        file_compress_name = file + '.gz'
        subprocess.call(['./index_file.sh', file, file_compress_name])

def compress_index_histone_file(file):
    file_no_head_name = file.split('.')[0] + '_noheader.txt' 
    subprocess.call(['./remove_header.sh', file, file_no_head_name])
    return file_no_head_name
        

if __name__ == '__main__':
    files = ['true_tss.bed', 'final_bg.bed', 'strict_enhancers.bed']
    files.append(compress_index_histone_file('alltogether_notnormed.txt'))
    compress_index_files(files)
