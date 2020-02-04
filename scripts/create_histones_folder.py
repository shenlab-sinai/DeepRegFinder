"""
This script creates the histones folder
for deepregfinder
"""
import os
import subprocess

################## Edit the following ##################
histone_marks = ['H3K9ac', 'H3K27me3']

cell_line = "K562"
########################################################

os.mkdir("histones")
os.chdir("histones")

for histone_mark in histone_marks:
	os.mkdir(histone_mark)
	os.chdir(histone_mark)
	with open("ENCODExplorer.R", "w") as outfile:
		outfile.write("""
			
library(ENCODExplorer)

encode_df <- get_encode_df()

query_results <- queryEncode(organism = "Homo sapiens",
                             biosample_type = "cell line",
                             assay = "Histone ChIP-seq",
                             file_format = "bam",
                             project = "ENCODE",
                             target = "{}",
                             biosample_name = "{}")

query_results <- query_results[query_results$output_type == "alignments" & query_results$assembly == "GRCh38", ]

downloadEncode(query_results, format = "bam")

	""".format(histone_mark, cell_line))

	subprocess.call(["Rscript", "ENCODExplorer.R"])

	with open("rename_files.sh", "w") as outfile:
		file_counter = 1
		for files in os.listdir(os.getcwd()):
			if files.endswith(".bam"):
				outfile.write("mv {} {}_rep{}.bam\n".format(files, histone_mark, file_counter))
				file_counter += 1

	subprocess.call(["sh", "rename_files.sh"])
	os.chdir("../")

