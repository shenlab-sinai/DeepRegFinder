#!/bin/bash

out1="${2}human_coding_clustered_tss.bed"
out2="${2}hcc_no_header.bed"
out3="${2}hcc_sorted.bed"
out4="${2}hcc_sorted_chr.bed"

#Formats to a bed file
awk 'OFS="\t" {print $2, $3, $3}' $1 > $out1

#Removing header line
tail -n +2 $out1 > $out2

#Sorting by chromosomes
sortBed -i $out2 > $out3

#Adding chr 
awk '{print "chr"$0}' $out3 > $out4



