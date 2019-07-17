#!/bin/bash

#TODO: Remove formation of so many temp files?

#Formats to a bed file
awk 'OFS="\t" {print $2, $3, $3}' $1 > human_coding_clustered_tss.bed

#Removing header line
tail -n +2 human_coding_clustered_tss.bed > hcc_no_header.bed

#Sorting by chromosomes
sortBed -i hcc_no_header.bed > hcc_sorted.bed

#Adding chr 
awk '{print "chr"$0}' hcc_sorted.bed > hcc_sorted_chr.bed



