#!/bin/bash

#Log transforms the last featureCounts column
awk 'OFS="\t" {print $1, $2, $3, $4, $5, $6, log($7 + 1)/log(2)}' $1 > $2