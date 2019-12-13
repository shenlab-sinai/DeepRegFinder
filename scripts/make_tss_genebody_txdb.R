library(Homo.sapiens)  # package from Bioconductor.
gtf_url <- 'ftp://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_32/gencode.v32.annotation.gtf.gz'
txdb <- makeTxDbFromGFF(gtf_url)

# TSS.
promoter_df <- as.data.frame(promoters(txdb, 0, 1))
promoter_bed <- promoter_df[c('seqnames', 'start', 'end', 'tx_name', 'width', 'strand')]
promoter_bed$start <- promoter_bed$start - 1
write.table(promoter_bed, 'gencode_v32_tss.bed', sep = "\t", 
            row.names = F, col.names = F, quote = F)

# Genebody.
gb_df <- as.data.frame(genes(txdb))
gb_bed <- gb_df[c('seqnames', 'start', 'end', 'gene_id', 'width', 'strand')]
gb_bed$start <- gb_bed$start - 1
write.table(gb_bed, 'gencode_v32_genebody.bed', sep = "\t", 
            row.names = F, col.names = F, quote = F)

