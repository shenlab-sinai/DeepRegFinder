### Preprocessing
- reduce file size by using gzipped bed and saf. NOT DO
- add #threads for featureCounts. DONE
- parallelize build_histone_tensors. USED MERGE INSTEAD. DONE
- change "enhancer_slopped" to "more_slopped". NOT DO
- use "__all__" to define what to import. DONE
- modify "process_genome" DONE
- require TSS and enhancer to overlap with DHS. DONE
- sample background using a fixed number instead of ratio. DONE
- process histone bin counts without log transform and evaluate performance. (not using log trans is a little better) DONE
- support for non-strand-specific GRO-seq. DONE
- support for lack of TFBS. DONE
- support for 3-way classification without using GRO-seq.
- whereabout of sox2 and CBP in H1.
- support for multiple enhancer markers.

### Training
- add tensorboard for training monitor. DONE
- understand why model outputs became NaNs. DONE
- add learning rate scheduler. DONE
- handle model blowup more gracefully. DONE
- add weight initialization. DONE
- tweak ConvNet: use larger 1st layer kernel size; DONE
- add an RNN model. DONE
- evaluate KimNet. DONE
- add conv-lstm. DONE
- class activation map.
- visualize the 1st conv layer.

### Prediction
- remove known TSSs from predicted enhancers (improved validation rate). DONE



