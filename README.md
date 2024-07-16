# Kernel-Bounded-Clustering-for-Spatial-Transcriptomics

# Installation

## python 3.8+
```
    pip install scikit-learn
    pip install h5py
    pip install pandas
    pip install python-igraph
    pip install scanpy
```

## Matlab install the toolbox:
1. Parallel Computing Toolbox
2. Statistics and Machine Learning Toolbox

# Dataset
The datasets in the turtorial are already preprocessed, download it from the following link and unzip it to the current directory tutorial/Dataset

https://drive.google.com/file/d/1xuB_kMPQDfUF__QY9GCpvazO6wAFLrO5/view?usp=drive_link


and the original dataset can be downloaded from the following link.
### DLPFC
http://spatial.libd.org/spatialLIBD/

### HER2st
https://github.com/almaan/her2st

### Slide-seq V2 mouse hippocampus
https://singlecell.broadinstitute.org/single_cell/study/SCP815/sensitive-spatial-genome-wideexpression-profiling-at-cellular-resolution#study-summary

### Stereo-seq mouse olfactory bulb
https://github.com/JinmiaoChenLab/SEDR_analyses


# DLPFC
1. cd /tutorial/wlikbc 
2. Run DLPFC_WL.py
3. Open Matlab and run the file turtorial/KBC/DLPFC_KBC.m

# HER2st
1. cd /tutorial/wlikbc 
2. Run HER2stWL.py
3. Open Matlab and run the file turtorial/KBC/HER2st_KBC.m

# Slide-seq V2 mouse hippocampus
1. cd /tutorial/wlikbc 
2. Run slideseqv2norwl.py
3. Open Matlab and run the file turtorial/KBC/Slideseqv2_KBC.m

# Stereo-seq mouse olfactory bulb
1. cd /tutorial/wlikbc 
2. Run sterseqwl.py
3. Open Matlab and run the file turtorial/KBC/sterseq_KBC.m
4. cd /tutorial/wlikbc and run plot_stereo.py



