# mc-wsj-aligned 
Preprocessing scripts for signal-level evaluation using MC-WSJ-AV

Example usage:
```
from preprocess import prepare_mc_wsj
prepare_mc_wsj(
    datapath="path/to/mc-wsj-av",
    savepath="path/to/output_csv",
    fs=8000, # controls decimation
    vocab="20k", # select subsets
    array=1, # which array to align from
    array_ch=1 # which channel within array to align
)
```
Please note that the aligned data is output to ```<datapath>/<vocab>_aligned_<sample_rate>```.
Information about the mixtures that can be used for writing dataloaders in PyTorch (or similar) is output to a CSV file in ```<savepath>```.

## Cite our work
```
@inproceedings{consept2023,
author = {W. Ravenscroft and S. Goetze and T. Hain},
booktitle = {2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2024)},
title = {Combining conformer and dual-path-transformer networks for single channel noisy reverberant speech separation},
publisher = {Institute of Electrical and Electronics Engineers (IEEE)},
year = {2024}
```
