# mc-wsj-aligned
Preprocessing scripts for signal level evaluation using MC-WSJ-AV

Example usage:
```
from preprocess import prepare_mc_wsj_csv
prepare_mc_wsj(
    datapath="path/to/mc-wsj-av,
    savepath="path/to/output_csv,
    fs=8000, # controls decimation
    vocab="20k", # select subsets
    array=1, # which array to align from
    array_ch=1 # which channel within array to align
)
```
