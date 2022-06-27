# AFHSD: Anchor-Free Hierarchical Saliency-based Detector for Macro- and Micro-expression Spotting in Long Videos

## Updates
- (June 25, 2022) Release training and inference code.

## Summary
- 1) An improved AFSD model by applying hierarchical feature
learning strategy has been proposed to implement MaE and ME
spotting task. This model efficiently utilizes the apex information
of facial expression and enjoys not only less hyper-parameter to
tune but also better performance in actual experiments.
- 2) We adopt “optical flow feature extraction-ROI selection”
strategy to eliminate the influence caused by pose swing and merge
overlapping intervals into an optimal interval based on weighted
relative offset to solve the problems of large output and large
F. Surname et al. adjustment of the position and size corresponding to different intervals.

## Getting Started

### Environment
- Python 3.7
- PyTorch == 1.4.0 **(Please make sure your pytorch version is 1.4)**
- NVIDIA GPU

### Setup
```shell script
pip3 install -r requirements.txt
python3 setup.py develop
```
### How to run the code

```shell script
# Data Preparation
python AFHSD/data/densflow.py
python AFHSD/data/gen_datasets.py

# train
python AFSD/train.py configs/config_flow.yaml

# test
python AFSD/test.py
```

## Citation
If you find this project useful for your research, please use the following BibTeX entry.
```
@InProceedings{MEGC_2022,
    author    = {Sai, Yang},
    title     = {Anchor-Free Hierarchical Saliency-based Detector for Macro- and Micro-expression Spotting in Long Videos},
    month     = {June},
    year      = {2022},
    email     = {yangsai@wxsbank.com}
}
```
