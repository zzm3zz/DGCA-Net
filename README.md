# DGCA-Net: A Dual-axis Generalized Cross Attention and Shape-Aware Network for Medical Image Segmentation
Zengmin Zhang Yanjun Peng

## Environments and Requirements

- Ubuntu 20.04.3 LTS
- GPU 1 NVIDIA RTX 2080Ti 11G
- CUDA version 10.8
- python version 3.6.9
  
## Dataset

- Synapse [link](https://github.com/Beckschen/TransUNet).
- FLARE2023 [link](https://codalab.lisn.upsaclay.fr/competitions/12239#learn_the_details-dataset).(You need to join this challenge.)
- ACDC [link](https://github.com/Beckschen/TransUNet).

## Preprocessing for FLARE2023

A brief description of the preprocessing method

- select samples
We analyzed data and selected 222 cases containing 13 organ annotations.

- cropping:
We only use the slice which contains interest regions.

- resampling:
[1, 1, 1] 

- intensity cropping:
[-175, 275]


### Preprocessing for boundary ground truth
```python
python makeboundry.py
```

## Training
```
python train_synapse.py
```

## Testing
```python
python test_synapse.py
```






