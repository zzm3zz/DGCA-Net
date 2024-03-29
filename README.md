# DGCA-Net: A Dual-axis Generalized Cross Attention and Shape-Aware Network for 2D Medical Image Segmentation
Zengmin Zhang, Yanjun Peng, Xiaomeng Duan, Qingfan Hou, Zhengyu Li

## Environments and Requirements

- Ubuntu 20.04.3 LTS
- GPU 1 NVIDIA RTX 2080Ti 11G
- CUDA version 10.8
- python version 3.6.9
  
## Dataset

- Synapse [link](https://github.com/Beckschen/TransUNet).
- FLARE2023 [link](https://codalab.lisn.upsaclay.fr/competitions/12239#learn_the_details-dataset).
- ACDC [link](https://github.com/Dootmaan/MT-UNet).

## Preprocessing for FLARE2023

A brief description of the preprocessing method

- select samples:
We analyzed data and selected 222 cases containing 13 organ annotations.

- cropping:
We only use the slice which contains interest regions.

- resampling:
[1, 1, 1] 

- intensity cropping:
[-175, 275]


## Training
```
python train.py
```

## Testing
```python
python test.py
```

## References

-[TransUNet](https://github.com/Beckschen/TransUNet)

-[SwinUNet](https://github.com/HuCaoFighting/Swin-Unet)

-[ScaleFormer](https://github.com/ZJUGiveLab/ScaleFormer)



