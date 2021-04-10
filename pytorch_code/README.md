# STP

## Introduction
This repo contains several models for video action recognition,
including [C3D](http://arxiv.org/pdf/1412.0767), [R2Plus1D](https://arxiv.org/abs/1711.11248v1), [R3D](https://arxiv.org/pdf/1703.07814.pdf), [P3D](http://openaccess.thecvf.com/content_ICCV_2017/papers/Qiu_Learning_Spatio-Temporal_Representation_ICCV_2017_paper.pdf), and [I3D](https://arxiv.org/abs/1705.07750).

## Dependencies

- Python 3.6 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch = 1.1.0](https://pytorch.org/)
- NVIDIA GPU
- Python packages: `pip install numpy opencv-python tensorboardX` 

## Training

   To train the model, please do:
   
    python train.py


## Testing
   To test by a model:
   
    python inference.py


## Reference
- The code is based on [source code](https://github.com/jfzhang95/pytorch-video-recognition).
- Results on UCF101 and HMDB51 is based on [source code](https://github.com/MichiganCOG/ViP).

----------

If you find this work useful, please consider citing it.
```
@article{wang2021stp, 
  title={Spatial-temporal Pooling for Action Recognition in Videos}, 
  author={Jiaming Wang, Zhenfeng Shao, Xiao Huang, Tao Lu, Ruiqian Zhang and Xianwei Lv},
  journal={NEUROCOMPUTING}, 
  year={2021},
  publisher={Elsevier}
}
```