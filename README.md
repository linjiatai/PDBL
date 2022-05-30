# PDBL: Improving Histopathological Tissue Classification with Plug-and-Play Pyramidal Deep-Broad Learning
![outline](PDBL.png)

## Introduction
The code of:

**PDBL: Improving Histopathological Tissue Classification with Plug-and-Play Pyramidal Deep-Broad Learning**, Jiatai Lin, Guoqiang Han, Xipeng Pan, Zaiyi Liu, Hao Chen, Danyi Li, Xiping Jia, Zhenwei Shi, Zhizhen Wang, Yanfen Cui, Haiming Li, Changhong Liang, Li Liang, Ying Wang, Chu Han, in IEEE Transactions on Medical Imaging, doi: 10.1109/TMI.2022.3161787.[[paper]](https://ieeexplore.ieee.org/document/9740140)

We propose a lightweight plug-and-play module (PDBL), which can be easily applied on almost any common CNN-based classification backbone. It can generally improve
all the three CNN backbones in the experiment for histopathological tissue classification with no re-training burden.

## Citation
If you find the code useful, please consider citing our paper using the following BibTeX entry.
```
@ARTICLE{9740140,
  author={Lin, Jiatai and Han, Guoqiang and Pan, Xipeng and Liu, Zaiyi and Chen, Hao and Li, Danyi and Jia, Xiping and Shi, Zhenwei and Wang, Zhizhen and Cui, Yanfen and Li, Haiming and Liang, Changhong and Liang, Li and Wang, Ying and Han, Chu},
  journal={IEEE Transactions on Medical Imaging}, 
  title={PDBL: Improving Histopathological Tissue Classification with Plug-and-Play Pyramidal Deep-Broad Learning}, 
  year={2022},
  pages={1-1},
  doi={10.1109/TMI.2022.3161787}}
```

## Prerequisite
* Tested on Ubuntu 20.04


## Usage
### Run an example experiment without re-training burde of ImageNet pretrained model.
- To run the whole pipeline, you need to specify the path to the saved model for each round. Please see the command in script.sh.
```
bash script.sh
```
