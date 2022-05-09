# 2022-TPAMI-SURE

PyTorch implementation for[ Robust Multi-view Clustering with Incomplete Information](https://ieeexplore.ieee.org/abstract/document/9723577) (TPAMI 2022).

## Requirements

pytorch==1.5.0 

numpy>=1.18.2

scikit-learn>=0.22.2

munkres>=1.1.2

logging>=0.5.1.2

## Datasets

The Scene-15 and Reuters-dim10 datasets are placed in "datasets" folder. Another datasets could be downloaded from [Google cloud](https://drive.google.com/drive/folders/1WFbxX1X_pNJX0bDRkbF577mRrviIcyKe?usp=sharing) or [Baidu cloud](https://pan.baidu.com/s/1NdgRH3k9Pq9SQjrorWSEeg) with password "rqv4".

## Demo

 Train a model with different settings

```bash
# Partially Aligned
python run.py --data 0 --gpu 0 --settings 0 --aligned-prop 0.5 --complete-prop 1.0
# Fully Aligned
python run.py --data 0 --gpu 0 --settings 0 --aligned-prop 1.0 --complete-prop 1.0
# Incomplete
python run.py --data 0 --gpu 0 --settings 1 --aligned-prop 1.0 --complete-prop 0.5
# Complete
python run.py --data 0 --gpu 0 --settings 1 --aligned-prop 1.0 --complete-prop 1.0
# PVP + PSP
python run.py --data 0 --gpu 0 --settings 2 --aligned-prop 0.5 --complete-prop 0.5
```

  - `--data`: choice of datasets.

  - `--gpu`:  which gpu to run.

  - `--settings`: 0-PVP, 1-PSP, 2-Both.

  - `--aligned-prop`: known aligned proportions for training.

  - `--complete-prop`: known complete proportions for training.

    **Parameters**: More parameters and descriptions can be found in the script.

    **Training Log**: The training log will be saved in `log/`

## Citation

If you find our work useful in your research, please consider citing:

```latex
@inproceedings{yang2021MvCLN,
   title={Partially View-aligned Representation Learning with Noise-robust Contrastive Loss},
   author={Mouxing Yang, Yunfan Li, Zhenyu Huang, Zitao Liu, Peng Hu, Xi Peng},
   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
   month={June},
   year={2021}
}
@article{yang2022SURE,
	title={Robust Multi-view Clustering with Incomplete Information},
  	author={Yang, Mouxing and Li, Yunfan and Hu, Peng and Bai, Jinfeng and Lv, Jian Cheng and Peng, Xi},  
	journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},     
 	year={2022},  
}
```

