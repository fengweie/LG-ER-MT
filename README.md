# LG-ER-MT
Pytorch implementation of our MICCAI 2020 paper "Local and Global Structure-Aware Entropy Regularized Mean Teacher Model for 3D Left Atrium Segmentation."

## Paper
[Local and Global Structure-Aware Entropy Regularized Mean Teacher Model for 3D Left Atrium Segmentation.](https://link.springer.com/chapter/10.1007/978-3-030-59710-8_55) <br />
Proceedings of the 2020 Conference on Medical Image Computing and Computer Assisted Intervention Early Access

Please cite our paper if you find it useful for your research.

```
@inproceedings{hang2020local,
  title={Local and global structure-aware entropy regularized mean teacher model for 3d left atrium segmentation},
  author={Hang, Wenlong and Feng, Wei and Liang, Shuang and Yu, Lequan and Wang, Qiong and Choi, Kup-Sze and Qin, Jing},
  booktitle={Medical Image Computing and Computer Assisted Intervention--MICCAI 2020: 23rd International Conference, Lima, Peru, October 4--8, 2020, Proceedings, Part I 23},
  pages={562--571},
  year={2020},
  organization={Springer}
}
```


### Dependencies
This code requires the following
* Python 3.6
* Pytorch 1.3.0
### Usage

1. Clone the repository:

   ```shell
   git clone https://github.com/3DMRIs/LG-ER-MT.git
   cd LG-ER-MT
   ```
2. Download dataset
   ```shell
   Please download Atrial Segmentation Challenge dataset (https://atriaseg2018.cardiacatlas.org/)
   Put the data in `data/2018LA_Seg_TrainingSet`.
    ```
3. Train model:
 
   ```shell
   cd code
   python train_LA_mt_ce_alldata_struct.py --gpu 0
   ```
3. Test model:
 
   ```shell
   cd code
   python test_LA.py --gpu 0
   ```

