# WeedsGalore Dataset :seedling::herb:	

This is the official implementation of the WACV 2025 paper **WeedsGalore: A Multispectral and Multitemporal UAV-based Dataset for Crop and Weed Segmentation in Agricultural Maize Fields.** 
WeedsGalore is a UAV-based multispectral dataset with dense annotations for crop and weed segmentation in maize fields. 
This repository contains code and download links for the dataset and pretrained models. 

[[`arXiv`](https://arxiv.org/abs/2502.13103)], [[`paper`](https://openaccess.thecvf.com/content/WACV2025/html/Celikkan_WeedsGalore_A_Multispectral_and_Multitemporal_UAV-Based_Dataset_for_Crop_and_WACV_2025_paper.html)], [[`dataset`](https://doidata.gfz.de/weedsgalore_e_celikkan_2024/)]

<a href="/img.png" target="_blank">
  <img src="/img.png" alt="WeedsGalore Preview" width="800"/>
</a>

## Dataset
### Download
Follow this [link](https://doidata.gfz.de/weedsgalore_e_celikkan_2024/) to download the dataset. The dataset (`weedsgalore-dataset`, 0.4GB) and full-field orthomosaics (`weedsgalore-orthomosaic`, 12GB, GeoTIFF) can be downloaded separately. 

### Structure

```
weedsgalore-dataset
└── 2023-05-25
    └── images
    └── semantics
    └── instances
    └── logs
└── 2023-05-30
    └── images
    └── ...  
└── ...
└── splits
    └── train.txt
    └── ... 
└── LICENSE.txt
```


### Licence
WeedsGalore dataset is distributed under the [Creative Commons Attribution (CC BY) Licence](https://creativecommons.org/licenses/by/4.0/).
Please refer to the [full licence text](https://doidata.gfz.de/weedsgalore_e_celikkan_2024/) for details. 

## Evaluation

### Requirements
Make sure to have the following installed:

- **Python**:`>= 3.7`
- **PyTorch**:`>= 1.13.0`
- **torchmetrics**: `>= 0.11.0`
- **absl**: `>= 1.3.0`
- **Pillow (PIL)**: `>= 9.0.1`

### Install Packages
Example: Create a conda environment and install dependencies:
```
conda create -n weedsgalore python=3.7.12
conda activate weedsgalore
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install absl-py=1.3.0
conda install pillow=9.0.1
pip install torchmetrics==0.11.4
```

Run the evaluation script, replacing the flags with your paths and parameters:
```
python src/evaluate.py --dataset_path <weedsgalore-dataset_directory> --split test --ckpt <ckpt_directory> --in_channels 5 --num_classes 6
```

Inference with probabilistic model:
```
python src/evaluate_vimc.py --dataset_path <weedsgalore-dataset_directory> --split test --ckpt <ckpt_directory> --in_channels 5 --num_classes 3 --mc_samples=5
```

You can download pretrained models for DeepLabv3+ [here](https://doidata.gfz.de/weedsgalore_e_celikkan_2024/ckpts.zip) (for both MSI and RGB input, uni-weed and multi-weed case, deterministic and probabilistic variants).

## License
This project is licensed under the Apache-2.0 License. See LICENSES folder for details. 
```
   Copyright 2024 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
   Copyright 2024 Ekin Celikkan

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

```


## Citation
If you use the dataset or code, please cite our paper:

```
@InProceedings{Celikkan_2025_WACV,
    author    = {Celikkan, Ekin and Kunzmann, Timo and Yeskaliyev, Yertay and Itzerott, Sibylle and Klein, Nadja and Herold, Martin},
    title     = {WeedsGalore: A Multispectral and Multitemporal UAV-Based Dataset for Crop and Weed Segmentation in Agricultural Maize Fields},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025},
    pages     = {4767-4777}
}
```
