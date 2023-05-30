# HumanNerf Auto Mask Refinement Experiment scripts

## Prerequesite 

### Hardware and system
This code is tested on Ubuntu 18.04.6 LTS
With 4 RTX6000 GPUs.

### Setup environment 
Install anaconda 

```
conda create -n humannerf_amr python=3.9
conda activate  humanerf_amr 
```
Install latest packages

```
torch
torchvision
numpy
scipy 
opencv-python
pillow
termcolor
pyyaml
tqdm
absl-py
gdown
```
If this is your first time try the SMPL based methods please downlaod SMPL model from https://smplify.is.tue.mpg.de/  unpack mpips_smplify_public_v2.zip

save the smpl model to the path

```
SMPL_DIR=/path/to/smpl
MODEL_DIR=$SMPL_DIR/smplify_public/code/models
cp $MODEL_DIR/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl third_parties/smpl/models
```

### Install modifed version of simple-romp
install modified simple-romp pacakge with modified vis-human. The modified version of simple-romp can output pure human mese without overlay with the original images. Two way to install such modifed package.


- First way: follow "https://github.com/Arthur151/ROMP/blob/master/simple_romp/README.md" to install the simple_romp. When import romp using `import romp` and `print(romp.__file__)` to find the package location and replace the file using the files in "simple_romp_modified"

- Second way: just use the modifed version of simple_romp in this repository. `cd simple_romp_modified` and `./reinstall.sh`. 

## How to reproduce the results
### layout
```
- configs 
- core 
- processed_data
- simple_romp_modified 
- third_parties 
- tools/prepare_wild
- videos 
-- cuvideo.sh #using ffmpeg to cut video
-- processing.py # generating files in processed_data from video
-- run.py # testing 
-- train.py # orginal humannerf method
-- train_combo_mask.py # refined mask
-- train_romp_mask.py # romp mask only 
```
Your videos are in the 'videos' folder



#### Train the model using `bash ./prepare.sh` or use the following command

check the configration after # before training

```python
python processing.py  
# experiment_id must match video name, imsize must match
cd tools/prepare_wild/ 
# the subject name in wild.yaml must match video name
python prepare_dataset.py

python train.py --cfg configs/human_nerf/wild/monocular/direct.yaml 
# subject name in direct yaml must match video name
python train_romp_mask.py --cfg configs/human_nerf/wild/monocular/romp_mask.yaml 
# subject name in romp_mask yaml must match video name
python train_combo_mask.py --cfg configs/human_nerf/wild/monocular/combo_mask.yaml 
# subject name in combo_mask yaml must match video name

```
### Citation
If this repository is helpful for your research please cite.
```
@article{inproceeding,
author = {Cao, Shengting and Zhao, Jiamiao and Hu, Fei and Gan, Yu},
title = {{Metaverse-Oriented Telerehabilitation with Signle-Camera-Based, Avatar-Free Rendering}}
}
```
### Note 
We laverage most of the code design from 
- https://github.com/Arthur151/ROMP
- https://github.com/chungyiweng/humannerf


