# python processing.py
# cd tools/prepare_wild/
# python prepare_dataset.py
# cd ../../

python train.py --cfg configs/human_nerf/wild/monocular/direct.yaml 
python train_romp_mask.py --cfg configs/human_nerf/wild/monocular/romp_mask.yaml 
python train_combo_mask.py --cfg configs/human_nerf/wild/monocular/combo_mask.yaml 