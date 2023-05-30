ffmpeg -i sample2_train.mp4 -ss 00:00:0.0 -t 2 -an sample2_train_2s

ffmpeg -r 30 -i experiments/human_nerf/wild/treadmill1_new_train/combo_mask/latest/freeview_0/%06d.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p out.mp4^C
