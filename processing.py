# Shengting Cao 
# Generate the traiing data for HumanNeRF training
# inclue image segmentation and romp shape parameter estimation 

import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
import time
import argparse
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import numpy as np
import shutil
import joblib
MIN_NUM_FRAMES = 25
import romp
print(romp.__file__)
import json

# Function to create segmentation mask
def directMask(img,model,preprocess,people_class,blur):
    # Scale input frame
	frame_data = torch.FloatTensor( img ) / 255.0
	input_tensor = preprocess(frame_data.permute(2, 0, 1))
    
    # Create mini-batch to be used by the model
	input_batch = input_tensor.unsqueeze(0)

    # Use GPU if supported, for better performance
	if torch.cuda.is_available():
		input_batch = input_batch.to('cuda')

	with torch.no_grad():
		output = model(input_batch)['out'][0]

	segmentation = output.argmax(0)

	bgOut = output[0:1][:][:]
	a = (1.0 - F.relu(torch.tanh(bgOut * 0.30 - 1.0))).pow(0.5) * 2.0

	people = segmentation.eq(torch.ones_like(segmentation).long().fill_(people_class) ).float()

	people.unsqueeze_(0).unsqueeze_(0)
	
	for i in range(3):
		people = F.conv2d(people, blur, stride=1, padding=1)

	# Activation function to combine masks - F.hardtanh(a * b)
	combined_mask = F.relu(F.hardtanh(a * (people.squeeze().pow(1.5)) ))
	combined_mask = combined_mask.expand(1, 3, -1, -1)

	res = (combined_mask * 255.0).cpu().squeeze().byte().permute(1, 2, 0).numpy()

	return res

#romp_mask 

# estimate metadata and save romp masks
def rompEstimation(dataset_dir,imsize,focal_length):
    print("estimating the romp parameters")
    input_path = os.path.join(dataset_dir,'images')
    settings = romp.main.default_settings 
    # settings is just a argparse Namespace. To change it, for instance, you can change mode via
    # settings.mode='video'
    settings.save_video=True
    settings.render_mesh = True
    settings.calc_smpl = True
    settings.show_items ="mesh_only"
    # settings.show = True
    settings.save_path = os.path.join(input_path,'romp')
    os.makedirs(os.path.join(dataset_dir,'romp'),exist_ok= True)
    romp_model = romp.ROMP(settings)
    print(romp_model.renderer)
    # crate humannerf metadata based on the romp estimation
    metadata = {}
    for image_path in sorted(os.listdir(input_path)):
        print(image_path)
        image_path_noextension = image_path.split(".")[0]
        # processing the images 
        outputs = romp_model(cv2.imread(os.path.join(input_path,image_path))) # please note that we take the input image in BGR format (cv2.imread).
        # print(outputs.keys())
        
        # cv2.imshow("image",outputs['rendered_image'])
        cv2.imwrite(os.path.join(dataset_dir,'romp',image_path.split('/')[-1]), outputs['rendered_image'])
        metadata[image_path_noextension] = {}
        # print("pose shape",outputs["smpl_thetas"].shape)
        metadata[image_path_noextension]["poses"] = outputs["smpl_thetas"].tolist()[0]
        metadata[image_path_noextension]["betas"] = outputs["smpl_betas"].tolist()[0]
        # focu length H/2 * 1/(tan(FOV/2)) = 1920/2. * 1./np.tan(np.radians(30)) = 1662.768
 

        metadata[image_path_noextension]["cam_intrinsics"] = [
        [focal_length, 0.0,imsize[0]//2], 
        [0.0, focal_length, imsize[1]//2 ],
        [0.0, 0.0, 1.0]
        ]
        metadata[image_path_noextension]["cam_extrinsics"] = np.eye(4)
        metadata[image_path_noextension]["cam_extrinsics"][:3, 3] = outputs["cam_trans"]
        metadata[image_path_noextension]["cam_extrinsics"] = metadata[image_path_noextension]["cam_extrinsics"].tolist() 
        
        # https://github.com/Arthur151/ROMP/issues/300 cam extrinsic calculation
        key = cv2.waitKey(10)
        if key == 27:
            break
    return metadata


# process image in directory to masks 
def images2masks(input_path, output_path, model, preprocess,people_class,blur):
    print("process images to segementation masks")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    model.eval()
    for image_path in sorted(os.listdir(input_path)):
        print(image_path)
        # processing the images 
        img = cv2.imread(os.path.join(input_path,image_path))
        mask = directMask(img,model,preprocess,people_class,blur)
        # Apply thresholding to convert mask to binary map
        t = time.time()
        ret,thresh = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
        print(time.time()-t)
        cv2.imwrite(os.path.join(output_path,image_path),thresh)
        # Allow early termination with Esc key
        key = cv2.waitKey(10)
        if key == 27:
            break

def video2images(video_path,output_path):
    print("video to images")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    romp.utils.video2frame(video_path, frame_save_dir=output_path)

def combine_mask(mask1_dir,mask2_dir,save_dir):
    for file_name in os.listdir(mask1_dir):
        image1 = cv2.imread(os.path.join(mask1_dir,file_name))
        image2 = cv2.imread(os.path.join(mask2_dir,file_name)) 
        bitwise_or = cv2.bitwise_or(image1, image2)
        # cv2.imshow("image1",image1)
        # cv2.imshow("image2",image2)
        # cv2.imshow("bitwise_or",bitwise_or)
        cv2.imwrite(os.path.join(save_dir,file_name),bitwise_or)
        cv2.waitKey(1)

if __name__ == '__main__':
	# Load pretrained model
    model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)
    # Segment people only for the purpose of human silhouette extraction
    people_class = 15
    # Evaluate model 
    model.eval()
    print("Model is loaded")
    blur = torch.FloatTensor([[[[1.0, 2.0, 1.0],[2.0, 4.0, 2.0],[1.0, 2.0, 1.0]]]]) / 16.0
     # Use GPU if supported, for better performance
    if torch.cuda.is_available():
        model.to('cuda')
        blur = blur.to('cuda')
    # Apply preprocessing (normalization)
    preprocess = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])  

    experiment_id = "treadmill3_train_new"
    # process video into image frames
    dataset_dir = os.path.join("processed_data",experiment_id)
    video_path = os.path.join("videos",f"{experiment_id}.mp4")
    video2images(video_path,os.path.join(dataset_dir,"images"))

    # get direct masks 
    images2masks(os.path.join(dataset_dir,"images"), os.path.join(dataset_dir,"masks"), model, preprocess,people_class,blur)

    # create romp mesh only estimations and meta data
    # imsize = (1920,1080) # horizontal image
    imsize = (1080,1920) # vertical image
    fov = 60   # iphone 50
    H = max(imsize)
    focal_length = H/2. * 1./np.tan(np.radians(fov/2))
    metadata = rompEstimation(dataset_dir,imsize,focal_length)
    with open(os.path.join(dataset_dir,"metadata.json"), "w") as outfile:
        json.dump(metadata, outfile)

    # generate romp mask based on romp imags
    images2masks(os.path.join(dataset_dir,"romp"),os.path.join(dataset_dir,"romp_masks"),model,preprocess,people_class,blur)

    # generate combo mask 
    mask1_dir = os.path.join(dataset_dir,"masks")
    mask2_dir = os.path.join(dataset_dir,"romp_masks")
    os.makedirs(os.path.join(dataset_dir,"combo_masks"),exist_ok=True)
    combine_mask(mask1_dir,mask2_dir,os.path.join(dataset_dir,"combo_masks"))

