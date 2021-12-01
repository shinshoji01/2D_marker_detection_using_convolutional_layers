import warnings
warnings.filterwarnings("ignore")
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import torch.nn as nn
import torch
import os

from util import *

def main():
    
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("representation_path", type=str, help="path of representation")
    parser.add_argument("result_dir", type=str, help="dir path of result")
    parser.add_argument("--scale_min", help="minimum of the scale of 1 bit", type=int, default=18)
    parser.add_argument("--scale_max", help="maximim of the scale of 1 bit", type=int, default=22)
    parser.add_argument("--angle_range", help="absolute value of the maximum angle", type=int, default=3)
    parser.add_argument("--basic_local_size", help="the basic filter size of local maximization", type=int, default=64)
    parser.add_argument("--step", help="# of stride in Convolutional Layer for filtering, the smaller it is, the more computational cost is required", type=int, default=2)
    parser.add_argument("--device", help="device you want use of filtering", type=str, default="cuda")
    
    args = parser.parse_args()
    
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)
    
    examples = []
    for i in range(50):
        aruco_image = aruco.drawMarker(dictionary, i, 8).reshape(1,8,8)
        if i==0:
            examples = aruco_image
        else:
            examples = np.concatenate([examples, aruco_image], axis=0)
    
    image = Image.open(args.representation_path)
    representation = np.asarray(image, dtype=np.float)[:,:,0]
    
    angle_scale = []
    for scale in range(args.scale_min, args.scale_max+1):
        for angle in range(-args.angle_range, args.angle_range+1):
            loc = get_loc(representation, scale, angle, step=args.step, device=args.device)
            if scale==args.scale_min and angle==-args.angle_range:
                loc_list = loc
            else:
                loc_list = np.concatenate([loc_list, loc], axis=1)
            angle_scale.append([scale, angle])
    
    local_size = int(args.basic_local_size/args.step)
    stride = int(args.basic_local_size/2/args.step)
    
    location = np.max(loc_list[0], axis=0)
    image_list = []
    for i in range(int(location.shape[-1]/stride)-1):
        for j in range(int(location.shape[-1]/stride)-1):
            local_loc = location[stride*i:stride*i+local_size, stride*j:stride*j+local_size]
            location[stride*i:stride*i+local_size, stride*j:stride*j+local_size] = local_maximization(local_loc)
    
    xy = np.arange(location.reshape(-1).shape[0])[(location>0.60).reshape(-1)]
    xy = np.concatenate([(xy//location.shape[-1]).reshape(1,-1), (xy%location.shape[-1]).reshape(1,-1)], axis=0).T
    
    for i in range(xy.shape[0]):
        x = xy[i][0]*args.step+int(args.step/2)
        y = xy[i][1]*args.step+int(args.step/2)
        scale, angle = angle_scale[np.argmax(loc_list[0,:,xy[i][0],xy[i][1]])]
        avepool = nn.AvgPool2d(scale, scale)
        for angle90 in range(4):
            individual = representation[x-int(scale*14/2)+1:x+int(scale*14/2)+1, y-int(scale*14/2)+1:y+int(scale*14/2)+1].copy()
            individual = representation[x-int(scale*14/2)+1:x+int(scale*14/2)+1, y-int(scale*14/2)+1:y+int(scale*14/2)+1].copy()
            individual[individual>128] = 255.0
            individual[individual<128] = 0.0
            individual = rotate(individual, degree=-angle+90*angle90, fill=255)
            target = individual[scale*4:-scale*4, scale*4:-scale*4]
            target = (cuda2numpy(avepool(torch.tensor(target.reshape(1, 1, target.shape[0], target.shape[1]))))[0]>128)*255
            score = np.mean(np.abs(examples[:,1:-1,1:-1] - target), axis=(1,2)).reshape(1, -1)
            if angle90==0:
                score_list = score
            else:
                score_list = np.concatenate([score_list, score], axis=0)

        rotated = np.argmin(score_list)//len(examples)*90
        id = np.argmin(score_list)%len(examples)
        if i==0:
            id_list = [id]
            rotated_list = [360-(rotated-angle)]
        else:
            id_list.append(id)
            rotated_list.append(360-(rotated-angle))

        print(f"id:{id}, {(x, y)}, {rotated_list[-1]} rotated")
    
    save_path = f"{args.result_dir}result_{os.path.basename(args.representation_path).split('.')[0]}.png"
    
    plt.figure(figsize=(30, 30))
    img = image_from_output(representation.reshape(1, 1, representation.shape[0], representation.shape[1]))[0]
    plt.imshow(img)
    for i in range(xy.shape[0]):
        x = xy[i][0]*args.step+int(args.step/2)
        y = xy[i][1]*args.step+int(args.step/2)
        plt.text(y, x, f"                    {(xy[i][0], xy[i][1])}: id-{id_list[i]}: {rotated_list[i]} rotated", bbox=dict(facecolor="red", alpha=0.5))
    plt.savefig(save_path)
    
    print(f"figure saved: {save_path}")
    
if __name__ == '__main__':
    main()