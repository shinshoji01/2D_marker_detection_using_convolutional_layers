import warnings
warnings.filterwarnings("ignore")
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import torch.nn as nn
import torch

from util import *

def main():
    
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("representation_path", type=str, help="path of representation")
    parser.add_argument("--scale_min", help="minimum of the scale of 1 bit", type=int, default=19)
    parser.add_argument("--scale_max", help="maximim of the scale of 1 bit", type=int, default=21)
    parser.add_argument("--angle_range", help="absolute value of the maximum angle", type=int, default=2)
    parser.add_argument("--noise", help="noise ratio", type=float, default=0.08)
    parser.add_argument("--size", help="size of the whole representation", type=int, default=2000)
    parser.add_argument("--num_per_line", help="# of samples in each row", type=int, default=4)
    
    args = parser.parse_args()
    
    
    representation = np.ones((args.size, args.size))*255
    step = int(args.size/args.num_per_line)
    for i in range(args.num_per_line):
        for j in range(args.num_per_line):
            aruco_image = aruco.drawMarker(dictionary, 4*i+j, 8*np.random.randint(args.scale_min, args.scale_max+1))
            aruco_image = add_frame(aruco_image)

            if i==0 and j==0:
                true_angle_list = [np.random.randint(0, args.angle_range*2+1)-args.angle_range+np.random.randint(4)*90]
            else:
                true_angle_list.append(np.random.randint(0, args.angle_range*2+1)-args.angle_range+np.random.randint(4)*90)

            aruco_image = rotate(aruco_image, true_angle_list[-1], fill=255.0)

            start = int((step-aruco_image.shape[0])/2)
            end = int((step-aruco_image.shape[0])/2)+aruco_image.shape[0]
            representation[start+step*i:end+step*i,start+step*j:end+step*j] = aruco_image
    representation = representation - np.random.randint(0, int(args.noise*255), (args.size,args.size))
    
    img = image_from_output(representation.reshape(1, 1, representation.shape[0], representation.shape[1]))[0]
    img.save(args.representation_path, format="png")
    print(f"representation saved: {args.representation_path}")
    
    
    
if __name__ == '__main__':
    main()