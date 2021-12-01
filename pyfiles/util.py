import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import torch.nn as nn

def min_max(x, axis=None, mean0=False, get_param=False):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min+1e-8)
    if mean0 :
        result = result*2 - 1
    if get_param:
        return result, min, max
    return result

def image_from_output(output):
    image_list = []
    for i in range(output.shape[0]):
        a = output[i]
        a = np.tile(np.transpose(a, axes=(1,2,0)), (1,1,int(3/a.shape[0])))
        a = min_max(a)*2**8 
        a[a>255] = 255
        a = np.uint8(a)
        a = Image.fromarray(a)
        image_list.append(a)
    return image_list

def degree_to_pi(degree):
    return degree/180*np.pi

def get_rotate(degree=40):
    theta = degree_to_pi(degree)
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

def rotate(image, degree=45, index_list=None, fill=1):
    size = image.shape[0]
    each = size // 14
    rot = get_rotate(degree)
    center = (int(size/2), int(size/2))
    
    background = np.ones((size, size))*fill
    value_list = list(set(list(image.reshape(-1))))
    for value in list(set(list(image.reshape(-1)))):
        a = np.arange(size*size)[(image==value).reshape(-1)]
        a = np.concatenate([(a//size).reshape(1,-1), (a%size).reshape(1,-1)], axis=0).T - center
        for index in np.array(np.round(np.dot(rot, a.T).T + center), dtype=np.int):
            try:
                background[index[0], index[1]] = value
            except: IndexError
            
    rotated = background.copy()
    kernel = np.ones((5,5),np.uint8)
    for value in list(set(list(image.reshape(-1))) - set([fill])):
        value_matrix = np.array(background==value, dtype=np.float64)
        closing = cv2.morphologyEx(value_matrix, cv2.MORPH_CLOSE, kernel)
        rotated[closing==closing.max()] = value
    
    return rotated

def add_frame(image):
    each = image.shape[0]//8
    size = each*14
    new_image = np.ones((size, size))*255
    new_image[3*each:-3*each,3*each:-3*each] = image
    return new_image

def get_loc(representation, frame_size, filter_angle, step=1, device="cuda"):
    frame = np.ones((12*frame_size,12*frame_size))*255.0
    frame[2*frame_size:-2*frame_size, 2*frame_size:-2*frame_size] = 0.0
    frame[4*frame_size:-4*frame_size, 4*frame_size:-4*frame_size] = 255.0
    frame = rotate(frame, filter_angle, fill=255.0)
    
    filter = nn.Conv2d(1, 1, frame_size*12-1, step, int((frame_size*12-1)/2)).to(device)
    
    parameters = filter.state_dict()
    bias = torch.zeros(parameters["bias"].shape)
    parameters["bias"] = bias
    weight = np.array(frame==0.0, dtype=np.float64)[:frame_size*12-1,:frame_size*12-1]
    weight = (weight/weight.sum()).reshape(1,1,weight.shape[0],weight.shape[1])
    parameters["weight"] = torch.tensor(weight)
    filter.load_state_dict(parameters)
    
    inputs_inner = torch.Tensor(1-min_max(representation).reshape(1,1,representation.shape[0], representation.shape[1])).to(device)
    loc = cuda2numpy(filter(inputs_inner))

    return loc

def cuda2numpy(x):
    return x.detach().to("cpu").numpy()

def local_maximization(local_loc):
    return (local_loc==local_loc.max())*local_loc.max()
