# system, image, math imports
import sys
import os
import time
import cv2 as cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Pool
from collections import OrderedDict

# machine learning imports
from torchvision import transforms as T
import torchvision.models as torch_models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from pytorch_wavelets import IDWT
from networks.decoders import DepthWaveProgressiveDecoder, SparseDepthWaveProgressiveDecoder
from networks.encoders import *

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
models_to_load = ["encoder", "depth"]

# model needed for pytorch depth detection
class DenseModel(nn.Module):
    def __init__(self, num_layers, output_scales, device="cpu"):
        super(DenseModel, self).__init__()
        device = torch.device("cpu" if device=="cpu" else "cuda")
        self.models = {}
        self.models["encoder"] = ResnetEncoder(num_layers, False)
        self.models["depth"] = DepthWaveProgressiveDecoder(self.models["encoder"].num_ch_enc, scales=output_scales)
        self.models["encoder"].to(device)
        self.models["depth"].to(device)
    
    def forward(self, x):
        features_encoder = self.models["encoder"](x)
        outputs = self.models["depth"](features_encoder)
        return outputs

# model needed for pytorch depth detection
class SparseModel(nn.Module):
    def __init__(self, num_layers, output_scales, sparse_scales, device="cpu"):
        super(SparseModel, self).__init__()
        device = torch.device("cpu" if device=="cpu" else "cuda")
        self.models = {}
        self.models["encoder"] = ResnetEncoder(num_layers, False)
        self.models["depth"] = SparseDepthWaveProgressiveDecoder(self.models["encoder"].num_ch_enc, scales=output_scales)
        self.models["encoder"].to(device)
        self.models["depth"].to(device)
        self.sparse_scales = sparse_scales
    
    def forward(self, x, thresh_ratio):
        features_encoder = self.models["encoder"](x)
        outputs = self.models["depth"](features_encoder, thresh_ratio, self.sparse_scales)
        return outputs


def establish_encoder_decoder_params():
    # Encoder Parameters
    num_layers = 50

    # Decoder Parameters
    output_scales = [0, 1, 2, 3]
    sparse_scales = [0, 1, 2, 3]

    device = "cpu"
    #device = "cuda"
    dense_model = DenseModel(num_layers, output_scales, device=device)
    dense_model.eval()
    sparse_model = SparseModel(num_layers, output_scales, sparse_scales, device=device)
    sparse_model.eval()

    return dense_model, sparse_model

# depth prediction conversion for a sigmoiud function
def disp_to_depth(disp, min_depth, max_depth):
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


# load ML models
def load_model(model, load_weights_folder):
    load_weights_folder = os.path.expanduser(load_weights_folder)
    assert os.path.isdir(load_weights_folder), \
        "Cannot find folder {}".format(load_weights_folder)
    print("loading model from folder {}".format(load_weights_folder))
    for n in models_to_load:
        print("Loading {} weights...".format(n))
        path = os.path.join(load_weights_folder, "{}.pth".format(n))
        model_dict = model.models[n].state_dict()
        pretrained_dict = torch.load(path, map_location={"cuda:0": "cpu"})
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.models[n].load_state_dict(model_dict)

# load model weights
def load_model_weights(dense_model, sparse_model):
    model_path = "HR_Res50"
    print("Loading weights for Dense model")
    load_model(dense_model, model_path)
    dense_model.models["encoder"].eval()
    print("Done")
    print("Loading weights for Sparse model")
    load_model(sparse_model, model_path)
    sparse_model.models["encoder"].eval()
    print("Done")

# converting images to tensor for ML processing
def to_torch(img):
    to_tensor = T.ToTensor()
    #resize = T.Resize((320, 1024), interpolation=T.InterpolationMode.BILINEAR, antialias=True)
    resize = T.Resize((640, 2048), interpolation=T.InterpolationMode.BILINEAR, antialias=True)
    img_tensor = to_tensor(resize(img)).unsqueeze(0)
    #img_tensor = to_tensor(img).unsqueeze(0)
    return img_tensor

# this creates the actual depth images
def calculate_depth_outputs(s_model, img_tensor):
    threshold = 0.05
    with torch.no_grad():
        sparse_outputs = s_model(img_tensor, thresh_ratio=threshold)
    return sparse_outputs

# convert the depth image model to an image save it
def visualize_sparse_outputs(sparse_outputs, output_path):
    # get the image as decimal
    image = sparse_outputs[('disp', 0)][0,0].numpy()/100
    # plot the image
    fig = plt.figure(figsize=(20.48, 6.4), dpi=100)
    
    # turn it to a heatmap
    plt.imshow(image, cmap="inferno")
    # hide what we don't care about and save it to our path
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight',pad_inches = 0)

def process_and_save_depth_image(input_image, output_image_path, sparse_model):
    # turn input image into a tensor
    image = Image.fromarray(input_image)
    img_tensor = to_torch(image)
    
    # calculate the depth image
    print("building " + output_image_path)
    sparse_outputs = calculate_depth_outputs(sparse_model, img_tensor)
    print(output_image_path + " finished")
    # save the depth image
    visualize_sparse_outputs(sparse_outputs, output_image_path)

def process_video(video):
    # input setup
    video_folder = "datasets/"
    # setup the tensor models that handle the depth processing
    dense_model, sparse_model = establish_encoder_decoder_params()
    load_model_weights(dense_model, sparse_model)

    # read through the .mp4 frames and calculate depth images
    video_file = video_folder + video + ".mp4"
    capture = cv2.VideoCapture(video_file)
    frame_count = 0
    video_list = []
    while capture.isOpened():
        ret, frame = capture.read()

        # build the output path
        output_path = video_folder + video + "/" + str(frame_count) + ".png"
        # find and save the depth image
        process_and_save_depth_image(frame, output_path, sparse_model)
        # read the file and save it to our video list
        video_list.append(cv2.imread(output_path))
        # increase frame count so we can build our output correctly
        frame_count += 1
    depth_video_name = video_folder + video + "/" + video + "_depth_video.mp4"
    depth_video = cv2.VideoWriter(depth_video_name, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, (2048, 640))
    for i in range(len(video_list)):
        depth_video.write(video_list[i])
    # attempting to create a video of the depth images, i think this doesn't work right though
    depth_video.release()


if __name__ == '__main__':
    input_videos = ["good_path", "floor", "right_wall"]
    
    # multiprocessing of the videos
    pool = multiprocessing.Pool(3)
    zip(pool.map(process_video, input_videos))
