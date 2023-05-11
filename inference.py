# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import json
import os

import cv2
import torch
from PIL import Image## import from Python Imaging Library (the original library that enabled Python to deal with images) the Image module
from torch import nn## import torch, ML library
from torchvision.transforms import Resize, ConvertImageDtype, Normalize## import a few classes from the torchvision transform module for transforamtions

import imgproc## import module defined in the same project
import model## models is a lightweight framework for mapping Python classes to schema-less databases, according to Google
from utils import load_state_dict## import function to work with dictionaries

model_names = sorted(## function that returns a sorted list of the specified iterable object filtered by the if condition
    name for name in model.__dict__ if name.islower() and not name.startswith("__") and callable(model.__dict__[name]))


def load_class_label(class_label_file: str, num_classes: int) -> list:
    class_label = json.load(open(class_label_file))## open the file given as parameter, load a json from it and assign it to class_label
    class_label_list = [class_label[str(i)] for i in range(num_classes)]## composes a list from the contents of the class_label dictionary

    return class_label_list


def choice_device(device_type: str) -> torch.device:
    # Select model processing equipment type
    if device_type == "cuda":## if the current device is a cuda (CUDA devices are nVidia devices that supports CUDA)
        device = torch.device("cuda", 0)## torch.device enables you to specify the device type responsible to load a tensor into memory, in this case, a cuda device
    else:
        device = torch.device("cpu")## the device is a CPU
    return device


def build_model(model_arch_name: str, model_num_classes: int, device: torch.device) -> [nn.Module, nn.Module]:## class used to build a module
    resnet_model = model.__dict__[model_arch_name](num_classes=model_num_classes)## created an instance of a ResNet model based on the specified architecture name model_arch_name and the number of classes model_num_classes.
    resnet_model = resnet_model.to(device=device, memory_format=torch.channels_last)

    return resnet_model


def preprocess_image(image_path: str, image_size: int, device: torch.device) -> torch.Tensor:
    image = cv2.imread(image_path)## read an image using open cv

    # BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)## convert it from the BGR space to the RGB space

    # OpenCV convert PIL
    image = Image.fromarray(image)## convert from the open cv format to the PIL format

    # Resize to 224
    image = Resize([image_size, image_size])(image)## resize to the sizes given as input parameters
    # Convert image data to pytorch format data
    tensor = imgproc.image_to_tensor(image, False, False).unsqueeze_(0)## convert to the pytorch format using the function defined in a different module
    # Convert a tensor image to the given ``dtype`` and scale the values accordingly
    tensor = ConvertImageDtype(torch.float)(tensor)## convert the image to a floar and scale 
    # Normalize a tensor image with mean and standard deviation.
    tensor = Normalize(args.model_mean_parameters, args.model_std_parameters)(tensor)

    # Transfer tensor channel image format data to CUDA device
    tensor = tensor.to(device=device, memory_format=torch.channels_last, non_blocking=True)## transfer the modified image to the cuda device

    return tensor


def main():
    # Get the label name corresponding to the drawing
    class_label_map = load_class_label(args.class_label_file, args.model_num_classes)## gets a list with the contents of the dictionary in the given file ("./data/ImageNet_1K_labels_map.txt")

    device = choice_device(args.device_type)## select model processing equipment type for the one given as argument

    # Initialize the model
    resnet_model = build_model(args.model_arch_name, args.model_num_classes, device)## build the ResNet model by calling the function defined earlier
    print(f"Build `{args.model_arch_name}` model successfully.")

    # Load model weights
    resnet_model, _, _, _, _, _ = load_state_dict(resnet_model, args.model_weights_path)## load the state dictionary from "./results/pretrained_models/ResNet18-ImageNet_1K-57bb63e.pth.tar"
    print(f"Load `{args.model_arch_name}` model weights `{os.path.abspath(args.model_weights_path)}` successfully.")## prints message to tell that the loading was successfuly completed

    # Start the verification mode of the model.
    resnet_model.eval()## sets ResNet the module in evaluation mode

    tensor = preprocess_image(args.image_path, args.image_size, device)## preprocesses the image from the argument list

    # Inference
    with torch.no_grad():## sets the context of torch and ensures that no gradients will be computed or stored, which can significantly speed up inference and reduce memory usage
        output = resnet_model(tensor)## passes the tensot through the model

    # Calculate the five categories with the highest classification probability
    prediction_class_index = torch.topk(output, k=5).indices.squeeze(0).tolist()## this is used to obtain the top-k predicted class indices from the output tensor

    # Print classification results
    for class_index in prediction_class_index:
        prediction_class_label = class_label_map[class_index]## get the label of each prediction from the class label map
        prediction_class_prob = torch.softmax(output, dim=1)[0, class_index].item()
        print(f"{prediction_class_label:<75} ({prediction_class_prob * 100:.2f}%)")## print the result

# Python has a built-in variable called __name__ that records the name of the currently running module or script
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_arch_name", type=str, default="resnet18")## defined the name of the architecture
    parser.add_argument("--model_mean_parameters", type=list, default=[0.485, 0.456, 0.406])## defines the model mean parameters
    parser.add_argument("--model_std_parameters", type=list, default=[0.229, 0.224, 0.225])## defines the model standars parameters
    parser.add_argument("--class_label_file", type=str, default="./data/ImageNet_1K_labels_map.txt")## defines the file name where to load a dictionary from
    parser.add_argument("--model_num_classes", type=int, default=1000)## defines the number of classes in the model
    parser.add_argument("--model_weights_path", type=str, default="./results/pretrained_models/ResNet18-ImageNet_1K-57bb63e.pth.tar")## defines the path to the weights on the model
    parser.add_argument("--image_path", type=str, default="./figure/n01440764_36.JPEG")## defines the path to the image
    parser.add_argument("--image_size", type=int, default=224)## defines the image size
    parser.add_argument("--device_type", type=str, default="cpu", choices=["cpu", "cuda"])## defines the type of the devide the model is running on
    args = parser.parse_args()

    main()
