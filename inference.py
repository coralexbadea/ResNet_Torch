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
from PIL import Image  # A module for opening, manipulating, and saving many different image file formats
from torch import nn  # Import the neural network module from PyTorch
from torchvision.transforms import Resize, ConvertImageDtype, Normalize  # Import specific transforms from torchvision

import imgproc  # Import the imgproc module (assuming it's a local module dealing with image processing)
import model  # Import the model module (assuming it's a local module defining the model architectures)

from utils import load_state_dict  # Import load_state_dict function from utils module (assuming it's for loading a model's state dict)

# Get a sorted list of model names from the model module, excluding private attributes/methods
model_names = sorted(
    name for name in model.__dict__ if name.islower() and not name.startswith("__") and callable(model.__dict__[name]))


def load_class_label(class_label_file: str, num_classes: int) -> list:
    # Load class label data from a JSON file
    class_label = json.load(open(class_label_file))
    # Create a list of class labels based on the number of classes
    class_label_list = [class_label[str(i)] for i in range(num_classes)]

    return class_label_list


def choice_device(device_type: str) -> torch.device:
    # Select model processing equipment type
    if device_type == "cuda":  # If the device type is specified as "cuda"
        device = torch.device("cuda", 0)  # Set device to use the first CUDA-capable GPU
    else:
        device = torch.device("cpu")  # If not, set device to use the CPU
    return device


def build_model(model_arch_name: str, model_num_classes: int, device: torch.device) -> [nn.Module, nn.Module]:
    # Build the model based on the architecture name
    resnet_model = model.__dict__[model_arch_name](num_classes=model_num_classes)
    # Move the model to the specified device and set the memory format to be channel last
    resnet_model = resnet_model.to(device=device, memory_format=torch.channels_last)

    return resnet_model


def preprocess_image(image_path: str, image_size: int, device: torch.device) -> torch.Tensor:
    # Load an image from a file using OpenCV
    image = cv2.imread(image_path)

    # Convert the color from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert the OpenCV image to a PIL image
    image = Image.fromarray(image)

    # Resize the image to the specified size using torchvision's Resize transform
    image = Resize([image_size, image_size])(image)

    # Convert the PIL image to a PyTorch tensor and add an extra dimension for batch size
    tensor = imgproc.image_to_tensor(image, False, False).unsqueeze_(0)

    # Convert the tensor's data type to float using torchvision's ConvertImageDtype transform
    tensor = ConvertImageDtype(torch.float)(tensor)

    # Normalize the tensor using the mean and standard deviation parameters
    tensor = Normalize(args.model_mean_parameters, args.model_std_parameters)(tensor)

    # Move the tensor to the specified device and set the memory format to be channel last
    tensor = tensor.to(device=device, memory_format=torch.channels_last, non_blocking=True)

    return tensor



def main():
    # Load the class labels
    class_label_map = load_class_label(args.class_label_file, args.model_num_classes)

    # Select device for computation (either "cuda" or "cpu")
    device = choice_device(args.device_type)

    # Initialize the model architecture
    resnet_model = build_model(args.model_arch_name, args.model_num_classes, device)
    print(f"Build `{args.model_arch_name}` model successfully.")

    # Load pre-trained model weights
    resnet_model, _, _, _, _, _ = load_state_dict(resnet_model, args.model_weights_path)
    print(f"Load `{args.model_arch_name}` model weights `{os.path.abspath(args.model_weights_path)}` successfully.")

    # Set the model in evaluation mode
    resnet_model.eval()

    # Preprocess the input image
    tensor = preprocess_image(args.image_path, args.image_size, device)

    # Inference
    with torch.no_grad():
        # Forward pass through the model
        output = resnet_model(tensor)

    # Get the top 5 predicted classes
    prediction_class_index = torch.topk(output, k=5).indices.squeeze(0).tolist()

    # Print the prediction results
    for class_index in prediction_class_index:
        # Get the predicted class label
        prediction_class_label = class_label_map[class_index]
        # Calculate the probability of the predicted class
        prediction_class_prob = torch.softmax(output, dim=1)[0, class_index].item()
        # Print the predicted class and its probability
        print(f"{prediction_class_label:<75} ({prediction_class_prob * 100:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_arch_name", type=str, default="resnet18")  # Model architecture name
    parser.add_argument("--model_mean_parameters", type=list, default=[0.485, 0.456, 0.406])  # Mean values for image normalization
    parser.add_argument("--model_std_parameters", type=list, default=[0.229, 0.224, 0.225])  # Std deviation values for image normalization
    parser.add_argument("--class_label_file", type=str, default="./data/ImageNet_1K_labels_map.txt")  # File path for class labels
    parser.add_argument("--model_num_classes", type=int, default=1000)  # Number of output classes for the model
    parser.add_argument("--model_weights_path", type=str, default="./results/pretrained_models/ResNet18-ImageNet_1K-57bb63e.pth.tar")  # Path to the pre-trained model weights
    parser.add_argument("--image_path", type=str, default="./figure/n01440764_36.JPEG")  # Path of the image to be classified
    parser.add_argument("--image_size", type=int, default=224)  # Input image size for the model
    parser.add_argument("--device_type", type=str, default="cpu", choices=["cpu", "cuda"])  # Device type for computation
    args = parser.parse_args()

    main()

