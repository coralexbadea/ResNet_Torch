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
import random  # Imports the random module, used for generating random numbers
from typing import Any  # Imports 'Any' from the 'typing' module for flexible type hinting
from torch import Tensor  # Imports the 'Tensor' data structure from the 'torch' library
from numpy import ndarray
import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F_vision  # Imports the 'functional' module from 'torchvision.transforms', and aliases it as 'F_vision' for easy access


__all__ = [
    "image_to_tensor", "tensor_to_image",
    "center_crop", "random_crop", "random_rotate", "random_vertically_flip", "random_horizontally_flip",
]


def image_to_tensor(image: np.ndarray, range_norm: bool, half: bool) -> torch.Tensor:  
    # Function to convert an image (numpy array) to a PyTorch tensor and apply optional transformations

    """
    Convert the image data type to the Tensor (NCHW) data type supported by PyTorch

    Args:
        image (np.ndarray): The image data read by ``OpenCV.imread``, the data range is [0,255] or [0, 1]
        range_norm (bool): Scale [0, 1] data to between [-1, 1]
        half (bool): Whether to convert torch.float32 similarly to torch.half type

    Returns:
        tensor (torch.Tensor): Data types supported by PyTorch

    Examples:
        >>> example_image = cv2.imread("example_image.bmp")
        >>> example_tensor = image_to_tensor(example_image, False, False)

    """
    # Convert image data type to Tensor data type
    tensor = F_vision.to_tensor(image)  # Converts a numpy image to PyTorch tensor
    
    # Scale the image data from [0, 1] to [-1, 1]
    if range_norm:
        tensor = tensor.mul(2.0).sub(1.0)  # Scales the tensor values from [0, 1] to [-1, 1]
    
    # Convert torch.float32 image data type to torch.half image data type
    if half:
        tensor = tensor.half()  # Converts the tensor data type to half-precision float (float16)
    
    return tensor  # Returns the final transformed tensor


def tensor_to_image(tensor: torch.Tensor, range_norm: bool, half: bool) -> Any:
    """Convert the Tensor(NCWH) data type supported by PyTorch to the np.ndarray(WHC) image data type

    Args:
        tensor (torch.Tensor): Data types supported by PyTorch (NCHW), the data range is [0, 1]
        range_norm (bool): Scale [-1, 1] data to between [0, 1]
        half (bool): Whether to convert torch.float32 similarly to torch.half type.

    Returns:
        image (np.ndarray): Data types supported by PIL or OpenCV

    Examples:
        >>> example_tensor = torch.randn([1,3, 256, 256], dtype=torch.float)
        >>> example_image = tensor_to_image(example_tensor, False, False)

    """
    # Scale the image data from [-1, 1] to [0, 1]
    if range_norm:  
        tensor = tensor.add(1.0).div(2.0)  # Scales the tensor values from [-1, 1] to [0, 1]
    
    # Convert torch.float32 image data type to torch.half image data type
    if half:
        tensor = tensor.half()  # Converts the tensor data type to half-precision float (float16)
    
    # Convert Tensor to numpy array and perform necessary transformations
    image = tensor.squeeze(0).permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype("uint8")  # Transforms the tensor to an image format

    return image  # Returns the final transformed image


def center_crop(
        images: ndarray | Tensor | list[ndarray] | list[Tensor],  # The input images to be cropped
        patch_size: int,  # The desired size of the cropped images
) -> [ndarray] or [Tensor] or [list[ndarray]] or [list[Tensor]]:  # The type of the returned cropped images
    if not isinstance(images, list):  # Check if the images are in a list
        images = [images]  # If not, wrap them into a list

    # Detect input image data type
    input_type = "Tensor" if torch.is_tensor(images[0]) else "Numpy"  # Check if images are of type Tensor or Numpy

    if input_type == "Tensor":  # For Tensor type images
        image_height, image_width = images[0].size()[-2:]  # Get the height and width of the first image in the list
    else:  # For Numpy type images
        image_height, image_width = images[0].shape[0:2]  # Get the height and width of the first image in the list

    # Calculate the start indices of the crop
    top = (image_height - patch_size) // 2  # Calculate the top index for cropping
    left = (image_width - patch_size) // 2  # Calculate the left index for cropping

    # Crop image patch
    if input_type == "Tensor":  # For Tensor type images
        images = [image[
                  :,  # All channels
                  :,  # All batches
                  top:top + patch_size,  # Height range for cropping
                  left:left + patch_size] for image in images]  # Width range for cropping
    else:
        images = [image[
                  top:top + patch_size,  # Height range for cropping
                  left:left + patch_size,  # Width range for cropping
                  ...] for image in images]  # All channels

    # When image number is 1
    if len(images) == 1:  # If there's only one image in the list
        images = images[0]  # Remove it from the list

    return images  # Return the cropped images


def random_crop(
        images: ndarray | Tensor | list[ndarray] | list[Tensor],  # The input images to be cropped
        patch_size: int,  # The desired size of the cropped images
) -> [ndarray] or [Tensor] or [list[ndarray]] or [list[Tensor]]:  # The type of the returned cropped images
    if not isinstance(images, list):
        images = [images]  # If not a list, wrap the input into a list

    # Detect input image data type
    input_type = "Tensor" if torch.is_tensor(images[0]) else "Numpy"  # Check if images are of type Tensor or Numpy

    if input_type == "Tensor":  # For Tensor type images
        image_height, image_width = images[0].size()[-2:]  # Get the height and width of the first image in the list
    else:  # For Numpy type images
        image_height, image_width = images[0].shape[0:2]  # Get the height and width of the first image in the list

    # Randomly choose the top and left coordinates for cropping
    top = random.randint(0, image_height - patch_size)  # Randomly choose the top index for cropping
    left = random.randint(0, image_width - patch_size)  # Randomly choose the left index for cropping

    # Crop image patch
    if input_type == "Tensor":  # For Tensor type images
        images = [image[
                  :,  # All channels
                  :,  # All batches
                  top:top + patch_size,  # Height range for cropping
                  left:left + patch_size] for image in images]  # Width range for cropping
    else:
        images = [image[
                  top:top + patch_size,  # Height range for cropping
                  left:left + patch_size,  # Width range for cropping
                  ...] for image in images]  # All channels

    # When image number is 1
    if len(images) == 1:  # If there's only one image in the list
        images = images[0]  # Remove it from the list

    return images  # Return the cropped images


def random_rotate(
        images: ndarray | Tensor | list[ndarray] | list[Tensor],  # The input images to be rotated
        angles: list,  # The list of angles one of which will be randomly chosen for rotation
        center: tuple = None,  # The center of rotation. If not provided, it defaults to the center of the image
        rotate_scale_factor: float = 1.0  # Scaling factor for the rotation
) -> [ndarray] or [Tensor] or [list[ndarray]] or [list[Tensor]]:  # The type of the returned rotated images
    # Random select specific angle
    angle = random.choice(angles)  # Choose a random angle from the given list

    if not isinstance(images, list):  # Check if the images are in a list
        images = [images]  # If not, wrap them into a list

    # Detect input image data type
    input_type = "Tensor" if torch.is_tensor(images[0]) else "Numpy"  # Check if images are of type Tensor or Numpy

    if input_type == "Tensor":  # For Tensor type images
        image_height, image_width = images[0].size()[-2:]  # Get the height and width of the first image in the list
    else:  # For Numpy type images
        image_height, image_width = images[0].shape[0:2]  # Get the height and width of the first image in the list

    # Define the center of rotation
    if center is None:  # If no center provided
        center = (image_width // 2, image_height // 2)  # Set the center of the image as the center of rotation

    # Define the rotation matrix
    matrix = cv2.getRotationMatrix2D(center, angle, rotate_scale_factor)  # Compute the affine matrix for rotation

    # Rotate images
    if input_type == "Tensor":  # For Tensor type images
        images = [F_vision.rotate(image, angle, center=center) for image in images]  # Rotate each image in the list using torchvision function
    else:  # For Numpy type images
        images = [cv2.warpAffine(image, matrix, (image_width, image_height)) for image in images]  # Rotate each image in the list using cv2 function

    # When image number is 1
    if len(images) == 1:  # If there's only one image in the list
        images = images[0]  # Remove it from the list

    return images  # Return the rotated images


def random_horizontally_flip(
        images: ndarray | Tensor | list[ndarray] | list[Tensor],  # The input images to be possibly flipped
        p: float = 0.5  # The probability threshold for performing the flip
) -> [ndarray] or [Tensor] or [list[ndarray]] or [list[Tensor]]:  # The type of the returned images, flipped or original
    # Get horizontal flip probability
    flip_prob = random.random()  # Generate a random float between 0.0 and 1.0

    if not isinstance(images, list):  # Check if the images are in a list
        images = [images]  # If not, wrap them into a list

    # Detect input image data type
    input_type = "Tensor" if torch.is_tensor(images[0]) else "Numpy"  # Check if images are of type Tensor or Numpy

    # Perform a horizontal flip if the generated probability is greater than the given threshold
    if flip_prob > p:  
        if input_type == "Tensor":  # For Tensor type images
            images = [F_vision.hflip(image) for image in images]  # Perform a horizontal flip on each image in the list using torchvision function
        else:  # For Numpy type images
            images = [cv2.flip(image, 1) for image in images]  # Perform a horizontal flip on each image in the list using cv2 function

    # When image number is 1
    if len(images) == 1:  # If there's only one image in the list
        images = images[0]  # Remove it from the list

    return images  # Return the flipped or original images


def random_vertically_flip(
        images: ndarray | Tensor | list[ndarray] | list[Tensor],
        p: float = 0.5
) -> [ndarray] or [Tensor] or [list[ndarray]] or [list[Tensor]]:
    # Get vertical flip probability
    flip_prob = random.random()

    if not isinstance(images, list):
        images = [images]

    # Detect input image data type
    input_type = "Tensor" if torch.is_tensor(images[0]) else "Numpy"

    if flip_prob > p:
        if input_type == "Tensor":
            images = [F_vision.vflip(image) for image in images]
        else:
            images = [cv2.flip(image, 0) for image in images]

    # When image number is 1
    if len(images) == 1:
        images = images[0]

    return images
