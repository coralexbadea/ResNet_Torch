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
import random## import module to be able to generate random numbers
from typing import Any## import module that provides runtime support for type hints
from torch import Tensor## import the Tensor class from torch module
from numpy import ndarray
import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F_vision## import the functional module from torchvision.transforms and alias it to F_vision

__all__ = [
    "image_to_tensor", "tensor_to_image",
    "center_crop", "random_crop", "random_rotate", "random_vertically_flip", "random_horizontally_flip",
]


def image_to_tensor(image: np.ndarray, range_norm: bool, half: bool) -> torch.Tensor:## defining a function that converts the image data type to the tensor data type
    """Convert the image data type to the Tensor (NCWH) data type supported by PyTorch

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
    tensor = F_vision.to_tensor(image)## calls the function from F_vision that transforms an image to a tensor

    # Scale the image data from [0, 1] to [-1, 1]
    if range_norm:
        tensor = tensor.mul(2.0).sub(1.0)## use operations defined in tensor module to scale the tensor

    # Convert torch.float32 image data type to torch.half image data type
    if half:
        tensor = tensor.half()## convert from the tensor with floats to a tensor with torch.halp data types

    return tensor## returns the processed tensor


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
    if range_norm:## if range_norm is true...
        tensor = tensor.add(1.0).div(2.0)## scale the image data in the tensor

    # Convert torch.float32 image data type to torch.half image data type
    if half:## if half variable is true...
        tensor = tensor.half()## convert the tnesor into half image data type

    image = tensor.squeeze(0).permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype("uint8")## performs some operations on the tensor, converts it to a numpy array and then converts it to an arrat of unsigned 8 bit integers

    return image## return the image with all the previously applied operations and transformations


def center_crop(## defines a function meant to crop an image given as parameter
        images: ndarray | Tensor | list[ndarray] | list[Tensor],## image input parameter which can be of multiple data types
        patch_size: int,## the patch size we want to crop to
) -> [ndarray] or [Tensor] or [list[ndarray]] or [list[Tensor]]:## possible return types of the function
    if not isinstance(images, list):## if the image provided is not in the format of a list...
        images = [images]## transform it in a list with one element

    # Detect input image data type
    input_type = "Tensor" if torch.is_tensor(images[0]) else "Numpy"## find the data type of the input image

    if input_type == "Tensor":## if the input type is a tensor
        image_height, image_width = images[0].size()[-2:]## get the size of the image
    else:## if it is not a tensor, but a numpy type
        image_height, image_width = images[0].shape[0:2]## get the size of the image

    # Calculate the start indices of the crop
    top = (image_height - patch_size) // 2## calculate where the top of the cropped image should be (// floor division = rounds the result down to the nearest whole number)
    left = (image_width - patch_size) // 2## calculate where the left of the cropped image should be

    # Crop lr image patch
    if input_type == "Tensor":## if we are working on a tensor
        # generates a new list of image patches by extracting rectangular subregions from each image in the original images list
        # the top and left variables define the starting position of the patch, and patch_size determines the size of the patch
        images = [image[## creates a new list
                  :,## select all rows of the image
                  :,## select all columns of the image
                  top:top + patch_size,## selects a range of rows from top to top + patch_size
                  left:left + patch_size] for image in images]## iterate through all the images in the images variable and does this for each one
    else:
        images = [image[## creates a new list
                  top:top + patch_size,## selects a range of rows from top to top + patch_size
                  left:left + patch_size,## selects a range of cols from left to left + patch_size
                  ...] for image in images]## iterate through all the images in the images variable and does this for each one

    # When image number is 1
    if len(images) == 1:## if there is only one image in the resulting array
        images = images[0]## images becomes just one object instead of a list

    return images## return the collected images (or image)


def random_crop(
        images: ndarray | Tensor | list[ndarray] | list[Tensor],## image input parameter which can be of multiple data types
        patch_size: int,## the patch size we want to crop to
) -> [ndarray] or [Tensor] or [list[ndarray]] or [list[Tensor]]:## possible return types of the function
    if not isinstance(images, list):
        images = [images]

    # Detect input image data type
    input_type = "Tensor" if torch.is_tensor(images[0]) else "Numpy"## find the data type of the input image

    if input_type == "Tensor":
        image_height, image_width = images[0].size()[-2:]## get the size of the image
    else:
        image_height, image_width = images[0].shape[0:2]## get the size of the image

    # Just need to find the top and left coordinates of the image
    top = random.randint(0, image_height - patch_size)## calculate where the top of the cropped image should be, this time using a random value
    left = random.randint(0, image_width - patch_size)## calculate where the left of the cropped image should be, this time using a random value

    # Crop lr image patch
    if input_type == "Tensor":## if we are working on a tensor
        images = [image[## creates a new list
                  :,## select all rows of the image
                  :,## select all cols of the image
                  top:top + patch_size,## selects a range of rows from top to top + patch_size
                  left:left + patch_size] for image in images]## iterate through all the images in the images variable and does this for each one
    else:
        images = [image[## creates a new list
                  top:top + patch_size,## selects a range of rows from top to top + patch_size
                  left:left + patch_size,## selects a range of rows from left to left + patch_size
                  ...] for image in images]## iterate through all the images in the images variable and does this for each one

    # When image number is 1
    if len(images) == 1:## if there is only one image in the resulting array
        images = images[0]## images becomes just one object instead of a list

    return images


def random_rotate(## defines a function that rotates an image
        images: ndarray | Tensor | list[ndarray] | list[Tensor],## image input parameter which can be of multiple data types
        angles: list,## list of angles to rotate the image
        center: tuple = None,
        rotate_scale_factor: float = 1.0## factor to scale the image while rotating, is initialisez with 1.0 so can be excluded when calling the funcion
) -> [ndarray] or [Tensor] or [list[ndarray]] or [list[Tensor]]:## possible return types of the function
    # Random select specific angle
    angle = random.choice(angles)## randomly select an angle from the list of angles

    if not isinstance(images, list):## if the image provided is not in the format of a list...
        images = [images]## transform it into a list with one element

    # Detect input image data type
    input_type = "Tensor" if torch.is_tensor(images[0]) else "Numpy"## find the data type of the input image

    if input_type == "Tensor":## if we are working on a tensor
        image_height, image_width = images[0].size()[-2:]## get the size of the image based on image data type
    else:
        image_height, image_width = images[0].shape[0:2]## get the size of the image based on image data type

    # Rotate LR image
    if center is None:## if a center of the image was not provided to the function...
        center = (image_width // 2, image_height // 2)## calculate the center

    matrix = cv2.getRotationMatrix2D(center, angle, rotate_scale_factor)## generate the transformation matrix that will rotate the image

    if input_type == "Tensor":## if we are working on a tensor
        images = [F_vision.rotate(image, angle, center=center) for image in images]## rotate all the images in the input list using therotate function from F_vision
    else:## else, if it is a numpy type
        # an affine transformation is any transformation that preserves collinearity, parallelism as well as the ratio of distances between the points
        images = [cv2.warpAffine(image, matrix, (image_width, image_height)) for image in images]## applies an affine transformation (defined by the rotation matrix earlier) on all the images in the input list

    # When image number is 1
    if len(images) == 1:## if there is only one image in the resulting array
        images = images[0]## images becomes just one object instead of a list

    return images


def random_horizontally_flip(## defines a function that flips an image horizontally with a random probability
        images: ndarray | Tensor | list[ndarray] | list[Tensor],## image input parameter which can be of multiple data types
        p: float = 0.5## (optional param) probability of the flip
) -> [ndarray] or [Tensor] or [list[ndarray]] or [list[Tensor]]:## possible return types of the function
    # Get horizontal flip probability
    flip_prob = random.random()## defines the random probability

    if not isinstance(images, list):## if the image provided is not in the format of a list...
        images = [images]## transform it into a list with one element

    # Detect input image data type
    input_type = "Tensor" if torch.is_tensor(images[0]) else "Numpy"## find the data type of the input image

    if flip_prob > p:## if the randomly generated probability os greater then the one given as parameter
        if input_type == "Tensor":## if the input data type is tensor
            images = [F_vision.hflip(image) for image in images]## flip the image using the flip function for tensors
        else:
            images = [cv2.flip(image, 1) for image in images]

    # When image number is 1
    if len(images) == 1:
        images = images[0]

    return images


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
