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
import queue
import sys
import threading
from glob import glob

import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets.folder import find_classes## imports find_classes from torchvision.datasets.folder
from torchvision.transforms import TrivialAugmentWide## imports TrivialAugmentWide from torchvision.transforms

import imgproc## imports imgproc

__all__ = [
    "ImageDataset",## a list of strings that are the names of the classes and functions that will be imported when using from <module> import *
    "PrefetchGenerator", "PrefetchDataLoader", "CPUPrefetcher", "CUDAPrefetcher",
]

# Image formats supported by the image processing library
IMG_EXTENSIONS = ("jpg", "jpeg", "png", "ppm", "bmp", "pgm", "tif", "tiff", "webp")## a tuple of strings that are image extensions

# The delimiter is not the same between different platforms
if sys.platform == "win32":## if the platform is win32 
    delimiter = "\\"## delimiter is \
else:## else
    delimiter = "/"## delimiter is /


class ImageDataset(Dataset):## class ImageDataset inherits from Dataset
    """Define training/valid dataset loading methods.

    Args:
        image_dir (str): Train/Valid dataset address.
        image_size (int): Image size.
        mode (str): Data set loading method, the training data set is for data enhancement,
            and the verification data set is not for data enhancement.
    """

    def __init__(self, image_dir: str, image_size: int, mean: list, std: list, mode: str) -> None:
        super(ImageDataset, self).__init__()## initialize the parent class
        # Iterate over all image paths
        self.image_file_paths = glob(f"{image_dir}/*/*")## image_file_paths is a list of strings that are the paths of the images
        # Form image class label pairs by the folder where the image is located
        _, self.class_to_idx = find_classes(image_dir)## find_classes returns a tuple of strings that are the classes and a dictionary that maps the classes to indices
        self.image_size = image_size## image_size is an integer
        self.mode = mode## mode is a string
        self.delimiter = delimiter## delimiter is a string

        if self.mode == "Train":## if mode is Train
            # Use PyTorch's own data enhancement to enlarge and enhance data
            self.pre_transform = transforms.Compose([## self.pre_transform is a Compose object
                transforms.RandomResizedCrop(self.image_size),## transforms.RandomResizedCrop returns a RandomResizedCrop object
                TrivialAugmentWide(),## performs data enhancement
                transforms.RandomRotation([0, 270]),## rotates the image by a random angle between 0 and 270 degrees
                transforms.RandomHorizontalFlip(0.5),## flips the image horizontally with a probability of 0.5
                transforms.RandomVerticalFlip(0.5),## flips the image vertically with a probability of 0.5
            ])
        elif self.mode == "Valid" or self.mode == "Test":## else if mode is Valid or Test
            # Use PyTorch's own data enhancement to enlarge and enhance data
            self.pre_transform = transforms.Compose([## self.pre_transform contains the following transforms
                transforms.Resize(256),## resizes the image to 256x256
                transforms.CenterCrop([self.image_size, self.image_size]),## crops the image to image_size x image_size
            ])
        else:
            raise "Unsupported data read type. Please use `Train` or `Valid` or `Test`"

        self.post_transform = transforms.Compose([## self.post_transform contains the following transforms
            transforms.ConvertImageDtype(torch.float),## converts the image to float
            transforms.Normalize(mean, std)## normalizes the image
        ])

    def __getitem__(self, batch_index: int) -> [torch.Tensor, int]:## returns a list of a tensor and an integer that are the image and the target
        image_dir, image_name = self.image_file_paths[batch_index].split(self.delimiter)[-2:]## splits the path of the image in the directory and the name of the image
        # Read a batch of image data
        if image_name.split(".")[-1].lower() in IMG_EXTENSIONS:## if the image extension is in IMG_EXTENSIONS
            image = cv2.imread(self.image_file_paths[batch_index])## reads the image
            target = self.class_to_idx[image_dir]## target is the index of the class
        else:
            raise ValueError(f"Unsupported image extensions, Only support `{IMG_EXTENSIONS}`, "## raises a ValueError
                             "please check the image file extensions.")

        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)## converts the image from BGR to RGB

        # OpenCV convert PIL
        image = Image.fromarray(image)## converts the image to a PIL image

        # Data preprocess
        image = self.pre_transform(image)## image is transformed by self.pre_transform

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        tensor = imgproc.image_to_tensor(image, False, False)## converts the image to a tensor

        # Data postprocess
        tensor = self.post_transform(tensor)## tensor is transformed by self.post_transform

        return {"image": tensor, "target": target}## returns a dictionary with the image and the target

    def __len__(self) -> int:
        return len(self.image_file_paths)## returns the length of the image_file_paths


class PrefetchGenerator(threading.Thread):
    """A fast data prefetch generator.

    Args:
        generator: Data generator.
        num_data_prefetch_queue (int): How many early data load queues.## undefined
    """

    def __init__(self, generator, num_data_prefetch_queue: int) -> None:## defines the constructor
        threading.Thread.__init__(self)## initializes the parent class
        self.queue = queue.Queue(num_data_prefetch_queue)
        self.generator = generator## sets the generator
        self.daemon = True## sets the daemon to True
        self.start()

    def run(self) -> None:
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __next__(self):## defines the next method
        next_item = self.queue.get()##  gets the next item
        if next_item is None:## if next_item is None
            raise StopIteration## raises a StopIteration
        return next_item

    def __iter__(self):
        return self


class PrefetchDataLoader(DataLoader):## defines the PrefetchDataLoader class
    """A fast data prefetch dataloader.

    Args:
        num_data_prefetch_queue (int): How many early data load queues.
        kwargs (dict): Other extended parameters.
    """

    def __init__(self, num_data_prefetch_queue: int, **kwargs) -> None:## defines the constructor
        self.num_data_prefetch_queue = num_data_prefetch_queue## sets the num_data_prefetch_queue
        super(PrefetchDataLoader, self).__init__(**kwargs)## calls the parent constructor

    def __iter__(self):
        return PrefetchGenerator(super().__iter__(), self.num_data_prefetch_queue)


class CPUPrefetcher:
    """Use the CPU side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler,
            and provides an iterable over the given dataset.
    """

    def __init__(self, dataloader) -> None:## defines the constructor
        self.original_dataloader = dataloader## sets the original_dataloader
        self.data = iter(dataloader)## sets the data

    def next(self):## defines the next method
        try:## tries to
            return next(self.data)## return the next item
        except StopIteration:
            return None

    def reset(self):## defines the reset method
        self.data = iter(self.original_dataloader)

    def __len__(self) -> int:## defines the __len__ method
        return len(self.original_dataloader)


class CUDAPrefetcher:## defines the CUDAPrefetcher class
    """Use the CUDA side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader, device: torch.device):## defines the constructor
        self.batch_data = None## sets the batch_data to None
        self.original_dataloader = dataloader## sets the original_dataloader
        self.device = device## sets the device

        self.data = iter(dataloader)## sets the data to the iterator of the dataloader 
        self.stream = torch.cuda.Stream()## sets the stream to a CUDA stream 
        self.preload()## calls the preload method to preload the data

    def preload(self):
        try:
            self.batch_data = next(self.data)## gets the next batch
        except StopIteration:## if there is no next batch
            self.batch_data = None
            return None
 
        with torch.cuda.stream(self.stream):
            for k, v in self.batch_data.items():
                if torch.is_tensor(v):
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)## moves the tensor to the device

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        self.preload()
        return batch_data

    def reset(self):
        self.data = iter(self.original_dataloader)
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)
