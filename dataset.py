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
from torchvision.datasets.folder import find_classes## import the find_classes method from the specified module 
from torchvision.transforms import TrivialAugmentWide## import the TrivialAugmentWide method or class from the specified module  

import imgproc## imports imgproc which is a small python image processing package compatible with Tensorflow/Pytorch

__all__ = [ # this is a list of strings defining what symbols in a module will be exported when from <module> import * is used on the module.
    "ImageDataset",## means that ImageDataset will be imported when we import * from the current module in a different file
    "PrefetchGenerator", "PrefetchDataLoader", "CPUPrefetcher", "CUDAPrefetcher",
]

# Image formats supported by the image processing library
IMG_EXTENSIONS = ("jpg", "jpeg", "png", "ppm", "bmp", "pgm", "tif", "tiff", "webp")## defines an array with possible image extensions

# The delimiter is not the same between different platforms
if sys.platform == "win32":## if the Python interpreter is running on a Windows platform...
    delimiter = "\\"## defines the delimiter in the image file path for windows
else:## else branch of the if statemenet
    delimiter = "/"## defines the delimiter in the image file path for other systems


class ImageDataset(Dataset):# defines the class ImageDataset which inherits from 
    """Define training/valid dataset loading methods.

    Args:
        image_dir (str): Train/Valid dataset address.
        image_size (int): Image size.
        mode (str): Data set loading method, the training data set is for data enhancement,
            and the verification data set is not for data enhancement.
    """

    def __init__(self, image_dir: str, image_size: int, mean: list, std: list, mode: str) -> None:
        super(ImageDataset, self).__init__()## call the init of the parent class in the init of the current class
        # Iterate over all image paths
        self.image_file_paths = glob(f"{image_dir}/*/*")## search for files that match the specific file pattern
        # Form image class label pairs by the folder where the image is located
        _, self.class_to_idx = find_classes(image_dir)## tries to find the folders named as the image_dir variable in the data set
        self.image_size = image_size## set the image_size class parameter with the value from the initializer
        self.mode = mode## set the mode class parameter with the value from the initializer
        self.delimiter = delimiter## set the delimiter class parameter with the value from the initializer

        if self.mode == "Train":## if we are running in training mode...
            # Use PyTorch's own data enhancement to enlarge and enhance data
            self.pre_transform = transforms.Compose([## composes the following transformations together and stored them in the pre_transform field
                transforms.RandomResizedCrop(self.image_size),## crops a random portion of image and resize it to the given size
                TrivialAugmentWide(),## init a TrivialAugmentWide object and give as parameter
                transforms.RandomRotation([0, 270]),## rotates the image with 270 degrees
                transforms.RandomHorizontalFlip(0.5),## flips the image horizontaly with the probability of 0.5
                transforms.RandomVerticalFlip(0.5),## flips the image verticaly with the probability of 0.5
            ])
        elif self.mode == "Valid" or self.mode == "Test":## else, if we are in validation or testing mode...
            # Use PyTorch's own data enhancement to enlarge and enhance data
            self.pre_transform = transforms.Compose([## again, composes the following transformations together
                transforms.Resize(256),## resize the input image to the given size
                transforms.CenterCrop([self.image_size, self.image_size]),## crops the given image at the center to the given size
            ])
        else:
            raise "Unsupported data read type. Please use `Train` or `Valid` or `Test`"

        self.post_transform = transforms.Compose([## and again, composes the following transformations together
            transforms.ConvertImageDtype(torch.float),## converts the image to the float dtype and scale the values accordingly
            transforms.Normalize(mean, std)## normalizes a tensor image with mean and standard deviation
        ])

    def __getitem__(self, batch_index: int) -> [torch.Tensor, int]:##  __getitem__ enabling the Python objects to behave like sequences or containers e.g lists, dictionaries, and tuples
        image_dir, image_name = self.image_file_paths[batch_index].split(self.delimiter)[-2:]## split the image file path based on the system delimiter astablised earlier
        # Read a batch of image data
        if image_name.split(".")[-1].lower() in IMG_EXTENSIONS:## check if the extension of the current image is in the defined extensions
            image = cv2.imread(self.image_file_paths[batch_index])## reads the image using openCV
            target = self.class_to_idx[image_dir]## gets the images in the image_dir folder set earlier
        else:
            raise ValueError(f"Unsupported image extensions, Only support `{IMG_EXTENSIONS}`, "## throws an exception if the extension of the image is not a supported one
                             "please check the image file extensions.")

        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)## converts the image from the BGR color space to the RGB color space using openCV

        # OpenCV convert PIL
        image = Image.fromarray(image)## creates an image memory from an object exporting the array interface

        # Data preprocess
        image = self.pre_transform(image)## applies the previously composed pre transformation on the image

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        tensor = imgproc.image_to_tensor(image, False, False)## converts the image into a tensor

        # Data postprocess
        tensor = self.post_transform(tensor)## applies the previously composed post transformation on the tensor

        return {"image": tensor, "target": target}## returns the composed tensor and target

    def __len__(self) -> int:# this method is a special method in Python that allows an object to define its length or size
        return len(self.image_file_paths)## defines the length of image_file_paths


class PrefetchGenerator(threading.Thread):
    """A fast data prefetch generator.

    Args:
        generator: Data generator.
        num_data_prefetch_queue (int): How many early data load queues.## undefined
    """

    def __init__(self, generator, num_data_prefetch_queue: int) -> None:## init function of the PrefetchGenerator class
        threading.Thread.__init__(self)## init the threading object
        self.queue = queue.Queue(num_data_prefetch_queue)
        self.generator = generator## sets the objects generator to the one given as argument
        self.daemon = True## sets the objects deamon to true
        self.start()

    def run(self) -> None:
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __next__(self):## an iterator object implements __next__, which is expected to return the next element of the iterable object
        next_item = self.queue.get()## get the next element from the queue
        if next_item is None:## if there is no next element...
            raise StopIteration## stop the iteration
        return next_item

    def __iter__(self):
        return self


class PrefetchDataLoader(DataLoader):## define the class PrefetchDataLoader which inherits from DataLoader
    """A fast data prefetch dataloader.

    Args:
        num_data_prefetch_queue (int): How many early data load queues.
        kwargs (dict): Other extended parameters.
    """

    def __init__(self, num_data_prefetch_queue: int, **kwargs) -> None:## init function of the PrefetchDataLoader class
        self.num_data_prefetch_queue = num_data_prefetch_queue## sets the objects num_data_prefetch_queue to the one given as argument
        super(PrefetchDataLoader, self).__init__(**kwargs)## calles the init of the super class

    def __iter__(self):
        return PrefetchGenerator(super().__iter__(), self.num_data_prefetch_queue)


class CPUPrefetcher:
    """Use the CPU side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler,
            and provides an iterable over the given dataset.
    """

    def __init__(self, dataloader) -> None:## init function of the CPUPrefetcher class
        self.original_dataloader = dataloader## sets the objects original_dataloader to the one given as argument
        self.data = iter(dataloader)## iter returns an iterator for the given argument (dataloader) and it sets the data of the current object with the iterator

    def next(self):## function the returns the next item in the iterator
        try:## try blocks let you test blocks of core for error and throw if they find any
            return next(self.data)## tries to return the next item from the objects daya
        except StopIteration:
            return None

    def reset(self):## set all members (of an iterator??) to their initial value
        self.data = iter(self.original_dataloader)

    def __len__(self) -> int:## this method is a special method in Python that allows an object to define its length or size
        return len(self.original_dataloader)


class CUDAPrefetcher:## define the CUDAPrefetcher, this time not inheriting from anything
    """Use the CUDA side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader, device: torch.device):## init function of the CUDAPrefetcher class
        self.batch_data = None## sets the objects batch_data to None
        self.original_dataloader = dataloader## sets the objects original_dataloader to the dataloader variable given as argument
        self.device = device## sets the objects device to the device variable given as argument

        self.data = iter(dataloader)## iter returns an iterator for the given argument (dataloader) and it sets the data of the current object with the iterator
        self.stream = torch.cuda.Stream()## creates a CUDA stream (which has some specific characteristics) object and assigns it to the stream attribute of the current object
        self.preload()## calles the preload method (defined below) on the object

    def preload(self):
        try:
            self.batch_data = next(self.data)## set the batch_data of the object to the next item in the objects data
        except StopIteration:## if previous operation could not be performed, catch the error
            self.batch_data = None
            return None

        with torch.cuda.stream(self.stream):
            for k, v in self.batch_data.items():
                if torch.is_tensor(v):
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)## moves the batch_data[k] tensor to the specified device

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
