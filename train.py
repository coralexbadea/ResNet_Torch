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
import os
import time

import torch
from torch import nn
from torch import optim## import package with optimization algorithms
from torch.cuda import amp## automatic mixed precision training, providesconvenience methods for mixed precision, where some operations use the torch.float32 data type and other operations use other data types
from torch.optim import lr_scheduler## class that decays the learning rate of each parameter group by gamma every step_size epochs
from torch.optim.swa_utils import AveragedModel## class that allows to compute running averages of the parameters
from torch.utils.data import DataLoader## class that combines a dataset and a sampler, and provides an iterable over the given dataset
from torch.utils.tensorboard import SummaryWriter## class that writes entries directly to event files in the log_dir to be consumed by TensorBoard

import config## import module defined in this project for configurations
import model## import the model module defined in this project
from dataset import CUDAPrefetcher, ImageDataset## import the data sets
from utils import accuracy, load_state_dict, make_directory, save_checkpoint, Summary, AverageMeter, ProgressMeter## import util functions defiend in this project

model_names = sorted(
    name for name in model.__dict__ if name.islower() and not name.startswith("__") and callable(model.__dict__[name]))


def main():## defines the starting point of this script
    # Initialize the number of training epochs
    start_epoch = 0## number of training epochs is initially 0

    # Initialize training network evaluation indicators
    best_acc1 = 0.0## network evaluation indicators are 0.0 at first

    train_prefetcher, valid_prefetcher = load_dataset()## load the datasets
    print(f"Load `{config.model_arch_name}` datasets successfully.")

    resnet_model, ema_resnet_model = build_model()## build the ResNEt model
    print(f"Build `{config.model_arch_name}` model successfully.")

    pixel_criterion = define_loss()## get definition of loss functions
    print("Define all loss functions successfully.")

    optimizer = define_optimizer(resnet_model)## get the optimiser for the model previously created
    print("Define all optimizer functions successfully.")

    scheduler = define_scheduler(optimizer)## get the scheduler for the optimizer
    print("Define all optimizer scheduler functions successfully.")

    print("Check whether to load pretrained model weights...")
    if config.pretrained_model_weights_path:## if the pretrained model has existing weights
        resnet_model, ema_resnet_model, start_epoch, best_acc1, optimizer, scheduler = load_state_dict(## load the model weights
            resnet_model,## load the weights on the model
            config.pretrained_model_weights_path,## specify the model weights path to know where to load from
            ema_resnet_model,## the ema model
            start_epoch,## starting epoch
            best_acc1,## best accuracy
            optimizer,## optimizer
            scheduler)## and scheduler, all to be used for different computations
        print(f"Loaded `{config.pretrained_model_weights_path}` pretrained model weights successfully.")
    else:
        print("Pretrained model weights not found.")

    print("Check whether the pretrained model is restored...")
    if config.resume:## undefined
        resnet_model, ema_resnet_model, start_epoch, best_acc1, optimizer, scheduler = load_state_dict(## undefined
            resnet_model,## undefined
            config.pretrained_model_weights_path,## undefined
            ema_resnet_model,## undefined
            start_epoch,## undefined
            best_acc1,## undefined
            optimizer,## undefined
            scheduler,## undefined
            "resume")
        print("Loaded pretrained generator model weights.")
    else:
        print("Resume training model not found. Start training from scratch.")

    # Create a experiment results
    samples_dir = os.path.join("samples", config.exp_name)## create path for sample directory
    results_dir = os.path.join("results", config.exp_name)## create path fot results directory
    make_directory(samples_dir)## make sample directory
    make_directory(results_dir)## make results directory

    # Create training process log file
    writer = SummaryWriter(os.path.join("samples", "logs", config.exp_name))## init the writer to make logging in files possible

    # Initialize the gradient scaler
    scaler = amp.GradScaler()## undefined

    for epoch in range(start_epoch, config.epochs):## iterate throung all the epochs 
        train(resnet_model, ema_resnet_model, train_prefetcher, pixel_criterion, optimizer, epoch, scaler, writer)## train the model each time
        acc1 = validate(ema_resnet_model, valid_prefetcher, epoch, writer, "Valid")## validate the result and save the accuracy
        print("\n")

        # Update LR
        scheduler.step()## update the scheduler

        # Automatically save the model with the highest index
        is_best = acc1 > best_acc1## iscalcualte if the currect accuracy is the best so far
        is_last = (epoch + 1) == config.epochs## calcualte if this is the last epoch
        best_acc1 = max(acc1, best_acc1)## update best accuracy so fat
        save_checkpoint({"epoch": epoch + 1,## save current epoch in the checkpoint
                         "best_acc1": best_acc1,## save best accuracy in the checkpoint
                         "state_dict": resnet_model.state_dict(),## save the state dictionary of the model in the checkpoint
                         "ema_state_dict": ema_resnet_model.state_dict(),## save the state dictionary of the ema model in the checkpoint
                         "optimizer": optimizer.state_dict(),## save optimizer in the checkpoint
                         "scheduler": scheduler.state_dict()},## and the scheduler
                        f"epoch_{epoch + 1}.pth.tar",
                        samples_dir,
                        results_dir,
                        is_best,
                        is_last)


def load_dataset() -> [CUDAPrefetcher, CUDAPrefetcher]:## function that load the data sets
    # Load train, test and valid datasets
    train_dataset = ImageDataset(config.train_image_dir,## provide the dirrectory with the training image
                                 config.image_size,## provide the image size
                                 config.model_mean_parameters,## provide the model mean parameters for the mean deviation on tensor later
                                 config.model_std_parameters,## and the standars parameters for the standard deviation on tensor later
                                 "Train")## set the mod in training
    valid_dataset = ImageDataset(config.valid_image_dir,
                                 config.image_size,
                                 config.model_mean_parameters,
                                 config.model_std_parameters,
                                 "Valid")

    # Generator all dataloader
    train_dataloader = DataLoader(train_dataset,## give the train data set as the data set of the data loader
                                  batch_size=config.batch_size,## how many samples per batch to load? parameter taken from config file
                                  shuffle=True,## true, to reshuffle data at each epoch
                                  num_workers=config.num_workers,## number of processes to use for data loading, parameter from config file
                                  pin_memory=True,## tell that the data loader will copy Tensors into device/CUDA pinned memory before returning them
                                  drop_last=True,## drop the last incomplete batch
                                  persistent_workers=True)## the data loader will not shutdown the worker processes after a dataset has been consumed once
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=False,
                                  num_workers=config.num_workers,
                                  pin_memory=True,
                                  drop_last=False,
                                  persistent_workers=True)

    # Place all data on the preprocessing data loader
    train_prefetcher = CUDAPrefetcher(train_dataloader, config.device)## use the CUDA prefetcher as the trining prefetcher
    valid_prefetcher = CUDAPrefetcher(valid_dataloader, config.device)## and as the validation prefetcher

    return train_prefetcher, valid_prefetcher## return the prefetchers


def build_model() -> [nn.Module, nn.Module]:## function to build and return the model
    # __dict__ is an attribute of objects, it is a dictionary that stores the attributes and their corresponding values for an object
    resnet_model = model.__dict__[config.model_arch_name](num_classes=config.model_num_classes)## takes the resnet18 attribute(in this case, function) from the model
    resnet_model = resnet_model.to(device=config.device, memory_format=torch.channels_last)## move the model and its associated parameters to a cuda device, enabling computations to be performed on that devic

    """The EMA technique involves maintaining a weighted average of the model's parameters over time. 
        It helps to stabilize the training process, reduce the impact of noisy updates, and improve the generalization ability of the model. 
    """

    ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: (1 - config.model_ema_decay) * averaged_model_parameter + config.model_ema_decay * model_parameter## undefined
    ema_resnet_model = AveragedModel(resnet_model, avg_fn=ema_avg)## init the ema model
    return resnet_model, ema_resnet_model## return the ResNet model and the ema model


def define_loss() -> nn.CrossEntropyLoss:
    criterion = nn.CrossEntropyLoss(label_smoothing=config.loss_label_smoothing)## this function is used to measure the discrepancy or difference between predicted class probabilities and the true class labels
    criterion = criterion.to(device=config.device, memory_format=torch.channels_last)## move the loss criterion and its associated parameters to a cuda device, enabling computations to be performed on that devic

    return criterion


def define_optimizer(model) -> optim.SGD:
    optimizer = optim.SGD(model.parameters(),
                          lr=config.model_lr,
                          momentum=config.model_momentum,
                          weight_decay=config.model_weight_decay)

    return optimizer


def define_scheduler(optimizer: optim.SGD) -> lr_scheduler.CosineAnnealingWarmRestarts:
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,## this function sets the learning rate of each parameter group using a cosine annealing schedule, this parameter is the wrapped optimizer to use for other operations
                                                         config.lr_scheduler_T_0,
                                                         config.lr_scheduler_T_mult,
                                                         config.lr_scheduler_eta_min)

    return scheduler


def train(## function used to train the model
        model: nn.Module,## model to train
        ema_model: nn.Module,## ema model 
        train_prefetcher: CUDAPrefetcher,## the train prefetcher
        criterion: nn.CrossEntropyLoss,## loss criterion
        optimizer: optim.Adam,## optimizer object
        epoch: int,## which epoch are we at
        scaler: amp.GradScaler,## scaler
        writer: SummaryWriter## writer
) -> None:
    # Calculate how many batches of data are in each Epoch
    batches = len(train_prefetcher)
    # Print information of progress bar during training
    batch_time = AverageMeter("Time", ":6.3f")## print the batch time meter
    data_time = AverageMeter("Data", ":6.3f")## print the data time meter
    losses = AverageMeter("Loss", ":6.6f")## print the losses meter
    acc1 = AverageMeter("Acc@1", ":6.2f")## print the accuracy 1 meter
    acc5 = AverageMeter("Acc@5", ":6.2f")## print the accuracy 5 meter
    progress = ProgressMeter(batches,
                             [batch_time, data_time, losses, acc1, acc5],
                             prefix=f"Epoch: [{epoch + 1}]")

    # Put the generative network model in training mode
    model.train()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    train_prefetcher.reset()## reset the training prefetcher
    batch_data = train_prefetcher.next()## initialize it again with the next item in the iteration

    # Get the initialization training time
    end = time.time()## initialization training time

    while batch_data is not None:## if the batch data is present
        # Calculate the time it takes to load a batch of data
        data_time.update(time.time() - end)## compute time it took to load the batch data

        # Transfer in-memory data to CUDA devices to speed up training
        images = batch_data["image"].to(device=config.device, memory_format=torch.channels_last, non_blocking=True)## transfer image data to cuda devices
        target = batch_data["target"].to(device=config.device, non_blocking=True)## tranfer target data to cuda devices

        # Get batch size
        batch_size = images.size(0)## get the size of the batch

        # Initialize generator gradients
        model.zero_grad(set_to_none=True)## initialize gradient

        # Mixed precision training
        with amp.autocast():
            output = model(images)## get the output
            loss = config.loss_weights * criterion(output, target)## compute the loss

        # Backpropagation
        scaler.scale(loss).backward()## scale the loss backwards to obtain the back propagation
        # update generator weights
        scaler.step(optimizer)## optimize the scaler step
        scaler.update()## update the weights

        # Update EMA
        ema_model.update_parameters(model)## update the ema as well

        # measure accuracy and record loss
        top1, top5 = accuracy(output, target, topk=(1, 5))## compute top1 and top5 accuracies
        losses.update(loss.item(), batch_size)## update losses for batch size
        acc1.update(top1[0].item(), batch_size)## update accuracy 1 for batch size
        acc5.update(top5[0].item(), batch_size)## update accuracy 5 for batch size

        # Calculate the time it takes to fully train a batch of data
        batch_time.update(time.time() - end)## calculate tme neede dto train a batch
        end = time.time()## reset timer

        # Write the data during training to the training log file
        if batch_index % config.train_print_frequency == 0:## undefined
            # Record loss during training and output to file
            writer.add_scalar("Train/Loss", loss.item(), batch_index + epoch * batches + 1)## write the training loss information in the log file
            progress.display(batch_index + 1)## move to the next batch

        # Preload the next batch of data
        batch_data = train_prefetcher.next()## preload the next batch of data

        # Add 1 to the number of data batches to ensure that the terminal prints data normally
        batch_index += 1## update the batch index to fit the new loaded batch


def validate(
        ema_model: nn.Module,
        data_prefetcher: CUDAPrefetcher,
        epoch: int,
        writer: SummaryWriter,
        mode: str
) -> float:
    # Calculate how many batches of data are in each Epoch
    batches = len(data_prefetcher)
    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    acc1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
    acc5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE)
    progress = ProgressMeter(batches, [batch_time, acc1, acc5], prefix=f"{mode}: ")

    # Put the exponential moving average model in the verification mode
    ema_model.eval()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    data_prefetcher.reset()
    batch_data = data_prefetcher.next()

    # Get the initialization test time
    end = time.time()

    with torch.no_grad():## any operations that occur within this block will not have their gradients computed or stored for later use in gradient-based optimization algorithms (backpropagation)
        while batch_data is not None:## is batch data is present
            # Transfer in-memory data to CUDA devices to speed up training
            images = batch_data["image"].to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
            target = batch_data["target"].to(device=config.device, non_blocking=True)

            # Get batch size
            batch_size = images.size(0)

            # Inference
            output = ema_model(images)

            # measure accuracy and record loss
            top1, top5 = accuracy(output, target, topk=(1, 5))
            acc1.update(top1[0].item(), batch_size)
            acc5.update(top5[0].item(), batch_size)

            # Calculate the time it takes to fully train a batch of data
            batch_time.update(time.time() - end)
            end = time.time()

            # Write the data during training to the training log file
            if batch_index % config.valid_print_frequency == 0:
                progress.display(batch_index + 1)

            # Preload the next batch of data
            batch_data = data_prefetcher.next()

            # Add 1 to the number of data batches to ensure that the terminal prints data normally
            batch_index += 1

    # print metrics
    progress.display_summary()

    if mode == "Valid" or mode == "Test":
        writer.add_scalar(f"{mode}/Acc@1", acc1.avg, epoch + 1)## write the accuracy information in the log file
    else:
        raise ValueError("Unsupported mode, please use `Valid` or `Test`.")

    return acc1.avg


if __name__ == "__main__":## only run the code inside the if statement when the program is run directly by the Python interprete
    main()## run the main function
